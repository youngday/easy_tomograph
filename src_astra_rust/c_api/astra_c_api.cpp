// ============================================================================
// astra_c_api.cpp — C shim: 将 ASTRA C++/CUDA 接口包装为 extern "C",
// 供 Rust 通过 FFI 调用。算法与 src_astra_cpp 完全一致:
//   * FP3D / FDK(hann)         — ASTRA CUDA 内核
//   * GPU 常驻 OS-SART         — 复刻 SIRT3D_CUDA 更新公式 (体积驻留 GPU)
//   * TV 去噪                  — 自写 CUDA 内核 (tv_kernel.cu)
// 布局常量与 src_astra_cpp 相同: N=512, nz=32, 180 角度, 10 子集
// ============================================================================
#include <astra/Globals.h>
#include <astra/Config.h>
#include <astra/ConeVecProjectionGeometry3D.h>
#include <astra/VolumeGeometry3D.h>
#include <astra/Data3D.h>
#include <astra/CudaProjector3D.h>
#include <astra/CudaFDKAlgorithm3D.h>
#include <astra/CudaForwardProjectionAlgorithm3D.h>
#include <astra/Filters.h>
#include <astra/GeometryUtil3D.h>
#include <astra/cuda/3d/astra3d.h>
#include <astra/cuda/3d/mem3d.h>
#include <astra/cuda/3d/arith3d.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

extern "C" int tv_denoise_cuda(const float* v, float* out,
                               int nz, int N, float beta, float w_z, float eps);

namespace {

constexpr int N = 512;
constexpr int nz = 32;
constexpr int n_angles = 180;
constexpr int n_subsets = 10;
constexpr int sub_size = n_angles / n_subsets;  // 18
constexpr int n_det_row = 64;
constexpr int n_det_col = (int)std::ceil(N * std::sqrt(2.0));  // 725
constexpr double DSO = 1000.0;
constexpr double DSD_det = 500.0;
constexpr double det_pix = 1.0;
constexpr double pitch_mm = 16.0;
constexpr size_t nvol = (size_t)nz * N * N;
constexpr size_t nsino = (size_t)n_det_row * n_angles * n_det_col;

void set_err(char* err, size_t err_len, const std::string& msg) {
    if (err && err_len) {
        size_t n = msg.size() < err_len - 1 ? msg.size() : err_len - 1;
        std::memcpy(err, msg.c_str(), n);
        err[n] = '\0';
    }
}

// 共享 cone_vec 几何 (探测器"角落"约定, 与 src_astra_cpp 一致)
std::vector<astra::SConeProjection> build_vectors(bool helical) {
    std::vector<astra::SConeProjection> vecs(n_angles);
    for (int i = 0; i < n_angles; ++i) {
        double th = 2.0 * M_PI * i / n_angles;
        double c = std::cos(th), s = std::sin(th);
        double z_src = helical ? pitch_mm * (th / (2.0 * M_PI) - 0.5) : 0.0;
        double dcx = -DSD_det * s, dcy = DSD_det * c, dcz = z_src;
        double ux = det_pix * c, uy = det_pix * s, uz = 0.0;
        double vx = 0.0, vy = 0.0, vz = det_pix;
        double sx = dcx - 0.5 * n_det_row * vx - 0.5 * n_det_col * ux;
        double sy = dcy - 0.5 * n_det_row * vy - 0.5 * n_det_col * uy;
        double sz = dcz - 0.5 * n_det_row * vz - 0.5 * n_det_col * uz;
        vecs[i] = {DSO * s, -DSO * c, z_src, sx, sy, sz, ux, uy, uz, vx, vy, vz};
    }
    return vecs;
}

// ---- 几何上下文 (FP / FDK) ----
struct GeomCtx {
    bool helical;
    std::unique_ptr<astra::CConeVecProjectionGeometry3D> projGeom;
    astra::CVolumeGeometry3D volGeom{N, N, nz};
    std::unique_ptr<astra::CCudaProjector3D> projector;
    astra::SFilterConfig filt;
    explicit GeomCtx(bool h) : helical(h) {
        auto v = build_vectors(h);
        projGeom = std::make_unique<astra::CConeVecProjectionGeometry3D>(
            n_angles, n_det_row, n_det_col, std::move(v));
        projector = std::make_unique<astra::CCudaProjector3D>(*projGeom, volGeom);
        filt.m_eType = astra::FILTER_HANN;  // 与 Python FilterType=hann 对齐
        filt.m_fD = 1.0f;
        filt.m_fParameter = -1.0f;
    }
};

// ---- GPU 常驻 OS-SART (与 src_astra_cpp/src/sart_gpu.cpp 逐行一致) ----
void free_gpu_data(astra::CData3D*& p) {
    if (p) {
        astraCUDA3d::freeGPUMemory(p);
        delete p;
        p = nullptr;
    }
}

struct Subset {
    std::unique_ptr<astra::CConeVecProjectionGeometry3D> geom;
    astra::Geometry3DParameters geom3;
    astra::CData3D* proj = nullptr;
    astra::CData3D* tmpProj = nullptr;
    astra::CData3D* lineWeight = nullptr;
    astra::CData3D* pixelWeight = nullptr;
    Subset() = default;
    Subset(const Subset&) = delete;
    Subset& operator=(const Subset&) = delete;
    Subset(Subset&& o) noexcept
        : geom(std::move(o.geom)), geom3(std::move(o.geom3)), proj(o.proj), tmpProj(o.tmpProj),
          lineWeight(o.lineWeight), pixelWeight(o.pixelWeight) {
        o.proj = o.tmpProj = o.lineWeight = o.pixelWeight = nullptr;
    }
    ~Subset() {
        free_gpu_data(proj);
        free_gpu_data(tmpProj);
        free_gpu_data(lineWeight);
        free_gpu_data(pixelWeight);
    }
};

void fill_subset_sino(astra::CFloat32ProjectionData3D* dst, const float* sino_noisy, int i) {
    float* d = dst->getFloat32Memory();
    for (int row = 0; row < n_det_row; ++row)
        for (int a = 0; a < sub_size; ++a)
            std::memcpy(d + ((size_t)row * sub_size + a) * n_det_col,
                        sino_noisy + ((size_t)row * n_angles + i * sub_size + a) * n_det_col,
                        n_det_col * sizeof(float));
}

struct SARTGpu {
    astra::CVolumeGeometry3D volGeom{N, N, nz};
    astra::CData3D* vol = nullptr;
    astra::CData3D* tmpVol = nullptr;
    std::vector<Subset> subsets;
    astraCUDA3d::SProjectorParams3D params;

    ~SARTGpu() {
        free_gpu_data(vol);
        free_gpu_data(tmpVol);
    }

    bool init(bool helical, const float* sino_noisy, std::string& err) {
        std::unique_ptr<astra::CFloat32VolumeData3D> volCpu(
            astra::createCFloat32VolumeData3DMemory(volGeom));
        vol = astraCUDA3d::createGPUData3DLike(volCpu.get());
        tmpVol = astraCUDA3d::createGPUData3DLike(volCpu.get());
        if (!vol || !tmpVol) { err = "体积 GPU 缓冲分配失败"; return false; }

        for (int i = 0; i < n_subsets; ++i) {
            Subset s;
            auto full = build_vectors(helical);
            std::vector<astra::SConeProjection> subvecs;
            subvecs.reserve(sub_size);
            for (int a = i * sub_size; a < (i + 1) * sub_size; ++a)
                subvecs.push_back(full[a]);
            s.geom = std::make_unique<astra::CConeVecProjectionGeometry3D>(
                sub_size, n_det_row, n_det_col, std::move(subvecs));
            s.geom3 = astra::convertAstraGeometry(&volGeom, s.geom.get());
            if (!s.geom3.isCone()) { err = "子集几何转换失败"; return false; }

            std::unique_ptr<astra::CFloat32ProjectionData3D> sinoCpu(
                astra::createCFloat32ProjectionData3DMemory(*s.geom));
            fill_subset_sino(sinoCpu.get(), sino_noisy, i);

            s.proj = astraCUDA3d::createGPUData3DLike(sinoCpu.get());
            s.tmpProj = astraCUDA3d::createGPUData3DLike(sinoCpu.get());
            s.lineWeight = astraCUDA3d::createGPUData3DLike(sinoCpu.get());
            s.pixelWeight = astraCUDA3d::createGPUData3DLike(volCpu.get());
            if (!s.proj || !s.tmpProj || !s.lineWeight || !s.pixelWeight) {
                err = "子集 GPU 缓冲分配失败"; return false;
            }
            if (!astraCUDA3d::copyToGPUMemory(sinoCpu.get(), s.proj)) {
                err = "子集 sinogram 上传失败"; return false;
            }
            sinoCpu.reset();

            astraCUDA3d::SProjectorParams3D p = params;
            p.volScale = s.geom3.getVolScale();
            astraCUDA3d::zeroGPUMemory(s.lineWeight);
            astraCUDA3d::processVol3D<astraCUDA3d::opSet>(tmpVol, 1.0f);
            if (!astraCUDA3d::FP(s.lineWeight, tmpVol, s.geom3, p)) {
                err = "lineWeight FP 失败"; return false;
            }
            astraCUDA3d::processVol3D<astraCUDA3d::opInvert>(s.lineWeight);
            astraCUDA3d::zeroGPUMemory(s.pixelWeight);
            astraCUDA3d::processVol3D<astraCUDA3d::opSet>(s.tmpProj, 1.0f);
            if (!astraCUDA3d::BP(s.tmpProj, s.pixelWeight, s.geom3, p)) {
                err = "pixelWeight BP 失败"; return false;
            }
            astraCUDA3d::processVol3D<astraCUDA3d::opInvert>(s.pixelWeight);

            subsets.push_back(std::move(s));
        }
        params.volScale = subsets[0].geom3.getVolScale();
        return true;
    }

    bool run(const float* vol_in, int n_epochs, float* vol_out, std::string& err) {
        {
            std::unique_ptr<astra::CFloat32VolumeData3D> volCpu(
                astra::createCFloat32VolumeData3DMemory(volGeom));
            std::memcpy(volCpu->getFloat32Memory(), vol_in, nvol * sizeof(float));
            if (!astraCUDA3d::copyToGPUMemory(volCpu.get(), vol)) {
                err = "体积上传失败"; return false;
            }
        }
        for (int e = 0; e < n_epochs; ++e) {
            for (auto& s : subsets) {
                astraCUDA3d::SProjectorParams3D p = params;
                p.volScale = s.geom3.getVolScale();
                astraCUDA3d::assignGPUMemory(s.tmpProj, s.proj);
                p.fOutputScale = -1.0f;
                if (!astraCUDA3d::FP(s.tmpProj, vol, s.geom3, p)) {
                    err = "残差 FP 失败"; return false;
                }
                astraCUDA3d::processVol3D<astraCUDA3d::opMul>(s.tmpProj, s.lineWeight);
                astraCUDA3d::zeroGPUMemory(tmpVol);
                p.fOutputScale = 1.0f;
                if (!astraCUDA3d::BP(s.tmpProj, tmpVol, s.geom3, p)) {
                    err = "残差 BP 失败"; return false;
                }
                astraCUDA3d::processVol3D<astraCUDA3d::opAddMul>(vol, tmpVol, s.pixelWeight);
            }
        }
        {
            std::unique_ptr<astra::CFloat32VolumeData3D> volCpu(
                astra::createCFloat32VolumeData3DMemory(volGeom));
            if (!astraCUDA3d::copyFromGPUMemory(volCpu.get(), vol)) {
                err = "体积下载失败"; return false;
            }
            std::memcpy(vol_out, volCpu->getFloat32Memory(), nvol * sizeof(float));
        }
        return true;
    }
};

}  // namespace

// ============================================================================
// extern "C" API (Rust FFI)
// ============================================================================
extern "C" {

size_t astra_rs_nvol() { return nvol; }
size_t astra_rs_nsino() { return nsino; }

void* astra_rs_geom_create(bool helical, char* err, size_t err_len) {
    try {
        return new GeomCtx(helical);
    } catch (const std::exception& e) {
        set_err(err, err_len, e.what());
        return nullptr;
    }
}
void astra_rs_geom_free(void* ctx) { delete static_cast<GeomCtx*>(ctx); }

int astra_rs_fp(void* ctx, const float* vol, float* sino_out, char* err, size_t err_len) {
    GeomCtx* g = static_cast<GeomCtx*>(ctx);
    try {
        std::unique_ptr<astra::CFloat32ProjectionData3D> sino(
            astra::createCFloat32ProjectionData3DMemory(*g->projGeom));
        std::memset(sino->getFloat32Memory(), 0, nsino * sizeof(float));
        std::unique_ptr<astra::CFloat32VolumeData3D> vd(
            astra::createCFloat32VolumeData3DMemory(g->volGeom));
        std::memcpy(vd->getFloat32Memory(), vol, nvol * sizeof(float));
        astra::CCudaForwardProjectionAlgorithm3D fp;
        if (!fp.initialize(g->projector.get(), sino.get(), vd.get()) || !fp.run(1)) {
            set_err(err, err_len, "FP3D_CUDA 运行失败"); return 1;
        }
        std::memcpy(sino_out, sino->getFloat32Memory(), nsino * sizeof(float));
        return 0;
    } catch (const std::exception& e) {
        set_err(err, err_len, e.what()); return 1;
    }
}

int astra_rs_fdk(void* ctx, const float* sino, float* vol_out, char* err, size_t err_len) {
    GeomCtx* g = static_cast<GeomCtx*>(ctx);
    try {
        std::unique_ptr<astra::CFloat32ProjectionData3D> sino_d(
            astra::createCFloat32ProjectionData3DMemory(*g->projGeom));
        std::memcpy(sino_d->getFloat32Memory(), sino, nsino * sizeof(float));
        std::unique_ptr<astra::CFloat32VolumeData3D> vol_d(
            astra::createCFloat32VolumeData3DMemory(g->volGeom));
        astra::CCudaFDKAlgorithm3D fdk(g->projector.get(), sino_d.get(), vol_d.get(), g->filt, false);
        if (!fdk.run(1)) { set_err(err, err_len, "FDK_CUDA 运行失败"); return 1; }
        std::memcpy(vol_out, vol_d->getFloat32Memory(), nvol * sizeof(float));
        return 0;
    } catch (const std::exception& e) {
        set_err(err, err_len, e.what()); return 1;
    }
}

void* astra_rs_sart_create(bool helical, const float* sino_noisy, size_t n,
                           char* err, size_t err_len) {
    if (n != nsino) { set_err(err, err_len, "sino_noisy 大小不匹配"); return nullptr; }
    SARTGpu* s = new SARTGpu();
    std::string e;
    if (!s->init(helical, sino_noisy, e)) {
        set_err(err, err_len, e);
        delete s;
        return nullptr;
    }
    return s;
}
void astra_rs_sart_free(void* h) { delete static_cast<SARTGpu*>(h); }

int astra_rs_sart_run(void* h, const float* vol_in, int n_epochs, float* vol_out,
                      char* err, size_t err_len) {
    SARTGpu* s = static_cast<SARTGpu*>(h);
    std::string e;
    if (!s->run(vol_in, n_epochs, vol_out, e)) {
        set_err(err, err_len, e);
        return 1;
    }
    return 0;
}

int astra_rs_tv(float* vol_inout, float beta, float w_z, char* err, size_t err_len) {
    static std::vector<float> buf;
    if (buf.size() != nvol) buf.resize(nvol);
    if (tv_denoise_cuda(vol_inout, buf.data(), nz, N, beta, w_z, 1e-8f) != 0) {
        set_err(err, err_len, "TV CUDA 内核失败");
        return 1;
    }
    std::memcpy(vol_inout, buf.data(), nvol * sizeof(float));
    return 0;
}

}  // extern "C"
