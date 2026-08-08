// ============================================================================
// astra_c_api.cpp — C shim: 将 ASTRA C++/CUDA 接口包装为 extern "C", 供 Rust FFI。
// 仅保留"ASTRA 对象/内核调用"这一层 (C++ 类无法跨 FFI):
//   * 几何构造 (向量由 Rust 传入)
//   * FP / FDK (ASTRA CUDA 算法)
//   * GPU 常驻 OS-SART 原语: upload / 单子集步 / download (循环控制在 Rust)
//   * TV CUDA 内核由 Rust 直接调用 (tv_kernel.cu 导出 extern "C")
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

#include <cuda_runtime.h>

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
constexpr size_t nvol = (size_t)nz * N * N;
constexpr size_t nsino = (size_t)n_det_row * n_angles * n_det_col;

void set_err(char* err, size_t err_len, const std::string& msg) {
    if (err && err_len) {
        size_t n = msg.size() < err_len - 1 ? msg.size() : err_len - 1;
        std::memcpy(err, msg.c_str(), n);
        err[n] = '\0';
    }
}

// 由 Rust 传入的 180×12 f64 向量数组 → SConeProjection
std::vector<astra::SConeProjection> vectors_to_projections(const double* v, size_t n) {
    std::vector<astra::SConeProjection> vecs(n_angles);
    for (int i = 0; i < n_angles; ++i) {
        size_t o = (size_t)i * 12;
        vecs[i] = {v[o + 0], v[o + 1], v[o + 2],  // src
                   v[o + 3], v[o + 4], v[o + 5],  // det corner
                   v[o + 6], v[o + 7], v[o + 8],  // u
                   v[o + 9], v[o + 10], v[o + 11]};  // v
    }
    (void)n;
    return vecs;
}

// ---- 几何上下文 (FP / FDK) ----
struct GeomCtx {
    std::unique_ptr<astra::CConeVecProjectionGeometry3D> projGeom;
    astra::CVolumeGeometry3D volGeom{N, N, nz};
    std::unique_ptr<astra::CCudaProjector3D> projector;
    astra::SFilterConfig filt;
    // 宿主侧数据缓冲 (CPU 内存, 缓存复用): 数据填充/读取由 Rust 编排,
    // shim 只做纯算法调用 (FP/FDK run)
    std::unique_ptr<astra::CFloat32ProjectionData3D> sinoBuf;
    std::unique_ptr<astra::CFloat32VolumeData3D> volBuf;
    GeomCtx(const double* vectors, size_t nv) {
        auto vecs = vectors_to_projections(vectors, nv);
        projGeom = std::make_unique<astra::CConeVecProjectionGeometry3D>(
            n_angles, n_det_row, n_det_col, std::move(vecs));
        sinoBuf.reset(astra::createCFloat32ProjectionData3DMemory(*projGeom));
        volBuf.reset(astra::createCFloat32VolumeData3DMemory(volGeom));
        projector = std::make_unique<astra::CCudaProjector3D>(*projGeom, volGeom);
        filt.m_eType = astra::FILTER_HANN;  // 与 Python FilterType=hann 对齐
        filt.m_fD = 1.0f;
        filt.m_fParameter = -1.0f;
    }
};

// ---- GPU 常驻 OS-SART (权重预计算 + 单子集迭代原语, 循环控制在 Rust) ----
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
    astra::CData3D* vol = nullptr;      // 重建 (GPU 常驻)
    astra::CData3D* tmpVol = nullptr;   // BP 累加缓冲 (GPU)
    std::vector<Subset> subsets;
    astraCUDA3d::SProjectorParams3D params;

    ~SARTGpu() {
        free_gpu_data(vol);
        free_gpu_data(tmpVol);
    }

    // 分配 GPU 缓冲 + 预计算权重; 子集 sinogram 由 Rust 切分后经
    // astra_rs_sart_subset_upload 逐个上传 (数据布局逻辑在 Rust 端)
    bool init(const double* vectors, size_t nv, std::string& err) {
        auto full = vectors_to_projections(vectors, nv);
        std::unique_ptr<astra::CFloat32VolumeData3D> volCpu(
            astra::createCFloat32VolumeData3DMemory(volGeom));
        vol = astraCUDA3d::createGPUData3DLike(volCpu.get());
        tmpVol = astraCUDA3d::createGPUData3DLike(volCpu.get());
        if (!vol || !tmpVol) { err = "体积 GPU 缓冲分配失败"; return false; }

        for (int i = 0; i < n_subsets; ++i) {
            Subset s;
            std::vector<astra::SConeProjection> subvecs;
            subvecs.reserve(sub_size);
            for (int a = i * sub_size; a < (i + 1) * sub_size; ++a)
                subvecs.push_back(full[a]);
            s.geom = std::make_unique<astra::CConeVecProjectionGeometry3D>(
                sub_size, n_det_row, n_det_col, std::move(subvecs));
            s.geom3 = astra::convertAstraGeometry(&volGeom, s.geom.get());
            if (!s.geom3.isCone()) { err = "子集几何转换失败"; return false; }

            // 仅用于 createGPUData3DLike 的形状; 内容由 Rust 上传
            std::unique_ptr<astra::CFloat32ProjectionData3D> sinoCpu(
                astra::createCFloat32ProjectionData3DMemory(*s.geom));
            std::memset(sinoCpu->getFloat32Memory(), 0,
                        (size_t)n_det_row * sub_size * n_det_col * sizeof(float));

            s.proj = astraCUDA3d::createGPUData3DLike(sinoCpu.get());
            s.tmpProj = astraCUDA3d::createGPUData3DLike(sinoCpu.get());
            s.lineWeight = astraCUDA3d::createGPUData3DLike(sinoCpu.get());
            s.pixelWeight = astraCUDA3d::createGPUData3DLike(volCpu.get());
            if (!s.proj || !s.tmpProj || !s.lineWeight || !s.pixelWeight) {
                err = "子集 GPU 缓冲分配失败"; return false;
            }
            sinoCpu.reset();

            // 权重预计算 (与 CudaSirtAlgorithm3D::precomputeWeights 一致)
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

    // 一次子集迭代 (复刻 SIRT3D_CUDA 更新公式), 体积驻留 GPU
    bool subset_step(int i, std::string& err) {
        if (i < 0 || i >= n_subsets) { err = "子集索引越界"; return false; }
        Subset& s = subsets[i];
        astraCUDA3d::SProjectorParams3D p = params;
        p.volScale = s.geom3.getVolScale();
        // tmpProj = b
        astraCUDA3d::assignGPUMemory(s.tmpProj, s.proj);
        // tmpProj = b - A·v
        p.fOutputScale = -1.0f;
        if (!astraCUDA3d::FP(s.tmpProj, vol, s.geom3, p)) { err = "残差 FP 失败"; return false; }
        // tmpProj = lineWeight * (b - A·v)
        astraCUDA3d::processVol3D<astraCUDA3d::opMul>(s.tmpProj, s.lineWeight);
        // tmpVol = A^T·tmpProj
        astraCUDA3d::zeroGPUMemory(tmpVol);
        p.fOutputScale = 1.0f;
        if (!astraCUDA3d::BP(s.tmpProj, tmpVol, s.geom3, p)) { err = "残差 BP 失败"; return false; }
        // v += pixelWeight * tmpVol
        astraCUDA3d::processVol3D<astraCUDA3d::opAddMul>(vol, tmpVol, s.pixelWeight);
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

// vectors: Rust 端生成的 180×12 f64 向量数组
void* astra_rs_geom_create(const double* vectors, size_t nv, char* err, size_t err_len) {
    if (nv != (size_t)n_angles * 12) {
        set_err(err, err_len, "向量数量不匹配");
        return nullptr;
    }
    try {
        GeomCtx* g = new GeomCtx(vectors, nv);
        if (!g->sinoBuf || !g->volBuf) {
            set_err(err, err_len, "宿主侧数据缓冲分配失败");
            delete g;
            return nullptr;
        }
        return g;
    } catch (const std::exception& e) {
        set_err(err, err_len, e.what());
        return nullptr;
    }
}
void astra_rs_geom_free(void* ctx) { delete static_cast<GeomCtx*>(ctx); }

// 宿主侧数据缓冲指针: kind=0 → sinogram, kind=1 → volume (Rust 直接读写)
float* astra_rs_data_ptr(void* ctx, int kind) {
    GeomCtx* g = static_cast<GeomCtx*>(ctx);
    return kind == 0 ? g->sinoBuf->getFloat32Memory() : g->volBuf->getFloat32Memory();
}

// 纯算法调用: 数据已由 Rust 填充进 ctx 的宿主侧缓冲
int astra_rs_fp_run(void* ctx, char* err, size_t err_len) {
    GeomCtx* g = static_cast<GeomCtx*>(ctx);
    try {
        astra::CCudaForwardProjectionAlgorithm3D fp;
        if (!fp.initialize(g->projector.get(), g->sinoBuf.get(), g->volBuf.get()) || !fp.run(1)) {
            set_err(err, err_len, "FP3D_CUDA 运行失败"); return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        set_err(err, err_len, e.what()); return 1;
    }
}

int astra_rs_fdk_run(void* ctx, char* err, size_t err_len) {
    GeomCtx* g = static_cast<GeomCtx*>(ctx);
    try {
        astra::CCudaFDKAlgorithm3D fdk(g->projector.get(), g->sinoBuf.get(), g->volBuf.get(), g->filt, false);
        if (!fdk.run(1)) { set_err(err, err_len, "FDK_CUDA 运行失败"); return 1; }
        return 0;
    } catch (const std::exception& e) {
        set_err(err, err_len, e.what()); return 1;
    }
}

void* astra_rs_sart_create(const double* vectors, size_t nv, char* err, size_t err_len) {
    if (nv != (size_t)n_angles * 12) {
        set_err(err, err_len, "向量数量不匹配");
        return nullptr;
    }
    SARTGpu* s = new SARTGpu();
    std::string e;
    if (!s->init(vectors, nv, e)) {
        set_err(err, err_len, e);
        delete s;
        return nullptr;
    }
    return s;
}
void astra_rs_sart_free(void* h) { delete static_cast<SARTGpu*>(h); }

// 上传第 i 个子集的 sinogram (Rust 端已完成子集切分, 此处仅做 GPU 上传)
int astra_rs_sart_subset_upload(void* h, int i, const float* sino_subset, size_t n,
                                char* err, size_t err_len) {
    SARTGpu* s = static_cast<SARTGpu*>(h);
    if (i < 0 || i >= n_subsets) {
        set_err(err, err_len, "子集索引越界"); return 1;
    }
    const size_t expect = (size_t)n_det_row * sub_size * n_det_col;
    if (n != expect) {
        set_err(err, err_len, "子集 sinogram 尺寸不匹配"); return 1;
    }
    Subset& ss = s->subsets[i];
    std::unique_ptr<astra::CFloat32ProjectionData3D> sinoCpu(
        astra::createCFloat32ProjectionData3DMemory(*ss.geom));
    std::memcpy(sinoCpu->getFloat32Memory(), sino_subset, expect * sizeof(float));
    if (!astraCUDA3d::copyToGPUMemory(sinoCpu.get(), ss.proj)) {
        set_err(err, err_len, "子集 sinogram 上传失败");
        return 1;
    }
    return 0;
}

// 上传初始重建到 GPU (循环控制在 Rust, 每 epoch 一次)
int astra_rs_sart_upload(void* h, const float* vol_in, char* err, size_t err_len) {
    SARTGpu* s = static_cast<SARTGpu*>(h);
    std::unique_ptr<astra::CFloat32VolumeData3D> volCpu(
        astra::createCFloat32VolumeData3DMemory(s->volGeom));
    std::memcpy(volCpu->getFloat32Memory(), vol_in, nvol * sizeof(float));
    if (!astraCUDA3d::copyToGPUMemory(volCpu.get(), s->vol)) {
        set_err(err, err_len, "体积上传失败");
        return 1;
    }
    return 0;
}

// 单子集迭代 (GPU 常驻)
int astra_rs_sart_subset_step(void* h, int i, char* err, size_t err_len) {
    SARTGpu* s = static_cast<SARTGpu*>(h);
    std::string e;
    if (!s->subset_step(i, e)) {
        set_err(err, err_len, e);
        return 1;
    }
    return 0;
}

// 下载重建回 CPU (完成后同步 GPU, 避免后续 TV 的 H2D 等待未完成的 D2H)
int astra_rs_sart_download(void* h, float* vol_out, char* err, size_t err_len) {
    SARTGpu* s = static_cast<SARTGpu*>(h);
    std::unique_ptr<astra::CFloat32VolumeData3D> volCpu(
        astra::createCFloat32VolumeData3DMemory(s->volGeom));
    if (!astraCUDA3d::copyFromGPUMemory(volCpu.get(), s->vol)) {
        set_err(err, err_len, "体积下载失败");
        return 1;
    }
    std::memcpy(vol_out, volCpu->getFloat32Memory(), nvol * sizeof(float));
    cudaDeviceSynchronize();  // 清空待处理 GPU 工作, 使后续内核计时干净
    return 0;
}

}  // extern "C"
