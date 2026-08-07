// ============================================================================
// GPU 常驻 OS-SART — 精确复刻 ASTRA SIRT3D_CUDA (CudaSirtAlgorithm3D.cpp)
// 更新公式 (逐子集):
//   tmpProj = b
//   tmpProj += (-1) * A·v               (FP, fOutputScale=-1)
//   tmpProj *= lineWeight               (lineWeight = 1/(A·1), 预计算)
//   tmpVol  = A^T·tmpProj               (BP, 累加进清零后的 tmpVol)
//   v      += pixelWeight * tmpVol      (pixelWeight = 1/(A^T·1), 预计算)
// 与 ASTRA 的差异只有数据驻留位置 (GPU 常驻), 运算顺序/内核完全相同
// → 结果与 CCudaSirtAlgorithm3D 逐位一致, 但每 epoch 只做 2 次 CPU↔GPU 传输
// ============================================================================

#include "sart_gpu.h"
#include "astra_geometry.h"

#include <astra/Globals.h>
#include <astra/Config.h>
#include <astra/ConeVecProjectionGeometry3D.h>
#include <astra/VolumeGeometry3D.h>
#include <astra/Data3D.h>
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
constexpr size_t nsino_sub = (size_t)n_det_row * sub_size * n_det_col;

void free_gpu_data(astra::CData3D*& p) {
    if (p) {
        astraCUDA3d::freeGPUMemory(p);
        delete p;
        p = nullptr;
    }
}

struct Subset {
    std::unique_ptr<astra::CConeVecProjectionGeometry3D> geom;
    astra::Geometry3DParameters geom3;    // convertAstraGeometry 结果 (dims/向量/volScale)
    astra::CData3D* proj = nullptr;       // 子集噪声 sinogram (GPU, 只读)
    astra::CData3D* tmpProj = nullptr;    // 残差缓冲 (GPU)
    astra::CData3D* lineWeight = nullptr; // 1/(A·1) (GPU)
    astra::CData3D* pixelWeight = nullptr;// 1/(A^T·1) (GPU, 体积大小)
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

// 从全角度 sinogram 复制第 i 个子集到 CPU 投影数据对象 (与 Python fast_sirt 的数据一致)
void fill_subset_sino(astra::CFloat32ProjectionData3D* dst, const std::vector<float>& sino_noisy, int i) {
    float* d = dst->getFloat32Memory();
    for (int row = 0; row < n_det_row; ++row)
        for (int a = 0; a < sub_size; ++a)
            std::memcpy(d + ((size_t)row * sub_size + a) * n_det_col,
                        sino_noisy.data() + ((size_t)row * n_angles + i * sub_size + a) * n_det_col,
                        n_det_col * sizeof(float));
}

}  // namespace

struct SARTGpu::Impl {
    astra::CVolumeGeometry3D volGeom;
    astra::CData3D* vol = nullptr;     // 重建 (GPU 常驻)
    astra::CData3D* tmpVol = nullptr;  // BP 累加缓冲 (GPU, 体积大小)
    std::vector<Subset> subsets;
    astraCUDA3d::SProjectorParams3D params;  // volScale 在 init 中设置

    Impl() : volGeom(N, N, nz) {}
    ~Impl() {
        free_gpu_data(vol);
        free_gpu_data(tmpVol);
    }
};

SARTGpu::SARTGpu() : m(std::make_unique<Impl>()) {}
SARTGpu::~SARTGpu() {}

bool SARTGpu::init(bool helical, const std::vector<float>& sino_noisy, std::string& err) {
    if (sino_noisy.size() != (size_t)n_det_row * n_angles * n_det_col) {
        err = "sino_noisy 大小不匹配";
        return false;
    }

    // 体积 GPU 缓冲 (以 CPU 侧体积对象为模板)
    std::unique_ptr<astra::CFloat32VolumeData3D> volCpu(
        astra::createCFloat32VolumeData3DMemory(m->volGeom));
    m->vol = astraCUDA3d::createGPUData3DLike(volCpu.get());
    m->tmpVol = astraCUDA3d::createGPUData3DLike(volCpu.get());
    if (!m->vol || !m->tmpVol) {
        err = "体积 GPU 缓冲分配失败";
        return false;
    }

    // 子集 (10 × 18 角度): 几何 + GPU 缓冲 + 权重预计算
    for (int i = 0; i < n_subsets; ++i) {
        Subset s;
        auto full = astra_cpp_build_vectors(helical, n_angles, n_det_row, n_det_col,
                                            DSO, DSD_det, det_pix, pitch_mm);
        std::vector<astra::SConeProjection> subvecs;
        subvecs.reserve(sub_size);
        for (int a = i * sub_size; a < (i + 1) * sub_size; ++a)
            subvecs.push_back(full[a]);
        s.geom = std::make_unique<astra::CConeVecProjectionGeometry3D>(
            sub_size, n_det_row, n_det_col, std::move(subvecs));
        s.geom3 = astra::convertAstraGeometry(&m->volGeom, s.geom.get());
        if (!s.geom3.isCone()) {
            err = "子集几何转换失败";
            return false;
        }

        std::unique_ptr<astra::CFloat32ProjectionData3D> sinoCpu(
            astra::createCFloat32ProjectionData3DMemory(*s.geom));
        fill_subset_sino(sinoCpu.get(), sino_noisy, i);

        s.proj = astraCUDA3d::createGPUData3DLike(sinoCpu.get());
        s.tmpProj = astraCUDA3d::createGPUData3DLike(sinoCpu.get());
        s.lineWeight = astraCUDA3d::createGPUData3DLike(sinoCpu.get());
        s.pixelWeight = astraCUDA3d::createGPUData3DLike(volCpu.get());  // 体积大小
        if (!s.proj || !s.tmpProj || !s.lineWeight || !s.pixelWeight) {
            err = "子集 GPU 缓冲分配失败";
            return false;
        }
        if (!astraCUDA3d::copyToGPUMemory(sinoCpu.get(), s.proj)) {
            err = "子集 sinogram 上传失败";
            return false;
        }
        sinoCpu.reset();  // CPU 模板释放, GPU 缓冲独立

        // ---- 权重预计算 (与 CudaSirtAlgorithm3D::precomputeWeights 完全一致) ----
        astraCUDA3d::SProjectorParams3D p = m->params;
        p.volScale = s.geom3.getVolScale();
        // lineWeight = 1 / (A·1)
        astraCUDA3d::zeroGPUMemory(s.lineWeight);
        astraCUDA3d::processVol3D<astraCUDA3d::opSet>(m->tmpVol, 1.0f);
        if (!astraCUDA3d::FP(s.lineWeight, m->tmpVol, s.geom3, p)) {
            err = "lineWeight FP 失败";
            return false;
        }
        astraCUDA3d::processVol3D<astraCUDA3d::opInvert>(s.lineWeight);
        // pixelWeight = 1 / (A^T·1)  (relaxation = 1 → 不再乘)
        astraCUDA3d::zeroGPUMemory(s.pixelWeight);
        astraCUDA3d::processVol3D<astraCUDA3d::opSet>(s.tmpProj, 1.0f);
        if (!astraCUDA3d::BP(s.tmpProj, s.pixelWeight, s.geom3, p)) {
            err = "pixelWeight BP 失败";
            return false;
        }
        astraCUDA3d::processVol3D<astraCUDA3d::opInvert>(s.pixelWeight);

        m->subsets.push_back(std::move(s));
    }
    m->params.volScale = m->subsets[0].geom3.getVolScale();  // 各子集相同 (1,1,1)
    return true;
}

bool SARTGpu::run(const std::vector<float>& vol_in, int n_epochs,
                  std::vector<float>& vol_out, std::string& err) {
    if (vol_in.size() != nvol) {
        err = "vol_in 大小不匹配";
        return false;
    }

    // 上传初始重建 (读 vol_in 完成后才可能写 vol_out, 支持就地)
    {
        std::unique_ptr<astra::CFloat32VolumeData3D> volCpu(
            astra::createCFloat32VolumeData3DMemory(m->volGeom));
        std::memcpy(volCpu->getFloat32Memory(), vol_in.data(), nvol * sizeof(float));
        if (!astraCUDA3d::copyToGPUMemory(volCpu.get(), m->vol)) {
            err = "体积上传失败";
            return false;
        }
    }

    for (int e = 0; e < n_epochs; ++e) {
        for (auto& s : m->subsets) {
            astraCUDA3d::SProjectorParams3D p = m->params;
            p.volScale = s.geom3.getVolScale();
            // tmpProj = b
            astraCUDA3d::assignGPUMemory(s.tmpProj, s.proj);
            // tmpProj = b - A·v
            p.fOutputScale = -1.0f;
            if (!astraCUDA3d::FP(s.tmpProj, m->vol, s.geom3, p)) {
                err = "残差 FP 失败";
                return false;
            }
            // tmpProj = lineWeight * (b - A·v)
            astraCUDA3d::processVol3D<astraCUDA3d::opMul>(s.tmpProj, s.lineWeight);
            // tmpVol = A^T·tmpProj
            astraCUDA3d::zeroGPUMemory(m->tmpVol);
            p.fOutputScale = 1.0f;
            if (!astraCUDA3d::BP(s.tmpProj, m->tmpVol, s.geom3, p)) {
                err = "残差 BP 失败";
                return false;
            }
            // v += pixelWeight * tmpVol
            astraCUDA3d::processVol3D<astraCUDA3d::opAddMul>(m->vol, m->tmpVol, s.pixelWeight);
        }
    }

    // 下载结果
    {
        std::unique_ptr<astra::CFloat32VolumeData3D> volCpu(
            astra::createCFloat32VolumeData3DMemory(m->volGeom));
        if (!astraCUDA3d::copyFromGPUMemory(volCpu.get(), m->vol)) {
            err = "体积下载失败";
            return false;
        }
        vol_out.assign(volCpu->getFloat32Memory(), volCpu->getFloat32Memory() + nvol);
    }
    return true;
}
