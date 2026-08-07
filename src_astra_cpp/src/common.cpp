// ============================================================================
// ASTRA 锥束混合重建 (C++ 移植版) — 核心实现
// 与 Python 版 src_3d_{axial,helical}/astra_cone_hybrid.py 逐段对齐:
//   A. Pure FDK (hann 滤波)
//   B. TV-OS-SART (10 子集 SIRT3D_CUDA + TV 去噪, β 递减)
//   C. Hybrid IR (OS-SART×10 + TV×10 + FDK 混合 10%)
// 差异说明:
//   * 噪声模型与 Python 相同 (泊松-高斯 + 环形伪影), 但 RNG 为 std::mt19937,
//     与 numpy 不完全逐位一致 → 指标会略有差异 (同数量级)
//   * 体模由 tools/make_phantom.py 生成 (tomophantom 无 C++ API)
// ============================================================================

#include "common.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <astra/Globals.h>
#include <astra/Config.h>
#include <astra/Algorithm.h>
#include <astra/ConeVecProjectionGeometry3D.h>
#include <astra/VolumeGeometry3D.h>
#include <astra/Data3D.h>
#include <astra/CudaProjector3D.h>
#include <astra/CudaFDKAlgorithm3D.h>
#include <astra/CudaSirtAlgorithm3D.h>
#include <astra/CudaForwardProjectionAlgorithm3D.h>
#include <astra/Filters.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tv_gpu.h"

namespace {

constexpr int N = 512;           // 体素网格 X/Y
constexpr int nz = 32;           // 体素网格 Z
constexpr int n_angles = 180;    // 投影角度数
constexpr int n_subsets = 10;    // OS 子集数
constexpr int sub_size = n_angles / n_subsets;  // 18
constexpr double DSO = 1000.0;   // source-isocenter
constexpr double DSD_det = 500.0;  // isocenter-detector
constexpr double det_pix = 1.0;  // 探测器像素尺寸
constexpr double pitch_mm = 16.0;  // 螺旋螺距 (mm/圈)

constexpr size_t nvol = (size_t)nz * N * N;
constexpr int n_det_row = 64;    // nz*2
constexpr int n_det_col = (int)std::ceil(N * std::sqrt(2.0));  // 725
constexpr size_t nsino = (size_t)n_det_row * n_angles * n_det_col;

constexpr const char* kSep60 = "============================================================";  // 60
constexpr const char* kSep55 = "-------------------------------------------------------";        // 55
constexpr const char* kSep70 = "======================================================================";  // 70
constexpr const char* kSep72 = "------------------------------------------------------------------------";  // 72

struct SubsetObjects {
    std::unique_ptr<astra::CConeVecProjectionGeometry3D> geom;
    std::unique_ptr<astra::CFloat32ProjectionData3D> sino;
    std::unique_ptr<astra::CFloat32VolumeData3D> vol;
    std::unique_ptr<astra::CCudaProjector3D> proj;
    std::unique_ptr<astra::CCudaSirtAlgorithm3D> alg;
};

// ---------------------------------------------------------------------------
// 计时
// ---------------------------------------------------------------------------
class Stopwatch {
    std::chrono::steady_clock::time_point t0_;
public:
    Stopwatch() { t0_ = std::chrono::steady_clock::now(); }
    double ms() const {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0_).count();
    }
};

// ---------------------------------------------------------------------------
// 文件 IO
// ---------------------------------------------------------------------------
std::vector<float> load_raw(const std::string& path, size_t n) {
    std::vector<float> buf(n);
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)(n * sizeof(float)));
    if (!f) return {};
    return buf;
}

void save_raw(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), (std::streamsize)(v.size() * sizeof(float)));
}

// ---------------------------------------------------------------------------
// 几何: 锥束 cone_vec (与 Python 逐段一致)
//   轴向: 源/探测器在 xy 平面旋转
//   螺旋: 同时沿 z 方向线性移动, z ∈ [-pitch/2, pitch/2]
// 注意: 直接构造函数接受的是探测器"角落"(bottom-left)约定, 而 Python 接口
//       给的是探测器"中心", 由 Config 初始化路径自动转换
//       (见 src/ConeVecProjectionGeometry3D.cpp initializeAngles)。
//       这里手动做同样的转换: fDetS -= 0.5*row*v + 0.5*col*u
// ---------------------------------------------------------------------------
std::vector<astra::SConeProjection> build_vectors(bool helical) {
    std::vector<astra::SConeProjection> vecs(n_angles);
    for (int i = 0; i < n_angles; ++i) {
        double th = 2.0 * M_PI * i / n_angles;  // linspace(0,360,180,endpoint=False)
        double c = std::cos(th), s = std::sin(th);
        double z_src = helical ? pitch_mm * (th / (2.0 * M_PI) - 0.5) : 0.0;
        // 探测器中心 (Python 接口约定)
        double dcx = -DSD_det * s, dcy = DSD_det * c, dcz = z_src;
        // 探测器 u/v 向量
        double ux = det_pix * c, uy = det_pix * s, uz = 0.0;
        double vx = 0.0, vy = 0.0, vz = det_pix;
        // 中心 → 角落 (bottom-left): 减半个探测器尺寸
        double sx = dcx - 0.5 * n_det_row * vx - 0.5 * n_det_col * ux;
        double sy = dcy - 0.5 * n_det_row * vy - 0.5 * n_det_col * uy;
        double sz = dcz - 0.5 * n_det_row * vz - 0.5 * n_det_col * uz;
        vecs[i] = {DSO * s, -DSO * c, z_src,   // source
                   sx, sy, sz,                 // detector bottom-left 角落
                   ux, uy, uz,                 // det u-vector
                   vx, vy, vz};                // det v-vector
    }
    return vecs;
}

// ---------------------------------------------------------------------------
// TV 去噪 (CPU, 与 Python 的 numpy 实现逐元素一致)
//   dx = v[x+1]-v[x] (x 最快), dy = v[y+1]-v[y], dz = v[z+1]-v[z]
//   mag = sqrt(dx^2+dy^2+(w_z*dz)^2+1e-8); u = (dx,dy,w_z*dz)/mag
//   div 为 u 的散度 (边界: 前向差分/后向差分); 返回 v + beta*div
// ---------------------------------------------------------------------------
void tv_denoise_inplace(std::vector<float>& v, float beta, float w_z) {
    constexpr size_t slice = (size_t)N * N;
    std::vector<float> ux(nvol), uy(nvol), uz(nvol);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i64 = 0; i64 < (int64_t)nvol; ++i64) {
        size_t i = (size_t)i64;
        size_t x = i % N;
        size_t y = (i / N) % N;
        size_t z = i / slice;
        float dx = (x + 1 < N) ? v[i + 1] - v[i] : 0.0f;
        float dy = (y + 1 < N) ? v[i + N] - v[i] : 0.0f;
        float dz = (z + 1 < nz) ? v[i + slice] - v[i] : 0.0f;
        float mag = std::sqrt(dx * dx + dy * dy + (w_z * dz) * (w_z * dz) + 1e-8f);
        ux[i] = dx / mag;
        uy[i] = dy / mag;
        uz[i] = w_z * dz / mag;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i64 = 0; i64 < (int64_t)nvol; ++i64) {
        size_t i = (size_t)i64;
        size_t x = i % N;
        size_t y = (i / N) % N;
        size_t z = i / slice;
        float div = 0.0f;
        if (x == 0)            div += ux[i];
        else if (x + 1 == N)   div += -ux[i - 1];
        else                   div += ux[i] - ux[i - 1];
        if (y == 0)            div += uy[i];
        else if (y + 1 == N)   div += -uy[i - N];
        else                   div += uy[i] - uy[i - N];
        if (z == 0)            div += uz[i];
        else if (z + 1 == nz)  div += -uz[i - slice];
        else                   div += uz[i] - uz[i - slice];
        v[i] = v[i] - beta * (-div);  // v + beta*div
    }
}

// ---------------------------------------------------------------------------
// 噪声/伪影: 与 ct_noise.py 的 add_artifacts(dose_level=0.5, rings=True) 同模型
//   quantum: (Poisson(λ) - λ)/scaling, λ = sino_norm*500, sino_norm = sino/sino.max()
//   electronic: N(0, 0.01/dose_level)
//   noisy = sino + sino.max()*(quantum*0.3 + electronic*0.7), clip>=0
//   rings: 15 个坏道, 每个加 uniform(-0.03, 0.03)*sino.max() 偏移
// RNG: std::mt19937 (seed 2024), 与 numpy 不逐位一致
// ---------------------------------------------------------------------------
std::vector<float> add_artifacts(const std::vector<float>& sino) {
    const float dose_level = 0.5f;
    const float smax = *std::max_element(sino.begin(), sino.end());

    std::vector<float> noisy(sino.size());
    {
        std::mt19937 rng(2024);
        const float scaling = 1000.0f * dose_level;
        std::poisson_distribution<int> pois(1.0);  // lambda 按元素重新设置
        std::normal_distribution<float> gauss(0.0f, 0.01f / dose_level);
        for (size_t i = 0; i < sino.size(); ++i) {
            float sn = sino[i] / smax;
            float lam = std::max(sn * scaling, 0.0f);
            float quantum = (pois(rng, std::poisson_distribution<int>::param_type((int)lam)) - lam) / scaling;
            float electronic = gauss(rng);
            noisy[i] = sino[i] + smax * (quantum * 0.3f + electronic * 0.7f);
        }
    }
    for (auto& v : noisy) v = std::max(v, 0.0f);

    // 环形伪影 (探测器坏道): 15 个, 无放回
    {
        std::mt19937 rng(2024);
        std::uniform_int_distribution<int> ch(0, n_det_col - 1);
        std::uniform_real_distribution<float> off(-0.03f, 0.03f);
        std::vector<int> picked;
        while ((int)picked.size() < 15) {
            int c = ch(rng);
            if (std::find(picked.begin(), picked.end(), c) == picked.end())
                picked.push_back(c);
        }
        const float noisy_max = *std::max_element(noisy.begin(), noisy.end());
        for (int c : picked) {
            float offset = off(rng) * noisy_max;
            for (size_t row = 0; row < n_det_row; ++row)
                for (size_t a = 0; a < n_angles; ++a)
                    noisy[(row * n_angles + a) * n_det_col + c] += offset;
        }
    }
    for (auto& v : noisy) v = std::max(v, 0.0f);
    return noisy;
}

// ---------------------------------------------------------------------------
// TV 去噪: GPU (CUDA 内核) → CPU 回退
//   与 Python tv_gpu.py 的 CuPy 内核算法等价 (单遍散度), 布局 [z][y][x]
// ---------------------------------------------------------------------------
namespace {
bool g_tv_gpu_ok = false;
std::vector<float> g_tv_buf;

void tv_denoise(std::vector<float>& v, float beta, float w_z) {
    if (g_tv_gpu_ok) {
        if (g_tv_buf.empty()) g_tv_buf.resize(v.size());
        if (tv_denoise_cuda(v.data(), g_tv_buf.data(), nz, N, beta, w_z, 1e-8f) == 0) {
            v.swap(g_tv_buf);
            return;
        }
        g_tv_gpu_ok = false;  // 设备出错 → 回退 CPU
        printf("   [TV] CUDA 不可用, 回退 CPU\n");
    }
    tv_denoise_inplace(v, beta, w_z);
}
}  // namespace

// ---------------------------------------------------------------------------
// 度量 (与 Python 逐段一致)
// ---------------------------------------------------------------------------
// 最小二乘线性标定: rec*a + b ≈ gt (掩码 gt>0.001)
std::vector<float> linear_scale(const std::vector<float>& rec, const std::vector<float>& gt) {
    double s_aa = 0, s_a = 0, s_ab = 0, s_b = 0;
    size_t m = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : s_aa, s_a, s_ab, s_b, m) schedule(static)
#endif
    for (int64_t i64 = 0; i64 < (int64_t)rec.size(); ++i64) {
        size_t i = (size_t)i64;
        if (gt[i] > 0.001f) {
            double r = rec[i];
            s_aa += r * r; s_a += r; s_ab += r * gt[i]; s_b += gt[i]; ++m;
        }
    }
    double det = s_aa * m - s_a * s_a;
    double a = (s_ab * m - s_a * s_b) / det;
    double b = (s_aa * s_b - s_a * s_ab) / det;
    std::vector<float> out(rec.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i64 = 0; i64 < (int64_t)rec.size(); ++i64)
        out[i64] = (float)(rec[i64] * a + b);
    return out;
}

double calc_rmse(const std::vector<float>& rec, const std::vector<float>& gt) {
    double s = 0; size_t m = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : s, m) schedule(static)
#endif
    for (int64_t i64 = 0; i64 < (int64_t)rec.size(); ++i64) {
        size_t i = (size_t)i64;
        if (gt[i] > 0.001f) { double e = gt[i] - rec[i]; s += e * e; ++m; }
    }
    return std::sqrt(s / m);
}

double calc_ssim(const std::vector<float>& rec, const std::vector<float>& gt) {
    double mux = 0, muy = 0;
    size_t m = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : mux, muy, m) schedule(static)
#endif
    for (int64_t i64 = 0; i64 < (int64_t)rec.size(); ++i64) {
        size_t i = (size_t)i64;
        if (gt[i] > 0.001f) { mux += gt[i]; muy += rec[i]; ++m; }
    }
    mux /= m; muy /= m;
    double sx = 0, sy = 0, sxy = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sx, sy, sxy) schedule(static)
#endif
    for (int64_t i64 = 0; i64 < (int64_t)rec.size(); ++i64) {
        size_t i = (size_t)i64;
        if (gt[i] > 0.001f) {
            double g = gt[i] - mux, r = rec[i] - muy;
            sx += g * g; sy += r * r; sxy += g * r;
        }
    }
    sx /= m; sy /= m; sxy /= m;
    double c1 = std::pow(0.01 * 0.05, 2.0);
    double c2 = std::pow(0.03 * 0.05, 2.0);
    return (2 * mux * muy + c1) * (2 * sxy + c2) /
           ((mux * mux + muy * muy + c1) * (sx + sy + c2));
}

ZProfile calc_z_profile(const std::vector<float>& rec, const std::vector<float>& gt) {
    constexpr size_t slice = (size_t)N * N;
    ZProfile zp;
    zp.per_slice.resize(nz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int z = 0; z < nz; ++z) {
        const float* g = gt.data() + (size_t)z * slice;
        const float* r = rec.data() + (size_t)z * slice;
        double s = 0; size_t m = 0;
        for (size_t i = 0; i < slice; ++i)
            if (g[i] > 0.001f) { double e = g[i] - r[i]; s += e * e; ++m; }
        zp.per_slice[z] = (m > 100) ? (float)std::sqrt(s / m) : 0.0f;
    }
    double sum = 0; float mx = 0;
    for (float v : zp.per_slice) { sum += v; mx = std::max(mx, v); }
    zp.mean = sum / nz;
    zp.max = mx;
    return zp;
}

// ---------------------------------------------------------------------------
// 输出: PNG 切片 (stb) + 摘要 JSON
// ---------------------------------------------------------------------------
void write_slice_png(const std::string& path, const std::vector<float>& vol,
                     int z, const std::vector<float>& soft_mask,
                     float vmin, float vmax) {
    const float* src = vol.data() + (size_t)z * N * N;
    std::vector<unsigned char> buf((size_t)N * N);
    float range = vmax - vmin;
    for (size_t y = 0; y < N; ++y)
        for (size_t x = 0; x < N; ++x) {
            size_t i = y * N + x;
            float v = (src[i] - vmin) / range * soft_mask[i];
            buf[i] = (unsigned char)std::max(0, std::min(255, (int)std::lround(v * 255.0f)));
        }
    stbi_write_png(path.c_str(), N, N, 1, buf.data(), N);
}

void write_error_png(const std::string& path, const std::vector<float>& vol,
                     const std::vector<float>& gt, int z,
                     const std::vector<float>& soft_mask) {
    const float* r = vol.data() + (size_t)z * N * N;
    const float* g = gt.data() + (size_t)z * N * N;
    std::vector<float> e((size_t)N * N);
    for (size_t i = 0; i < (size_t)N * N; ++i) e[i] = r[i] - g[i];
    std::vector<float> sorted = e;
    std::sort(sorted.begin(), sorted.end());
    float v = std::max(0.005f, (float)std::fabs(sorted[(size_t)((size_t)N * N * 95 / 100)]) * 1.2f);
    std::vector<unsigned char> buf((size_t)N * N);
    for (size_t i = 0; i < (size_t)N * N; ++i) {
        float t = 0.5f + e[i] / (2.0f * v);
        buf[i] = (unsigned char)std::max(0, std::min(255, (int)std::lround(t * 255.0f)));
    }
    // 误差图同样乘软遮罩 (避免外部为 0 的假误差)
    for (size_t i = 0; i < (size_t)N * N; ++i)
        buf[i] = (unsigned char)(buf[i] * soft_mask[i]);
    stbi_write_png(path.c_str(), N, N, 1, buf.data(), N);
}

std::string json_arr(const std::vector<float>& v, int prec) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ", ";
        os << std::fixed << std::setprecision(prec) << (double)v[i];
    }
    os << "]";
    return os.str();
}

std::string json_result(const AlgorithmResult& r) {
    std::ostringstream os;
    os << std::fixed
       << "{\"rmse\": " << std::setprecision(5) << r.rmse
       << ", \"ssim\": " << std::setprecision(4) << r.ssim
       << ", \"time_ms\": " << std::setprecision(1) << r.time_ms << "}";
    return os.str();
}

}  // namespace

// ============================================================================
// 主流水线
// ============================================================================
int run_pipeline(bool helical, const std::string& phantom_path, const std::string& outdir) {
    constexpr size_t slice = (size_t)N * N;

    printf("%s\n", kSep60);
    printf("%s  [锥束 CBCT | ASTRA CUDA C++]\n", helical ? "螺旋(Helical) 混合重建" : "FBP + IR 混合重建");
    printf("%s\n", kSep60);

    // ---- 1. 载入体模 (tools/make_phantom.py 生成) ----
    std::vector<float> vol_gt = load_raw(phantom_path, nvol);
    if (vol_gt.empty()) {
        printf("错误: 无法读取体模文件 %s (先运行 tools/make_phantom.py)\n", phantom_path.c_str());
        return 1;
    }
    float gt_min = *std::min_element(vol_gt.begin(), vol_gt.end());
    float gt_max = *std::max_element(vol_gt.begin(), vol_gt.end());
    printf("   体模: [%.5f, %.5f]\n", (double)gt_min, (double)gt_max);

    // ---- 2. 几何 (与 TIGRE/ASTRA Python 对齐) ----
    auto vecs = build_vectors(helical);
    astra::CConeVecProjectionGeometry3D projGeom(n_angles, n_det_row, n_det_col, std::move(vecs));
    astra::CVolumeGeometry3D volGeom(N, N, nz);
    astra::CCudaProjector3D projector(projGeom, volGeom);

    // ---- 3. GPU 正向投影 (FP3D_CUDA) ----
    printf("\nGPU 正向投影...\n");
    Stopwatch sw_fp;
    std::unique_ptr<astra::CFloat32ProjectionData3D> sino_data(
        astra::createCFloat32ProjectionData3DMemory(projGeom));
    std::memset(sino_data->getFloat32Memory(), 0, nsino * sizeof(float));
    std::unique_ptr<astra::CFloat32VolumeData3D> vol_data(
        astra::createCFloat32VolumeData3DMemory(volGeom));
    std::memcpy(vol_data->getFloat32Memory(), vol_gt.data(), nvol * sizeof(float));

    astra::CCudaForwardProjectionAlgorithm3D fp;
    if (!fp.initialize(&projector, sino_data.get(), vol_data.get())) {
        printf("错误: FP3D_CUDA 初始化失败\n");
        return 1;
    }
    if (!fp.run(1)) { printf("错误: FP3D_CUDA 运行失败\n"); return 1; }
    std::vector<float> sino(nsino);
    std::memcpy(sino.data(), sino_data->getFloat32Memory(), nsino * sizeof(float));
    printf("   完成: %.0fms, 形状 (%d, %d, %d)\n", sw_fp.ms(), n_det_row, n_angles, n_det_col);

    // ---- 软遮罩 (可视化用) ----
    std::vector<float> soft_mask((size_t)N * N);
    for (size_t y = 0; y < N; ++y)
        for (size_t x = 0; x < N; ++x) {
            float dist = std::sqrt((float)((double)x - N / 2.0) * (x - N / 2.0) +
                                   ((double)y - N / 2.0) * (y - N / 2.0));
            float body_r = N * 0.42f;
            soft_mask[y * N + x] = std::max(0.0f, std::min(1.0f, (body_r + 20 - dist) / 20));
        }

    // ============================
    // A. Pure FDK
    // ============================
    printf("%s\nA. Pure FDK\n%s\n", kSep55, kSep55);
    Stopwatch sw_fdk;
    std::unique_ptr<astra::CFloat32ProjectionData3D> sino_fdk(
        astra::createCFloat32ProjectionData3DMemory(projGeom));
    std::memcpy(sino_fdk->getFloat32Memory(), sino.data(), nsino * sizeof(float));
    std::unique_ptr<astra::CFloat32VolumeData3D> vol_fdk(
        astra::createCFloat32VolumeData3DMemory(volGeom));
    astra::SFilterConfig filt;
    filt.m_eType = astra::FILTER_HANN;  // 与 Python FilterType=hann 对齐
    filt.m_fD = 1.0f;
    filt.m_fParameter = -1.0f;
    astra::CCudaFDKAlgorithm3D fdk(&projector, sino_fdk.get(), vol_fdk.get(), filt, false);
    if (!fdk.run(1)) { printf("错误: FDK 运行失败\n"); return 1; }
    std::vector<float> fdk_raw(nvol);
    std::memcpy(fdk_raw.data(), vol_fdk->getFloat32Memory(), nvol * sizeof(float));
    double fdk_t = sw_fdk.ms();
    std::vector<float> fdk_rec = linear_scale(fdk_raw, vol_gt);
    double fdk_rmse = calc_rmse(fdk_rec, vol_gt);
    double fdk_ssim = calc_ssim(fdk_rec, vol_gt);
    ZProfile fdk_zprof = calc_z_profile(fdk_rec, vol_gt);
    printf("   RMSE=%.5f, SSIM=%.4f, %.0fms\n", fdk_rmse, fdk_ssim, fdk_t);
    printf("   z-profile: mean=%.5f, max=%.5f\n", fdk_zprof.mean, fdk_zprof.max);

    // ---- 噪声数据 ----
    std::vector<float> sino_noisy = add_artifacts(sino);
    printf("%s\n有噪声数据 (dose=0.5, rings=15)...\n", kSep55);

    // 有噪声 FDK (一次性, 作为 B/C 的起点)
    std::unique_ptr<astra::CFloat32ProjectionData3D> sino_n(
        astra::createCFloat32ProjectionData3DMemory(projGeom));
    std::memcpy(sino_n->getFloat32Memory(), sino_noisy.data(), nsino * sizeof(float));
    std::unique_ptr<astra::CFloat32VolumeData3D> vol_n(
        astra::createCFloat32VolumeData3DMemory(volGeom));
    astra::CCudaFDKAlgorithm3D fdk_n(&projector, sino_n.get(), vol_n.get(), filt, false);
    if (!fdk_n.run(1)) { printf("错误: FDK(noisy) 运行失败\n"); return 1; }
    std::vector<float> rec_fdk_n(nvol);
    std::memcpy(rec_fdk_n.data(), vol_n->getFloat32Memory(), nvol * sizeof(float));
    std::vector<float> rec_fdk_n_ls = linear_scale(rec_fdk_n, vol_gt);
    double fdk_noisy_rmse = calc_rmse(rec_fdk_n_ls, vol_gt);
    printf("   FDK(noisy) RMSE=%.5f\n", fdk_noisy_rmse);

    // ---- 预分配 10 个子集 SIRT3D_CUDA 对象 (与 Python 复用对象一致) ----
    std::vector<SubsetObjects> subs;
    subs.reserve(n_subsets);
    for (int i = 0; i < n_subsets; ++i) {
        std::vector<astra::SConeProjection> subvecs;
        subvecs.reserve(sub_size);
        std::vector<astra::SConeProjection> full = build_vectors(helical);
        for (int a = i * sub_size; a < (i + 1) * sub_size; ++a)
            subvecs.push_back(full[a]);
        SubsetObjects s;
        s.geom = std::make_unique<astra::CConeVecProjectionGeometry3D>(
            sub_size, n_det_row, n_det_col, std::move(subvecs));
        s.sino.reset(astra::createCFloat32ProjectionData3DMemory(*s.geom));
        float* dst = s.sino->getFloat32Memory();
        size_t k = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t row64 = 0; row64 < n_det_row; ++row64) {
            int row = (int)row64;
            size_t rk = (size_t)row * sub_size * n_det_col;
            const float* src = sino_noisy.data() + (size_t)row * n_angles * n_det_col + (size_t)i * sub_size * n_det_col;
            for (int a = 0; a < sub_size; ++a)
                std::memcpy(dst + rk + (size_t)a * n_det_col,
                            src + (size_t)a * n_det_col, n_det_col * sizeof(float));
        }
        (void)k;
        s.vol.reset(astra::createCFloat32VolumeData3DMemory(volGeom));
        s.proj = std::make_unique<astra::CCudaProjector3D>(*s.geom, volGeom);
        s.alg = std::make_unique<astra::CCudaSirtAlgorithm3D>(s.proj.get(), s.sino.get(), s.vol.get());
        subs.push_back(std::move(s));
    }

    // OS-SART 一步: 每子集 1 次 SIRT 迭代 (与 Python fast_sirt/fast_ossart 一致)
    auto ossart_step = [&](std::vector<float>& rec) {
        for (auto& s : subs) {
            std::memcpy(s.vol->getFloat32Memory(), rec.data(), nvol * sizeof(float));
            s.alg->run(1);
            std::memcpy(rec.data(), s.vol->getFloat32Memory(), nvol * sizeof(float));
        }
    };

    // ============================
    // B. TV-OS-SART
    // ============================
    printf("%s\nB. TV-OS-SART\n%s\n", kSep55, kSep55);
    const float beta0 = 0.002f, decay = 0.8f, w_z = 1.5f;
    std::vector<float> rec_tv = rec_fdk_n;
    double t_tv_total = 0.0;
    double best_rmse = 1e9, best_ssim = 0, best_t = 0;
    int best_ni = 0;
    std::vector<float> best_rec;
    double t_sirt = 0.0, t_tv = 0.0;  // 耗时构成统计
    g_tv_gpu_ok = true;               // 优先 GPU TV, 失败自动回退 CPU
    for (int ni = 1; ni <= 10; ++ni) {
        Stopwatch sw_sirt;
        ossart_step(rec_tv);
        t_sirt += sw_sirt.ms();
        Stopwatch sw_tv;
        tv_denoise(rec_tv, beta0 * std::pow(decay, ni - 1), w_z);
        t_tv += sw_tv.ms();
        t_tv_total += sw_sirt.ms() + sw_tv.ms();
        std::vector<float> ls = linear_scale(rec_tv, vol_gt);
        double r = calc_rmse(ls, vol_gt), s = calc_ssim(ls, vol_gt);
        if (r < best_rmse) { best_rmse = r; best_ssim = s; best_rec = ls; best_t = t_tv_total; best_ni = ni; }
        printf("   TV-OS-SART x%3d (β=%.4f): RMSE=%.5f, SSIM=%.4f, 累计%.0fms\n",
               ni, (double)(beta0 * std::pow(decay, ni - 1)), r, s, t_tv_total);
    }
    printf("   >> 最优: TV-OS-SART x%d: RMSE=%.5f\n", best_ni, best_rmse);
    double tv_improv = (1 - best_rmse / fdk_noisy_rmse) * 100;
    printf("   TV 改善 vs 噪声FDK(%.5f): %+.1f%%\n", fdk_noisy_rmse, tv_improv);
    ZProfile best_tv_zprof = calc_z_profile(best_rec, vol_gt);
    printf("   TV-OS-SART z-profile: mean=%.5f, max=%.5f\n", best_tv_zprof.mean, best_tv_zprof.max);

    // ============================
    // C. Hybrid IR (OS10 + TV10(β↓) + FDK 混合 10%)
    // ============================
    printf("%s\nC. Hybrid IR (OS-SART×10 + TV×10(β递减) + FDK混合 10%%)\n%s\n", kSep55, kSep55);
    Stopwatch sw_h;
    std::vector<float> rec_h = rec_fdk_n;
    for (int ni = 0; ni < 10; ++ni) {
        Stopwatch sw_sirt2;
        ossart_step(rec_h);
        t_sirt += sw_sirt2.ms();
        Stopwatch sw_tv2;
        tv_denoise(rec_h, beta0 * std::pow(decay, ni), w_z);
        t_tv += sw_tv2.ms();
    }
    for (size_t i = 0; i < nvol; ++i)
        rec_h[i] = 0.9f * rec_h[i] + 0.1f * rec_fdk_n[i];
    double t_hybrid = sw_h.ms();
    std::vector<float> rec_h_ls = linear_scale(rec_h, vol_gt);
    double r_hybrid = calc_rmse(rec_h_ls, vol_gt), s_hybrid = calc_ssim(rec_h_ls, vol_gt);
    ZProfile hybrid_zprof = calc_z_profile(rec_h_ls, vol_gt);
    printf("   Hybrid IR: RMSE=%.5f, SSIM=%.4f, %.0fms\n", r_hybrid, s_hybrid, t_hybrid);
    printf("   z-profile: mean=%.5f, max=%.5f\n", hybrid_zprof.mean, hybrid_zprof.max);

    // ============================
    // D. 输出: raw / PNG / JSON
    // ============================
    printf("\n生成输出...\n");
    // std::filesystem::create_directories
    std::string cmd = "mkdir -p " + outdir;
    if (std::system(cmd.c_str()) != 0)
        printf("警告: 无法创建输出目录 %s\n", outdir.c_str());

    save_raw(outdir + "/cpp_fdk.raw", fdk_rec);
    save_raw(outdir + "/cpp_tv.raw", best_rec);
    save_raw(outdir + "/cpp_hybrid.raw", rec_h_ls);

    int mid = nz / 2;
    write_slice_png(outdir + "/cpp_gt.png", vol_gt, mid, soft_mask, 0.0f, 0.05f);
    write_slice_png(outdir + "/cpp_fdk.png", fdk_rec, mid, soft_mask, 0.0f, 0.05f);
    write_slice_png(outdir + "/cpp_tv.png", best_rec, mid, soft_mask, 0.0f, 0.05f);
    write_slice_png(outdir + "/cpp_hybrid.png", rec_h_ls, mid, soft_mask, 0.0f, 0.05f);
    write_error_png(outdir + "/cpp_err_fdk.png", fdk_rec, vol_gt, mid, soft_mask);
    write_error_png(outdir + "/cpp_err_tv.png", best_rec, vol_gt, mid, soft_mask);
    write_error_png(outdir + "/cpp_err_hybrid.png", rec_h_ls, vol_gt, mid, soft_mask);

    // 汇总表
    printf("\n%s\n", kSep70);
    printf("汇总对比 (32x512x512, %d角度, %d子集)\n", n_angles, n_subsets);
    printf("%s\n", kSep70);
    printf("%-30s %10s %12s %8s %10s\n", "算法", "耗时(ms)", "RMSE", "SSIM", "z-RMSE");
    printf("%s\n", kSep72);
    printf("%-30s %8.0f ms  %10.5f  %8.4f %10.5f\n", "Pure FDK", fdk_t, fdk_rmse, fdk_ssim, fdk_zprof.mean);
    std::string tv_name = "TV-OS-SART x" + std::to_string(best_ni);
    printf("%-30s %8.0f ms  %10.5f  %8.4f %10.5f\n", tv_name.c_str(), best_t, best_rmse, best_ssim, best_tv_zprof.mean);
    printf("%-30s %8.0f ms  %10.5f  %8.4f %10.5f\n", "Hybrid IR", t_hybrid, r_hybrid, s_hybrid, hybrid_zprof.mean);
    printf("%s\n", kSep72);
    printf("耗时构成: SIRT(200次子集迭代)=%.0fms, TV(20次)=%.0fms%s\n",
           t_sirt, t_tv, g_tv_gpu_ok ? " (GPU)" : " (CPU)");

    // 摘要 JSON (结构对齐 Python 版 summary.json)
    std::ostringstream js;
    js << std::fixed
       << "{\n  \"backend\": \"ASTRA CUDA " << (helical ? "helical cone-beam" : "cone-beam") << " (C++)\",\n"
       << "  \"config\": {\"N\": " << N << ", \"nz\": " << nz << ", \"n_angles\": " << n_angles
       << ", \"n_subsets\": " << n_subsets << ", \"DSO\": 1000, \"iso_det\": 500";
    if (helical) js << ", \"pitch\": 16.0";
    js << "},\n  \"results\": {\n"
       << "    \"Pure FDK\": " << json_result({fdk_rmse, fdk_ssim, fdk_t}) << ",\n";
    if (helical)
        js << "    \"Hybrid IR\": " << json_result({r_hybrid, s_hybrid, t_hybrid}) << ",\n"
           << "    \"TV-OS-SART x" << best_ni << "\": " << json_result({best_rmse, best_ssim, best_t}) << "\n";
    else
        js << "    \"TV-OS-SART x" << best_ni << "\": " << json_result({best_rmse, best_ssim, best_t}) << ",\n"
           << "    \"Hybrid IR\": " << json_result({r_hybrid, s_hybrid, t_hybrid}) << "\n";
    js << "  },\n  \"z_profile\": {\n"
       << "    \"FDK\": " << json_arr(fdk_zprof.per_slice, 5);
    if (!helical)
        js << ",\n    \"Hybrid IR\": " << json_arr(hybrid_zprof.per_slice, 5);
    js << ",\n    \"TV-OS-SART x" << best_ni << "\": " << json_arr(best_tv_zprof.per_slice, 5)
       << "\n  }\n}\n";
    std::ofstream js_out(outdir + "/cpp_summary.json");
    js_out << js.str();

    // z-profile CSV
    std::ofstream zcsv(outdir + "/cpp_zprofile.csv");
    zcsv << "z,FDK,Hybrid,TV\n";
    for (int z = 0; z < nz; ++z)
        zcsv << z << "," << fdk_zprof.per_slice[z] << "," << hybrid_zprof.per_slice[z]
             << "," << best_tv_zprof.per_slice[z] << "\n";

    printf("   => %s/cpp_*.png, cpp_*.raw, cpp_zprofile.csv\n", outdir.c_str());
    printf("\nDone!\n");
    return 0;
}
