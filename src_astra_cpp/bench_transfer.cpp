// 微基准: 拆解单次 SIRT 子集迭代的耗时构成 (传输 vs 内核)
#include <astra/Globals.h>
#include <astra/Config.h>
#include <astra/Algorithm.h>
#include <astra/ConeVecProjectionGeometry3D.h>
#include <astra/VolumeGeometry3D.h>
#include <astra/Data3D.h>
#include <astra/CudaProjector3D.h>
#include <astra/CudaSirtAlgorithm3D.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

int main() {
    const int N = 512, nz = 32, n_angles = 18;  // 单子集
    const int n_det_row = 64, n_det_col = 725;
    const double DSO = 1000.0, DSD_det = 500.0, det_pix = 1.0;
    const size_t nvol = (size_t)nz * N * N;
    const size_t nsino = (size_t)n_det_row * n_angles * n_det_col;

    std::vector<astra::SConeProjection> vecs(n_angles);
    for (int i = 0; i < n_angles; ++i) {
        double th = 2.0 * M_PI * i / n_angles;
        double c = std::cos(th), s = std::sin(th);
        double sx = -DSD_det * s - 0.5 * n_det_col * det_pix * c;
        double sy = DSD_det * c - 0.5 * n_det_col * det_pix * s;
        double sz = -0.5 * n_det_row * det_pix;
        vecs[i] = {DSO * s, -DSO * c, 0.0, sx, sy, sz,
                   det_pix * c, det_pix * s, 0.0, 0.0, 0.0, det_pix};
    }
    astra::CConeVecProjectionGeometry3D projGeom(n_angles, n_det_row, n_det_col, std::move(vecs));
    astra::CVolumeGeometry3D volGeom(N, N, nz);
    astra::CCudaProjector3D projector(projGeom, volGeom);

    auto* sino = astra::createCFloat32ProjectionData3DMemory(projGeom);
    auto* vol = astra::createCFloat32VolumeData3DMemory(volGeom);
    for (size_t i = 0; i < nsino; ++i) sino->getFloat32Memory()[i] = 0.01f * (i % 7);
    std::memset(vol->getFloat32Memory(), 0, nvol * sizeof(float));
    std::unique_ptr<astra::CCudaSirtAlgorithm3D> alg(
        new astra::CCudaSirtAlgorithm3D(&projector, sino, vol));

    std::vector<float> rec(nvol, 0.01f);
    const int R = 50;

    // 1) 纯 CPU memcpy 往返 (rec -> vol, vol -> rec)
    {
        auto t0 = clk::now();
        for (int r = 0; r < R; ++r) {
            std::memcpy(vol->getFloat32Memory(), rec.data(), nvol * sizeof(float));
            std::memcpy(rec.data(), vol->getFloat32Memory(), nvol * sizeof(float));
        }
        printf("1) CPU memcpy 往返: %.2f ms/次 (2×33MB)\n", ms_since(t0) / R);
    }
    // 2) 纯 run(1) (内部 upload + 内核 + download, 无外部 memcpy)
    {
        auto t0 = clk::now();
        for (int r = 0; r < R; ++r) alg->run(1);
        printf("2) run(1) (GPU):   %.2f ms/次\n", ms_since(t0) / R);
    }
    // 3) 完整迭代 (与真实循环一致)
    {
        auto t0 = clk::now();
        for (int r = 0; r < R; ++r) {
            std::memcpy(vol->getFloat32Memory(), rec.data(), nvol * sizeof(float));
            alg->run(1);
            std::memcpy(rec.data(), vol->getFloat32Memory(), nvol * sizeof(float));
        }
        printf("3) 完整迭代:       %.2f ms/次  (1+2 应≈3)\n", ms_since(t0) / R);
    }
    return 0;
}
