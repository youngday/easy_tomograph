#pragma once
// 共享 cone_vec 几何构建: 轴向/螺旋向量 (探测器"角落"约定, 与 Python 接口一致)
// 供 common.cpp 与 sart_gpu.cpp 共用, 避免两处各自实现导致约定漂移
#include <astra/GeometryUtil3D.h>  // SConeProjection

#include <cmath>
#include <vector>

inline std::vector<astra::SConeProjection> astra_cpp_build_vectors(
    bool helical, int n_angles, int n_det_row, int n_det_col,
    double DSO, double DSD_det, double det_pix, double pitch_mm) {
    std::vector<astra::SConeProjection> vecs(n_angles);
    for (int i = 0; i < n_angles; ++i) {
        double th = 2.0 * M_PI * i / n_angles;  // linspace(0,360,n,endpoint=False)
        double c = std::cos(th), s = std::sin(th);
        double z_src = helical ? pitch_mm * (th / (2.0 * M_PI) - 0.5) : 0.0;
        // 探测器中心 (Python 接口约定) → 角落 (bottom-left, 后端约定)
        double dcx = -DSD_det * s, dcy = DSD_det * c, dcz = z_src;
        double ux = det_pix * c, uy = det_pix * s, uz = 0.0;
        double vx = 0.0, vy = 0.0, vz = det_pix;
        double sx = dcx - 0.5 * n_det_row * vx - 0.5 * n_det_col * ux;
        double sy = dcy - 0.5 * n_det_row * vy - 0.5 * n_det_col * uy;
        double sz = dcz - 0.5 * n_det_row * vz - 0.5 * n_det_col * uz;
        vecs[i] = {DSO * s, -DSO * c, z_src,  // source
                   sx, sy, sz,                // detector corner
                   ux, uy, uz,                // det u-vector
                   vx, vy, vz};               // det v-vector
    }
    return vecs;
}
