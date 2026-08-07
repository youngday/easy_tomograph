#pragma once
// ASTRA 锥束混合重建 (C++ 移植版) — 公共接口
// 对应 Python 版: src_3d_axial/astra_cone_hybrid.py 与 src_3d_helical/astra_cone_hybrid.py

#include <string>
#include <vector>

// 单个算法结果
struct AlgorithmResult {
    double rmse = 0.0;
    double ssim = 0.0;
    double time_ms = 0.0;
};

// 沿 z 方向逐片 RMSE (z-profile)
struct ZProfile {
    std::vector<float> per_slice;  // nz 个值
    double mean = 0.0;
    double max = 0.0;
};

// 运行完整的 ASTRA 锥束混合重建流水线 (FDK / TV-OS-SART / Hybrid IR)
//   helical: true  = 螺旋轨迹 (pitch=16mm/圈), false = 轴向圆轨迹
//   phantom_path: 输入体模 (float32, nz*N*N, [z][y][x] 布局)
//   outdir:        输出目录 (JSON 摘要 / 结果图 / .raw 结果体)
//   max_epochs:    迭代轮数上限 (默认 10)
//   target_rmse:   RMSE 达标即提前停止 (默认 0.001; 设为 0 则跑满 max_epochs)
// 返回 0 表示成功
int run_pipeline(bool helical, const std::string& phantom_path, const std::string& outdir,
                 int max_epochs = 10, double target_rmse = 0.001);
