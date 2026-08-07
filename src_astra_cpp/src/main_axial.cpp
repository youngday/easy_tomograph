// ASTRA 锥束轴向 (axial) 混合重建入口
// 用法: astra_axial [phantom.raw] [outdir] [max_epochs=10] [target_rmse=0.001]
//   默认从仓库根目录运行: 输入 src_astra_cpp/data/vol_gt.raw, 输出 img_3d_axial/astra_cpp/
//   target_rmse=0 表示跑满 max_epochs (不做提前停止)
#include "common.h"

#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    std::string phantom = "src_astra_cpp/data/vol_gt.raw";
    std::string outdir = "img_3d_axial/astra_cpp";
    int max_epochs = 10;
    double target_rmse = 0.001;
    if (argc > 1) phantom = argv[1];
    if (argc > 2) outdir = argv[2];
    if (argc > 3) max_epochs = std::atoi(argv[3]);
    if (argc > 4) target_rmse = std::atof(argv[4]);
    return run_pipeline(false, phantom, outdir, max_epochs, target_rmse);
}
