// ASTRA 锥束轴向 (axial) 混合重建入口
// 用法: astra_axial [phantom.raw] [outdir]
//   默认从仓库根目录运行: 输入 src_astra_cpp/data/vol_gt.raw, 输出 img_3d_axial/astra_cpp/
#include "common.h"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    std::string phantom = "src_astra_cpp/data/vol_gt.raw";
    std::string outdir = "img_3d_axial/astra_cpp";
    if (argc > 1) phantom = argv[1];
    if (argc > 2) outdir = argv[2];
    return run_pipeline(false, phantom, outdir);
}
