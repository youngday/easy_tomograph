// ASTRA 锥束轴向 (axial) 混合重建 (Rust 版)
// 用法: astra_rs_axial [phantom.raw] [outdir] [max_epochs=10] [target_rmse=0.001]
//   默认从仓库根目录运行: 输入 src_astra_cpp/data/vol_gt.raw, 输出 img_3d_axial/astra_rs/
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let phantom = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "src_astra_cpp/data/vol_gt.raw".to_string());
    let outdir = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "img_3d_axial/astra_rs".to_string());
    let max_epochs: i32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let target_rmse: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.001);
    std::process::exit(astra_rs::pipeline::run_pipeline(
        false,
        &phantom,
        &outdir,
        max_epochs,
        target_rmse,
    ));
}
