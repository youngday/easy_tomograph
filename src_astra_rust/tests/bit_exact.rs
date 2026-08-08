//! GPU 回归测试: Rust 管线输出 vs C++ 版参考。
//! GPU 内核路径 (FP/FDK/SART/TV) 与数据搬运逐位一致; CPU 端 linear_scale 的
//! 并行浮点归约顺序不同 (C++ OpenMP vs Rust 分块), 导致最终 raw 有 ≤~4e-9 的
//! ulp 级差异, 故对比容差取 1e-5 (远小于目标 RMSE 1e-3, 能抓住真实回归)。
//! 前置: 已生成体模/共享噪声, 且已运行 C++ 版生成参考输出:
//!   python3 src_astra_cpp/tools/make_phantom.py
//!   python3 src_astra_cpp/tools/make_sino_noisy.py
//!   (运行 src_astra_cpp 的 axial/helical 可执行文件, 产出 img_3d_{axial,helical}/astra_cpp/)
//! 运行: cargo test --release -- --ignored   (需要 GPU + libastra)
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const TOL: f32 = 1e-5;

// 两个测试共用 GPU: 串行执行, 避免 GTX 1660 (6GB) 显存不足
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

fn load_raw(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("无法读取 {}: {e}", path.display()));
    assert_eq!(
        bytes.len() % 4,
        0,
        "raw 文件大小不是 4 的倍数: {}",
        path.display()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "长度不一致: {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn compare_raw(got: &Path, cpp_ref: &Path, name: &str) {
    let a = load_raw(got);
    let b = load_raw(cpp_ref);
    let d = max_diff(&a, &b);
    assert!(
        d <= TOL,
        "{name}: 与 C++ 参考差异过大, max-diff={d:e} > {TOL:e}\n  got={}\n  ref={}",
        got.display(),
        cpp_ref.display()
    );
}

/// 从 cpp_summary.json 提取 "epochs": {"tv_ossart": N, "hybrid": M}
fn read_epochs(path: &Path) -> (i64, i64) {
    let s = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("无法读取 {}: {e}", path.display()));
    let tv = s
        .split("\"tv_ossart\"")
        .nth(1)
        .and_then(|r| r.split([':', ',', '}']).nth(1))
        .and_then(|x| x.trim().parse().ok())
        .unwrap_or_else(|| panic!("JSON 中找不到 tv_ossart: {}", path.display()));
    let hy = s
        .split("\"hybrid\"")
        .nth(1)
        .and_then(|r| r.split([':', ',', '}']).nth(1))
        .and_then(|x| x.trim().parse().ok())
        .unwrap_or_else(|| panic!("JSON 中找不到 hybrid: {}", path.display()));
    (tv, hy)
}

fn run_mode(helical: bool, mode: &str) {
    let _guard = GPU_LOCK.lock().unwrap();
    let root = repo_root();
    let phantom = root.join("src_astra_cpp/data/vol_gt.raw");
    assert!(
        phantom.exists(),
        "体模缺失: {} (先运行 make_phantom.py)",
        phantom.display()
    );
    let noisy = root.join(format!("src_astra_cpp/data/sino_noisy_{mode}.raw"));
    assert!(
        noisy.exists(),
        "共享噪声缺失: {} (先运行 make_sino_noisy.py)",
        noisy.display()
    );
    let cpp_dir = root.join(format!("img_3d_{mode}/astra_cpp"));
    for f in [
        "cpp_fdk.raw",
        "cpp_fdk_noisy.raw",
        "cpp_tv.raw",
        "cpp_hybrid.raw",
    ] {
        assert!(
            cpp_dir.join(f).exists(),
            "C++ 参考缺失: {} (先运行 src_astra_cpp)",
            cpp_dir.join(f).display()
        );
    }

    let outdir = std::env::temp_dir().join(format!("astra_rs_bit_exact_{mode}"));
    let _ = std::fs::remove_dir_all(&outdir);
    let rc = astra_rs::pipeline::run_pipeline(
        helical,
        phantom.to_str().unwrap(),
        outdir.to_str().unwrap(),
        10,    // max_epochs (与 C++ 参考一致)
        0.001, // target_rmse
    );
    assert_eq!(rc, 0, "管线返回失败");

    for f in [
        "cpp_fdk.raw",
        "cpp_fdk_noisy.raw",
        "cpp_tv.raw",
        "cpp_hybrid.raw",
    ] {
        compare_raw(&outdir.join(f), &cpp_dir.join(f), f);
    }
    let (tv_got, hy_got) = read_epochs(&outdir.join("cpp_summary.json"));
    let (tv_ref, hy_ref) = read_epochs(&cpp_dir.join("cpp_summary.json"));
    assert_eq!(
        (tv_got, hy_got),
        (tv_ref, hy_ref),
        "提前停止轮次不一致: got TV={tv_got}/Hybrid={hy_got}, ref TV={tv_ref}/Hybrid={hy_ref}"
    );
    println!("✓ {mode}: 4 个输出与 C++ 参考一致 (max-diff≤{TOL:e}), 提前停止 TV x{tv_got}/Hybrid x{hy_got} 一致");
}

#[test]
#[ignore = "需要 GPU + C++ 参考输出 (先跑 src_astra_cpp 生成 img_3d_*/astra_cpp/cpp_*.raw)"]
fn axial_bit_exact_with_cpp() {
    run_mode(false, "axial");
}

#[test]
#[ignore = "需要 GPU + C++ 参考输出 (先跑 src_astra_cpp 生成 img_3d_*/astra_cpp/cpp_*.raw)"]
fn helical_bit_exact_with_cpp() {
    run_mode(true, "helical");
}
