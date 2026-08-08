//! 主流水线 (镜像 src_astra_cpp/src/common.cpp 的 run_pipeline):
//! FP → 噪声 → FDK → TV-OS-SART(提前停止) → Hybrid(提前停止) → 输出
use crate::ffi::{self, Geom, Sart, N, NSINO, NVOL, NZ, N_ANGLES, N_DET_COL, N_DET_ROW, N_SUBSETS};
use crate::metrics;
use std::path::Path;

const SEP60: &str = "============================================================";
const SEP55: &str = "-------------------------------------------------------";
const SEP70: &str = "======================================================================";
const SEP72: &str = "------------------------------------------------------------------------";

struct Stopwatch(std::time::Instant);
impl Stopwatch {
    fn new() -> Self {
        Self(std::time::Instant::now())
    }
    fn ms(&self) -> f64 {
        self.0.elapsed().as_secs_f64() * 1000.0
    }
}

fn load_raw(path: &str, n: usize) -> Option<Vec<f32>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() != n * 4 {
        return None;
    }
    let mut v = vec![0.0f32; n];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        v[i] = f32::from_ne_bytes(chunk.try_into().unwrap());
    }
    Some(v)
}

fn save_raw(path: &str, v: &[f32]) {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &x in v {
        bytes.extend_from_slice(&x.to_ne_bytes());
    }
    let _ = std::fs::write(path, bytes);
}

fn make_dir(path: &str) {
    let _ = std::fs::create_dir_all(path);
}

fn json_arr(v: &[f32]) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{:.5}", x)).collect();
    format!("[{}]", items.join(", "))
}

fn json_result(rmse: f64, ssim: f64, time_ms: f64) -> String {
    format!(
        "{{\"rmse\": {:.5}, \"ssim\": {:.4}, \"time_ms\": {:.1}}}",
        rmse, ssim, time_ms
    )
}

fn fmt_f32(v: f32) -> String {
    format!("{:.5}", v)
}

/// 运行完整流水线; 返回 0 表示成功
pub fn run_pipeline(
    helical: bool,
    phantom_path: &str,
    outdir: &str,
    max_epochs: i32,
    target_rmse: f64,
) -> i32 {
    println!("{}", SEP60);
    println!(
        "{}  [锥束 CBCT | ASTRA CUDA Rust]",
        if helical {
            "螺旋(Helical) 混合重建"
        } else {
            "FBP + IR 混合重建"
        }
    );
    println!("{}", SEP60);

    if let Err(e) = ffi::check_sizes() {
        println!("错误: {}", e);
        return 1;
    }

    // ---- 1. 载入体模 ----
    let vol_gt = match load_raw(phantom_path, NVOL) {
        Some(v) => v,
        None => {
            println!(
                "错误: 无法读取体模文件 {} (先运行 src_astra_cpp/tools/make_phantom.py)",
                phantom_path
            );
            return 1;
        }
    };
    let gt_min = vol_gt.iter().fold(f32::MAX, |a, &b| a.min(b));
    let gt_max = vol_gt.iter().fold(f32::MIN, |a, &b| a.max(b));
    println!("   体模: [{:.5}, {:.5}]", gt_min, gt_max);

    // ---- 2. 几何 + FP ----
    let geom = match Geom::create(helical) {
        Ok(g) => g,
        Err(e) => {
            println!("错误: 几何创建失败: {}", e);
            return 1;
        }
    };
    println!("\nGPU 正向投影...");
    let sw_fp = Stopwatch::new();
    let mut sino = vec![0.0f32; NSINO];
    if let Err(e) = geom.fp(&vol_gt, &mut sino) {
        println!("错误: FP 失败: {}", e);
        return 1;
    }
    println!(
        "   完成: {:.0}ms, 形状 ({}, {}, {})",
        sw_fp.ms(),
        N_DET_ROW,
        N_ANGLES,
        N_DET_COL
    );

    // ---- 3. 噪声 (优先共享文件) ----
    let sino_noisy: Vec<f32>;
    {
        let phantom_dir = Path::new(phantom_path)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string());
        let mode = if helical { "helical" } else { "axial" };
        let noise_path = format!("{}/sino_noisy_{}.raw", phantom_dir, mode);
        if let Some(s) = load_raw(&noise_path, NSINO) {
            sino_noisy = s;
            println!(
                "{}\n使用共享噪声文件 {} (与 Python 版逐位一致)",
                SEP55, noise_path
            );
        } else {
            sino_noisy = crate::noise::add_artifacts(&sino);
            println!(
                "{}\n未找到 {}, 使用内置噪声 (MT19937, 与 Python 版略有差异)",
                SEP55, noise_path
            );
        }
    }

    // ---- 4. A. Pure FDK ----
    println!("{}\nA. Pure FDK\n{}", SEP55, SEP55);
    let sw_fdk = Stopwatch::new();
    let mut fdk_raw = vec![0.0f32; NVOL];
    if let Err(e) = geom.fdk(&sino, &mut fdk_raw) {
        println!("错误: FDK 失败: {}", e);
        return 1;
    }
    let fdk_t = sw_fdk.ms();
    let fdk_rec = metrics::linear_scale(&fdk_raw, &vol_gt);
    let fdk_rmse = metrics::calc_rmse(&fdk_rec, &vol_gt);
    let fdk_ssim = metrics::calc_ssim(&fdk_rec, &vol_gt);
    let fdk_zprof = metrics::calc_z_profile(&fdk_rec, &vol_gt);
    println!(
        "   RMSE={:.5}, SSIM={:.4}, {:.0}ms",
        fdk_rmse, fdk_ssim, fdk_t
    );
    println!(
        "   z-profile: mean={:.5}, max={:.5}",
        fdk_zprof.mean, fdk_zprof.max
    );

    // ---- 5. FDK(noisy) ----
    let sw_fdk_n = Stopwatch::new();
    let mut rec_fdk_n = vec![0.0f32; NVOL];
    if let Err(e) = geom.fdk(&sino_noisy, &mut rec_fdk_n) {
        println!("错误: FDK(noisy) 失败: {}", e);
        return 1;
    }
    let fdk_noisy_t = sw_fdk_n.ms();
    let rec_fdk_n_ls = metrics::linear_scale(&rec_fdk_n, &vol_gt);
    let fdk_noisy_rmse = metrics::calc_rmse(&rec_fdk_n_ls, &vol_gt);
    let fdk_noisy_ssim = metrics::calc_ssim(&rec_fdk_n_ls, &vol_gt);
    let fdk_noisy_zprof = metrics::calc_z_profile(&rec_fdk_n_ls, &vol_gt);
    println!(
        "   FDK(noisy) RMSE={:.5}, SSIM={:.4}, {:.0}ms",
        fdk_noisy_rmse, fdk_noisy_ssim, fdk_noisy_t
    );

    // ---- 6. GPU 常驻 OS-SART ----
    let sart = match Sart::create(helical, &sino_noisy) {
        Ok(s) => s,
        Err(e) => {
            println!("错误: GPU 常驻 SART 初始化失败: {}", e);
            return 1;
        }
    };
    let mut sart_err = String::new();
    let mut ossart_epoch = |rec: &mut Vec<f32>| -> Result<(), String> {
        let tmp = rec.clone(); // 避免 sart.run(&tmp, …, rec) 的别名借用
        match sart.run(&tmp, 1, rec) {
            Ok(()) => Ok(()),
            Err(e) => {
                sart_err = e;
                Err(sart_err.clone())
            }
        }
    };

    // ---- 7. B. TV-OS-SART (提前停止) ----
    println!(
        "{}\nB. TV-OS-SART (RMSE≤{:.3} 提前停止, 上限{}轮)\n{}",
        SEP55, target_rmse, max_epochs, SEP55
    );
    let beta0 = 0.002f32;
    let decay = 0.8f32;
    let w_z = 1.5f32;
    let mut rec_tv = rec_fdk_n.clone();
    let mut t_tv_total = 0.0f64;
    let mut best_rmse = 1e9f64;
    let mut best_ssim = 0.0f64;
    let mut best_t = 0.0f64;
    let mut best_ni = 0i32;
    let mut best_rec: Vec<f32> = Vec::new();
    let mut t_sirt = 0.0f64;
    let mut t_tv = 0.0f64;
    for ni in 1..=max_epochs {
        let sw_sirt = Stopwatch::new();
        if let Err(e) = ossart_epoch(&mut rec_tv) {
            println!("错误: SART 运行失败: {}", e);
            return 1;
        }
        t_sirt += sw_sirt.ms();
        let sw_tv = Stopwatch::new();
        if let Err(e) = ffi::tv(&mut rec_tv, beta0 * decay.powi(ni - 1), w_z) {
            println!("错误: TV 失败: {}", e);
            return 1;
        }
        t_tv += sw_tv.ms();
        t_tv_total += sw_sirt.ms() + sw_tv.ms();
        let ls = metrics::linear_scale(&rec_tv, &vol_gt);
        let r = metrics::calc_rmse(&ls, &vol_gt);
        let s = metrics::calc_ssim(&ls, &vol_gt);
        if r < best_rmse {
            best_rmse = r;
            best_ssim = s;
            best_rec = ls;
            best_t = t_tv_total;
            best_ni = ni;
        }
        println!(
            "   TV-OS-SART x{:3} (β={:.4}): RMSE={:.5}, SSIM={:.4}, 累计{:.0}ms",
            ni,
            beta0 * decay.powi(ni - 1),
            r,
            s,
            t_tv_total
        );
        if target_rmse > 0.0 && r <= target_rmse {
            println!(
                "   ✓ RMSE={:.5} ≤ {:.3}, 提前停止于 x{}",
                r, target_rmse, ni
            );
            break;
        }
    }
    println!("   >> 最优: TV-OS-SART x{}: RMSE={:.5}", best_ni, best_rmse);
    let tv_improv = (1.0 - best_rmse / fdk_noisy_rmse) * 100.0;
    println!(
        "   TV 改善 vs 噪声FDK({:.5}): {:+.1}%",
        fdk_noisy_rmse, tv_improv
    );
    let best_tv_zprof = metrics::calc_z_profile(&best_rec, &vol_gt);
    println!(
        "   TV-OS-SART z-profile: mean={:.5}, max={:.5}",
        best_tv_zprof.mean, best_tv_zprof.max
    );

    // ---- 8. C. Hybrid IR (提前停止) ----
    println!(
        "{}\nC. Hybrid IR (OS-SART×{} + TV×{}(β递减) + FDK混合 10%, RMSE≤{:.3} 提前停止)\n{}",
        SEP55, max_epochs, max_epochs, target_rmse, SEP55
    );
    let sw_h = Stopwatch::new();
    let mut rec_h = rec_fdk_n.clone();
    let mut hyb_epochs = max_epochs;
    for ni in 0..max_epochs {
        let sw_sirt2 = Stopwatch::new();
        if let Err(e) = ossart_epoch(&mut rec_h) {
            println!("错误: SART 运行失败: {}", e);
            return 1;
        }
        t_sirt += sw_sirt2.ms();
        let sw_tv2 = Stopwatch::new();
        if let Err(e) = ffi::tv(&mut rec_h, beta0 * decay.powi(ni), w_z) {
            println!("错误: TV 失败: {}", e);
            return 1;
        }
        t_tv += sw_tv2.ms();
        hyb_epochs = ni + 1;
        if target_rmse > 0.0 {
            let blend: Vec<f32> = rec_h
                .iter()
                .zip(rec_fdk_n.iter())
                .map(|(a, b)| 0.9 * a + 0.1 * b)
                .collect();
            let r = metrics::calc_rmse(&metrics::linear_scale(&blend, &vol_gt), &vol_gt);
            if r <= target_rmse {
                println!(
                    "   ✓ Hybrid RMSE={:.5} ≤ {:.3}, 提前停止于 x{}",
                    r, target_rmse, hyb_epochs
                );
                break;
            }
        }
    }
    for i in 0..NVOL {
        rec_h[i] = 0.9 * rec_h[i] + 0.1 * rec_fdk_n[i];
    }
    let rec_h_ls = metrics::linear_scale(&rec_h, &vol_gt);
    let r_hybrid = metrics::calc_rmse(&rec_h_ls, &vol_gt);
    let s_hybrid = metrics::calc_ssim(&rec_h_ls, &vol_gt);
    let t_hybrid = sw_h.ms();
    let hybrid_zprof = metrics::calc_z_profile(&rec_h_ls, &vol_gt);
    println!(
        "   Hybrid IR: RMSE={:.5}, SSIM={:.4}, {:.0}ms",
        r_hybrid, s_hybrid, t_hybrid
    );
    println!(
        "   z-profile: mean={:.5}, max={:.5}",
        hybrid_zprof.mean, hybrid_zprof.max
    );

    // ---- 9. 输出 ----
    println!("\n生成输出...");
    make_dir(outdir);
    save_raw(&format!("{}/cpp_fdk.raw", outdir), &fdk_rec);
    save_raw(&format!("{}/cpp_fdk_noisy.raw", outdir), &rec_fdk_n_ls);
    save_raw(&format!("{}/cpp_tv.raw", outdir), &best_rec);
    save_raw(&format!("{}/cpp_hybrid.raw", outdir), &rec_h_ls);

    // 汇总表
    println!(
        "\n{}\n汇总对比 (32x512x512, {}角度, {}子集)\n{}",
        SEP70, N_ANGLES, N_SUBSETS, SEP70
    );
    println!(
        "{:<30} {:>10} {:>12} {:>8} {:>10}",
        "算法", "耗时(ms)", "RMSE", "SSIM", "z-RMSE"
    );
    println!("{}", SEP72);
    println!(
        "{:<30} {:>8.0} ms  {:>10.5}  {:>8.4} {:>10.5}",
        "Pure FDK", fdk_t, fdk_rmse, fdk_ssim, fdk_zprof.mean
    );
    println!(
        "{:<30} {:>8.0} ms  {:>10.5}  {:>8.4} {:>10.5}",
        "FDK(noisy)", fdk_noisy_t, fdk_noisy_rmse, fdk_noisy_ssim, fdk_noisy_zprof.mean
    );
    let tv_name = format!("TV-OS-SART x{}", best_ni);
    println!(
        "{:<30} {:>8.0} ms  {:>10.5}  {:>8.4} {:>10.5}",
        tv_name, best_t, best_rmse, best_ssim, best_tv_zprof.mean
    );
    println!(
        "{:<30} {:>8.0} ms  {:>10.5}  {:>8.4} {:>10.5}",
        "Hybrid IR", t_hybrid, r_hybrid, s_hybrid, hybrid_zprof.mean
    );
    println!("{}", SEP72);
    let n_sirt = (best_ni + hyb_epochs) * N_SUBSETS as i32;
    println!(
        "耗时构成: SIRT({}次子集迭代)={:.0}ms, TV({}次)={:.0}ms (GPU)",
        n_sirt,
        t_sirt,
        best_ni + hyb_epochs,
        t_tv
    );

    // 摘要 JSON
    let backend = if helical {
        "ASTRA CUDA helical cone-beam (Rust)"
    } else {
        "ASTRA CUDA cone-beam (Rust)"
    };
    let mut js = String::new();
    js.push_str(&format!("{{\n  \"backend\": \"{}\",\n", backend));
    js.push_str(&format!(
        "  \"config\": {{\"N\": {}, \"nz\": {}, \"n_angles\": {}, \"n_subsets\": {}, \"DSO\": 1000, \"iso_det\": 500",
        N, NZ, N_ANGLES, N_SUBSETS
    ));
    if helical {
        js.push_str(", \"pitch\": 16.0");
    }
    js.push_str(&format!(
        ", \"target_rmse\": {:.3}, \"epochs\": {{\"tv_ossart\": {}, \"hybrid\": {}}}",
        target_rmse, best_ni, hyb_epochs
    ));
    js.push_str("},\n  \"results\": {\n");
    js.push_str(&format!(
        "    \"Pure FDK\": {},\n",
        json_result(fdk_rmse, fdk_ssim, fdk_t)
    ));
    js.push_str(&format!(
        "    \"FDK(noisy)\": {},\n",
        json_result(fdk_noisy_rmse, fdk_noisy_ssim, fdk_noisy_t)
    ));
    if helical {
        js.push_str(&format!(
            "    \"Hybrid IR\": {},\n",
            json_result(r_hybrid, s_hybrid, t_hybrid)
        ));
        js.push_str(&format!(
            "    \"TV-OS-SART x{}\": {}\n",
            best_ni,
            json_result(best_rmse, best_ssim, best_t)
        ));
    } else {
        js.push_str(&format!(
            "    \"TV-OS-SART x{}\": {},\n",
            best_ni,
            json_result(best_rmse, best_ssim, best_t)
        ));
        js.push_str(&format!(
            "    \"Hybrid IR\": {}\n",
            json_result(r_hybrid, s_hybrid, t_hybrid)
        ));
    }
    js.push_str("  },\n  \"z_profile\": {\n");
    js.push_str(&format!(
        "    \"FDK\": {},\n",
        json_arr(&fdk_zprof.per_slice)
    ));
    js.push_str(&format!(
        "    \"FDK(noisy)\": {},\n",
        json_arr(&fdk_noisy_zprof.per_slice)
    ));
    js.push_str(&format!(
        "    \"Hybrid IR\": {},\n",
        json_arr(&hybrid_zprof.per_slice)
    ));
    js.push_str(&format!(
        "    \"TV-OS-SART x{}\": {}\n  }}\n}}\n",
        best_ni,
        json_arr(&best_tv_zprof.per_slice)
    ));
    let _ = std::fs::write(format!("{}/cpp_summary.json", outdir), &js);

    // z-profile CSV
    let mut csv = String::from("z,FDK,Hybrid,TV\n");
    for z in 0..NZ {
        csv.push_str(&format!(
            "{},{},{},{}\n",
            z,
            fmt_f32(fdk_zprof.per_slice[z]),
            fmt_f32(hybrid_zprof.per_slice[z]),
            fmt_f32(best_tv_zprof.per_slice[z])
        ));
    }
    let _ = std::fs::write(format!("{}/cpp_zprofile.csv", outdir), csv);

    println!(
        "   => {}/cpp_*.raw, cpp_summary.json, cpp_zprofile.csv",
        outdir
    );

    // ---- 渲染结果图 (与 C++ 共用 render_results.py) ----
    {
        let py = std::env::var("PYTHON").unwrap_or_else(|_| {
            if Path::new(".venv/bin/python").exists() {
                ".venv/bin/python".to_string()
            } else {
                "python3".to_string()
            }
        });
        let mode = if helical { "helical" } else { "axial" };
        let rcmd = format!(
            "{} src_astra_cpp/tools/render_results.py {} \"{}\"",
            py, mode, outdir
        );
        println!("渲染结果图: {}", rcmd);
        let status = std::process::Command::new(&py)
            .args(["src_astra_cpp/tools/render_results.py", mode, outdir])
            .status();
        if let Ok(st) = status {
            if !st.success() {
                println!("   (渲染失败, 可手动运行: {})", rcmd);
            }
        }
    }

    println!("\nDone!");
    0
}
