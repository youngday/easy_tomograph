//! 度量: linear_scale / RMSE / SSIM / z-profile — 与 Python/C++ 逐段一致,
//! 用 std 线程按块并行归约 (无第三方依赖)
use crate::ffi::{N, NZ};

pub struct ZProfile {
    pub per_slice: Vec<f32>,
    pub mean: f64,
    pub max: f32,
}

fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(8)
}

/// 并行归约: 对闭包 f 在块内累加 (f 返回 (sum, count) 元组)
fn par_reduce<T: Send + Sync + Default + Copy>(
    n: usize,
    f: impl Fn(usize, usize) -> (T, usize) + Sync + Send,
    g: impl Fn(Vec<(T, usize)>) -> (T, usize),
) -> (T, usize) {
    let nthreads = num_threads().min(n.max(1));
    if nthreads <= 1 {
        return f(0, n);
    }
    let chunk = n.div_ceil(nthreads);
    let f_ref = &f; // 共享引用, 供各线程闭包捕获 (&T: Send ⇐ T: Sync)
    let mut acc: Vec<(T, usize)> = Vec::with_capacity(nthreads);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..nthreads)
            .map(|t| {
                let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(n));
                s.spawn(move || f_ref(lo, hi))
            })
            .collect();
        for h in handles {
            acc.push(h.join().unwrap());
        }
    });
    g(acc)
}

/// 最小二乘线性标定: rec*a + b ≈ gt (掩码 gt>0.001)
pub fn linear_scale(rec: &[f32], gt: &[f32]) -> Vec<f32> {
    let n = rec.len();
    let ((s_aa, s_a, s_ab, s_b), m) = par_reduce(
        n,
        |lo, hi| {
            let mut aa = 0.0f64;
            let mut a = 0.0f64;
            let mut ab = 0.0f64;
            let mut b = 0.0f64;
            let mut m = 0usize;
            for i in lo..hi {
                if gt[i] > 0.001 {
                    let r = rec[i] as f64;
                    aa += r * r;
                    a += r;
                    ab += r * gt[i] as f64;
                    b += gt[i] as f64;
                    m += 1;
                }
            }
            ((aa, a, ab, b), m)
        },
        |acc| {
            acc.iter()
                .fold(((0.0, 0.0, 0.0, 0.0), 0usize), |(t, m), (x, mm)| {
                    ((t.0 + x.0, t.1 + x.1, t.2 + x.2, t.3 + x.3), m + mm)
                })
        },
    );
    let m = m.max(1) as f64;
    let det = s_aa * m - s_a * s_a;
    let ca = (s_ab * m - s_a * s_b) / det;
    let cb = (s_aa * s_b - s_a * s_ab) / det;
    rec.iter().map(|&r| (r as f64 * ca + cb) as f32).collect()
}

pub fn calc_rmse(rec: &[f32], gt: &[f32]) -> f64 {
    let n = rec.len();
    let (s, m) = par_reduce(
        n,
        |lo, hi| {
            let mut s = 0.0f64;
            let mut m = 0usize;
            for i in lo..hi {
                if gt[i] > 0.001 {
                    let e = gt[i] as f64 - rec[i] as f64;
                    s += e * e;
                    m += 1;
                }
            }
            (s, m)
        },
        |acc| {
            acc.iter()
                .fold((0.0, 0usize), |(s, m), (x, mm)| (s + x, m + mm))
        },
    );
    (s / m.max(1) as f64).sqrt()
}

pub fn calc_ssim(rec: &[f32], gt: &[f32]) -> f64 {
    let n = rec.len();
    // 均值
    let ((mux, muy), m) = par_reduce(
        n,
        |lo, hi| {
            let mut ux = 0.0f64;
            let mut uy = 0.0f64;
            let mut m = 0usize;
            for i in lo..hi {
                if gt[i] > 0.001 {
                    ux += gt[i] as f64;
                    uy += rec[i] as f64;
                    m += 1;
                }
            }
            ((ux, uy), m)
        },
        |acc| {
            acc.iter().fold(((0.0, 0.0), 0usize), |(t, m), (x, mm)| {
                ((t.0 + x.0, t.1 + x.1), m + mm)
            })
        },
    );
    let m = m.max(1) as f64;
    let mux = mux / m;
    let muy = muy / m;
    // 方差/协方差
    let ((sx, sy, sxy), _) = par_reduce(
        n,
        |lo, hi| {
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            let mut sxy = 0.0f64;
            let mut m = 0usize;
            for i in lo..hi {
                if gt[i] > 0.001 {
                    let g = gt[i] as f64 - mux;
                    let r = rec[i] as f64 - muy;
                    sx += g * g;
                    sy += r * r;
                    sxy += g * r;
                    m += 1;
                }
            }
            ((sx, sy, sxy), m)
        },
        |acc| {
            acc.iter().fold(
                ((0.0, 0.0, 0.0), 0usize),
                |((sx, sy, sxy), m), ((x, y, z), mm)| ((sx + x, sy + y, sxy + z), m + mm),
            )
        },
    );
    let (sx, sy, sxy) = (sx / m, sy / m, sxy / m);
    let c1 = (0.01f64 * 0.05f64).powi(2);
    let c2 = (0.03f64 * 0.05f64).powi(2);
    (2.0 * mux * muy + c1) * (2.0 * sxy + c2) / ((mux * mux + muy * muy + c1) * (sx + sy + c2))
}

/// 沿 z 方向逐片 RMSE
pub fn calc_z_profile(rec: &[f32], gt: &[f32]) -> ZProfile {
    let slice = N * N;
    let per_slice: Vec<f32> = (0..NZ)
        .map(|z| {
            let base = z * slice;
            let mut s = 0.0f64;
            let mut m = 0usize;
            for i in base..base + slice {
                if gt[i] > 0.001 {
                    let e = gt[i] as f64 - rec[i] as f64;
                    s += e * e;
                    m += 1;
                }
            }
            if m > 100 {
                (s / m as f64).sqrt() as f32
            } else {
                0.0
            }
        })
        .collect();
    let mean = per_slice.iter().map(|&v| v as f64).sum::<f64>() / NZ as f64;
    let max = per_slice.iter().fold(0.0f32, |a, &b| a.max(b));
    ZProfile {
        per_slice,
        mean,
        max,
    }
}
