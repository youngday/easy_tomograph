//! 锥束 cone_vec 几何向量计算 (纯 Rust, 探测器"角落"约定 — 与 src_astra_cpp 一致)
use crate::ffi::{N_ANGLES, N_DET_COL, N_DET_ROW};

pub const DSO: f64 = 1000.0; // source-isocenter
pub const DSD_DET: f64 = 500.0; // isocenter-detector
pub const DET_PIX: f64 = 1.0; // 探测器像素尺寸
pub const PITCH_MM: f64 = 16.0; // 螺旋螺距

/// 生成 180×12 的向量数组 (每行: src xyz, det 角落 xyz, u xyz, v xyz)
/// 与 C++ 版逐位一致 (f64 计算, 探测器中心→角落转换)
pub fn build_vectors(helical: bool) -> Vec<f64> {
    let mut v = vec![0.0f64; N_ANGLES * 12];
    for i in 0..N_ANGLES {
        let th = 2.0 * std::f64::consts::PI * i as f64 / N_ANGLES as f64; // linspace(0,360,180,endpoint=False)
        let (c, s) = (th.cos(), th.sin());
        let z_src = if helical {
            PITCH_MM * (th / (2.0 * std::f64::consts::PI) - 0.5)
        } else {
            0.0
        };
        // 探测器中心 (Python 接口约定)
        let (dcx, dcy, dcz) = (-DSD_DET * s, DSD_DET * c, z_src);
        // 探测器 u/v 向量
        let (ux, uy, uz) = (DET_PIX * c, DET_PIX * s, 0.0);
        let (vx, vy, vz) = (0.0, 0.0, DET_PIX);
        // 中心 → 角落 (bottom-left): 减半个探测器尺寸
        let sx = dcx - 0.5 * N_DET_ROW as f64 * vx - 0.5 * N_DET_COL as f64 * ux;
        let sy = dcy - 0.5 * N_DET_ROW as f64 * vy - 0.5 * N_DET_COL as f64 * uy;
        let sz = dcz - 0.5 * N_DET_ROW as f64 * vz - 0.5 * N_DET_COL as f64 * uz;
        let o = i * 12;
        v[o..o + 12].copy_from_slice(&[
            DSO * s,
            -DSO * c,
            z_src, // source
            sx,
            sy,
            sz, // detector corner
            ux,
            uy,
            uz, // det u-vector
            vx,
            vy,
            vz, // det v-vector
        ]);
    }
    v
}
