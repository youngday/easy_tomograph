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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} vs {b}");
    }

    /// 轴向第 0 帧: 源在 (0,-1000,0), 探测器角落 (-362.5, 500, 0)
    #[test]
    fn axial_frame0() {
        let v = build_vectors(false);
        // source
        assert_close(v[0], 0.0, 1e-9);
        assert_close(v[1], -DSO, 1e-9);
        assert_close(v[2], 0.0, 1e-9);
        // det corner (中心 (0, 500, 0) − 半个探测器: u 沿 +x, v 沿 +z)
        assert_close(v[3], -0.5 * N_DET_COL as f64 * DET_PIX, 1e-9);
        assert_close(v[4], DSD_DET, 1e-9);
        assert_close(v[5], -0.5 * N_DET_ROW as f64 * DET_PIX, 1e-9);
        // u/v 向量
        assert_close(v[6], DET_PIX, 1e-9);
        assert_close(v[7], 0.0, 1e-9);
        assert_close(v[8], 0.0, 1e-9);
        assert_close(v[9], 0.0, 1e-9);
        assert_close(v[10], 0.0, 1e-9);
        assert_close(v[11], DET_PIX, 1e-9);
    }

    /// 轴向第 45 帧 (θ=90°): 源在 (1000, 0, 0), 探测器角落 (500, -362.5, 0)
    #[test]
    fn axial_frame45() {
        let v = build_vectors(false);
        let o = 45 * 12;
        assert_close(v[o + 0], DSO, 1e-9);
        assert_close(v[o + 1], 0.0, 1e-9);
        assert_close(v[o + 2], 0.0, 1e-9);
        // det corner: 中心 (−500, ~0, 0) − 半个探测器 (u 沿 +y, v 沿 +z)
        assert_close(v[o + 3], -DSD_DET, 1e-9);
        assert_close(v[o + 4], -0.5 * N_DET_COL as f64 * DET_PIX, 1e-9);
        assert_close(v[o + 5], -0.5 * N_DET_ROW as f64 * DET_PIX, 1e-9);
    }

    /// 螺旋 z: 源/探测器 z = PITCH*(θ/2π − 0.5), 第 0 帧 -8, 中间帧 0
    #[test]
    fn helical_z_positions() {
        let v = build_vectors(true);
        assert_close(v[2], -PITCH_MM * 0.5, 1e-9); // i=0
        let mid = (N_ANGLES / 2) * 12;
        assert_close(v[mid + 2], 0.0, 1e-9); // i=90
        assert_close(v[mid + 5], -0.5 * N_DET_ROW as f64 * DET_PIX, 1e-9); // det corner z 同步 −32
        let last = (N_ANGLES - 1) * 12;
        let z_last = PITCH_MM * ((N_ANGLES - 1) as f64 / N_ANGLES as f64 - 0.5);
        assert_close(v[last + 2], z_last, 1e-9);
    }

    /// 轴向 z 恒为 0; 每帧 12 个元素, 共 180 帧
    #[test]
    fn vector_layout() {
        for helical in [false, true] {
            let v = build_vectors(helical);
            assert_eq!(v.len(), N_ANGLES * 12);
        }
        let v = build_vectors(false);
        for i in 0..N_ANGLES {
            assert_close(v[i * 12 + 2], 0.0, 1e-9);
        }
    }
}
