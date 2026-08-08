//! 内置噪声回退 (仅在共享噪声文件 sino_noisy_*.raw 缺失时使用):
//! 与 ct_noise.py 相同模型 (泊松-高斯 + 环形伪影, seed=2024), RNG 为 MT19937
//! (与 numpy/std::mt19937 不逐位一致 → 结果与 Python/C++ 内置噪声略有差异, 同数量级)
use crate::ffi::{NSINO, N_ANGLES, N_DET_COL, N_DET_ROW};

struct Mt19937 {
    mt: [u32; 624],
    idx: usize,
}

impl Mt19937 {
    fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { mt, idx: 624 }
    }
    fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            for i in 0..624 {
                let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
                self.mt[i] =
                    self.mt[(i + 397) % 624] ^ (y >> 1) ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }
    fn uniform(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }
    fn uniform_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.uniform() * (hi - lo)
    }
    fn randint(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }
    /// 泊松 (Knuth)
    fn poisson(&mut self, lam: f64) -> u32 {
        if lam <= 0.0 {
            return 0;
        }
        let l = (-lam).exp();
        let mut k = 0u32;
        let mut p = 1.0f64;
        loop {
            k += 1;
            p *= self.uniform() as f64;
            if p <= l {
                break;
            }
        }
        k - 1
    }
    /// 正态 (Box-Muller)
    fn normal(&mut self, mu: f32, sigma: f32) -> f32 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        mu + sigma * z
    }
}

/// 与 Python add_artifacts(sino, dose_level=0.5, hardening=False, rings=True, scatter=False) 同模型
pub fn add_artifacts(sino: &[f32]) -> Vec<f32> {
    debug_assert_eq!(sino.len(), NSINO);
    let smax = sino.iter().fold(0.0f32, |a, &b| a.max(b));

    // 量子噪声 (泊松) + 电子噪声 (高斯), seed=2024
    let mut noisy = vec![0.0f32; NSINO];
    {
        let mut rng = Mt19937::new(2024);
        let dose_level = 0.5f32;
        let scaling = 1000.0 * dose_level;
        for i in 0..NSINO {
            let sn = sino[i] / smax;
            let lam = (sn * scaling).max(0.0) as f64;
            let quantum = (rng.poisson(lam) as f64 - lam) / scaling as f64;
            let electronic = rng.normal(0.0, 0.01 / dose_level);
            noisy[i] = sino[i] + smax * (quantum as f32 * 0.3 + electronic * 0.7);
        }
    }
    for v in noisy.iter_mut() {
        *v = v.max(0.0);
    }

    // 环形伪影: 15 个坏道 (无放回), seed=2024
    {
        let mut rng = Mt19937::new(2024);
        let mut picked = Vec::with_capacity(15);
        while picked.len() < 15 {
            let c = rng.randint(N_DET_COL);
            if !picked.contains(&c) {
                picked.push(c);
            }
        }
        let noisy_max = noisy.iter().fold(0.0f32, |a, &b| a.max(b));
        for &c in &picked {
            let offset = rng.uniform_range(-0.03, 0.03) * noisy_max;
            for row in 0..N_DET_ROW {
                for a in 0..N_ANGLES {
                    noisy[(row * N_ANGLES + a) * N_DET_COL + c] += offset;
                }
            }
        }
    }
    for v in noisy.iter_mut() {
        *v = v.max(0.0);
    }
    noisy
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MT19937 标准验证向量 (seed=5489 是参考实现默认种子)
    #[test]
    fn mt19937_reference_sequence() {
        let mut rng = Mt19937::new(5489);
        assert_eq!(rng.next_u32(), 3499211612); // 第 1 个 tempered 输出
        assert_eq!(rng.next_u32(), 581869302); // 第 2 个
    }

    #[test]
    fn mt19937_deterministic() {
        let mut a = Mt19937::new(2024);
        let mut b = Mt19937::new(2024);
        for _ in 0..10000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
        let mut c = Mt19937::new(2025);
        assert_ne!(c.next_u32(), a.next_u32(), "不同种子应产生不同序列");
    }

    #[test]
    fn uniform_in_unit_range() {
        let mut rng = Mt19937::new(7);
        for _ in 0..1000 {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u), "uniform 越界: {u}");
        }
    }

    /// add_artifacts: 确定性 + 非负 + 尺寸正确
    #[test]
    fn add_artifacts_properties() {
        let sino = vec![0.5f32; NSINO];
        let a = add_artifacts(&sino);
        let b = add_artifacts(&sino);
        assert_eq!(a, b, "同输入两次调用应逐位一致");
        assert_eq!(a.len(), NSINO);
        assert!(a.iter().all(|&v| v >= 0.0), "噪声输出不应为负");
    }

    /// 有值输入 vs 零输入: 噪声后均值有变化 (确实注入了噪声)
    #[test]
    fn add_artifacts_changes_signal() {
        let sino = vec![0.5f32; NSINO];
        let a = add_artifacts(&sino);
        let changed = a.iter().zip(sino.iter()).filter(|(x, y)| x != y).count();
        assert!(changed > NSINO / 2, "仅 {} 个元素被改动", changed);
    }
}
