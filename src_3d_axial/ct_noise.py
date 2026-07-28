"""
CT 噪声/伪影模拟工具
======================
用于测试迭代重建算法的噪声鲁棒性
"""

import numpy as np
from scipy.ndimage import gaussian_filter

def quantum_noise(sino, dose_level=1.0, seed=None):
    """
    量子噪声 + 电子噪声 (泊松-高斯混合)
    
    Args:
        sino: 干净投影数据
        dose_level: 剂量水平 (1.0=正常, 0.1=低剂量)
        seed: 随机种子
    
    Returns:
        含噪声的投影数据
    """
    if seed is not None:
        np.random.seed(seed)
    
    sino = np.asarray(sino, dtype=np.float32)
    sino_norm = sino / max(sino.max(), 1e-10)
    
    # 泊松量子噪声 (信号越强噪声越大)
    scaling = 1000 * dose_level
    quantum = np.random.poisson(np.maximum(sino_norm * scaling, 0)).astype(np.float32)
    quantum = (quantum - sino_norm * scaling) / scaling  # 零均值噪声
    
    # 高斯电子噪声 (与信号无关)
    electronic = np.random.normal(0, 0.01 / dose_level, sino.shape).astype(np.float32)
    
    noisy = sino + sino.max() * (quantum * 0.3 + electronic * 0.7)
    return np.clip(noisy, 0, None)


def beam_hardening(sino, poly_coeff=(0.8, 0.15, 0.05)):
    """
    射束硬化伪影 (多项式模型)
    
    模拟多能谱X射线穿过物质后的硬化效应:
    I_detected ≈ I_0 * exp(-∫μ(E,x)dx) → 重建出杯状伪影
    
    Args:
        sino: 干净投影数据 (已取-log)
        poly_coeff: 硬化多项式系数 (高阶项产生杯状伪影)
    
    Returns:
        含硬化效应的投影数据
    """
    sino = np.asarray(sino, dtype=np.float32)
    result = np.zeros_like(sino)
    for i, c in enumerate(poly_coeff):
        result += c * (sino ** (i + 1))
    return result


def ring_artifacts(sino, n_rings=15, intensity=0.03, seed=None):
    """
    环形伪影 (探测器通道增益不一致)
    
    模拟探测器个别通道增益漂移产生的环形伪影。
    Args:
        sino: 形状 (det_row, n_angles, det_col)
        n_rings: 坏道数量
        intensity: 坏道偏移强度
        seed: 随机种子
    """
    if seed is not None:
        np.random.seed(seed)
    
    sino = np.asarray(sino, dtype=np.float32)
    # 在探测器列方向随机选坏道
    det_col = sino.shape[2]
    bad_channels = np.random.choice(det_col, n_rings, replace=False)
    
    noisy = sino.copy()
    for ch in bad_channels:
        # 该通道所有角度增益偏移
        offset = np.random.uniform(-intensity, intensity) * sino.max()
        noisy[:, :, ch] += offset
    
    return np.clip(noisy, 0, None)


def scatter_conv(sino, kernel_size=7, scatter_frac=0.1):
    """
    散射伪影 (卷积模型)
    
    模拟康普顿散射产生的低频背景。
    用高斯卷积核近似散射分布。
    
    Args:
        sino: 投影数据
        kernel_size: 散射核大小
        scatter_frac: 散射占总信号比例
    """
    sino = np.asarray(sino, dtype=np.float32)
    # 在探测器平面做高斯平滑模拟散射
    scatter = np.zeros_like(sino)
    for i in range(sino.shape[1]):  # 每个角度独立
        scatter[:, i, :] = gaussian_filter(sino[:, i, :], sigma=kernel_size)
    
    return sino + scatter * scatter_frac


def add_artifacts(sino, dose_level=1.0, hardening=True, rings=True,
                  scatter=True, seed=2024, **kwargs):
    """
    综合噪声+伪影生成
    
    Args:
        sino: 干净投影
        dose_level: 剂量水平
        hardening: 是否加射束硬化
        rings: 是否加环形伪影
        scatter: 是否加散射
        seed: 随机种子
    
    Returns:
        含噪声/伪影的投影
    """
    result = sino.copy()
    
    if dose_level < 1.0:
        result = quantum_noise(result, dose_level, seed=seed)
    
    if hardening:
        result = beam_hardening(result)
    
    if rings:
        result = ring_artifacts(result, seed=seed if seed else None)
    
    if scatter:
        result = scatter_conv(result)
    
    return np.clip(result, 0, None)


def make_low_dose(sino, dose_level=0.25, seed=2024):
    """
    低剂量CT模拟 (25% 剂量)
    快捷函数
    """
    return add_artifacts(sino, dose_level=dose_level, 
                         hardening=True, rings=True, scatter=False, seed=seed)


if __name__ == "__main__":
    # 测试
    print("测试 CT 噪声/伪影工具...")
    sino_test = np.random.rand(10, 100, 50).astype(np.float32) * 0.1
    
    for name, fn in [("量子噪声", quantum_noise), 
                      ("射束硬化", beam_hardening),
                      ("环形伪影", ring_artifacts),
                      ("散射", scatter_conv),
                      ("低剂量(25%)", lambda s: make_low_dose(s, 0.25))]:
        result = fn(sino_test.copy())
        diff = np.abs(result - sino_test).mean()
        print(f"   {name}: 平均变化 {diff:.6f}")
    
    print("OK")
