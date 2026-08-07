"""
生成与 Python 版 (src_3d_{axial,helical}/astra_cone_hybrid.py) 完全一致的
含噪声 sinogram, 供 C++ 重建加载 —— 使 C++ 的 FDK(noisy)/TV-OS-SART/Hybrid
与 Python 版噪声逐位相同。

注意: 轴向与螺旋的干净 sinogram 不同 (轨迹不同), 噪声必须分别生成。

噪声 = 原版 add_artifacts(sino, dose_level=0.5, hardening=False, rings=True, scatter=False)
     (numpy RandomState seed=2024, 见 src_3d_axial/ct_noise.py)
正向投影 = 与 Python 脚本相同的 cone_vec 几何 + FP3D_CUDA

用法: python tools/make_sino_noisy.py [axial|helical|both] [体模路径] [输出目录]
   默认 both, 生成 data/sino_noisy_{axial,helical}.raw
"""
import os
import sys

import numpy as np

N, nz, n_angles = 512, 32, 180
n_det_row, n_det_col = 64, int(np.ceil(N * np.sqrt(2)))
DSO, DSD_det, det_pix, pitch = 1000.0, 500.0, 1.0, 16.0

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src_astra_cpp
args = sys.argv[1:]
modes = [a for a in args if a in ("axial", "helical", "both")] or ["both"]
phantom = next((a for a in args if "vol_gt" in a), os.path.join(root, "data", "vol_gt.raw"))
outdir = next((a for a in args if a not in ("axial", "helical", "both") and "vol_gt" not in a), os.path.join(root, "data"))

# 原版噪声函数 (与 Python 脚本相同的 add_artifacts)
sys.path.insert(0, os.path.join(os.path.dirname(root), "src_3d_axial"))
from ct_noise import add_artifacts  # noqa: E402

import astra  # noqa: E402

vol_gt = np.fromfile(phantom, dtype=np.float32).reshape(nz, N, N)


# 与 Python 脚本相同的 cone_vec 几何 (helical: 源/探测器沿 z 移动)
def make_vectors(helical):
    angles_rad = np.deg2rad(np.linspace(0, 360, n_angles, endpoint=False)).astype(np.float32)
    vectors = np.zeros((n_angles, 12), dtype=np.float32)
    for i, th in enumerate(angles_rad):
        c, s = np.cos(th), np.sin(th)
        z_src = pitch * (th / (2 * np.pi) - 0.5) if helical else 0.0
        vectors[i, :3] = [DSO * s, -DSO * c, z_src]
        vectors[i, 3:6] = [-DSD_det * s, DSD_det * c, z_src]
        vectors[i, 6:9] = [det_pix * c, det_pix * s, 0.0]
        vectors[i, 9:12] = [0.0, 0.0, det_pix]
    return angles_rad, vectors


for mode in (["axial", "helical"] if "both" in modes else modes):
    helical = mode == "helical"
    _, vectors = make_vectors(helical)
    proj_geom = astra.create_proj_geom("cone_vec", n_det_row, n_det_col, vectors)
    vol_geom = astra.create_vol_geom(N, N, nz)

    # 正向投影 (与脚本的 FP3D_CUDA 相同)
    vid = astra.data3d.create("-vol", vol_geom, vol_gt.astype(np.float32))
    sid = astra.data3d.create("-sino", proj_geom, 0.0)
    cfg = astra.astra_dict("FP3D_CUDA")
    cfg["ProjectionDataId"] = sid
    cfg["VolumeDataId"] = vid
    aid = astra.algorithm.create(cfg)
    astra.algorithm.run(aid)
    sino = astra.data3d.get(sid)
    astra.algorithm.delete(aid)
    astra.data3d.delete(sid)
    astra.data3d.delete(vid)

    # 与脚本相同的噪声参数
    sino_noisy = add_artifacts(sino, dose_level=0.5, hardening=False, rings=True, scatter=False)
    out = os.path.join(outdir, f"sino_noisy_{mode}.raw")
    os.makedirs(outdir, exist_ok=True)
    np.ascontiguousarray(sino_noisy, dtype=np.float32).tofile(out)
    print(f"{mode}: 含噪声 sinogram 已写出: {out}  (形状 {sino_noisy.shape})")

    # 参考值: Python 版 FDK(noisy) RMSE
    def linear_scale(rec):
        mask = vol_gt > 0.001
        A = np.column_stack([rec.ravel()[mask.ravel()], np.ones(mask.sum())])
        b = vol_gt.ravel()[mask.ravel()]
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        return rec * coef[0] + coef[1]

    sido_n = astra.data3d.create("-sino", proj_geom, sino_noisy.astype(np.float32))
    rido_n = astra.data3d.create("-vol", vol_geom)
    c = astra.astra_dict("FDK_CUDA")
    c["ProjectionDataId"] = sido_n
    c["ReconstructionDataId"] = rido_n
    c["option"] = {"FilterType": "hann"}
    a = astra.algorithm.create(c)
    astra.algorithm.run(a)
    rec_fdk_n = astra.data3d.get(rido_n)
    astra.algorithm.delete(a)
    astra.data3d.delete(rido_n)
    astra.data3d.delete(sido_n)

    mask = vol_gt > 0.001
    rmse = float(np.sqrt(np.mean((vol_gt[mask] - linear_scale(rec_fdk_n)[mask]) ** 2)))
    print(f"{mode}: Python 参考 FDK(noisy) RMSE = {rmse:.5f}  (C++ 运行输出应与此一致)")
