"""
生成 C++ 重建所需的体模文件 (tomophantom 无 C++ API, 故用 Python 生成一次)
输出: float32, 形状 (nz, N, N), [z][y][x] 布局, 与 src_3d_axial|helical/astra_cone_hybrid.py
      中的 vol_gt 逐元素一致 (Model 4, 512x512x32, 掩码后)

用法: python tools/make_phantom.py [输出路径, 默认 data/vol_gt.raw]
"""
import os
import sys

import numpy as np
import tomophantom
from tomophantom import TomoP3D

N, nz = 512, 32
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "vol_gt.raw")

tp_lib = os.path.join(os.path.dirname(tomophantom.__file__), "phantomlib", "Phantom3DLibrary.dat")
ph = TomoP3D.Model(4, (N, N, nz), tp_lib).astype(np.float32)
vol_gt = np.transpose(ph, (2, 0, 1)).copy()
vol_gt = (vol_gt - 0.2) * 0.035
vol_gt = np.clip(vol_gt, 0, 0.05)
Y, X = np.ogrid[:N, :N]
circ_mask = ((X - N / 2) ** 2 + (Y - N / 2) ** 2) <= (N * 0.42) ** 2
vol_gt[:, ~circ_mask] = 0.0

os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
vol_gt.tofile(out)
print(f"体模已写出: {out}")
print(f"   形状 {vol_gt.shape}, 范围 [{vol_gt.min():.5f}, {vol_gt.max():.5f}]")
