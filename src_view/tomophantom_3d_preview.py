"""
TomoPhantom 3D 体模预览
========================
生成 Phantom3DLibrary.dat 中 15 种 3D 体模的中间切片预览
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tomophantom
from tomophantom import TomoP3D

N = 128
nz = 32
lib = os.path.join(
    os.path.dirname(tomophantom.__file__), "phantomlib", "Phantom3DLibrary.dat"
)

models = {
    1:  "Simple ellipsoids",
    2:  "Modified Shepp-Logan",
    3:  "Defrise-like",
    4:  "QRM-like",
    5:  "Single cone",
    6:  "Rectangle",
    7:  "3 components",
    8:  "Gaussian+parabola",
    9:  "5 components",
    10: "5 components",
    11: "SPECTRAL-like",
    12: "Rect+ellipse",
    13: "Resolution",
    14: "Composite",
    15: "DLS phantom",
}

fig, axes = plt.subplots(3, 5, figsize=(18, 10))
for idx, (model_no, name) in enumerate(models.items()):
    r, c = idx // 5, idx % 5
    ph = TomoP3D.Model(model_no, (N, N, nz), lib)
    ph = np.transpose(ph, (2, 0, 1))  # (N,N,nz) -> (nz,N,N)
    mid = nz // 2
    axes[r, c].imshow(ph[mid], cmap="gray")
    axes[r, c].set_title(
        f"Model {model_no:02d}\n{name}\n"
        f"{ph.shape}  [{ph.min():.3f}~{ph.max():.3f}]",
        fontsize=7,
    )
    axes[r, c].axis("off")

plt.tight_layout()
os.makedirs("img_out", exist_ok=True)
plt.savefig("img_out/tomophantom_3d_models.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ img_out/tomophantom_3d_models.png")
