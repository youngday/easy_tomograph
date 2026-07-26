"""
TomoPhantom 2D 体模预览
========================
生成 Phantom2DLibrary.dat 中 15 种 2D 体模的预览图
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import tomophantom
from tomophantom import TomoP2D

N = 256  # 预览分辨率
lib = os.path.join(
    os.path.dirname(tomophantom.__file__), "phantomlib", "Phantom2DLibrary.dat"
)

models = {
    1: "Classic Shepp-Logan\n10 ellipses",
    2: "Piecewise-Smooth\nS-L (gaussian+ellipse)",
    3: "Defrise\n7 vertical ellipses",
    4: "QRM phantom\n41 components",
    5: "1 cone",
    6: "1 rectangle",
    7: "3 components",
    8: "6 gaussians+parabola",
    9: "5 components",
    10: "5 components",
    11: "SPECTRAL\n25 components",
    12: "rect+ellipse\n61 components",
    13: "Resolution\n40 rectangles",
    14: "Composite",
    15: "DLS phantom",
}

fig, axes = plt.subplots(3, 5, figsize=(18, 10))
for idx, (model_no, name) in enumerate(models.items()):
    r, c = idx // 5, idx % 5
    ph = TomoP2D.Model(model_no, N, lib)
    axes[r, c].imshow(ph, cmap="gray")
    axes[r, c].set_title(f"Model {model_no:02d}\n{name}", fontsize=8)
    axes[r, c].axis("off")

plt.tight_layout()
plt.savefig("img_out/tomophantom_models.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ img_out/tomophantom_models.png")
