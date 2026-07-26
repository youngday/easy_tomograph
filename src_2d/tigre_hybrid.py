"""
FBP + IR 混合重建 (TIGRE GPU)
=============================
核心思想: 用 FBP 的快速重建结果作为迭代法 (SIRT) 的初始值,
          让 IR 从更好的起点开始迭代 -> 更快收敛 + 更高质量

对比组:
  - Pure FBP (基线)
  - FBP + SIRT (混合)

模式:
  - GPU 模式: 使用 TIGRE Toolbox (CUDA 加速)
"""

from time import time

import matplotlib.pyplot as plt
import numpy as np
import tomophantom
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from tomophantom import TomoP2D

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import json
import os

try:
    import tigre
    import tigre.algorithms as algs
except ImportError:
    print("=" * 60)
    print("错误: 需要 TIGRE Toolbox (GPU 版本)")
    print("安装: git+https://github.com/CERN/TIGRE.git")
    print("=" * 60)
    exit(1)

print("=" * 60)
print("FBP + IR 混合重建对比  [后端: GPU (TIGRE CUDA)]")
print("=" * 60)

N = 512
n_angles = 360
print(f"体模: {N}x{N}, 角度: {n_angles}")

tp_lib = os.path.join(
    os.path.dirname(tomophantom.__file__), "phantomlib", "Phantom2DLibrary.dat"
)
ph = TomoP2D.Model(4, N, tp_lib)
ct = (ph - 0.65) * 2000 / 0.65
ct = ct.astype(np.float32)

Y, X = np.ogrid[:N, :N]
head_r = 235
circ_mask = (X - N / 2) ** 2 + (Y - N / 2) ** 2 <= head_r**2
ct[~circ_mask] = -1000

dist = np.sqrt((X - N / 2) ** 2 + (Y - N / 2) ** 2)
soft_mask = np.clip((head_r + 20 - dist) / 20, 0, 1)

theta_deg = np.linspace(0, 180, n_angles, endpoint=False)
theta_rad = np.deg2rad(theta_deg).astype(np.float32)
D = int(np.ceil(N * np.sqrt(2)))

geo = tigre.geometry()
geo.DSD = 1536
geo.DSO = 1000
geo.nVoxel = np.array([1, N, N])
geo.sVoxel = np.array([1, N, N])
geo.dVoxel = geo.sVoxel / geo.nVoxel
geo.nDetector = np.array([1, D])
geo.dDetector = np.array([1.0, N / D])
geo.sDetector = geo.nDetector * geo.dDetector
geo.offOrigin = np.array([0, 0, 0])
geo.offDetector = np.array([0, 0])
geo.mode = "parallel"

angles = theta_rad
vol_gt = ct[np.newaxis, :, :].astype(np.float32)

print("GPU 正向投影...")
t0 = time()
sino_3d = tigre.Ax(vol_gt, geo, angles)
print(f"   完成: 耗时 {(time() - t0) * 1000:.0f}ms, 投影形状 {sino_3d.shape}")
sino = sino_3d[:, 0, :]


def linear_scale(rec):
    rec_clip = np.clip(rec, -5000, 5000)
    mask = circ_mask
    A = np.column_stack([rec_clip.ravel()[mask.ravel()], np.ones(mask.sum())])
    b = ct.ravel()[mask.ravel()]
    coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return rec * coef[0] + coef[1]


def calc_rmse(rec):
    return np.sqrt(np.mean((ct[circ_mask] - rec[circ_mask]) ** 2))


def calc_ssim(rec):
    c1, c2 = (0.01 * 2000) ** 2, (0.03 * 2000) ** 2
    mu_x, mu_y = ct[circ_mask].mean(), rec[circ_mask].mean()
    sig_x, sig_y = ct[circ_mask].var(), rec[circ_mask].var()
    sig_xy = np.mean((ct[circ_mask] - mu_x) * (rec[circ_mask] - mu_y))
    return (
        (2 * mu_x * mu_y + c1)
        * (2 * sig_xy + c2)
        / ((mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2))
    )


print("GPU 预热...")
_ = algs.fdk(sino_3d, geo, angles, filter="shepp_logan")
_ = algs.sirt(sino_3d, geo, angles, 1)
print("   预热完成\n")

# ---- A. Pure FBP (基线) ----
print("-" * 55)
print("A. Pure FBP (TIGRE FDK shepp_logan)")
print("-" * 55)
t0 = time()
rec_fdk_3d = algs.fdk(sino_3d, geo, angles, filter="shepp_logan")
fbp_raw = rec_fdk_3d[0]
fbp_denoised = gaussian_filter(fbp_raw, sigma=0.5)
fbp_rec = linear_scale(fbp_denoised)
fbp_t = time() - t0
fbp_rmse = calc_rmse(fbp_rec)
fbp_ssim = calc_ssim(fbp_rec)
print(f"   RMSE={fbp_rmse:.2f}, SSIM={fbp_ssim:.4f}, {fbp_t * 1000:.0f}ms")

# ---- B. FBP + SIRT (混合) ----
print("-" * 55)
print("B. FBP + SIRT (FBP 初始值)")
print("-" * 55)
fbs_hist = []
best_fs = {"rmse": 1e9, "ssim": -1, "rec": None, "t": 0, "n": 0}
rec_fbs_3d = rec_fdk_3d.copy()
sirt_iters = [10, 20, 50, 100, 200]
prev_n = 0
for n_iter in sirt_iters:
    t0 = time()
    rec_fbs_3d = algs.sirt(
        sino_3d, geo, angles, niter=n_iter - prev_n, init=rec_fbs_3d, noneg=False
    )
    t = time() - t0
    rec2d = linear_scale(rec_fbs_3d[0])
    r, s = calc_rmse(rec2d), calc_ssim(rec2d)
    fbs_hist.append((n_iter, t, r, s))
    if r < best_fs["rmse"]:
        best_fs = {"rmse": r, "ssim": s, "rec": rec2d, "t": t, "n": n_iter}
    print(f"   x{n_iter:4d}: RMSE={r:.2f}, SSIM={s:.4f}, {t * 1000:.0f}ms")
    prev_n = n_iter
print(
    f"   >> 最优: FBP+SIRT x{best_fs['n']}: RMSE={best_fs['rmse']:.2f}, SSIM={best_fs['ssim']:.4f}, {best_fs['t'] * 1000:.0f}ms"
)

# ---- C. TIGRE 算法限制说明 ----
print("-" * 55)
print("C. 注: TIGRE 的 SART/OS-SART 不兼容平行束几何")
print("-" * 55)
print("   仅保留 FBP+SIRT 对比 (ASTRA 版包含完整 SART/OS-SART)")
print()

# ============================================================
# 结果列表
# ============================================================
results = [
    ("Pure FBP (shepp_logan)", fbp_t, fbp_rmse, fbp_ssim, 0),
    (
        "FBP + SIRT (hybrid)",
        best_fs["t"],
        best_fs["rmse"],
        best_fs["ssim"],
        (fbp_rmse - best_fs["rmse"]) / fbp_rmse * 100,
    ),
]

# ============================================================
# 汇总对比
# ============================================================
print("=" * 60)
print("汇总对比")
print("=" * 60)
print(f"{'算法':35s} {'耗时(ms)':>10s} {'RMSE':>8s} {'SSIM':>8s} {'提升':>10s}")
print("-" * 71)
for name, t, r, s, imp in results:
    imp_str = f"{imp:+.1f}%" if imp != 0 else "  基线"
    print(f"{name:35s} {t * 1000:>8.0f} ms {r:>8.2f} {s:>8.4f} {imp_str:>10s}")

print("\n推荐配置 (性价比):")
print(f"   速度优先:    FBP (FDK)     -> {fbp_t * 1000:.0f}ms, RMSE={fbp_rmse:.1f}")
if any(h[0] == 10 for h in fbs_hist):
    f10_s = next(h for h in fbs_hist if h[0] == 10)
    print(
        f"   高质量:      FBP+SIRT x10 -> {f10_s[1] * 1000:.0f}ms, RMSE={f10_s[2]:.1f}"
    )

print("\n生成可视化...")
os.makedirs("img_out", exist_ok=True)

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3, width_ratios=[1, 1, 1])

gray_cmap = "gray"
err_cmap = "RdBu_r"

plot_items = [
    ("Ground Truth", ct, None, None, None),
    ("Pure FBP", fbp_rec, fbp_rmse, fbp_ssim, fbp_t),
    ("FBP+SIRT (best)", best_fs["rec"], best_fs["rmse"], best_fs["ssim"], best_fs["t"]),
]

for i, (title, img, rmse, ssim, t) in enumerate(plot_items):
    ax = fig.add_subplot(gs[0, i])
    if rmse is not None:
        img_display = img * soft_mask + ct * (1 - soft_mask)
    else:
        img_display = img
    ax.imshow(img_display, cmap=gray_cmap, vmin=-200, vmax=600)
    tstr = title
    if rmse is not None:
        tstr += f"\nRMSE={rmse:.1f} SSIM={ssim:.4f}\n{t * 1000:.0f}ms"
    ax.set_title(tstr, fontsize=9)
    ax.axis("off")

err_items = [
    ("", ct - ct),
    ("FBP Error", fbp_rec - ct),
    ("FBP+SIRT Error", best_fs["rec"] - ct),
]

cax = None
for i, (title, err_img) in enumerate(err_items):
    ax = fig.add_subplot(gs[1, i])
    err_img_masked = err_img.copy()
    err_img_masked[~circ_mask] = 0
    vmax = max(30, np.percentile(np.abs(err_img_masked[circ_mask]), 95) * 1.2)
    print(f"       {title}: vmax={vmax:.1f}")
    im = ax.imshow(err_img_masked, cmap=err_cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    cax = im

fig.colorbar(cax, ax=fig.get_axes(), shrink=0.6, pad=0.02)
plt.suptitle(
    "FBP + IR Hybrid Reconstruction (GPU: TIGRE CUDA)",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)
plt.savefig("img_out/tigre_hybrid.png", dpi=150, bbox_inches="tight")
plt.close()
print("   => img_out/tigre_hybrid.png")

summary = {
    "backend": "GPU (TIGRE CUDA)",
    "config": {"N": N, "n_angles": n_angles},
    "results": {
        name: {"rmse": round(r, 2), "ssim": round(s, 4), "time_ms": round(t * 1000, 1)}
        for name, t, r, s, _ in results
    },
}
with open("img_out/tigre_hybrid_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("   => img_out/tigre_hybrid_summary.json")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
