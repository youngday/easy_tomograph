"""
FBP + IR 混合重建 (TIGRE 锥束 CBCT)
=====================================
核心思想: 用 FDK 作为初始值, 对比 SIRT / OS-SART
"""

from time import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import json
import os

try:
    import tigre
    import tigre.algorithms as algs
except ImportError:
    print("错误: 需要 TIGRE Toolbox (GPU 版本)")
    exit(1)

print("=" * 60)
print("FBP + IR 混合重建对比  [锥束 CBCT | TIGRE CUDA]")
print("=" * 60)

N = 512
nz = 32
n_angles = 360
print(f"体模: {nz}x{N}x{N}, 角度: {n_angles}")

vol_gt = np.zeros((nz, N, N), dtype=np.float32)
Z, Y, X = np.ogrid[:nz, :N, :N]
cz, cy, cx = nz / 2, N / 2, N / 2
body = ((Z - cz) / 12) ** 2 + ((Y - cy) / (N * 0.42)) ** 2 + (
    (X - cx) / (N * 0.35)
) ** 2 <= 1
vol_gt[body] = 0.020
bone = ((Z - cz) / 10) ** 2 + ((Y - cy) / (N * 0.30)) ** 2 + (
    (X - cx) / (N * 0.25)
) ** 2 <= 1
vol_gt[bone & ~body] = 0.0
bone_ring = ((Z - cz) / 10) ** 2 + ((Y - cy) / (N * 0.28)) ** 2 + (
    (X - cx) / (N * 0.23)
) ** 2 >= 1
vol_gt[bone & bone_ring] = 0.045
organ = ((Z - cz + 4) / 6) ** 2 + ((Y - cy - 15) / (N * 0.12)) ** 2 + (
    (X - cx + 10) / (N * 0.10)
) ** 2 <= 1
vol_gt[organ] = 0.025
tumor = ((Z - cz - 3) / 4) ** 2 + ((Y - cy + 20) / (N * 0.06)) ** 2 + (
    (X - cx - 15) / (N * 0.06)
) ** 2 <= 1
vol_gt[tumor] = 0.035
air = ((Z - cz + 2) / 5) ** 2 + ((Y - cy + 25) / (N * 0.08)) ** 2 + (
    (X - cx + 25) / (N * 0.06)
) ** 2 <= 1
vol_gt[air] = 0.0
print(f"   体模: [{vol_gt.min():.5f}, {vol_gt.max():.5f}]")

angles = np.deg2rad(np.linspace(0, 360, n_angles, endpoint=False)).astype(np.float32)
D = int(np.ceil(N * np.sqrt(2)))
geo = tigre.geometry()
geo.DSD = 1536.0
geo.DSO = 1000.0
geo.nVoxel = np.array([nz, N, N])
geo.sVoxel = np.array([nz * 1.5, N, N])
geo.dVoxel = geo.sVoxel / geo.nVoxel
geo.nDetector = np.array([nz * 2, D])
geo.dDetector = np.array([1.0, N / D])
geo.sDetector = geo.nDetector * geo.dDetector
geo.offOrigin = np.array([0, 0, 0])
geo.offDetector = np.array([0, 0])
geo.mode = "cone"
geo.filter = None

print("\nGPU 正向投影...")
t0 = time()
sino = tigre.Ax(vol_gt, geo, angles)
print(f"   完成: {(time() - t0) * 1000:.0f}ms, 形状 {sino.shape}")


def linear_scale(rec):
    mask = vol_gt > 0.001
    A = np.column_stack([rec.ravel()[mask.ravel()], np.ones(mask.sum())])
    b = vol_gt.ravel()[mask.ravel()]
    coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return rec * coef[0] + coef[1]


def calc_rmse(rec):
    mask = vol_gt > 0.001
    return np.sqrt(np.mean((vol_gt[mask] - rec[mask]) ** 2))


def calc_ssim(rec):
    mask = vol_gt > 0.001
    c1, c2 = (0.01 * 0.05) ** 2, (0.03 * 0.05) ** 2
    mu_x, mu_y = vol_gt[mask].mean(), rec[mask].mean()
    sig_x, sig_y = vol_gt[mask].var(), rec[mask].var()
    sig_xy = np.mean((vol_gt[mask] - mu_x) * (rec[mask] - mu_y))
    return (
        (2 * mu_x * mu_y + c1)
        * (2 * sig_xy + c2)
        / ((mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2))
    )


def tv_gradient(vol, eps=1e-8):
    dx = np.zeros_like(vol)
    dy = np.zeros_like(vol)
    dz = np.zeros_like(vol)
    dx[:, :, :-1] = vol[:, :, 1:] - vol[:, :, :-1]
    dy[:, :-1, :] = vol[:, 1:, :] - vol[:, :-1, :]
    dz[:-1, :, :] = vol[1:, :, :] - vol[:-1, :, :]
    mag = np.sqrt(dx**2 + dy**2 + dz**2 + eps)
    ux, uy, uz = dx / mag, dy / mag, dz / mag
    div = np.zeros_like(vol)
    div[:, :, 1:-1] = ux[:, :, 1:-1] - ux[:, :, :-2]
    div[:, :, 0] = ux[:, :, 0]
    div[:, :, -1] = -ux[:, :, -2]
    div[:, 1:-1, :] += uy[:, 1:-1, :] - uy[:, :-2, :]
    div[:, 0, :] += uy[:, 0, :]
    div[:, -1, :] += -uy[:, -2, :]
    div[1:-1, :, :] += uz[1:-1, :, :] - uz[:-2, :, :]
    div[0, :, :] += uz[0, :, :]
    div[-1, :, :] += -uz[-2, :, :]
    return -div


print("GPU 预热...")
_ = algs.fdk(sino, geo, angles, filter="shepp_logan")
_ = algs.sirt(sino, geo, angles, 1)
_ = algs.ossart(sino, geo, angles, 1, blocksize=9, verbose=False)
print("   预热完成\n")

# ---- A. Pure FDK ----
print("-" * 55)
print("A. Pure FDK")
print("-" * 55)
t0 = time()
rec_fdk = algs.fdk(sino, geo, angles, filter="shepp_logan")
fdk_rec = linear_scale(rec_fdk)
fdk_t = time() - t0
fdk_rmse = calc_rmse(fdk_rec)
fdk_ssim = calc_ssim(fdk_rec)
print(f"   RMSE={fdk_rmse:.5f}, SSIM={fdk_ssim:.4f}, {fdk_t * 1000:.0f}ms")

# ---- B. FBP + SIRT ----
print("-" * 55)
print("B. FBP + SIRT")
print("-" * 55)
b_hist = []
best_b = {"rmse": 1e9, "ssim": -1, "rec": None, "t": 0, "n": 0}
rec_b = rec_fdk.copy()
for n_iter in [5, 10, 20, 50]:
    t0 = time()
    rec_b = algs.sirt(sino, geo, angles, niter=n_iter, init=rec_b, noneg=False)
    t = time() - t0
    rec2d = linear_scale(rec_b)
    r, s = calc_rmse(rec2d), calc_ssim(rec2d)
    b_hist.append((n_iter, t, r, s))
    if r < best_b["rmse"]:
        best_b = {"rmse": r, "ssim": s, "rec": rec2d, "t": t, "n": n_iter}
    print(f"   x{n_iter:3d}: RMSE={r:.5f}, SSIM={s:.4f}, {t * 1000:.0f}ms")
print(f"   >> 最优: SIRT x{best_b['n']}: RMSE={best_b['rmse']:.5f}")

# ---- C. FBP + OS-SART ----
print("-" * 55)
print("C. FBP + OS-SART (20子集)")
print("-" * 55)
c_hist = []
best_c = {"rmse": 1e9, "ssim": -1, "rec": None, "t": 0, "n": 0}
rec_c = rec_fdk.copy()
os_iters = [1, 2, 5, 10]
for n_iter in os_iters:
    t0 = time()
    rec_c = algs.ossart(
        sino, geo, angles, niter=n_iter, init=rec_c, blocksize=9, verbose=False
    )
    t = time() - t0
    rec2d = linear_scale(rec_c)
    r, s = calc_rmse(rec2d), calc_ssim(rec2d)
    c_hist.append((n_iter, t, r, s))
    if r < best_c["rmse"]:
        best_c = {"rmse": r, "ssim": s, "rec": rec2d, "t": t, "n": n_iter}
    print(f"   x{n_iter:3d}: RMSE={r:.5f}, SSIM={s:.4f}, {t * 1000:.0f}ms")
print(f"   >> 最优: OS-SART x{best_c['n']}: RMSE={best_c['rmse']:.5f}")

# ---- D. TV-OS-SART (OS-SART + TV) ----
print("-" * 55)
print("D. TV-OS-SART (OS-SART + TV 去噪)")
print("-" * 55)
d_hist = []
best_d = {"rmse": 1e9, "ssim": -1, "rec": None, "t": 0, "n": 0}
rec_d = rec_fdk.copy()
beta = 0.0005
for n_iter in os_iters:
    t0 = time()
    rec_d = algs.ossart(
        sino, geo, angles, niter=n_iter, init=rec_d, blocksize=9, verbose=False
    )
    rec_d = rec_d - beta * tv_gradient(rec_d)
    t = time() - t0
    rec2d = linear_scale(rec_d)
    r, s = calc_rmse(rec2d), calc_ssim(rec2d)
    d_hist.append((n_iter, t, r, s))
    if r < best_d["rmse"]:
        best_d = {"rmse": r, "ssim": s, "rec": rec2d, "t": t, "n": n_iter}
    print(f"   x{n_iter:3d}: RMSE={r:.5f}, SSIM={s:.4f}, {t * 1000:.0f}ms")
print(f"   >> 最优: TV-OS-SART x{best_d['n']}: RMSE={best_d['rmse']:.5f}")

# ---- 汇总 ----
print("\n" + "=" * 70)
print("汇总对比 (32x512x512, 360角度)")
print("=" * 70)
print(f"{'算法':30s} {'耗时(ms)':>10s} {'RMSE':>12s} {'SSIM':>8s} {'vsFDK':>10s}")
print("-" * 72)
s_imp = f"{(1 - best_b['rmse'] / fdk_rmse) * 100:+.1f}%"
c_imp = f"{(1 - best_c['rmse'] / fdk_rmse) * 100:+.1f}%"
d_imp = f"{(1 - best_d['rmse'] / fdk_rmse) * 100:+.1f}%"
print(
    f"{'Pure FDK':30s} {fdk_t * 1000:>8.0f} ms {fdk_rmse:>12.5f} {fdk_ssim:>8.4f} {'-':>10s}"
)
print(
    f"{'SIRT x' + str(best_b['n']):30s} {best_b['t'] * 1000:>8.0f} ms {best_b['rmse']:>12.5f} {best_b['ssim']:>8.4f} {s_imp:>10s}"
)
print(
    f"{'OS-SART x' + str(best_c['n']):30s} {best_c['t'] * 1000:>8.0f} ms {best_c['rmse']:>12.5f} {best_c['ssim']:>8.4f} {c_imp:>10s}"
)
print(
    f"{'TV-OS-SART x' + str(best_d['n']):30s} {best_d['t'] * 1000:>8.0f} ms {best_d['rmse']:>12.5f} {best_d['ssim']:>8.4f} {d_imp:>10s}"
)

results = [
    ("Pure FDK", fdk_t, fdk_rmse, fdk_ssim),
    ("SIRT x" + str(best_b["n"]), best_b["t"], best_b["rmse"], best_b["ssim"]),
    ("OS-SART x" + str(best_c["n"]), best_c["t"], best_c["rmse"], best_c["ssim"]),
    ("TV-OS-SART x" + str(best_d["n"]), best_d["t"], best_d["rmse"], best_d["ssim"]),
]

# 可视化
print("\n生成可视化...")
os.makedirs("img_out", exist_ok=True)
mid = nz // 2
fig = plt.figure(figsize=(20, 9))
gs = GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.3)
all_data = [("Ground Truth", vol_gt[mid], None, None, None)]
for name, t, rmse, ssim in results:
    if "FDK" in name:
        rec = fdk_rec
    elif "SIRT" in name and "TV" not in name:
        rec = best_b["rec"]
    elif "OS-SART" in name and "TV" not in name:
        rec = best_c["rec"]
    elif "TV-OS" in name:
        rec = best_d["rec"]
    all_data.append((name, rec[mid], rmse, ssim, t))
for i, (title, img, rmse, ssim, t) in enumerate(all_data):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img, cmap="gray", vmin=0, vmax=0.05)
    tstr = (
        title
        if rmse is None
        else f"{title}\nRMSE={rmse:.5f} SSIM={ssim:.4f}\n{t * 1000:.0f}ms"
    )
    ax.set_title(tstr, fontsize=7)
    ax.axis("off")
    ax2 = fig.add_subplot(gs[1, i])
    e = np.zeros_like(img) if rmse is None else img - vol_gt[mid]
    v = max(0.005, np.percentile(np.abs(e), 95) * 1.2) if rmse is not None else 0.005
    im = ax2.imshow(e, cmap="RdBu_r", vmin=-v, vmax=v)
    ax2.set_title("Error", fontsize=8)
    ax2.axis("off")
plt.suptitle(
    "Cone-beam CBCT: TIGRE CUDA (512x512x32)", fontsize=13, fontweight="bold", y=0.98
)
plt.savefig("img_out/tigre_cone_hybrid.png", dpi=150, bbox_inches="tight")
plt.close()
print("   => img_out/tigre_cone_hybrid.png")

summary = {
    "backend": "GPU (TIGRE CUDA cone-beam)",
    "config": {"N": N, "nz": nz, "n_angles": n_angles},
    "results": {
        name: {"rmse": round(r, 5), "ssim": round(s, 4), "time_ms": round(t * 1000, 1)}
        for name, t, r, s in results
    },
}
with open("img_out/tigre_cone_hybrid_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("   => img_out/tigre_cone_hybrid_summary.json")
print("\nDone!")
