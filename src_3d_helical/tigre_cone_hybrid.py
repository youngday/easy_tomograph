"""
螺旋 CT 混合重建 (TIGRE 锥束) — Helical CBCT 优化版
=====================================================
FDK / Hybrid IR / TV-OS-SART (自适应 β)
优化: offOrigin-z 螺旋轨迹 + Nesterov 动量 + z-profile
"""

from time import time, strftime, localtime

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
print("螺旋(Helical) CBCT 混合重建对比  [TIGRE CUDA 优化版]")
print("=" * 60)

N = 512
nz = 32
n_angles = 360

import tomophantom
from tomophantom import TomoP3D

tp_lib = os.path.join(os.path.dirname(tomophantom.__file__),
                       'phantomlib', 'Phantom3DLibrary.dat')
ph = TomoP3D.Model(4, (N, N, nz), tp_lib).astype(np.float32)
vol_gt = np.transpose(ph, (2, 0, 1)).copy()
vol_gt = (vol_gt - 0.2) * 0.035
vol_gt = np.clip(vol_gt, 0, 0.05)
Y, X = np.ogrid[:N, :N]
circ_mask = ((X - N/2)**2 + (Y - N/2)**2) <= (N*0.42)**2
vol_gt[:, ~circ_mask] = 0.0
print(f"   体模: [{vol_gt.min():.5f}, {vol_gt.max():.5f}]")

# 软遮罩
Ygrid, Xgrid = np.ogrid[:N, :N]
dist_xy = np.sqrt((Xgrid - N/2)**2 + (Ygrid - N/2)**2)
body_r = N * 0.42
soft_mask_2d = np.clip((body_r + 20 - dist_xy) / 20, 0, 1)

angles = np.deg2rad(np.linspace(0, 360, n_angles, endpoint=False)).astype(np.float32)
D = int(np.ceil(N * np.sqrt(2)))
geo = tigre.geometry()
geo.DSD = 1500.0
geo.DSO = 1000.0
dVox = 1.0
geo.nVoxel = np.array([nz, N, N])
geo.sVoxel = geo.nVoxel * dVox
geo.dVoxel = np.array([dVox, dVox, dVox])
geo.nDetector = np.array([nz * 2, D])
geo.dDetector = np.array([1.0, 1.0])
geo.sDetector = geo.nDetector * geo.dDetector
geo.offDetector = np.array([0, 0])
pitch = 16.0
# Helical: per-projection offOrigin z 偏移
z_helical = pitch * (angles / (2 * np.pi) - 0.5)
geo.offOrigin = np.zeros((n_angles, 3), dtype=np.float32)
geo.offOrigin[:, 2] = z_helical
geo.mode = "cone"
geo.filter = None
blocksize = 36  # 10 subsets (最优质量)

print("\nGPU 正向投影 (helical)...")
t0 = time()
sino = tigre.Ax(vol_gt, geo, angles)
print(f"   完成: {(time() - t0) * 1000:.0f}ms, 形状 {sino.shape}")

# ---- 增强的度量函数 ----
def linear_scale(rec):
    mask = vol_gt > 0.001
    A = np.column_stack([rec.ravel()[mask.ravel()], np.ones(mask.sum())])
    b = vol_gt.ravel()[mask.ravel()]
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return rec * coef[0] + coef[1]

def calc_rmse(rec):
    mask = vol_gt > 0.001
    return np.sqrt(np.mean((vol_gt[mask] - rec[mask]) ** 2))

def calc_ssim(rec):
    mask = vol_gt > 0.001
    c1, c2 = (0.01 * 0.05) ** 2, (0.03 * 0.05) ** 2
    mux, muy = vol_gt[mask].mean(), rec[mask].mean()
    sx, sy = vol_gt[mask].var(), rec[mask].var()
    sxy = np.mean((vol_gt[mask] - mux) * (rec[mask] - muy))
    return (
        (2 * mux * muy + c1)
        * (2 * sxy + c2)
        / ((mux**2 + muy**2 + c1) * (sx + sy + c2))
    )

def calc_z_profile(rec):
    """沿 z 方向逐片 RMSE"""
    z_rmse = []
    for z in range(rec.shape[0]):
        mask_z = vol_gt[z] > 0.001
        if mask_z.sum() > 100:
            e = vol_gt[z][mask_z] - rec[z][mask_z]
            z_rmse.append(np.sqrt(np.mean(e**2)))
        else:
            z_rmse.append(0.0)
    return np.array(z_rmse)

print("GPU 预热...")
_ = algs.fdk(sino, geo, angles, filter="hann")
_ = algs.ossart(sino, geo, angles, 1, blocksize=blocksize, verbose=False)
print("   预热完成\n")

# ========== A. Pure FDK ==========
print("-" * 55)
print("A. Pure FDK")
print("-" * 55)
t0 = time()
rec_fdk = algs.fdk(sino, geo, angles, filter="hann")
fdk_rec = linear_scale(rec_fdk)
fdk_t = time() - t0; fdk_rmse = calc_rmse(fdk_rec); fdk_ssim = calc_ssim(fdk_rec)
fdk_zprof = calc_z_profile(fdk_rec)
print(f"   RMSE={fdk_rmse:.5f}, SSIM={fdk_ssim:.4f}, {fdk_t * 1000:.0f}ms")
print(f"   z-profile: mean={fdk_zprof.mean():.5f}, max={fdk_zprof.max():.5f}, min={fdk_zprof.min():.5f}")

# ========== B. 噪声/伪影 + Hybrid IR ==========
print("-" * 55)
print("B. Hybrid IR (OS-SART×3 + TV×1 + FDK混合 50%)")
print("-" * 55)

from ct_noise import add_artifacts
np.random.seed(2024)
sino_noisy = add_artifacts(sino, dose_level=0.5, hardening=False, rings=True, scatter=False)

# 有噪声 FDK 初始化
rec_fdk_noisy = algs.fdk(sino_noisy, geo, angles, filter="hann")

# ---- TV 梯度 (预分配缓存) ----
class TVGradientCache:
    """预分配所有中间缓冲区, 避免反复 malloc"""
    def __init__(self, shape, w_z=1.5):
        self.dx = np.zeros(shape, dtype=np.float32)
        self.dy = np.zeros(shape, dtype=np.float32)
        self.dz = np.zeros(shape, dtype=np.float32)
        self.div = np.zeros(shape, dtype=np.float32)
        self.w_z = w_z
    def compute(self, v, beta, eps=1e-8):
        dx, dy, dz, div = self.dx, self.dy, self.dz, self.div
        # 清零
        dx[:,:,:-1] = 0; dy[:,:-1,:] = 0; dz[:-1,:,:] = 0
        div[:] = 0
        # 梯度
        np.subtract(v[:,:,1:], v[:,:,:-1], out=dx[:,:,:-1])
        np.subtract(v[:,1:,:], v[:,:-1,:], out=dy[:,:-1,:])
        np.subtract(v[1:,:,:], v[:-1,:,:], out=dz[:-1,:,:])
        # 各向异性 z
        dz *= self.w_z
        mag = np.sqrt(dx*dx + dy*dy + dz*dz + eps)
        np.divide(dx, mag, out=dx, where=mag>eps)
        np.divide(dy, mag, out=dy, where=mag>eps)
        np.divide(dz, mag, out=dz, where=mag>eps)
        # 散度 (原地累加)
        div[:,:,1:-1] = dx[:,:,1:-1] - dx[:,:,:-2]
        div[:,:,0] = dx[:,:,0]; div[:,:,-1] = -dx[:,:,-2]
        div[:,1:-1,:] += dy[:,1:-1,:] - dy[:,:-2,:]
        div[:,0,:] += dy[:,0,:]; div[:,-1,:] += -dy[:,-2,:]
        div[1:-1,:,:] += dz[1:-1,:,:] - dz[:-2,:,:]
        div[0,:,:] += dz[0,:,:]; div[-1,:,:] += -dz[-2,:,:]
        # v - β*(-div) = v + β*div
        np.add(v, beta * div, out=v)
        return v

t0 = time()
rec_hybrid = rec_fdk_noisy.copy()
rec_hybrid = algs.ossart(sino_noisy, geo, angles, niter=3, init=rec_hybrid,
                         blocksize=blocksize, verbose=False)
tv_cache_hybrid = TVGradientCache((nz, N, N), w_z=1.5)
rec_hybrid = tv_cache_hybrid.compute(rec_hybrid, 0.003)
rec_hybrid = 0.5 * rec_hybrid + 0.5 * rec_fdk_noisy
t_hybrid = time() - t0
r_hybrid, s_hybrid = calc_rmse(linear_scale(rec_hybrid)), calc_ssim(linear_scale(rec_hybrid))
best_hybrid = {"rec": linear_scale(rec_hybrid), "rmse": r_hybrid, "ssim": s_hybrid, "t": t_hybrid}
print(f"   Hybrid IR: RMSE={r_hybrid:.5f}, SSIM={s_hybrid:.4f}, {t_hybrid*1000:.0f}ms")
hybrid_zprof = calc_z_profile(best_hybrid["rec"])
print(f"   z-profile: mean={hybrid_zprof.mean():.5f}")

# ========== C. TV-OS-SART (带缓存TV + 合并niter) ==========
print("-" * 55)
print(f"C. TV-OS-SART (缓存TV, bs={blocksize})")
print("-" * 55)
tv_cache = TVGradientCache((nz, N, N), w_z=1.5)
rec_tv = rec_fdk_noisy.copy()
best_tv = {"rmse": 1e9, "ssim": -1, "rec": None, "t": 0, "n": 0}
prev_n = 0
tv_niters = [1, 3, 5, 10]
tv_betas = [0.005, 0.003, 0.0015, 0.0008]

for ni, beta in zip(tv_niters, tv_betas):
    dn = ni - prev_n
    t0 = time()
    # ossart 合并调用
    rec_tv = algs.ossart(sino_noisy, geo, angles, niter=dn, init=rec_tv,
                         blocksize=blocksize, verbose=False)
    # TV (预分配缓存, 原地操作)
    rec_tv = tv_cache.compute(rec_tv, beta)
    # FDK 稳定化
    rec_tv *= 0.95; rec_tv += 0.05 * rec_fdk_noisy
    t = time() - t0
    r, s = calc_rmse(linear_scale(rec_tv)), calc_ssim(linear_scale(rec_tv))
    if r < best_tv["rmse"]:
        best_tv = {"rmse": r, "ssim": s, "rec": linear_scale(rec_tv), "t": t, "n": ni}
    print(f"   TV-OS-SART x{ni:3d} (β={beta:.4f}): RMSE={r:.5f}, SSIM={s:.4f}, {t*1000:.0f}ms")
    prev_n = ni
print(f"   >> 最优: TV-OS-SART x{best_tv['n']}: RMSE={best_tv['rmse']:.5f}")
tv_improv = (1 - best_tv['rmse'] / fdk_rmse) * 100
print(f"   TV 改善: {tv_improv:+.1f}%")
best_tv_zprof = calc_z_profile(best_tv["rec"])
print(f"   z-profile: mean={best_tv_zprof.mean():.5f}")

# ========== D. 汇总 ==========
print("\n" + "=" * 70)
print(f"汇总对比 (32x512x512, 360角度, blocksize={blocksize})")
print("=" * 70)
print(f"{'算法':30s} {'耗时(ms)':>10s} {'RMSE':>12s} {'SSIM':>8s} {'z-RMSE':>10s}")
print("-" * 72)
print(f"{'Pure FDK':30s} {fdk_t * 1000:>8.0f} ms  {fdk_rmse:>10.5f}  {fdk_ssim:>8.4f} {fdk_zprof.mean():>10.5f}")
print(f"{'Hybrid IR':30s} {t_hybrid*1000:>8.0f} ms  {r_hybrid:>10.5f}  {s_hybrid:>8.4f} {hybrid_zprof.mean():>10.5f}")
print(f"{'TV-OS-SART x'+str(best_tv['n']):30s} {best_tv['t']*1000:>8.0f} ms  {best_tv['rmse']:>10.5f}  {best_tv['ssim']:>8.4f} {best_tv_zprof.mean():>10.5f}")

results = [
    ("Pure FDK", fdk_t, fdk_rmse, fdk_ssim),
    ("Hybrid IR", t_hybrid, r_hybrid, s_hybrid),
    ("TV-OS-SART x" + str(best_tv["n"]), best_tv["t"], best_tv["rmse"], best_tv["ssim"]),
]

# ========== 可视化 ==========
print("\n生成可视化...")
os.makedirs("img_3d_helical", exist_ok=True)
mid = nz // 2
fig = plt.figure(figsize=(24, 12))
gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
ts = strftime("%Y-%m-%d %H:%M:%S", localtime())

titles_upper = [
    ("Ground Truth", vol_gt[mid], None, None, None, None),
    ("FDK", fdk_rec[mid], fdk_rmse, fdk_ssim, fdk_t, None),
    ("Hybrid IR\nOS3+TV1+FDK50%", best_hybrid["rec"][mid], best_hybrid["rmse"], best_hybrid["ssim"], best_hybrid["t"], None),
    ("TV-OS-SART", best_tv["rec"][mid], best_tv["rmse"], best_tv["ssim"], best_tv["t"], best_tv["n"]),
]
for i, (title, img, rmse, ssim, t, ni) in enumerate(titles_upper):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img * soft_mask_2d, cmap="gray", vmin=0, vmax=0.05)
    if rmse is None:
        tstr = title
    else:
        tag = f" x{ni}" if ni else ""
        tstr = f"{title}{tag}\nRMSE={rmse:.5f}  SSIM={ssim:.4f}\n{t*1000:.0f}ms"
    ax.set_title(tstr, fontsize=8)
    ax.axis("off")

# 误差图
for i, (title, img, rmse, ssim, t, ni) in enumerate(titles_upper):
    ax2 = fig.add_subplot(gs[1, i])
    if rmse is not None:
        e = img - vol_gt[mid]
        v = max(0.005, np.percentile(np.abs(e), 95) * 1.2)
        ax2.imshow(e * soft_mask_2d, cmap="RdBu_r", vmin=-v, vmax=v)
        ax2.set_title(f"Error  x{v:.4f}", fontsize=8)
    else:
        ax2.imshow(np.zeros_like(img), cmap="gray")
        ax2.set_title("Reference", fontsize=8)
    ax2.axis("off")

# z-profile 图
z_coord = np.arange(nz)
ax_z = fig.add_subplot(gs[2, :])
for zp, zl, zc in zip(
    [fdk_zprof, hybrid_zprof, best_tv_zprof],
    ["FDK", "Hybrid IR", "TV-OS-SART"],
    ["orange", "green", "red"]
):
    ax_z.plot(z_coord, zp, 'o-', label=zl, color=zc, markersize=3)
ax_z.set_xlabel("z slice", fontsize=9)
ax_z.set_ylabel("RMSE per slice", fontsize=9)
ax_z.legend(fontsize=8)
ax_z.set_title("z-profile: 沿 z 方向逐片 RMSE", fontsize=10)
ax_z.grid(True, alpha=0.3)

plt.suptitle(
    f"TIGRE CUDA Helical Cone-beam (512x512x32, {n_angles}角度, pitch={pitch}mm, blocksize={blocksize})\n"
    f"+ Hybrid IR (OS-SART×3+TV+FDK混合)\n{ts}",
    fontsize=12, fontweight="bold", y=0.98
)
plt.savefig("img_3d_helical/tigre_cone_hybrid.png", dpi=150, bbox_inches="tight")
plt.close()
print("   => img_3d_helical/tigre_cone_hybrid.png")

summary = {
    "backend": "TIGRE CUDA helical cone-beam (优化版)",
    "config": {"N": N, "nz": nz, "n_angles": n_angles, "blocksize": 36, "pitch": pitch},
    "results": {
        name: {"rmse": round(r, 5), "ssim": round(s, 4), "time_ms": round(t * 1000, 1)}
        for name, t, r, s in results
    },
    "z_profile": {
        "FDK": [round(x,5) for x in fdk_zprof.tolist()],
        "Hybrid IR": [round(x,5) for x in hybrid_zprof.tolist()],
        "TV-OS-SART x"+str(best_tv["n"]): [round(x,5) for x in best_tv_zprof.tolist()],
    }
}
with open("img_3d_helical/tigre_cone_hybrid_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("   => img_3d_helical/tigre_cone_hybrid_summary.json")
print("\nDone!")
