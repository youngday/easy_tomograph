"""
FBP + IR 混合重建 (TIGRE 锥束 CBCT) — 优化版
===============================================
FDK / OS-SART (warm-start) / TV-OS-SART
优化: 统一体模(M4) + 加速blocksize + 图像打印时间
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
print("FBP + IR 混合重建对比  [锥束 CBCT | TIGRE CUDA 优化]")
print("=" * 60)

N = 512
nz = 32
n_angles = 360

import tomophantom
from tomophantom import TomoP3D

tp_lib = os.path.join(os.path.dirname(tomophantom.__file__),
                       'phantomlib', 'Phantom3DLibrary.dat')
# 统一使用 Model 4 (与 ASTRA 一致)
ph = TomoP3D.Model(4, (N, N, nz), tp_lib).astype(np.float32)
vol_gt = np.transpose(ph, (2, 0, 1)).copy()
vol_gt = (vol_gt - 0.2) * 0.035
vol_gt = np.clip(vol_gt, 0, 0.05)
Y, X = np.ogrid[:N, :N]
circ_mask = ((X - N/2)**2 + (Y - N/2)**2) <= (N*0.42)**2
vol_gt[:, ~circ_mask] = 0.0
print(f"   体模: [{vol_gt.min():.5f}, {vol_gt.max():.5f}]")

# 软遮罩 (消除外围辉光)
Ygrid, Xgrid = np.ogrid[:N, :N]
dist_xy = np.sqrt((Xgrid - N/2)**2 + (Ygrid - N/2)**2)
body_r = N * 0.42
soft_mask_2d = np.clip((body_r + 20 - dist_xy) / 20, 0, 1)

angles = np.deg2rad(np.linspace(0, 360, n_angles, endpoint=False)).astype(np.float32)
D = int(np.ceil(N * np.sqrt(2)))
geo = tigre.geometry()
geo.DSD = 1500.0  # 与 ASTRA 对齐: DSO(1000)+iso-det(500)=1500
geo.DSO = 1000.0
dVox = 1.0
geo.nVoxel = np.array([nz, N, N])
geo.sVoxel = geo.nVoxel * dVox  # 各向同性 1.0mm
geo.dVoxel = np.array([dVox, dVox, dVox])
geo.nDetector = np.array([nz * 2, D])  # (v, u)
geo.dDetector = np.array([1.0, 1.0])   # 各向同性 1.0mm
geo.sDetector = geo.nDetector * geo.dDetector
geo.offOrigin = np.array([0, 0, 0])
geo.offDetector = np.array([0, 0])
geo.mode = "cone"
geo.filter = None

print("\nGPU 正向投影...")
t0 = time()
sino = tigre.Ax(vol_gt, geo, angles)
print(f"   完成: {(time() - t0) * 1000:.0f}ms, 形状 {sino.shape}")

# ---- 度量函数 ----
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

print("GPU 预热...")
# blocksize=36 → 10 subsets (快2倍于18)
_ = algs.fdk(sino, geo, angles, filter="hann")
_ = algs.ossart(sino, geo, angles, 1, blocksize=36, verbose=False)
print("   预热完成\n")

# ========== A. Pure FDK ==========
print("-" * 55)
print("A. Pure FDK")
print("-" * 55)
t0 = time()
rec_fdk = algs.fdk(sino, geo, angles, filter="hann")
fdk_rec = linear_scale(rec_fdk)
fdk_t = time() - t0
fdk_rmse = calc_rmse(fdk_rec)
fdk_ssim = calc_ssim(fdk_rec)
print(f"   RMSE={fdk_rmse:.5f}, SSIM={fdk_ssim:.4f}, {fdk_t * 1000:.0f}ms")

# ========== B. OS-SART (clean, warm-start from FDK) ==========
print("-" * 55)
print("B. FBP + OS-SART (warm-start, blocksize=36)")
print("-" * 55)
c_hist = []
best_c = {"rmse": 1e9, "ssim": -1, "rec": None, "t": 0, "n": 0}
rec_c = rec_fdk.copy()
prev_n = 0
for n_iter in [1, 3, 5, 10]:
    dn = n_iter - prev_n
    t0 = time()
    rec_c = algs.ossart(sino, geo, angles, niter=dn, init=rec_c, blocksize=36, verbose=False)
    t = time() - t0
    r = calc_rmse(linear_scale(rec_c))
    s = calc_ssim(linear_scale(rec_c))
    c_hist.append((n_iter, t, r, s))
    if r < best_c["rmse"]:
        best_c = {"rmse": r, "ssim": s, "rec": linear_scale(rec_c), "t": t, "n": n_iter}
    print(f"   x{n_iter:3d} (+{dn}): RMSE={r:.5f}, SSIM={s:.4f}, {t*1000:.0f}ms")
    prev_n = n_iter
print(f"   >> 最优: OS-SART x{best_c['n']}: RMSE={best_c['rmse']:.5f}")

# ========== C. 噪声/伪影对比 (OS-SART vs TV-OS-SART) ==========
print("-" * 55)
print("C. 噪声/伪影对比 (量子噪声 + 环伪影)")
print("-" * 55)
print("   比较 OS-SART 与 TV-OS-SART 的鲁棒性")

from ct_noise import add_artifacts

np.random.seed(2024)
sino_noisy = add_artifacts(sino, dose_level=0.5, hardening=False, rings=True, scatter=False)

def tv_gradient(v, eps=1e-8):
    dx = np.zeros_like(v); dy = np.zeros_like(v); dz = np.zeros_like(v)
    dx[:,:,:-1] = v[:,:,1:]-v[:,:,:-1]; dy[:,:-1,:] = v[:,1:,:]-v[:,:-1,:]; dz[:-1,:,:] = v[1:,:,:]-v[:-1,:,:]
    mag = np.sqrt(dx**2+dy**2+dz**2+eps); ux,uy,uz = dx/mag, dy/mag, dz/mag
    div = np.zeros_like(v)
    div[:,:,1:-1] = ux[:,:,1:-1]-ux[:,:,:-2]; div[:,:,0]=ux[:,:,0]; div[:,:,-1]=-ux[:,:,-2]
    div[:,1:-1,:] += uy[:,1:-1,:]-uy[:,:-2,:]; div[:,0,:]+=uy[:,0,:]; div[:,-1,:]+=-uy[:,-2,:]
    div[1:-1,:,:] += uz[1:-1,:,:]-uz[:-2,:,:]; div[0,:,:]+=uz[0,:,:]; div[-1,:,:]+=-uz[-2,:,:]
    return -div

# 有噪声 OS-SART
rec_fdk_noisy = algs.fdk(sino_noisy, geo, angles, filter="hann")
rec_n = rec_fdk_noisy.copy()
n_hist = []; best_n = {"rmse":1e9,"ssim":-1,"rec":None,"t":0,"n":0}
prev_n = 0
for ni in [1, 3, 5]:
    dn = ni - prev_n
    t0=time(); rec_n=algs.ossart(sino_noisy,geo,angles,niter=dn,init=rec_n,blocksize=36,verbose=False)
    t=time()-t0; r,s=calc_rmse(linear_scale(rec_n)),calc_ssim(linear_scale(rec_n))
    n_hist.append((ni,t,r,s))
    if r<best_n["rmse"]: best_n={"rmse":r,"ssim":s,"rec":linear_scale(rec_n),"t":t,"n":ni}
    print(f"   有噪声 OS-SART x{ni:3d} (+{dn}): RMSE={r:.5f}, SSIM={s:.4f}, {t*1000:.0f}ms")
    prev_n = ni
print(f"   >> 最优: {best_n['n']}: RMSE={best_n['rmse']:.5f}")

# TV-OS-SART (β scheduling: high→low for denoise→detail)
rec_tv=rec_fdk_noisy.copy()
tv_hist=[]; best_tv={"rmse":1e9,"ssim":-1,"rec":None,"t":0,"n":0}
prev_n = 0
for ni, beta in zip([1, 3, 5, 10], [0.003, 0.002, 0.001, 0.0005]):
    dn = ni - prev_n
    t0=time(); rec_tv=algs.ossart(sino_noisy,geo,angles,niter=dn,init=rec_tv,blocksize=36,verbose=False)
    rec_tv=rec_tv-beta*tv_gradient(rec_tv)
    t=time()-t0; r,s=calc_rmse(linear_scale(rec_tv)),calc_ssim(linear_scale(rec_tv))
    tv_hist.append((ni,t,r,s))
    if r<best_tv["rmse"]: best_tv={"rmse":r,"ssim":s,"rec":linear_scale(rec_tv),"t":t,"n":ni}
    print(f"   TV-OS-SART x{ni:3d} (+{dn}, \u03b2={beta}): RMSE={r:.5f}, SSIM={s:.4f}, {t*1000:.0f}ms")
    prev_n = ni
print(f"   >> 最优: TV-OS-SART x{best_tv['n']}: RMSE={best_tv['rmse']:.5f}")
tv_improv = (1-best_tv['rmse']/best_n['rmse'])*100
print(f"   TV 改善: {tv_improv:+.1f}%")

# ========== D. 汇总 ==========
print("\n" + "=" * 70)
print("汇总对比 (32x512x512, 360角度, blocksize=36)")
print("=" * 70)
print(f"{'算法':30s} {'耗时(ms)':>10s} {'RMSE':>12s} {'SSIM':>8s} {'vsFDK':>10s}")
print("-" * 72)
c_imp = f"{(1 - best_c['rmse'] / fdk_rmse) * 100:+.1f}%"
print(f"{'Pure FDK':30s} {fdk_t * 1000:>8.0f} ms  {fdk_rmse:>10.5f}  {fdk_ssim:>8.4f} {'-':>10s}")
print(f"{'OS-SART x' + str(best_c['n']):30s} {best_c['t'] * 1000:>8.0f} ms  {best_c['rmse']:>10.5f}  {best_c['ssim']:>8.4f} {c_imp:>10s}")
print(f"{'有噪声 OS-SART x'+str(best_n['n']):30s} {best_n['t']*1000:>8.0f} ms  {best_n['rmse']:>10.5f}  {best_n['ssim']:>8.4f} {'':>10s}")
print(f"{'TV-OS-SART x'+str(best_tv['n']):30s} {best_tv['t']*1000:>8.0f} ms  {best_tv['rmse']:>10.5f}  {best_tv['ssim']:>8.4f} {'':>10s}")

results = [
    ("Pure FDK", fdk_t, fdk_rmse, fdk_ssim),
    ("OS-SART x" + str(best_c["n"]), best_c["t"], best_c["rmse"], best_c["ssim"]),
    ("有噪声 OS-SART x" + str(best_n["n"]), best_n["t"], best_n["rmse"], best_n["ssim"]),
    ("TV-OS-SART x" + str(best_tv["n"]), best_tv["t"], best_tv["rmse"], best_tv["ssim"]),
]

# ========== 可视化 ==========
print("\n生成可视化...")
os.makedirs("img_3d_axial", exist_ok=True)
mid = nz // 2
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.3)

ts = strftime("%Y-%m-%d %H:%M:%S", localtime())

# 上排: 重建图像
titles_upper = [
    ("Ground Truth", vol_gt[mid], None, None, None, None),
    ("FDK", fdk_rec[mid], fdk_rmse, fdk_ssim, fdk_t, None),
    ("OS-SART", best_c["rec"][mid], best_c["rmse"], best_c["ssim"], best_c["t"], best_c["n"]),
    ("Noisy OS-SART", best_n["rec"][mid], best_n["rmse"], best_n["ssim"], best_n["t"], best_n["n"]),
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

# 下排: 误差图
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

plt.suptitle(
    f"TIGRE CUDA Cone-beam  (512x512x32, {n_angles}角度, blocksize=36)\n{ts}",
    fontsize=12, fontweight="bold", y=0.98
)
plt.savefig("img_3d_axial/tigre_cone_hybrid.png", dpi=150, bbox_inches="tight")
plt.close()
print("   => img_3d_axial/tigre_cone_hybrid.png")

summary = {
    "backend": "TIGRE CUDA cone-beam (optimized v2)",
    "config": {"N": N, "nz": nz, "n_angles": n_angles, "blocksize": 36},
    "results": {
        name: {"rmse": round(r, 5), "ssim": round(s, 4), "time_ms": round(t * 1000, 1)}
        for name, t, r, s in results
    },
}
with open("img_3d_axial/tigre_cone_hybrid_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("   => img_3d_axial/tigre_cone_hybrid_summary.json")
print("\nDone!")
