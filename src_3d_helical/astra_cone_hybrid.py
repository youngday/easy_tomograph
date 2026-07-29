"""
FBP + IR 混合重建 (ASTRA 锥束 CBCT) — 与 TIGRE 对齐版
======================================================
FDK / 噪声OS-SART / TV-OS-SART
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
    import astra
except ImportError:
    print("错误: 需要 ASTRA Toolbox")
    exit(1)

print("=" * 60)
print("FBP + IR 混合重建  [锥束 CBCT | ASTRA CUDA]")
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

# ---- 软遮罩 ----
Ygrid, Xgrid = np.ogrid[:N, :N]
dist_xy = np.sqrt((Xgrid - N/2)**2 + (Ygrid - N/2)**2)
body_r = N * 0.42
soft_mask_2d = np.clip((body_r + 20 - dist_xy) / 20, 0, 1)

# ---- 锥束几何 (与 TIGRE 对齐) ----
# 对齐: DSO=1000, iso-detector=500 → 总 source-detector=1500
# voxel 各向同性 1.0mm, detector 各向同性 1.0mm
angles_rad = np.deg2rad(np.linspace(0, 360, n_angles, endpoint=False)).astype(np.float32)
DSO, DSD_det = 1000.0, 500.0  # source-isocenter, isocenter-detector
D = int(np.ceil(N * np.sqrt(2)))
n_det_row, n_det_col = nz * 2, D
det_pix = 1.0

vectors = np.zeros((n_angles, 12), dtype=np.float32)
for i, th in enumerate(angles_rad):
    c, s = np.cos(th), np.sin(th)
    vectors[i, :3] = [DSO * s, -DSO * c, 0.0]          # source
    vectors[i, 3:6] = [-DSD_det * s, DSD_det * c, 0.0]  # detector center
    vectors[i, 6:9] = [det_pix * c, det_pix * s, 0.0]   # det u-vector
    vectors[i, 9:12] = [0.0, 0.0, det_pix]               # det v-vector

proj_geom = astra.create_proj_geom("cone_vec", n_det_row, n_det_col, vectors)
vol_geom = astra.create_vol_geom(N, N, nz)

print("\nGPU 正向投影...")
t0 = time()
vid = astra.data3d.create("-vol", vol_geom, vol_gt.astype(np.float32))
sid = astra.data3d.create("-sino", proj_geom, 0.0)
cfg = astra.astra_dict("FP3D_CUDA")
cfg["ProjectionDataId"] = sid
cfg["VolumeDataId"] = vid
aid = astra.algorithm.create(cfg)
astra.algorithm.run(aid)
sino = astra.data3d.get(sid)
print(f"   完成: {(time()-t0)*1000:.0f}ms, 形状 {sino.shape}")
astra.algorithm.delete(aid)
astra.data3d.delete(sid)
astra.data3d.delete(vid)

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

# GPU 预热
print("GPU 预热...")
sid_w = astra.data3d.create("-sino", proj_geom, sino)
rid_w = astra.data3d.create("-vol", vol_geom)
c = astra.astra_dict("FDK_CUDA")
c["ProjectionDataId"] = sid_w; c["ReconstructionDataId"] = rid_w
a = astra.algorithm.create(c); astra.algorithm.run(a, 1); astra.algorithm.delete(a)
astra.data3d.delete(rid_w); astra.data3d.delete(sid_w)
print("   预热完成\n")

# ============================
# A. Pure FDK
# ============================
print("-" * 55)
print("A. Pure FDK")
print("-" * 55)
t0 = time()
sid_fdk = astra.data3d.create("-sino", proj_geom, sino)
rid_fdk = astra.data3d.create("-vol", vol_geom)
c = astra.astra_dict("FDK_CUDA")
c["ProjectionDataId"] = sid_fdk; c["ReconstructionDataId"] = rid_fdk
a = astra.algorithm.create(c); astra.algorithm.run(a)
fdk_raw = astra.data3d.get(rid_fdk).copy()
astra.algorithm.delete(a); astra.data3d.delete(rid_fdk); astra.data3d.delete(sid_fdk)
fdk_rec = linear_scale(fdk_raw)
fdk_t = time() - t0; fdk_rmse = calc_rmse(fdk_rec); fdk_ssim = calc_ssim(fdk_rec)
print(f"   RMSE={fdk_rmse:.5f}, SSIM={fdk_ssim:.4f}, {fdk_t*1000:.0f}ms")

# ============================
# B. 噪声/伪影 + TV-OS-SART
# ============================
n_subsets = 10
sub_size = n_angles // n_subsets

from ct_noise import add_artifacts
np.random.seed(2024)
sino_noisy = add_artifacts(sino, dose_level=0.5, hardening=False, rings=True, scatter=False)

# 占位: clean OS-SART 已移除
best_c = {"rmse": 1e9}

# TV 梯度算子
def tv_gradient(v, eps=1e-8):
    dx = np.zeros_like(v); dy = np.zeros_like(v); dz = np.zeros_like(v)
    dx[:,:,:-1]=v[:,:,1:]-v[:,:,:-1]; dy[:,:-1,:]=v[:,1:,:]-v[:,:-1,:]; dz[:-1,:,:]=v[1:,:,:]-v[:-1,:,:]
    mag = np.sqrt(dx**2+dy**2+dz**2+eps); ux,uy,uz = dx/mag, dy/mag, dz/mag
    div = np.zeros_like(v)
    div[:,:,1:-1]=ux[:,:,1:-1]-ux[:,:,:-2]; div[:,:,0]=ux[:,:,0]; div[:,:,-1]=-ux[:,:,-2]
    div[:,1:-1,:]+=uy[:,1:-1,:]-uy[:,:-2,:]; div[:,0,:]+=uy[:,0,:]; div[:,-1,:]+=-uy[:,-2,:]
    div[1:-1,:,:]+=uz[1:-1,:,:]-uz[:-2,:,:]; div[0,:,:]+=uz[0,:,:]; div[-1,:,:]+=-uz[-2,:,:]
    return -div

# 对噪声数据建子集
subsets_n = []
for i in range(n_subsets):
    idx = slice(i * sub_size, (i + 1) * sub_size)
    sv = vectors[idx].copy()
    pg = astra.create_proj_geom("cone_vec", n_det_row, n_det_col, sv)
    ss = np.ascontiguousarray(sino_noisy[:, idx, :])
    sid_sub = astra.data3d.create("-sino", pg, ss)
    subsets_n.append((pg, sid_sub))

# 有噪声 FDK
sid_n = astra.data3d.create("-sino", proj_geom, sino_noisy)
rid_n = astra.data3d.create("-vol", vol_geom)
c = astra.astra_dict("FDK_CUDA")
c["ProjectionDataId"] = sid_n; c["ReconstructionDataId"] = rid_n
a = astra.algorithm.create(c); astra.algorithm.run(a)
rec_fdk_n = astra.data3d.get(rid_n).copy()
astra.algorithm.delete(a); astra.data3d.delete(rid_n); astra.data3d.delete(sid_n)

# OS-SART on noisy
best_n = {"rmse": 1e9}
rec_n = rec_fdk_n.copy()
prev_n = 0
for ni in [1, 3, 5]:
    dn = ni - prev_n
    t0 = time()
    for _ in range(dn):
        for _, sid_sub in subsets_n:
            rid = astra.data3d.create("-vol", vol_geom, data=rec_n.astype(np.float32))
            c = astra.astra_dict("SIRT3D_CUDA")
            c["ProjectionDataId"] = sid_sub; c["ReconstructionDataId"] = rid
            c["option"] = {"GPUindex": 0}
            a = astra.algorithm.create(c); astra.algorithm.run(a, 1)
            rec_n = astra.data3d.get(rid).copy()
            astra.algorithm.delete(a); astra.data3d.delete(rid)
    t = time() - t0
    r, s = calc_rmse(linear_scale(rec_n)), calc_ssim(linear_scale(rec_n))
    if r < best_n["rmse"]: best_n = {"rmse": r, "ssim": s, "rec": linear_scale(rec_n), "t": t, "n": ni}
    print(f"   有噪声 OS-SART x{ni:3d}: RMSE={r:.5f}, SSIM={s:.4f}, {t*1000:.0f}ms")
    prev_n = ni
print(f"   >> 最优: x{best_n['n']}: RMSE={best_n['rmse']:.5f}")

# TV-OS-SART (β scheduling: high→low for denoise→detail)
best_tv = {"rmse": 1e9}
rec_tv = rec_fdk_n.copy()
prev_n = 0
for ni, beta in zip([1, 3, 5, 10], [0.003, 0.002, 0.001, 0.0005]):
    dn = ni - prev_n
    t0 = time()
    for _ in range(dn):
        for _, sid_sub in subsets_n:
            rid = astra.data3d.create("-vol", vol_geom, data=rec_tv.astype(np.float32))
            c = astra.astra_dict("SIRT3D_CUDA")
            c["ProjectionDataId"] = sid_sub; c["ReconstructionDataId"] = rid
            c["option"] = {"GPUindex": 0}
            a = astra.algorithm.create(c); astra.algorithm.run(a, 1)
            rec_tv = astra.data3d.get(rid).copy()
            astra.algorithm.delete(a); astra.data3d.delete(rid)
        rec_tv = rec_tv - beta * tv_gradient(rec_tv)
    t = time() - t0
    r, s = calc_rmse(linear_scale(rec_tv)), calc_ssim(linear_scale(rec_tv))
    if r < best_tv["rmse"]: best_tv = {"rmse": r, "ssim": s, "rec": linear_scale(rec_tv), "t": t, "n": ni}
    print(f"   TV-OS-SART x{ni:3d}: RMSE={r:.5f}, SSIM={s:.4f}, {t*1000:.0f}ms")
    prev_n = ni
for _, sid in subsets_n: astra.data3d.delete(sid)
print(f"   >> 最优: TV-OS-SART x{best_tv['n']}: RMSE={best_tv['rmse']:.5f}")
tv_improv = (1 - best_tv['rmse']/best_n['rmse']) * 100
print(f"   TV 改善: {tv_improv:+.1f}%")

# ============================
# C. 汇总
# ============================
print("\n" + "=" * 70)
print("汇总对比 (32x512x512, 360角度, 10子集)")
print("=" * 70)
print(f"{'算法':30s} {'耗时(ms)':>10s} {'RMSE':>12s} {'SSIM':>8s} {'vsFDK':>10s}")
print("-" * 72)
c_imp = f"{(1 - best_n['rmse'] / fdk_rmse) * 100:+.1f}%"
print(f"{'Pure FDK':30s} {fdk_t*1000:>8.0f} ms  {fdk_rmse:>10.5f}  {fdk_ssim:>8.4f} {'-':>10s}")
print(f"{'有噪声 OS-SART x'+str(best_n['n']):30s} {best_n['t']*1000:>8.0f} ms  {best_n['rmse']:>10.5f}  {best_n['ssim']:>8.4f} {c_imp:>10s}")
print(f"{'TV-OS-SART x'+str(best_tv['n']):30s} {best_tv['t']*1000:>8.0f} ms  {best_tv['rmse']:>10.5f}  {best_tv['ssim']:>8.4f} {'':>10s}")

results = [
    ("Pure FDK", fdk_t, fdk_rmse, fdk_ssim),
    ("有噪声 OS-SART x"+str(best_n["n"]), best_n["t"], best_n["rmse"], best_n["ssim"]),
    ("TV-OS-SART x"+str(best_tv["n"]), best_tv["t"], best_tv["rmse"], best_tv["ssim"]),
]

# 可视化
print("\n生成可视化...")
os.makedirs("img_3d_helical", exist_ok=True)
mid = nz // 2
fig = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)
ts = strftime("%Y-%m-%d %H:%M:%S", localtime())

titles_upper = [
    ("Ground Truth", vol_gt[mid], None, None, None, None),
    ("FDK", fdk_rec[mid], fdk_rmse, fdk_ssim, fdk_t, None),
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
    ax.set_title(tstr, fontsize=8); ax.axis("off")
for i, (title, img, rmse, ssim, t, ni) in enumerate(titles_upper):
    ax2 = fig.add_subplot(gs[1, i])
    if rmse is not None:
        e = img - vol_gt[mid]; v = max(0.005, np.percentile(np.abs(e), 95) * 1.2)
        ax2.imshow(e * soft_mask_2d, cmap="RdBu_r", vmin=-v, vmax=v)
        ax2.set_title(f"Error  x{v:.4f}", fontsize=8)
    else:
        ax2.imshow(np.zeros_like(img), cmap="gray"); ax2.set_title("Reference", fontsize=8)
    ax2.axis("off")
plt.suptitle(f"ASTRA CUDA Cone-beam (32x512x512, {n_angles}角度, 10子集)\n{ts}",
             fontsize=12, fontweight="bold", y=0.98)
plt.savefig("img_3d_helical/astra_cone_hybrid.png", dpi=150, bbox_inches="tight")
plt.close()
print("   => img_3d_helical/astra_cone_hybrid.png")

summary = {
    "backend": "ASTRA CUDA cone-beam (对齐TIGRE版)",
    "config": {"N": N, "nz": nz, "n_angles": n_angles, "n_subsets": n_subsets, "DSO":1000, "iso_det":500},
    "results": {name: {"rmse": round(r,5), "ssim": round(s,4), "time_ms": round(t*1000,1)}
                for name,t,r,s in results},
}
with open("img_3d_helical/astra_cone_hybrid_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("   => img_3d_helical/astra_cone_hybrid_summary.json")
print("\nDone!")
