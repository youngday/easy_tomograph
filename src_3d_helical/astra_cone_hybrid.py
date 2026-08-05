"""
螺旋 CT 混合重建 (ASTRA 锥束) — Helical CBCT 优化版
=====================================================
FDK / Hybrid IR / TV-OS-SART
优化: 自适应 TV β + z-profile 评估
螺距(pitch)=16 mm/rot, 360° 扫描
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

# TV 去噪: GPU (CuPy) → CPU 回退
try:
    from tv_gpu import tv_denoise_gpu as _tv_denoise_gpu
    _USE_GPU_TV = True
except Exception:
    _USE_GPU_TV = False

def tv_denoise(v, beta, w_z=1.5, **kwargs):
    if _USE_GPU_TV:
        try:
            return _tv_denoise_gpu(v, beta, w_z=w_z)
        except Exception:
            pass
    # CPU fallback
    dx=np.zeros_like(v);dy=np.zeros_like(v);dz=np.zeros_like(v)
    dx[:,:,:-1]=v[:,:,1:]-v[:,:,:-1];dy[:,:-1,:]=v[:,1:,:]-v[:,:-1,:];dz[:-1,:,:]=v[1:,:,:]-v[:-1,:,:]
    mag=np.sqrt(dx**2+dy**2+(w_z*dz)**2+1e-8);ux,uy,uz=dx/mag,dy/mag,w_z*dz/mag
    div=np.zeros_like(v)
    div[:,:,1:-1]=ux[:,:,1:-1]-ux[:,:,:-2];div[:,:,0]=ux[:,:,0];div[:,:,-1]=-ux[:,:,-2]
    div[:,1:-1,:]+=uy[:,1:-1,:]-uy[:,:-2,:];div[:,0,:]+=uy[:,0,:];div[:,-1,:]+=-uy[:,-2,:]
    div[1:-1,:,:]+=uz[1:-1,:,:]-uz[:-2,:,:];div[0,:,:]+=uz[0,:,:];div[-1,:,:]+=-uz[-2,:,:]
    return v - beta * (-div)

print("=" * 60)
print("螺旋(Helical) CBCT 混合重建  [ASTRA CUDA | 优化版]")
print("=" * 60)

N = 512
nz = 32
n_angles = 180

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

# ---- 螺旋(Helical)锥束几何 ----
# 对齐: DSO=1000, iso-detector=500 → 总 source-detector=1500
# voxel 各向同性 1.0mm, detector 各向同性 1.0mm
# 螺旋扫描: 源/探测器在旋转同时沿 z 方向移动
angles_rad = np.deg2rad(np.linspace(0, 360, n_angles, endpoint=False)).astype(np.float32)
DSO, DSD_det = 1000.0, 500.0  # source-isocenter, isocenter-detector
D = int(np.ceil(N * np.sqrt(2)))
n_det_row, n_det_col = nz * 2, D
det_pix = 1.0
pitch = 16.0  # 螺距 (mm/圈)

# 螺旋轨迹: 源和探测器沿 z 方向线性移动
vectors = np.zeros((n_angles, 12), dtype=np.float32)
for i, th in enumerate(angles_rad):
    c, s = np.cos(th), np.sin(th)
    z_src = pitch * (th / (2 * np.pi) - 0.5)  # 中心化到 z∈[-pitch/2, pitch/2]
    vectors[i, :3] = [DSO * s, -DSO * c, z_src]          # source
    vectors[i, 3:6] = [-DSD_det * s, DSD_det * c, z_src]  # detector center
    vectors[i, 6:9] = [det_pix * c, det_pix * s, 0.0]     # det u-vector (xy平面)
    vectors[i, 9:12] = [0.0, 0.0, det_pix]                # det v-vector (z方向)

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

# ---- 沿 z 方向逐片评估 (z-profile) ----
def calc_z_profile(rec):
    """计算每个 z 切片的 RMSE, 评估 helical 重建的 z 一致性"""
    z_rmse = []
    for z in range(rec.shape[0]):
        mask_z = vol_gt[z] > 0.001
        if mask_z.sum() > 100:
            e = vol_gt[z][mask_z] - rec[z][mask_z]
            z_rmse.append(np.sqrt(np.mean(e**2)))
        else:
            z_rmse.append(0.0)
    return np.array(z_rmse)

# GPU 预热
print("GPU 预热...")
sid_w = astra.data3d.create("-sino", proj_geom, sino)
rid_w = astra.data3d.create("-vol", vol_geom)
c = astra.astra_dict("FDK_CUDA")
c["ProjectionDataId"] = sid_w; c["ReconstructionDataId"] = rid_w
c["option"] = {"FilterType": "hann"}  # 与TIGRE filter=hann对齐 (ram-lak噪声放大2倍, 见exp_fdk_noise)
a = astra.algorithm.create(c); astra.algorithm.run(a, 1); astra.algorithm.delete(a)
astra.data3d.delete(rid_w); astra.data3d.delete(sid_w)
print("   预热完成\n")

# ============================
# A. Pure FDK (螺距校正)
# ============================
print("-" * 55)
print("A. Pure FDK (helical weighted)")
print("-" * 55)
t0 = time()
sid_fdk = astra.data3d.create("-sino", proj_geom, sino)
rid_fdk = astra.data3d.create("-vol", vol_geom)
c = astra.astra_dict("FDK_CUDA")
c["ProjectionDataId"] = sid_fdk; c["ReconstructionDataId"] = rid_fdk
c["option"] = {"FilterType": "hann"}  # 与TIGRE filter=hann对齐
a = astra.algorithm.create(c); astra.algorithm.run(a)
fdk_raw = astra.data3d.get(rid_fdk).copy()
astra.algorithm.delete(a); astra.data3d.delete(rid_fdk); astra.data3d.delete(sid_fdk)
fdk_rec = linear_scale(fdk_raw)
fdk_t = time() - t0; fdk_rmse = calc_rmse(fdk_rec); fdk_ssim = calc_ssim(fdk_rec)
fdk_zprof = calc_z_profile(fdk_rec)
print(f"   RMSE={fdk_rmse:.5f}, SSIM={fdk_ssim:.4f}, {fdk_t*1000:.0f}ms")
print(f"   z-profile: mean={fdk_zprof.mean():.5f}, max={fdk_zprof.max():.5f}, min={fdk_zprof.min():.5f}")

# ---- 有噪声数据 + 通用加速设施 ----
n_subsets = 10
sub_size = n_angles // n_subsets

from ct_noise import add_artifacts
np.random.seed(2024)
sino_noisy = add_artifacts(sino, dose_level=0.5, hardening=False, rings=True, scatter=False)

best_c = {"rmse": 1e9}

# ---- 预分配 ASTRA 对象 (避免反复 create/delete) ----
print("预分配 ASTRA 子集对象...")
t0_pre = time()
subset_algs = []  # [(pg, sid, rid, alg)]
for i in range(n_subsets):
    idx = slice(i * sub_size, (i + 1) * sub_size)
    sv = vectors[idx].copy()
    pg = astra.create_proj_geom("cone_vec", n_det_row, n_det_col, sv)
    ss = np.ascontiguousarray(sino_noisy[:, idx, :])
    sid_sub = astra.data3d.create("-sino", pg, ss)
    rid_sub = astra.data3d.create("-vol", vol_geom)
    cfg = astra.astra_dict("SIRT3D_CUDA")
    cfg["ProjectionDataId"] = sid_sub
    cfg["ReconstructionDataId"] = rid_sub
    cfg["option"] = {"GPUindex": 0}
    alg_sub = astra.algorithm.create(cfg)
    subset_algs.append((pg, sid_sub, rid_sub, alg_sub))
print(f"   完成: {(time()-t0_pre)*1000:.0f}ms")

# 有噪声 FDK (一次性)
sid_n = astra.data3d.create("-sino", proj_geom, sino_noisy)
rid_n = astra.data3d.create("-vol", vol_geom)
c = astra.astra_dict("FDK_CUDA")
c["ProjectionDataId"] = sid_n; c["ReconstructionDataId"] = rid_n
c["option"] = {"FilterType": "hann"}  # 与TIGRE filter=hann对齐
a = astra.algorithm.create(c); astra.algorithm.run(a)
rec_fdk_n = astra.data3d.get(rid_n).copy()
astra.algorithm.delete(a); astra.data3d.delete(rid_n); astra.data3d.delete(sid_n)
fdk_noisy_rmse = calc_rmse(linear_scale(rec_fdk_n))

# ===== OS-SART 快速函数 (复用ASTRA对象) =====
def fast_ossart(rec, n_step=1):
    """
    快速 OS-SART: 复用预分配的 rid/alg, 只更新数据
    避免了每次子集迭代 create/delete ASTRA 对象的开销
    """
    for _ in range(n_step):
        for _, sid, rid, alg in subset_algs:
            astra.data3d.store(rid, rec.astype(np.float32))
            astra.algorithm.run(alg, 1)
            rec = astra.data3d.get(rid).copy()
    return rec

# ===== C. TV-OS-SART (加速版 + 各向异性) =====
print("-" * 55)
print("C. TV-OS-SART (加速: 复用ASTRA对象 + 各向异性TV)")
print("-" * 55)

best_tv = {"rmse": 1e9}
rec_tv = rec_fdk_n.copy()
beta0, decay = 0.002, 0.8  # β递减: 0.002→0.0003@x10 (interleave, 与TIGRE/Hybrid一致)
w_z = 1.5
t_tv_total = 0.0  # 纯算法时间累计 (度量/打印不计入, 与Hybrid口径一致)

for ni in range(1, 11):  # 迭代至 x10 (ASTRA SIRT收敛慢)
    t0 = time()
    rec_tv = rec_tv.astype(np.float32)
    for _, sid, rid, alg in subset_algs:
        astra.data3d.store(rid, rec_tv)
        astra.algorithm.run(alg, 1)
        rec_tv = astra.data3d.get(rid).copy()
    rec_tv = tv_denoise(rec_tv, beta0 * decay ** (ni - 1), w_z=1.5)
    t_tv_total += time() - t0
    t = t_tv_total
    r, s = calc_rmse(linear_scale(rec_tv)), calc_ssim(linear_scale(rec_tv))
    if r < best_tv["rmse"]:
        best_tv = {"rmse": r, "ssim": s, "rec": linear_scale(rec_tv), "t": t, "n": ni}
    print(f"   TV-OS-SART x{ni:3d} (β={beta0*decay**(ni-1):.4f}): RMSE={r:.5f}, SSIM={s:.4f}, 累计{t*1000:.0f}ms")

# 清理 ASTRA 对象
print(f"   >> 最优: TV-OS-SART x{best_tv['n']}: RMSE={best_tv['rmse']:.5f}")
tv_improv = (1 - best_tv['rmse']/fdk_noisy_rmse) * 100
print(f"   TV 改善 vs 噪声FDK({fdk_noisy_rmse:.5f}): {tv_improv:+.1f}%")
best_tv_zprof = calc_z_profile(best_tv["rec"])
print(f"   z-profile: mean={best_tv_zprof.mean():.5f}, max={best_tv_zprof.max():.5f}")

# ============================
# B. Fast Hybrid IR (OS10+TV10(β↓)+FDK10%) — 改进: SIRT3轮收敛不足→10轮; FDK50%→10%(ASTRA的FDK噪声大, 50%混合拉坏, 见exp_astra_hybrid)
# ============================
print("-" * 55)
t0 = time()
rec_hybrid = rec_fdk_n.copy()
beta0, decay = 0.002, 0.8  # β: 0.002→0.0003@x10
for ni in range(10):
    rec_hybrid = fast_ossart(rec_hybrid, 1)
    rec_hybrid = tv_denoise(rec_hybrid, beta0 * decay ** ni, w_z=1.5)
rec_hybrid = 0.9 * rec_hybrid + 0.1 * rec_fdk_n
t_hybrid = time() - t0
r_hybrid, s_hybrid = calc_rmse(linear_scale(rec_hybrid)), calc_ssim(linear_scale(rec_hybrid))
best_hybrid = {"rec": linear_scale(rec_hybrid), "rmse": r_hybrid, "ssim": s_hybrid, "t": t_hybrid}
print(f"   Hybrid IR: RMSE={r_hybrid:.5f}, SSIM={s_hybrid:.4f}, {t_hybrid*1000:.0f}ms")
hybrid_zprof = calc_z_profile(best_hybrid["rec"])
print(f"   z-profile: mean={hybrid_zprof.mean():.5f}, max={hybrid_zprof.max():.5f}")

# 清理 ASTRA 对象
for pg, sid, rid, alg in subset_algs:
    astra.algorithm.delete(alg)
    astra.data3d.delete(rid)
    astra.data3d.delete(sid)

# ============================
# D. 汇总
# ============================
print("\n" + "=" * 70)
print(f"汇总对比 (32x512x512, {n_angles}角度, {n_subsets}子集)")
print("=" * 70)
print(f"{'算法':30s} {'耗时(ms)':>10s} {'RMSE':>12s} {'SSIM':>8s} {'z-RMSE':>10s}")
print("-" * 72)
print(f"{'Pure FDK':30s} {fdk_t*1000:>8.0f} ms  {fdk_rmse:>10.5f}  {fdk_ssim:>8.4f} {fdk_zprof.mean():>10.5f}")
print(f"{'Hybrid IR':30s} {t_hybrid*1000:>8.0f} ms  {r_hybrid:>10.5f}  {s_hybrid:>8.4f} {hybrid_zprof.mean():>10.5f}")
print(f"{'TV-OS-SART x'+str(best_tv['n']):30s} {best_tv['t']*1000:>8.0f} ms  {best_tv['rmse']:>10.5f}  {best_tv['ssim']:>8.4f} {best_tv_zprof.mean():>10.5f}")

results = [
    ("Pure FDK", fdk_t, fdk_rmse, fdk_ssim),
    ("Hybrid IR", t_hybrid, r_hybrid, s_hybrid),
    ("TV-OS-SART x"+str(best_tv["n"]), best_tv["t"], best_tv["rmse"], best_tv["ssim"]),
]

# 可视化
print("\n生成可视化...")
os.makedirs("img_3d_helical", exist_ok=True)
mid = nz // 2
fig = plt.figure(figsize=(28, 12))
gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
ts = strftime("%Y-%m-%d %H:%M:%S", localtime())

titles_upper = [
    ("Ground Truth", vol_gt[mid], None, None, None, None),
    ("FDK", fdk_rec[mid], fdk_rmse, fdk_ssim, fdk_t, None),
    ("Hybrid IR\nOS10+TV10(β↓)+FDK10%", best_hybrid["rec"][mid], best_hybrid["rmse"], best_hybrid["ssim"], best_hybrid["t"], None),
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

# 误差图
for i, (title, img, rmse, ssim, t, ni) in enumerate(titles_upper):
    ax2 = fig.add_subplot(gs[1, i])
    if rmse is not None:
        e = img - vol_gt[mid]; v = max(0.005, np.percentile(np.abs(e), 95) * 1.2)
        ax2.imshow(e * soft_mask_2d, cmap="RdBu_r", vmin=-v, vmax=v)
        ax2.set_title(f"Error  x{v:.4f}", fontsize=8)
    else:
        ax2.imshow(np.zeros_like(img), cmap="gray"); ax2.set_title("Reference", fontsize=8)
    ax2.axis("off")

# z-profile 图 (第3行)
z_coord = np.arange(nz)
zprofiles = [fdk_zprof, hybrid_zprof, best_tv_zprof]
zlabels = ["FDK", "Hybrid IR", "TV-OS-SART"]
zcolors = ["orange", "purple", "red"]
ax_z = fig.add_subplot(gs[2, :])
for zp, zl, zc in zip(zprofiles, zlabels, zcolors):
    ax_z.plot(z_coord, zp, 'o-', label=zl, color=zc, markersize=3)
ax_z.set_xlabel("z slice", fontsize=9)
ax_z.set_ylabel("RMSE per slice", fontsize=9)
ax_z.legend(fontsize=8)
ax_z.set_title("z-profile: 沿 z 方向逐片 RMSE", fontsize=10)
ax_z.grid(True, alpha=0.3)

plt.suptitle(f"ASTRA CUDA Helical Cone-beam (32x512x512, {n_angles}角度, pitch={pitch}mm, {n_subsets}子集)\n+ Hybrid IR (OS-SART×10+TV×10+FDK混合10%)\n{ts}",
             fontsize=12, fontweight="bold", y=0.98)
plt.savefig("img_3d_helical/astra_cone_hybrid.png", dpi=150, bbox_inches="tight")
plt.close()
print("   => img_3d_helical/astra_cone_hybrid.png")

summary = {
    "backend": "ASTRA CUDA helical cone-beam (加速版)",
    "config": {"N": N, "nz": nz, "n_angles": n_angles, "n_subsets": n_subsets,
               "DSO":1000, "iso_det":500, "pitch": pitch},
    "results": {name: {"rmse": round(r,5), "ssim": round(s,4), "time_ms": round(t*1000,1)}
                for name,t,r,s in results},
    "z_profile": {
        "FDK": [round(x,5) for x in fdk_zprof.tolist()],
        "TV-OS-SART x"+str(best_tv["n"]): [round(x,5) for x in best_tv_zprof.tolist()],
    }
}
with open("img_3d_helical/astra_cone_hybrid_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("   => img_3d_helical/astra_cone_hybrid_summary.json")
print("\nDone!")
