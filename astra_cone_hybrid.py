"""
FBP + IR 混合重建 (ASTRA 锥束 CBCT)
=====================================
核心思想: 用 FDK 的快速重建作为迭代法 (SIRT3D) 的初始值

对比组:
  - Pure FDK (基线)
  - FBP + SIRT (混合)
"""

import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import os, json

try:
    import astra
except ImportError:
    print("错误: 需要 ASTRA Toolbox")
    exit(1)

print("=" * 60)
print("FBP + IR 混合重建对比  [锥束 CBCT | ASTRA CUDA]")
print("=" * 60)

# ============================================================
# 参数设置
# ============================================================
N = 512
nz = 32
n_angles = 360
print(f"体模: {nz}x{N}x{N}, 角度: {n_angles}")

# ============================================================
# 1. 生成 3D 体模 (与 tigre_cone_hybrid.py 一致)
# ============================================================
vol_gt = np.zeros((nz, N, N), dtype=np.float32)
Z, Y, X = np.ogrid[:nz, :N, :N]
cz, cy, cx = nz/2, N/2, N/2

body = ((Z - cz)/12)**2 + ((Y - cy)/(N*0.42))**2 + ((X - cx)/(N*0.35))**2 <= 1
vol_gt[body] = 0.020

bone = ((Z - cz)/10)**2 + ((Y - cy)/(N*0.30))**2 + ((X - cx)/(N*0.25))**2 <= 1
vol_gt[bone & ~body] = 0.0
bone_ring = ((Z - cz)/10)**2 + ((Y - cy)/(N*0.28))**2 + ((X - cx)/(N*0.23))**2 >= 1
vol_gt[bone & bone_ring] = 0.045

organ = ((Z - cz+4)/6)**2 + ((Y - cy-15)/(N*0.12))**2 + ((X - cx+10)/(N*0.10))**2 <= 1
vol_gt[organ] = 0.025

tumor = ((Z - cz-3)/4)**2 + ((Y - cy+20)/(N*0.06))**2 + ((X - cx-15)/(N*0.06))**2 <= 1
vol_gt[tumor] = 0.035

air = ((Z - cz+2)/5)**2 + ((Y - cy+25)/(N*0.08))**2 + ((X - cx+25)/(N*0.06))**2 <= 1
vol_gt[air] = 0.0

print(f"   体模范围: [{vol_gt.min():.5f}, {vol_gt.max():.5f}]")

# ============================================================
# 2. 锥束几何 (ASTRA cone_vec)
# ============================================================
theta_deg = np.linspace(0, 360, n_angles, endpoint=False)
angles_rad = np.deg2rad(theta_deg).astype(np.float32)
DSO = 1000.0   # 源-物体距离
DSD = 500.0    # 物体-探测器距离

# 探测器尺寸 (覆盖体模)
D = int(np.ceil(N * np.sqrt(2)))
n_det_row = nz * 2
n_det_col = D
det_pix = 1.0

# 构造 cone_vec 几何 (每投影 12 维向量)
# [srcX, srcY, srcZ, detX, detY, detZ, colX, colY, colZ, rowX, rowY, rowZ]
vectors = np.zeros((n_angles, 12), dtype=np.float32)
for i, th in enumerate(angles_rad):
    cos, sin = np.cos(th), np.sin(th)
    # 源位置: 绕 Z 轴旋转
    vectors[i, :3] = [DSO * sin, -DSO * cos, 0.0]
    # 探测器中心 (源对面)
    vectors[i, 3:6] = [-DSD * sin, DSD * cos, 0.0]
    # 列向量 (det 水平方向, 乘像素大小)
    vectors[i, 6:9] = [det_pix * cos, det_pix * sin, 0.0]
    # 行向量 (det 垂直方向, 沿 Z)
    vectors[i, 9:12] = [0.0, 0.0, det_pix]

proj_geom = astra.create_proj_geom('cone_vec', n_det_row, n_det_col, vectors)
vol_geom = astra.create_vol_geom(N, N, nz)  # ASTRA: (X, Y, Z)

# ============================================================
# 3. 正向投影 (FP3D_CUDA)
# ============================================================
print("\nGPU 正向投影...")
t0 = time()
vol_id = astra.data3d.create('-vol', vol_geom, vol_gt.astype(np.float32))
sino_id = astra.data3d.create('-sino', proj_geom, 0.0)
cfg = astra.astra_dict('FP3D_CUDA')
cfg['ProjectionDataId'] = sino_id
cfg['VolumeDataId'] = vol_id
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
sino = astra.data3d.get(sino_id)
print(f"   完成: {(time()-t0)*1000:.0f}ms, 投影形状 {sino.shape}")
astra.algorithm.delete(alg_id)
astra.data3d.delete(sino_id)
astra.data3d.delete(vol_id)

# ============================================================
# 辅助函数
# ============================================================
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
    return (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (sig_x + sig_y + c2))

# GPU 预热
print("GPU 预热...")
sino_id = astra.data3d.create('-sino', proj_geom, sino)
rec_id = astra.data3d.create('-vol', vol_geom)
for algo in ['FDK_CUDA', 'SIRT3D_CUDA']:
    cfg = astra.astra_dict(algo)
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id
    if algo == 'SIRT3D_CUDA':
        cfg['option'] = {'GPUindex': 0}
    aid = astra.algorithm.create(cfg)
    astra.algorithm.run(aid, 1)
    astra.algorithm.delete(aid)
astra.data3d.delete(rec_id)
astra.data3d.delete(sino_id)
print("   预热完成\n")

# ============================================================
# A. Pure FDK (基线)
# ============================================================
print("-" * 55)
print("A. Pure FDK (ASTRA FDK_CUDA)")
print("-" * 55)
t0 = time()
sino_id = astra.data3d.create('-sino', proj_geom, sino)
rec_id = astra.data3d.create('-vol', vol_geom)
cfg = astra.astra_dict('FDK_CUDA')
cfg['ProjectionDataId'] = sino_id
cfg['ReconstructionDataId'] = rec_id
aid = astra.algorithm.create(cfg)
astra.algorithm.run(aid)
fdk_raw = astra.data3d.get(rec_id).copy()
astra.algorithm.delete(aid)
astra.data3d.delete(rec_id)
astra.data3d.delete(sino_id)

fdk_rec = linear_scale(fdk_raw)
fdk_t = time() - t0
fdk_rmse = calc_rmse(fdk_rec)
fdk_ssim = calc_ssim(fdk_rec)
print(f"   RMSE={fdk_rmse:.5f}, SSIM={fdk_ssim:.4f}, {fdk_t*1000:.0f}ms")

# ============================================================
# B. FBP + SIRT3D (混合)
# ============================================================
print("-" * 55)
print("B. FBP + SIRT3D (FDK 初始值)")
print("-" * 55)
sirt_hist = []
best_sirt = {'rmse': 1e9, 'ssim': -1, 'rec': None, 't': 0, 'n': 0}
sino_id = astra.data3d.create('-sino', proj_geom, sino)
sirt_iters = [5, 10, 20, 50]
for n_iter in sirt_iters:
    rec_id = astra.data3d.create('-vol', vol_geom, data=fdk_raw.astype(np.float32))
    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = rec_id
    cfg['option'] = {'GPUindex': 0}
    aid = astra.algorithm.create(cfg)
    t0 = time()
    astra.algorithm.run(aid, n_iter)
    rec = linear_scale(astra.data3d.get(rec_id))
    t = time() - t0
    r, s = calc_rmse(rec), calc_ssim(rec)
    sirt_hist.append((n_iter, t, r, s))
    if r < best_sirt['rmse']:
        best_sirt = {'rmse': r, 'ssim': s, 'rec': rec, 't': t, 'n': n_iter}
    astra.algorithm.delete(aid)
    astra.data3d.delete(rec_id)
    print(f"   x{n_iter:3d}: RMSE={r:.5f}, SSIM={s:.4f}, {t*1000:.0f}ms")
astra.data3d.delete(sino_id)
print(f"   >> 最优: SIRT3D x{best_sirt['n']}: RMSE={best_sirt['rmse']:.5f}, SSIM={best_sirt['ssim']:.4f}")

# ============================================================
# C. FBP + OS-SART (SIRT3D 子集交替, 20子集)
# ============================================================
print("-" * 55)
print("C. FBP + OS-SART (20子集, SIRT3D 子集交替)")
print("-" * 55)
print("   ASTRA 无原生 OS-SART, 用 SIRT3D_CUDA 在各子集交替迭代实现")

n_subsets = 20
subset_size = n_angles // n_subsets

# 为每个子集创建独立的投影几何 + sinogram
subsets = []
for i in range(n_subsets):
    idx = slice(i * subset_size, (i + 1) * subset_size)
    sub_vec = vectors[idx].copy()
    pg_sub = astra.create_proj_geom('cone_vec', n_det_row, n_det_col, sub_vec)
    sino_sub = np.ascontiguousarray(sino[:, idx, :])
    sid_sub = astra.data3d.create('-sino', pg_sub, sino_sub)
    subsets.append((pg_sub, sid_sub))

fbo_hist = []
best_fo = {'rmse': 1e9, 'ssim': -1, 'rec': None, 't': 0, 'n': 0}
vol_os = fdk_raw.copy()
os_iters = [1, 2, 5, 10]
prev_n = 0

for n_iter in os_iters:
    t0 = time()
    for _ in range(n_iter - prev_n):
        for pg_sub, sid_sub in subsets:
            rid_sub = astra.data3d.create('-vol', vol_geom, data=vol_os.astype(np.float32))
            cfg = astra.astra_dict('SIRT3D_CUDA')
            cfg['ProjectionDataId'] = sid_sub
            cfg['ReconstructionDataId'] = rid_sub
            cfg['option'] = {'GPUindex': 0}
            aid = astra.algorithm.create(cfg)
            astra.algorithm.run(aid, 1)
            vol_os = astra.data3d.get(rid_sub).copy()
            astra.algorithm.delete(aid)
            astra.data3d.delete(rid_sub)
    t = time() - t0
    rec = linear_scale(vol_os)
    r, s = calc_rmse(rec), calc_ssim(rec)
    fbo_hist.append((n_iter, t, r, s))
    if r < best_fo['rmse']:
        best_fo = {'rmse': r, 'ssim': s, 'rec': rec, 't': t, 'n': n_iter}
    print(f"   x{n_iter:3d}轮: RMSE={r:.5f}, SSIM={s:.4f}, {t*1000:.0f}ms")
    prev_n = n_iter

for _, sid in subsets:
    astra.data3d.delete(sid)
print(f"   >> 最优: OS-SART x{best_fo['n']}: RMSE={best_fo['rmse']:.5f}, SSIM={best_fo['ssim']:.4f}")

# ============================================================
# 汇总对比
# ============================================================
print("\n" + "=" * 60)
print("汇总对比")
print("=" * 60)
print(f"{'算法':35s} {'耗时(ms)':>10s} {'RMSE':>10s} {'SSIM':>8s}")
print("-" * 65)
print(f"{'Pure FDK':35s} {fdk_t*1000:>8.0f} ms {fdk_rmse:>10.5f} {fdk_ssim:>8.4f}")
print(f"{'FBP+SIRT3D x'+str(best_sirt['n']):35s} {best_sirt['t']*1000:>8.0f} ms {best_sirt['rmse']:>10.5f} {best_sirt['ssim']:>8.4f}")
print(f"{'FBP+OS-SART x'+str(best_fo['n']):35s} {best_fo['t']*1000:>8.0f} ms {best_fo['rmse']:>10.5f} {best_fo['ssim']:>8.4f}")

# 等迭代对比
print("\n" + "=" * 60)
print("等迭代对比 (SIRT vs OS-SART)")
print("=" * 60)
print(f"{'轮次':>6s} {'SIRT3D':>22s} {'OS-SART(手动)':>22s}")
print(f"{'':>6s} {'RMSE/耗时':>22s} {'RMSE/耗时':>22s}")
print("-" * 52)
for n in [5, 10]:
    sr = next((h for h in sirt_hist if h[0] == n), None)
    fo = next((h for h in fbo_hist if h[0] == n), None)
    r1 = f"{sr[2]:.5f}/{sr[1]*1000:.0f}ms" if sr else "-"
    r2 = f"{fo[2]:.5f}/{fo[1]*1000:.0f}ms" if fo else "-"
    print(f"  x{n:3d}    {r1:>22s}  {r2:>22s}")

# ============================================================
# 与 TIGRE 锥束结果对比
# ============================================================
print("\n" + "=" * 60)
print("ASTRA vs TIGRE 锥束对比 (同体模, 同参数)")
print("=" * 60)
print(f"{'指标':25s} {'ASTRA':>18s} {'TIGRE':>18s}")
print("-" * 61)
print(f"{'FDK 耗时':25s} {fdk_t*1000:>8.0f} ms {'62ms':>18s}")
print(f"{'FDK RMSE':25s} {fdk_rmse:>18.5f} {'0.00521':>18s}")
print(f"{'SIRT x50 耗时':25s} {best_sirt['t']*1000:>8.0f} ms {'2287ms':>18s}")
print(f"{'FBP+SIRT x50 RMSE':25s} {best_sirt['rmse']:>18.5f} {'0.00331':>18s}")
print(f"{'FBP+OS-SART x5 RMSE':25s} {best_fo['rmse']:>18.5f} {'0.00324':>18s}")

# ============================================================
# 结果列表
# ============================================================
results = [
    ('Pure FDK', fdk_t, fdk_rmse, fdk_ssim),
    ('FBP+SIRT3D x'+str(best_sirt['n']), best_sirt['t'], best_sirt['rmse'], best_sirt['ssim']),
    ('FBP+OS-SART x'+str(best_fo['n']), best_fo['t'], best_fo['rmse'], best_fo['ssim']),
]

# ============================================================
# 可视化
# ============================================================
print("\n生成可视化...")
os.makedirs("img_out", exist_ok=True)
mid = nz // 2

fig = plt.figure(figsize=(16, 8))
gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

titles = ['Ground Truth', 'Pure FDK', 'FBP+SIRT3D', 'FBP+OS-SART']
imgs = [vol_gt[mid], fdk_rec[mid], best_sirt['rec'][mid], best_fo['rec'][mid]]
for i, (t, im) in enumerate(zip(titles, imgs)):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(im, cmap='gray', vmin=0, vmax=0.05)
    ax.set_title(t, fontsize=9)
    ax.axis('off')

errs = [np.zeros_like(vol_gt[mid]), fdk_rec[mid] - vol_gt[mid],
        best_sirt['rec'][mid] - vol_gt[mid], best_fo['rec'][mid] - vol_gt[mid]]
err_titles = ['Error (GT)', 'FDK Error', 'SIRT3D Error', 'OS-SART Error']
for i, (t, e) in enumerate(zip(err_titles, errs)):
    ax = fig.add_subplot(gs[1, i])
    vmax = max(0.005, np.percentile(np.abs(e), 95) * 1.2)
    im = ax.imshow(e, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(t, fontsize=9)
    ax.axis('off')

plt.suptitle('Cone-beam CBCT: ASTRA CUDA (512x12x32, 360 angles)', fontsize=13, fontweight='bold', y=0.98)
plt.savefig("img_out/astra_cone_hybrid.png", dpi=150, bbox_inches='tight')
plt.close()
print("   => img_out/astra_cone_hybrid.png")

summary = {
    'backend': 'GPU (ASTRA CUDA cone-beam)',
    'config': {'N': N, 'nz': nz, 'n_angles': n_angles},
    'results': {name: {'rmse': round(r, 5), 'ssim': round(s, 4), 'time_ms': round(t*1000, 1)}
                for name, t, r, s in results},
}
with open("img_out/astra_cone_hybrid_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("   => img_out/astra_cone_hybrid_summary.json")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
