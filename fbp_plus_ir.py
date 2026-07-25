"""
FBP + IR 混合重建
=================
核心思想: 用 FBP 的快速重建结果作为迭代法 (CGLS/SIRT) 的初始值,
          让 IR 从更好的起点开始迭代 → 更快收敛 + 更高质量

对比组:
  - Pure FBP (基线 — 最快, 质量较差)
  - Pure CGLS from zero (基线 — 慢, 质量好)
  - FBP + CGLS (混合)
  - Pure SIRT from zero (基线)
  - FBP + SIRT (混合)

模式:
  - GPU 模式: 使用 ASTRA toolbox (CUDA 加速), 需安装 astra-toolbox + CUDA

与 fbp_vs_ir.py 的区别:
  - gpu_fbp_vs_ir.py: 纯方法对比 (FBP vs CGLS vs SIRT)
  - fbp_plus_ir.py:   混合方法 (FBP初始化+IR迭代)
"""

import numpy as np
from time import time
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os, json

# ============================================================
# 后端检测: GPU (ASTRA) vs CPU (numpy)
# ============================================================
try:
    import astra
except ImportError:
    print("=" * 60)
    print("错误: 需要 ASTRA Toolbox (GPU 版本)")
    print("安装: uv pip install astra-toolbox")
    print("=" * 60)
    exit(1)

print("=" * 60)
print("FBP + IR 混合重建对比  [后端: GPU (ASTRA CUDA)]")
print("=" * 60)

# ============================================================
# 参数设置
# ============================================================
N = 512
n_angles = 360
print(f"体模: {N}x{N}, 角度: {n_angles}")

# ============================================================
# 1. 生成体模
# ============================================================
np.random.seed(42)
p = resize(shepp_logan_phantom(), (N, N), anti_aliasing=True)
ct = p * 2000 - 1000  # 缩放到近似 HU 值
Y, X = np.ogrid[:N, :N]
circ_mask = (X - N / 2) ** 2 + (Y - N / 2) ** 2 <= (N / 2 * 0.95) ** 2
ct[~circ_mask] = -1000

# ============================================================
# 2. 正向投影
# ============================================================
theta_deg = np.linspace(0, 180, n_angles, endpoint=False)
sino = radon(ct, theta=theta_deg, circle=False)
D = sino.shape[0]

# ============================================================
# 辅助函数
# ============================================================
def linear_scale(rec):
    mask = circ_mask & (np.abs(rec) < 2000)
    A = np.column_stack([rec.ravel()[mask.ravel()], np.ones(mask.sum())])
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
    return (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (sig_x + sig_y + c2))

# ============================================================
# GPU 路径 (ASTRA)
# ============================================================
# 同一几何坐标系
theta_rad = np.deg2rad(theta_deg).astype(np.float32)
proj_geom = astra.create_proj_geom('parallel', 1.0, D, theta_rad)
vol_geom = astra.create_vol_geom(N, N)

# GPU 预热
print("GPU 预热...")
sino = np.ascontiguousarray(sino.T)
sid = astra.data2d.create('-sino', proj_geom, sino)
rid = astra.data2d.create('-vol', vol_geom)
for algo in ['FBP_CUDA', 'CGLS_CUDA']:
    cfg = astra.astra_dict(algo)
    cfg['ProjectionDataId'] = sid
    cfg['ReconstructionDataId'] = rid
    if algo == 'CGLS_CUDA':
        cfg['option'] = {'GPUindex': 0}
    aid = astra.algorithm.create(cfg)
    astra.algorithm.run(aid, 1)
    astra.algorithm.delete(aid)
astra.data2d.delete(rid)
astra.data2d.delete(sid)
print("   预热完成\n")

# ---- A. Pure FBP (基线) ----
print("-" * 55)
print("A. Pure FBP (ASTRA FBP_CUDA shepp-logan)")
print("-" * 55)
t0 = time()
sid = astra.data2d.create('-sino', proj_geom, sino)
rid = astra.data2d.create('-vol', vol_geom)
cfg = astra.astra_dict('FBP_CUDA')
cfg['ProjectionDataId'] = sid
cfg['ReconstructionDataId'] = rid
cfg['option'] = {'FilterType': 'shepp-logan'}
aid = astra.algorithm.create(cfg)
astra.algorithm.run(aid)
t_fbp = time() - t0
rec_fbp = astra.data2d.get(rid).copy()
fbp_rec = linear_scale(astra.data2d.get(rid))
fbp_t = time() - t0
fbp_rmse = calc_rmse(fbp_rec)
fbp_ssim = calc_ssim(fbp_rec)
astra.algorithm.delete(aid)
astra.data2d.delete(rid)
astra.data2d.delete(sid)
print(f"   RMSE={fbp_rmse:.2f}, SSIM={fbp_ssim:.4f}, {fbp_t*1000:.0f}ms")
print("   注: 与 gpu_fbp_vs_ir.py 的 FBP_CUDA shepp-logan 结果可比")

# ---- B. Pure CGLS (从零开始) ----
print("-" * 55)
print("B. Pure CGLS (从零开始)")
print("-" * 55)
sid = astra.data2d.create('-sino', proj_geom, sino)
cgls_hist = []
best_cgls = {'rmse': 1e9, 'ssim': -1, 'rec': None, 't': 0, 'n': 0}
cgls_iters = [5, 10, 20, 30, 50, 80, 100]
for n_iter in cgls_iters:
    rid = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('CGLS_CUDA')
    cfg['ProjectionDataId'] = sid
    cfg['ReconstructionDataId'] = rid
    cfg['option'] = {'GPUindex': 0}
    aid = astra.algorithm.create(cfg)
    t0 = time()
    astra.algorithm.run(aid, n_iter)
    rec = linear_scale(astra.data2d.get(rid))
    t = time() - t0
    r, s = calc_rmse(rec), calc_ssim(rec)
    cgls_hist.append((n_iter, t, r, s))
    if r < best_cgls['rmse']:
        best_cgls = {'rmse': r, 'ssim': s, 'rec': rec, 't': t, 'n': n_iter}
    astra.algorithm.delete(aid)
    astra.data2d.delete(rid)
    print(f"   x{n_iter:3d}: RMSE={r:.2f}, SSIM={s:.4f}, {t*1000:.0f}ms")
astra.data2d.delete(sid)
print(f"   >> 最优: CGLS x{best_cgls['n']}: RMSE={best_cgls['rmse']:.2f}, SSIM={best_cgls['ssim']:.4f}, {best_cgls['t']*1000:.0f}ms")

# ---- C. FBP + CGLS (混合) ----
print("-" * 55)
print("C. FBP + CGLS (FBP 初始值)")
print("-" * 55)
sid = astra.data2d.create('-sino', proj_geom, sino)
fbc_hist = []
best_fc = {'rmse': 1e9, 'ssim': -1, 'rec': None, 't': 0, 'n': 0}
for n_iter in cgls_iters:
    rid = astra.data2d.create('-vol', vol_geom, data=fbp_rec.astype(np.float32))
    cfg = astra.astra_dict('CGLS_CUDA')
    cfg['ProjectionDataId'] = sid
    cfg['ReconstructionDataId'] = rid
    cfg['option'] = {'GPUindex': 0}
    aid = astra.algorithm.create(cfg)
    t0 = time()
    astra.algorithm.run(aid, n_iter)
    rec = linear_scale(astra.data2d.get(rid))
    t = time() - t0
    r, s = calc_rmse(rec), calc_ssim(rec)
    fbc_hist.append((n_iter, t, r, s))
    if r < best_fc['rmse']:
        best_fc = {'rmse': r, 'ssim': s, 'rec': rec, 't': t, 'n': n_iter}
    astra.algorithm.delete(aid)
    astra.data2d.delete(rid)
    print(f"   x{n_iter:3d}: RMSE={r:.2f}, SSIM={s:.4f}, {t*1000:.0f}ms")
astra.data2d.delete(sid)
print(f"   >> 最优: FBP+CGLS x{best_fc['n']}: RMSE={best_fc['rmse']:.2f}, SSIM={best_fc['ssim']:.4f}, {best_fc['t']*1000:.0f}ms")

# ---- D. Pure SIRT (从零开始) ----
print("-" * 55)
print("D. Pure SIRT (从零开始)")
print("-" * 55)
sid = astra.data2d.create('-sino', proj_geom, sino)
sirt_hist = []
best_sirt = {'rmse': 1e9, 'ssim': -1, 'rec': None, 't': 0, 'n': 0}
sirt_iters = [10, 20, 50, 100, 200, 500]
for n_iter in sirt_iters:
    rid = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('SIRT_CUDA')
    cfg['ProjectionDataId'] = sid
    cfg['ReconstructionDataId'] = rid
    cfg['option'] = {'GPUindex': 0}
    aid = astra.algorithm.create(cfg)
    t0 = time()
    astra.algorithm.run(aid, n_iter)
    rec = linear_scale(astra.data2d.get(rid))
    t = time() - t0
    r, s = calc_rmse(rec), calc_ssim(rec)
    sirt_hist.append((n_iter, t, r, s))
    if r < best_sirt['rmse']:
        best_sirt = {'rmse': r, 'ssim': s, 'rec': rec, 't': t, 'n': n_iter}
    astra.algorithm.delete(aid)
    astra.data2d.delete(rid)
    print(f"   x{n_iter:4d}: RMSE={r:.2f}, SSIM={s:.4f}, {t*1000:.0f}ms")
astra.data2d.delete(sid)
print(f"   >> 最优: SIRT x{best_sirt['n']}: RMSE={best_sirt['rmse']:.2f}, SSIM={best_sirt['ssim']:.4f}, {best_sirt['t']*1000:.0f}ms")

# ---- E. FBP + SIRT (混合) ----
print("-" * 55)
print("E. FBP + SIRT (FBP 初始值)")
print("-" * 55)
sid = astra.data2d.create('-sino', proj_geom, sino)
fbs_hist = []
best_fs = {'rmse': 1e9, 'ssim': -1, 'rec': None, 't': 0, 'n': 0}
for n_iter in sirt_iters:
    rid = astra.data2d.create('-vol', vol_geom, data=fbp_rec.astype(np.float32))
    cfg = astra.astra_dict('SIRT_CUDA')
    cfg['ProjectionDataId'] = sid
    cfg['ReconstructionDataId'] = rid
    cfg['option'] = {'GPUindex': 0}
    aid = astra.algorithm.create(cfg)
    t0 = time()
    astra.algorithm.run(aid, n_iter)
    rec = linear_scale(astra.data2d.get(rid))
    t = time() - t0
    r, s = calc_rmse(rec), calc_ssim(rec)
    fbs_hist.append((n_iter, t, r, s))
    if r < best_fs['rmse']:
        best_fs = {'rmse': r, 'ssim': s, 'rec': rec, 't': t, 'n': n_iter}
    astra.algorithm.delete(aid)
    astra.data2d.delete(rid)
    print(f"   x{n_iter:4d}: RMSE={r:.2f}, SSIM={s:.4f}, {t*1000:.0f}ms")
astra.data2d.delete(sid)
print(f"   >> 最优: FBP+SIRT x{best_fs['n']}: RMSE={best_fs['rmse']:.2f}, SSIM={best_fs['ssim']:.4f}, {best_fs['t']*1000:.0f}ms")

# 结果列表
results = [
    ('Pure FBP (shepp-logan)', fbp_t, fbp_rmse, fbp_ssim, 0),
    ('Pure CGLS (from zero)', best_cgls['t'], best_cgls['rmse'], best_cgls['ssim'],
     (fbp_rmse - best_cgls['rmse']) / fbp_rmse * 100),
    ('FBP + CGLS (hybrid)', best_fc['t'], best_fc['rmse'], best_fc['ssim'],
     (fbp_rmse - best_fc['rmse']) / fbp_rmse * 100),
    ('Pure SIRT (from zero)', best_sirt['t'], best_sirt['rmse'], best_sirt['ssim'],
     (fbp_rmse - best_sirt['rmse']) / fbp_rmse * 100),
    ('FBP + SIRT (hybrid)', best_fs['t'], best_fs['rmse'], best_fs['ssim'],
     (fbp_rmse - best_fs['rmse']) / fbp_rmse * 100),
]

# ============================================================
# 汇总结果
# ============================================================
print("\n" + "=" * 60)
print("汇总对比")
print("=" * 60)
print(f"{'算法':35s} {'耗时(ms)':>10s} {'RMSE':>8s} {'SSIM':>8s} {'提升':>10s}")
print("-" * 71)
for name, t, r, s, imp in results:
    imp_str = f"{imp:+.1f}%" if imp != 0 else "  基线"
    print(f"{name:35s} {t*1000:>8.0f} ms {r:>8.2f} {s:>8.4f} {imp_str:>10s}")

fc_imp = (best_cgls['rmse'] - best_fc['rmse']) / max(best_cgls['rmse'], 1e-10) * 100
fs_imp = (best_sirt['rmse'] - best_fs['rmse']) / max(best_sirt['rmse'], 1e-10) * 100
print("\n混合 vs 纯 IR 对比:")
print(f"   FBP+CGLS vs Pure CGLS: RMSE {'降低' if fc_imp>=0 else '升高'} {abs(fc_imp):+.1f}%")
print(f"   FBP+SIRT vs Pure SIRT: RMSE {'降低' if fs_imp>=0 else '升高'} {abs(fs_imp):+.1f}%")

# ============================================================
# 可视化
# ============================================================
print("\n生成可视化...")
os.makedirs("img_out", exist_ok=True)

fig = plt.figure(figsize=(18, 14))
gs = GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.35,
              width_ratios=[1, 1, 1, 0.05, 1, 0.05])

gray_cmap = 'gray'
err_cmap = 'RdBu_r'

# 第1行: 重建对比
plot_items = [('Ground Truth', ct, None, None, None),
              ('Pure FBP', fbp_rec, fbp_rmse, fbp_ssim, fbp_t),
              ('CGLS (best)', best_cgls['rec'], best_cgls['rmse'], best_cgls['ssim'], best_cgls['t']),
              ('FBP+CGLS (best)', best_fc['rec'], best_fc['rmse'], best_fc['ssim'], best_fc['t'])]

for i, (title, img, rmse, ssim, t) in enumerate(plot_items):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img, cmap=gray_cmap, vmin=-200, vmax=600)
    tstr = title
    if rmse is not None:
        tstr += f"\nRMSE={rmse:.1f} SSIM={ssim:.4f}\n{t*1000:.0f}ms"
    ax.set_title(tstr, fontsize=9)
    ax.axis('off')

# 第2行: 误差图
err_items = [('FBP Error', fbp_rec - ct),
             ('CGLS Error', best_cgls['rec'] - ct),
             ('FBP+CGLS Error', best_fc['rec'] - ct),
             ('FBP+SIRT Error', best_fs['rec'] - ct)]

for i, (title, err_img) in enumerate(err_items):
    ax = fig.add_subplot(gs[1, i])
    err_img_masked = err_img.copy()
    err_img_masked[~circ_mask] = 0
    vmax = max(30, np.percentile(np.abs(err_img_masked[circ_mask]), 95) * 1.2)
    print(f"       {title}: vmax={vmax:.1f}")
    im = ax.imshow(err_img_masked, cmap=err_cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis('off')

cax = fig.add_subplot(gs[1, 3])
plt.colorbar(im, cax=cax)
cax.set_ylabel('HU Error', fontsize=8)
plt.suptitle('FBP + IR Hybrid Reconstruction (GPU: ASTRA CUDA)', fontsize=15, fontweight='bold', y=0.98)
plt.savefig("img_out/fbp_plus_ir.png", dpi=150, bbox_inches='tight')
plt.close()
print("   => img_out/fbp_plus_ir.png")

# ============================================================
# 保存总结
# ============================================================
summary = {
    'backend': 'GPU (ASTRA CUDA)',
    'config': {'N': N, 'n_angles': n_angles},
    'results': {name: {'rmse': round(r, 2), 'ssim': round(s, 4), 'time_ms': round(t*1000, 1)}
                for name, t, r, s, _ in results},
}
with open("img_out/fbp_plus_ir_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("   => img_out/fbp_plus_ir_summary.json")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
