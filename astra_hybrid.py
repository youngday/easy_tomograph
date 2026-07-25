"""
FBP + IR 混合重建
=================
核心思想: 用 FBP 的快速重建结果作为迭代法 (SIRT/SART) 的初始值,
          让 IR 从更好的起点开始迭代 → 更快收敛 + 更高质量

对比组:
  - Pure FBP (基线)
  - FBP + SIRT (混合)
  - FBP + SART (混合)

模式:
  - GPU 模式: 使用 ASTRA toolbox (CUDA 加速), 需安装 astra-toolbox + CUDA

"""

import numpy as np
from time import time
import tomophantom
from tomophantom import TomoP2D
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
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
# 1. 生成体模 (TomoPhantom Model 4 - QRM 多椭圆体模)
# ============================================================
tp_lib = os.path.join(os.path.dirname(tomophantom.__file__),
                       'phantomlib', 'Phantom2DLibrary.dat')
ph = TomoP2D.Model(4, N, tp_lib)
ct = (ph - 0.65) * 2000 / 0.65
ct = ct.astype(np.float32)
Y, X = np.ogrid[:N, :N]
head_r = 235
circ_mask = (X - N / 2) ** 2 + (Y - N / 2) ** 2 <= head_r ** 2
ct[~circ_mask] = -1000

# ============================================================
# 2. 正向投影 (ASTRA GPU FP 算法, 与重建算子完全匹配)
# ============================================================
# 之前: skimage.transform.radon → ASTRA 重建, 算子不匹配导致精度天花板
# 现在: ASTRA FP algorithm → ASTRA 重建, 算子完全匹配
theta_deg = np.linspace(0, 180, n_angles, endpoint=False)
theta_rad = np.deg2rad(theta_deg).astype(np.float32)
D = int(np.ceil(N * np.sqrt(2)))
proj_geom = astra.create_proj_geom('parallel', 1.0, D, theta_rad)
vol_geom = astra.create_vol_geom(N, N)

# GPU 前向投影 (使用 ASTRA FP 算法, 与重建使用相同的几何)
ct_32f = np.ascontiguousarray(ct.astype(np.float32))
vol_id = astra.data2d.create('-vol', vol_geom, ct_32f)
sino_id = astra.data2d.create('-sino', proj_geom, 0.0)
cfg = astra.astra_dict('FP_CUDA')
cfg['ProjectionDataId'] = sino_id
cfg['VolumeDataId'] = vol_id
cfg['option'] = {'GPUindex': 0}
alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
sino = astra.data2d.get(sino_id)  # (n_angles, D)
print(f"   前向投影: 形状 {sino.shape}, D={D}")
# 清理
astra.algorithm.delete(alg_id)
astra.data2d.delete(sino_id)
astra.data2d.delete(vol_id)

# ============================================================
# 辅助函数
# ============================================================
def linear_scale(rec):
    # 先裁剪极端值, 再用全圆形遮罩做线性拟合
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
    return (2 * mu_x * mu_y + c1) * (2 * sig_xy + c2) / \
           ((mu_x ** 2 + mu_y ** 2 + c1) * (sig_x + sig_y + c2))

# ============================================================
# GPU 预热
# ============================================================
print("GPU 预热...")
sid_warm = astra.data2d.create('-sino', proj_geom, sino)
rid_warm = astra.data2d.create('-vol', vol_geom)
for algo in ['FBP_CUDA']:
    cfg = astra.astra_dict(algo)
    cfg['ProjectionDataId'] = sid_warm
    cfg['ReconstructionDataId'] = rid_warm
    if algo == 'FBP_CUDA':
        cfg['option'] = {'FilterType': 'shepp-logan'}
    aid = astra.algorithm.create(cfg)
    astra.algorithm.run(aid, 1)
    astra.algorithm.delete(aid)
astra.data2d.delete(rid_warm)
astra.data2d.delete(sid_warm)
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
print("   注: ASTRA FP 前向投影, 算子完全匹配")

# ---- B. FBP + SIRT (混合) ----
print("-" * 55)
print("B. FBP + SIRT (FBP 初始値)")
print("-" * 55)
sid = astra.data2d.create('-sino', proj_geom, sino)
fbs_hist = []
best_fs = {'rmse': 1e9, 'ssim': -1, 'rec': None, 't': 0, 'n': 0}
sirt_iters = [10, 20, 50, 100, 200]
for n_iter in sirt_iters:
    rid = astra.data2d.create('-vol', vol_geom, data=rec_fbp.astype(np.float32))
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

# ---- C. FBP + SART (混合) ----
print("-" * 55)
print("C. FBP + SART (FBP 初始値)")
print("-" * 55)
sid = astra.data2d.create('-sino', proj_geom, sino)
fbsa_hist = []
best_fsa = {'rmse': 1e9, 'ssim': -1, 'rec': None, 't': 0, 'n': 0}
sart_iters = [2, 5, 10, 20]
for n_iter in sart_iters:
    rid = astra.data2d.create('-vol', vol_geom, data=rec_fbp.astype(np.float32))
    cfg = astra.astra_dict('SART_CUDA')
    cfg['ProjectionDataId'] = sid
    cfg['ReconstructionDataId'] = rid
    cfg['option'] = {'GPUindex': 0}
    aid = astra.algorithm.create(cfg)
    t0 = time()
    astra.algorithm.run(aid, n_iter)
    rec = linear_scale(astra.data2d.get(rid))
    t = time() - t0
    r, s = calc_rmse(rec), calc_ssim(rec)
    fbsa_hist.append((n_iter, t, r, s))
    if r < best_fsa['rmse']:
        best_fsa = {'rmse': r, 'ssim': s, 'rec': rec, 't': t, 'n': n_iter}
    astra.algorithm.delete(aid)
    astra.data2d.delete(rid)
    print(f"   x{n_iter:3d}: RMSE={r:.2f}, SSIM={s:.4f}, {t*1000:.0f}ms")
astra.data2d.delete(sid)
print(f"   >> 最优: FBP+SART x{best_fsa['n']}: RMSE={best_fsa['rmse']:.2f}, SSIM={best_fsa['ssim']:.4f}, {best_fsa['t']*1000:.0f}ms")

# ============================================================
# D. FBP+SIRT vs FBP+SART 对比 (等迭代次数)
# ============================================================
print("\n" + "=" * 60)
print("D. FBP+SIRT vs FBP+SART 对比")
print("=" * 60)
print(f"{'迭代数':>8s} {'FBP+SIRT':>12s} {'耗时':>10s} {'FBP+SART':>12s} {'耗时':>10s} {'速度比'}")
print("-" * 60)
for n in sorted(set([10, 20])):
    fsr = next((h for h in fbs_hist if h[0] == n), None)
    fsa = next((h for h in fbsa_hist if h[0] == n), None)
    fsr_r = f"{fsr[2]:.1f}" if fsr else "-"
    fsr_t = f"{fsr[1]*1000:.0f}ms" if fsr else "-"
    fsa_r = f"{fsa[2]:.1f}" if fsa else "-"
    fsa_t = f"{fsa[1]*1000:.0f}ms" if fsa else "-"
    ratio = f"{(fsr[1]/fsa[1]):.1f}x" if fsr and fsa else "-"
    print(f"  x{n:4d}      {fsr_r:>8s}    {fsr_t:>8s}    {fsa_r:>8s}    {fsa_t:>8s}    {ratio}")

# ============================================================
# 结果列表
# ============================================================
results = [
    ('Pure FBP (shepp-logan)', fbp_t, fbp_rmse, fbp_ssim, 0),
    ('FBP + SIRT (hybrid)', best_fs['t'], best_fs['rmse'], best_fs['ssim'],
     (fbp_rmse - best_fs['rmse']) / fbp_rmse * 100),
    ('FBP + SART (hybrid)', best_fsa['t'], best_fsa['rmse'], best_fsa['ssim'],
     (fbp_rmse - best_fsa['rmse']) / fbp_rmse * 100),
]

# ============================================================
# 汇总对比
# ============================================================
print("\n" + "=" * 60)
print("汇总对比")
print("=" * 60)
print(f"{'算法':35s} {'耗时(ms)':>10s} {'RMSE':>8s} {'SSIM':>8s} {'提升':>10s}")
print("-" * 71)
for name, t, r, s, imp in results:
    imp_str = f"{imp:+.1f}%" if imp != 0 else "  基线"
    print(f"{name:35s} {t*1000:>8.0f} ms {r:>8.2f} {s:>8.4f} {imp_str:>10s}")

print("\n推荐配置 (性价比):")
print(f"   速度优先:    FBP (FDK)       → {fbp_t*1000:.0f}ms, RMSE={fbp_rmse:.1f}")
if any(h[0] == 5 for h in fbsa_hist):
    fsa5 = next(h for h in fbsa_hist if h[0] == 5)
    print(f"   产品级首选:  FBP+SART x5    → {fsa5[1]*1000:.0f}ms, RMSE={fsa5[2]:.1f}")
if any(h[0] == 10 for h in fbs_hist):
    f10_s = next(h for h in fbs_hist if h[0] == 10)
    print(f"   高质量:      FBP+SIRT x10   → {f10_s[1]*1000:.0f}ms, RMSE={f10_s[2]:.1f}")

print("\n生成可视化...")
os.makedirs("img_out", exist_ok=True)

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3,
              width_ratios=[1, 1, 1, 1])

gray_cmap = 'gray'
err_cmap = 'RdBu_r'

# 第1行: 重建对比
plot_items = [('Ground Truth', ct, None, None, None),
              ('Pure FBP', fbp_rec, fbp_rmse, fbp_ssim, fbp_t),
              ('FBP+SIRT (best)', best_fs['rec'], best_fs['rmse'], best_fs['ssim'], best_fs['t']),
              ('FBP+SART (best)', best_fsa['rec'], best_fsa['rmse'], best_fsa['ssim'], best_fsa['t'])]

for i, (title, img, rmse, ssim, t) in enumerate(plot_items):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img, cmap=gray_cmap, vmin=-200, vmax=600)
    tstr = title
    if rmse is not None:
        tstr += f"\nRMSE={rmse:.1f} SSIM={ssim:.4f}\n{t*1000:.0f}ms"
    ax.set_title(tstr, fontsize=9)
    ax.axis('off')

# 第2行: 误差图
err_items = [('', ct - ct),
             ('FBP Error', fbp_rec - ct),
             ('FBP+SIRT Error', best_fs['rec'] - ct),
             ('FBP+SART Error', best_fsa['rec'] - ct)]

cax = None
for i, (title, err_img) in enumerate(err_items):
    ax = fig.add_subplot(gs[1, i])
    err_img_masked = err_img.copy()
    err_img_masked[~circ_mask] = 0
    vmax = max(30, np.percentile(np.abs(err_img_masked[circ_mask]), 95) * 1.2)
    print(f"       {title}: vmax={vmax:.1f}")
    im = ax.imshow(err_img_masked, cmap=err_cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    cax = im

fig.colorbar(im, ax=fig.get_axes(), shrink=0.6, pad=0.02)
plt.suptitle('FBP + IR Hybrid Reconstruction (GPU: ASTRA CUDA)', fontsize=15, fontweight='bold', y=0.98)
plt.savefig("img_out/astra_hybrid.png", dpi=150, bbox_inches='tight')
plt.close()
print("   => img_out/astra_hybrid.png")

# ============================================================
# 保存总结
# ============================================================
summary = {
    'backend': 'GPU (ASTRA CUDA)',
    'config': {'N': N, 'n_angles': n_angles},
    'results': {name: {'rmse': round(r, 2), 'ssim': round(s, 4), 'time_ms': round(t*1000, 1)}
                for name, t, r, s, _ in results},
}
with open("img_out/astra_hybrid_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("   => img_out/astra_hybrid_summary.json")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
