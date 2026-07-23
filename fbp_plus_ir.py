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
  - GPU 模式: 使用 ASTRA toolbox (CUDA 加速)
  - CPU 模式: 使用 numpy 实现 (无 GPU 依赖)
  自动检测, 无需手动切换

与 gpu_fbp_vs_ir.py 的区别:
  - gpu_fbp_vs_ir.py: 纯方法对比 (FBP vs CGLS vs SIRT)
  - fbp_plus_ir.py:   混合方法 (FBP初始化+IR迭代)
"""

import numpy as np
from time import time
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os, json

# ============================================================
# 后端检测: GPU (ASTRA) vs CPU (numpy)
# ============================================================
HAVE_ASTRA = False
try:
    import astra
    HAVE_ASTRA = True
except ImportError:
    pass

BACKEND = 'GPU (ASTRA CUDA)' if HAVE_ASTRA else 'CPU (numpy)'
print("=" * 60)
print(f"FBP + IR 混合重建对比  [后端: {BACKEND}]")
print("=" * 60)

# ============================================================
# 参数设置
# ============================================================
N = 512 if HAVE_ASTRA else 256
n_angles = 360 if HAVE_ASTRA else 180
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
if HAVE_ASTRA:
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
    print(f"   注: 与 gpu_fbp_vs_ir.py 的 FBP_CUDA shepp-logan 结果可比")

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
# CPU 路径 (skimage + numpy)
# ============================================================
else:
    print("-" * 55)
    print("A. Pure FBP (skimage iradon)")
    print("-" * 55)
    t0 = time()
    fbp_rec = iradon(sino, theta=theta_deg, filter_name='shepp-logan', circle=False)
    fbp_rec = linear_scale(fbp_rec)
    fbp_t = time() - t0
    fbp_rmse = calc_rmse(fbp_rec)
    fbp_ssim = calc_ssim(fbp_rec)
    print(f"   RMSE={fbp_rmse:.2f}, SSIM={fbp_ssim:.4f}, {fbp_t*1000:.0f}ms")

    def sirt_numpy(sino, theta, n_iter, init=None):
        N = int(np.sqrt(2) * sino.shape[0])
        N = N if N % 2 == 0 else N + 1
        rec = np.zeros((N, N), dtype=np.float64) if init is None else init.copy()
        for it in range(n_iter):
            update = np.zeros_like(rec)
            weight = np.zeros_like(rec) + 1e-10
            for ang, proj in zip(theta, sino.T):
                sino_est = radon(rec, theta=[ang], circle=False)[:, 0]
                diff = proj - sino_est
                bp = iradon(diff.reshape(-1, 1), theta=[ang],
                            filter_name=None, circle=False).squeeze()
                bp = resize(bp, rec.shape, anti_aliasing=False)
                update += bp
                weight += np.ones_like(rec)
            rec += update / weight
            if it % 5 == 0 or it == n_iter - 1:
                r = calc_rmse(linear_scale(rec))
                print(f"   SIRT x{it+1:3d}: RMSE={r:.2f}")
        return rec

    print("\nB. Pure SIRT (从零开始)")
    t0 = time()
    sirt_cpu = sirt_numpy(sino, theta_deg, n_iter=30)
    sirt_cpu = linear_scale(sirt_cpu)
    sirt_cpu = resize(sirt_cpu, (N, N), anti_aliasing=False)
    sirt_t = time() - t0
    sirt_rmse = calc_rmse(sirt_cpu)
    sirt_ssim = calc_ssim(sirt_cpu)
    print(f"   >> Pure SIRT: RMSE={sirt_rmse:.2f}, SSIM={sirt_ssim:.4f}, {sirt_t*1000:.0f}ms")

    print("\nC. FBP + SIRT (FBP 初始值)")
    t0 = time()
    init_size = int(np.sqrt(2) * D)
    init_size = init_size if init_size % 2 == 0 else init_size + 1
    fbs_cpu = sirt_numpy(sino, theta_deg, n_iter=30,
                          init=resize(fbp_rec, (init_size, init_size), anti_aliasing=False))
    fbs_cpu = linear_scale(fbs_cpu)
    fbs_cpu = resize(fbs_cpu, (N, N), anti_aliasing=False)
    fbs_t = time() - t0
    fbs_rmse = calc_rmse(fbs_cpu)
    fbs_ssim = calc_ssim(fbs_cpu)
    print(f"   >> FBP+SIRT: RMSE={fbs_rmse:.2f}, SSIM={fbs_ssim:.4f}, {fbs_t*1000:.0f}ms")

    # 占位
    cgls_hist = []
    fbc_hist = []
    sirt_hist = [(30, sirt_t, sirt_rmse, sirt_ssim)]
    fbs_hist = [(30, fbs_t, fbs_rmse, fbs_ssim)]
    best_fc = best_cgls = {'rmse': fbp_rmse, 'ssim': fbp_ssim, 'rec': fbp_rec, 't': 0, 'n': 0}
    best_sirt = {'rmse': sirt_rmse, 'ssim': sirt_ssim, 'rec': sirt_cpu, 't': sirt_t, 'n': 30}
    best_fs = {'rmse': fbs_rmse, 'ssim': fbs_ssim, 'rec': fbs_cpu, 't': fbs_t, 'n': 30}

    results = [
        ('Pure FBP (skimage)', fbp_t, fbp_rmse, fbp_ssim, 0),
        ('Pure SIRT (numpy)', sirt_t, sirt_rmse, sirt_ssim,
         (fbp_rmse - sirt_rmse) / fbp_rmse * 100),
        ('FBP + SIRT (hybrid)', fbs_t, fbs_rmse, fbs_ssim,
         (fbp_rmse - fbs_rmse) / fbp_rmse * 100),
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

if HAVE_ASTRA:
    fc_imp = (best_cgls['rmse'] - best_fc['rmse']) / max(best_cgls['rmse'], 1e-10) * 100
    fs_imp = (best_sirt['rmse'] - best_fs['rmse']) / max(best_sirt['rmse'], 1e-10) * 100
    print(f"\n混合 vs 纯 IR 对比:")
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
              ('Pure FBP', fbp_rec, fbp_rmse, fbp_ssim, fbp_t)]
if HAVE_ASTRA:
    plot_items += [('CGLS (best)', best_cgls['rec'], best_cgls['rmse'], best_cgls['ssim'], best_cgls['t']),
                   ('FBP+CGLS (best)', best_fc['rec'], best_fc['rmse'], best_fc['ssim'], best_fc['t'])]
else:
    plot_items += [('SIRT (best)', best_sirt['rec'], best_sirt['rmse'], best_sirt['ssim'], best_sirt['t']),
                   ('FBP+SIRT (best)', best_fs['rec'], best_fs['rmse'], best_fs['ssim'], best_fs['t'])]

for i, (title, img, rmse, ssim, t) in enumerate(plot_items):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img, cmap=gray_cmap, vmin=-200, vmax=600)
    tstr = title
    if rmse is not None:
        tstr += f"\nRMSE={rmse:.1f} SSIM={ssim:.4f}\n{t*1000:.0f}ms"
    ax.set_title(tstr, fontsize=9)
    ax.axis('off')

# 第2行: 误差图
err_items = [('FBP Error', fbp_rec - ct)]
if HAVE_ASTRA:
    err_items += [('CGLS Error', best_cgls['rec'] - ct),
                  ('FBP+CGLS Error', best_fc['rec'] - ct),
                  ('FBP+SIRT Error', best_fs['rec'] - ct)]
else:
    err_items += [('SIRT Error', best_sirt['rec'] - ct),
                  ('FBP+SIRT Error', best_fs['rec'] - ct),
                  ('', fbp_rec - ct)]

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

# 第3行: 收敛曲线 (GPU) 或 柱状图 (CPU)
if HAVE_ASTRA:
    ax3 = fig.add_subplot(gs[2, :3])
    cg_n = [h[0] for h in cgls_hist]; cg_r = [h[2] for h in cgls_hist]
    fc_n = [h[0] for h in fbc_hist]; fc_r = [h[2] for h in fbc_hist]
    ax3.semilogy(cg_n, cg_r, 'o-', color='#1f77b4', lw=2, ms=6, label=f'Pure CGLS (best={best_cgls["rmse"]:.1f})')
    ax3.semilogy(fc_n, fc_r, 's-', color='#d62728', lw=2, ms=6, label=f'FBP+CGLS (best={best_fc["rmse"]:.1f})')
    ax3.axhline(y=fbp_rmse, color='gray', ls='--', lw=1.5, label=f'Pure FBP (baseline={fbp_rmse:.1f})')
    ax3.set_xlabel('Iterations', fontsize=11)
    ax3.set_ylabel('RMSE', fontsize=11)
    ax3.set_title('CGLS Convergence: From Zero vs FBP Initialized', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    best_fc_i = min(range(len(fc_r)), key=lambda i: fc_r[i])
    ax3.annotate(f'FBP+CGLS x{fc_n[best_fc_i]}\nRMSE={best_fc["rmse"]:.1f}',
                 xy=(fc_n[best_fc_i], fc_r[best_fc_i]),
                 xytext=(fc_n[best_fc_i] + 10, fc_r[best_fc_i] * 0.7),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 fontsize=9, color='red',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    ax4 = fig.add_subplot(gs[2, 4:])
    sn = [h[0] for h in sirt_hist]; sr = [h[2] for h in sirt_hist]
    fs_n = [h[0] for h in fbs_hist]; fs_r = [h[2] for h in fbs_hist]
    ax4.semilogy(sn, sr, 'o-', color='#2ca02c', lw=2, ms=6, label=f'Pure SIRT (best={best_sirt["rmse"]:.1f})')
    ax4.semilogy(fs_n, fs_r, 's-', color='#ff7f0e', lw=2, ms=6, label=f'FBP+SIRT (best={best_fs["rmse"]:.1f})')
    ax4.axhline(y=fbp_rmse, color='gray', ls='--', lw=1.5, label=f'Pure FBP (baseline={fbp_rmse:.1f})')
    ax4.set_xlabel('Iterations', fontsize=11)
    ax4.set_ylabel('RMSE', fontsize=11)
    ax4.set_title('SIRT Convergence: From Zero vs FBP Initialized', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    best_fs_i = min(range(len(fs_r)), key=lambda i: fs_r[i])
    ax4.annotate(f'FBP+SIRT x{fs_n[best_fs_i]}\nRMSE={best_fs["rmse"]:.1f}',
                 xy=(fs_n[best_fs_i], fs_r[best_fs_i]),
                 xytext=(fs_n[best_fs_i] + 60, fs_r[best_fs_i] * 0.7),
                 arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                 fontsize=9, color='orange',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    ax_fs = fig.add_subplot(gs[2, 3])
    ax_fs.imshow(best_fs['rec'], cmap=gray_cmap, vmin=-200, vmax=600)
    ax_fs.set_title(f'FBP+SIRT\nRMSE={best_fs["rmse"]:.1f}\n{best_fs["t"]*1000:.0f}ms', fontsize=9)
    ax_fs.axis('off')
    ax_sirt = fig.add_subplot(gs[3, 3])
    ax_sirt.imshow(best_sirt['rec'], cmap=gray_cmap, vmin=-200, vmax=600)
    ax_sirt.set_title(f'Pure SIRT\nRMSE={best_sirt["rmse"]:.1f}\n{best_sirt["t"]*1000:.0f}ms', fontsize=9)
    ax_sirt.axis('off')
else:
    ax3 = fig.add_subplot(gs[2, :3])
    methods = ['FBP', 'SIRT', 'FBP+SIRT']
    rmse_vals = [fbp_rmse, best_sirt['rmse'], best_fs['rmse']]
    cls = ['gray', '#2ca02c', '#ff7f0e']
    bars = ax3.bar(methods, rmse_vals, color=cls, width=0.5)
    for bar, val in zip(bars, rmse_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
    ax3.set_ylabel('RMSE', fontsize=11)
    ax3.set_title('CPU Mode: RMSE Comparison', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')

    ax4 = fig.add_subplot(gs[2, 4:])
    ssim_vals = [fbp_ssim, best_sirt['ssim'], best_fs['ssim']]
    bars2 = ax4.bar(methods, ssim_vals, color=cls, width=0.5)
    for bar, val in zip(bars2, ssim_vals):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', fontsize=9)
    ax4.set_ylabel('SSIM', fontsize=11)
    ax4.set_title('SSIM Comparison', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    ax_fs = fig.add_subplot(gs[2, 3])
    ax_fs.imshow(best_fs['rec'], cmap=gray_cmap, vmin=-200, vmax=600)
    ax_fs.set_title(f'FBP+SIRT\nRMSE={best_fs["rmse"]:.1f}', fontsize=9)
    ax_fs.axis('off')
    ax_sirt = fig.add_subplot(gs[3, 3])
    ax_sirt.text(0.5, 0.5, 'CPU mode\n(no CGLS)', ha='center', va='center', fontsize=12)
    ax_sirt.axis('off')

# 第4行: 剖面线
ax_profile = fig.add_subplot(gs[3, :3])
center = N // 2
ax_profile.plot(ct[center, :], 'k-', label='GT', lw=2, alpha=0.8)
ax_profile.plot(fbp_rec[center, :], '--', color='gray', lw=1.5, label=f'FBP ({fbp_rmse:.1f})')
if HAVE_ASTRA:
    ax_profile.plot(best_cgls['rec'][center, :], '--', color='#1f77b4', lw=1.5,
                    label=f'CGLS ({best_cgls["rmse"]:.1f})')
    ax_profile.plot(best_fc['rec'][center, :], ':', color='#d62728', lw=2,
                    label=f'FBP+CGLS ({best_fc["rmse"]:.1f})')
else:
    ax_profile.plot(best_sirt['rec'][center, :], '--', color='#2ca02c', lw=1.5,
                    label=f'SIRT ({best_sirt["rmse"]:.1f})')
    ax_profile.plot(best_fs['rec'][center, :], ':', color='#ff7f0e', lw=2,
                    label=f'FBP+SIRT ({best_fs["rmse"]:.1f})')
ax_profile.set_title(f'Center Row Profile (Row {center})', fontsize=12)
ax_profile.set_xlabel('Pixel Position', fontsize=10)
ax_profile.set_ylabel('HU Value', fontsize=10)
ax_profile.legend(fontsize=8, ncol=2)
ax_profile.grid(True, alpha=0.3)

plt.suptitle(f'FBP + IR Hybrid Reconstruction ({BACKEND})', fontsize=15, fontweight='bold', y=0.98)
plt.savefig("img_out/fbp_plus_ir.png", dpi=150, bbox_inches='tight')
plt.close()
print("   => img_out/fbp_plus_ir.png")

# ============================================================
# 保存总结
# ============================================================
summary = {
    'backend': BACKEND,
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
