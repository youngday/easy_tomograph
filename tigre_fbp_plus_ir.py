"""
FBP + IR 混合重建 (TIGRE GPU)
=============================
核心思想: 用 FBP 的快速重建结果作为迭代法 (CGLS/SIRT) 的初始值,
          让 IR 从更好的起点开始迭代 → 更快收敛 + 更高质量

对比组:
  - Pure FBP (基线 — 最快, 质量较差)
  - Pure CGLS from zero (基线 — 慢, 质量好)
  - FBP + CGLS (混合)
  - Pure SIRT from zero (基线)
  - FBP + SIRT (混合)

模式:
  - GPU 模式: 使用 TIGRE Toolbox (CUDA 加速)
"""

import numpy as np
from time import time
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import os, json

# ============================================================
# 后端检测: 需要 TIGRE
# ============================================================
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

# ============================================================
# 参数设置
# ============================================================
N = 512
n_angles = 360
print(f"体模: {N}x{N}, 角度: {n_angles}")

# ============================================================
# 1. 生成体模 (自定义头部横断面, 12种组织)
# ============================================================
def _add_ellipse(img, cx, cy, rx, ry, angle, value):
    cos_a = np.cos(np.deg2rad(angle))
    sin_a = np.sin(np.deg2rad(angle))
    xr = (X - cx) * cos_a + (Y - cy) * sin_a
    yr = -(X - cx) * sin_a + (Y - cy) * cos_a
    img[(xr / rx)**2 + (yr / ry)**2 <= 1] = value

Y, X = np.ogrid[:N, :N]
ct = np.full((N, N), -1000, dtype=np.float32)
# 头皮/软组织  (~50 HU)
_add_ellipse(ct, 256, 256, 210, 170, 0, 50)
# 颅骨外板      (~800 HU)
_add_ellipse(ct, 256, 256, 185, 150, 0, 800)
# 颅骨内板/松质 (~300 HU)
_add_ellipse(ct, 256, 256, 175, 142, 0, 300)
# 灰质          (~35 HU)
_add_ellipse(ct, 256, 256, 160, 130, 0, 35)
# 白质          (~28 HU)
_add_ellipse(ct, 256, 246, 110, 90, 0, 28)
_add_ellipse(ct, 260, 270, 90, 80, 0, 28)
# 侧脑室/CSF   (~5 HU)
_add_ellipse(ct, 240, 235, 30, 18, -15, 5)
_add_ellipse(ct, 272, 235, 30, 18, 15, 5)
# 第三脑室      (~5 HU)
_add_ellipse(ct, 256, 220, 12, 6, 0, 5)
# 丘脑          (~38 HU)
_add_ellipse(ct, 245, 230, 12, 10, 0, 38)
_add_ellipse(ct, 267, 230, 12, 10, 0, 38)
# 眼球          (~20 HU)
_add_ellipse(ct, 235, 420, 22, 22, 0, 20)
_add_ellipse(ct, 277, 420, 22, 22, 0, 20)
# 晶状体        (~120 HU)
_add_ellipse(ct, 235, 410, 8, 4, 0, 120)
_add_ellipse(ct, 277, 410, 8, 4, 0, 120)
# 鼻腔/额窦     (-1000 / -800 HU)
_add_ellipse(ct, 256, 370, 18, 8, 0, -1000)
_add_ellipse(ct, 256, 340, 15, 5, 0, -800)
# 小肿瘤        (~50 HU, 稍高于灰质)
_add_ellipse(ct, 210, 210, 10, 8, 30, 50)
# 钙化点        (~400 HU)
_add_ellipse(ct, 250, 260, 3, 3, 0, 400)

head_r = 215  # 头部半径 (略大于头皮 210)
circ_mask = (X - N / 2) ** 2 + (Y - N / 2) ** 2 <= head_r ** 2
ct[~circ_mask] = -1000

# 余弦软遮罩: 在圆形边缘过渡带平滑到背景
dist = np.sqrt((X - N / 2) ** 2 + (Y - N / 2) ** 2)
soft_mask = np.clip((head_r + 20 - dist) / 20, 0, 1)

# ============================================================
# 2. 正向投影 (TIGRE GPU)
# ============================================================
theta_deg = np.linspace(0, 180, n_angles, endpoint=False)
theta_rad = np.deg2rad(theta_deg).astype(np.float32)

# TIGRE 2D 几何: nVoxel = (z, y, x), nDetector = (v, u)
D = int(np.ceil(N * np.sqrt(2)))  # 探测器覆盖体模对角线

geo = tigre.geometry()
geo.DSD = 1536
geo.DSO = 1000
geo.nVoxel = np.array([1, N, N])          # (z, y, x)
geo.sVoxel = np.array([1, N, N])          # (mm)
geo.dVoxel = geo.sVoxel / geo.nVoxel      # (1, 1, 1) mm
geo.nDetector = np.array([1, D])          # (v, u)
geo.dDetector = np.array([1.0, N/D])       # 探测器 FOV = D * N/D = N, 匹配体模, 完全消除竖纹
geo.sDetector = geo.nDetector * geo.dDetector
geo.offOrigin = np.array([0, 0, 0])
geo.offDetector = np.array([0, 0])
geo.mode = 'parallel'

angles = theta_rad  # 弧度

# 体模 → TIGRE 3D 格式 (1, N, N)
vol_gt = ct[np.newaxis, :, :].astype(np.float32)

print("GPU 正向投影...")
t0 = time()
sino_3d = tigre.Ax(vol_gt, geo, angles)
print(f"   完成: 耗时 {(time()-t0)*1000:.0f}ms, 投影形状 {sino_3d.shape}")
# sino_3d: (n_angles, 1, D)
sino = sino_3d[:, 0, :]

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

# GPU 预热
print("GPU 预热...")
_ = algs.fdk(sino_3d, geo, angles)
_ = algs.cgls(sino_3d, geo, angles, 1)
_ = algs.sirt(sino_3d, geo, angles, 1)
print("   预热完成\n")

# ============================================================
# A. Pure FBP (基线)
# ============================================================
print("-" * 55)
print("A. Pure FBP (TIGRE FDK shepp_logan)")
print("-" * 55)
t0 = time()
rec_fdk_3d = algs.fdk(sino_3d, geo, angles, filter='shepp_logan')
fbp_raw = rec_fdk_3d[0]
# 轻量高斯去振铃 (σ=0.5): 消除 TIGRE FDK 在均匀背景的宽振铃
fbp_denoised = gaussian_filter(fbp_raw, sigma=0.5)
fbp_rec = linear_scale(fbp_denoised)
fbp_t = time() - t0
fbp_rmse = calc_rmse(fbp_rec)
fbp_ssim = calc_ssim(fbp_rec)
print(f"   RMSE={fbp_rmse:.2f}, SSIM={fbp_ssim:.4f}, {fbp_t*1000:.0f}ms (TIGRE FDK shepp_logan)")

# ============================================================
# B. Pure CGLS (从零开始)
# ============================================================
print("-" * 55)
print("B. Pure CGLS (从零开始)")
print("-" * 55)
cgls_hist = []
best_cgls = {'rmse': 1e9, 'ssim': -1, 'rec': None, 't': 0, 'n': 0}
cgls_iters = [5, 10, 20, 30, 50]  # x50 后收益递减

rec_cgls_3d = None
prev_n = 0
for n_iter in cgls_iters:
    t0 = time()
    if rec_cgls_3d is None:
        rec_cgls_3d = algs.cgls(sino_3d, geo, angles, niter=n_iter)
    else:
        rec_cgls_3d = algs.cgls(sino_3d, geo, angles, niter=n_iter - prev_n,
                                init=rec_cgls_3d)
    t = time() - t0
    rec2d = linear_scale(rec_cgls_3d[0])
    r, s = calc_rmse(rec2d), calc_ssim(rec2d)
    cgls_hist.append((n_iter, t, r, s))
    if r < best_cgls['rmse']:
        best_cgls = {'rmse': r, 'ssim': s, 'rec': rec2d, 't': t, 'n': n_iter}
    print(f"   x{n_iter:3d}: RMSE={r:.2f}, SSIM={s:.4f}, {t*1000:.0f}ms")
    prev_n = n_iter

print(f"   >> 最优: CGLS x{best_cgls['n']}: RMSE={best_cgls['rmse']:.2f}, SSIM={best_cgls['ssim']:.4f}, {best_cgls['t']*1000:.0f}ms")

# ============================================================
# C. FBP + CGLS (混合)
# ============================================================
print("-" * 55)
print("C. FBP + CGLS (FBP 初始值)")
print("-" * 55)
fbc_hist = []
best_fc = {'rmse': 1e9, 'ssim': -1, 'rec': None, 't': 0, 'n': 0}
rec_fbc_3d = rec_fdk_3d.copy()  # 用原始 FDK (未去振铃) 做初始化
# 去振铃仅用于 FBP 显示, IR 初始化用原始 FDK 保留更多信息
prev_n = 0
for n_iter in cgls_iters:
    t0 = time()
    rec_fbc_3d = algs.cgls(sino_3d, geo, angles, niter=n_iter - prev_n,
                           init=rec_fbc_3d)
    t = time() - t0
    rec2d = linear_scale(rec_fbc_3d[0])
    r, s = calc_rmse(rec2d), calc_ssim(rec2d)
    fbc_hist.append((n_iter, t, r, s))
    if r < best_fc['rmse']:
        best_fc = {'rmse': r, 'ssim': s, 'rec': rec2d, 't': t, 'n': n_iter}
    print(f"   x{n_iter:3d}: RMSE={r:.2f}, SSIM={s:.4f}, {t*1000:.0f}ms")
    prev_n = n_iter

print(f"   >> 最优: FBP+CGLS x{best_fc['n']}: RMSE={best_fc['rmse']:.2f}, SSIM={best_fc['ssim']:.4f}, {best_fc['t']*1000:.0f}ms")

# ============================================================
# D. Pure SIRT (从零开始)
# ============================================================
print("-" * 55)
print("D. Pure SIRT (从零开始)")
print("-" * 55)
sirt_hist = []
best_sirt = {'rmse': 1e9, 'ssim': -1, 'rec': None, 't': 0, 'n': 0}
sirt_iters = [10, 20, 50, 100, 200]  # x200 后太慢

rec_sirt_3d = None
prev_n = 0
for n_iter in sirt_iters:
    t0 = time()
    if rec_sirt_3d is None:
        rec_sirt_3d = algs.sirt(sino_3d, geo, angles, niter=n_iter, noneg=False)
    else:
        rec_sirt_3d = algs.sirt(sino_3d, geo, angles, niter=n_iter - prev_n,
                                init=rec_sirt_3d, noneg=False)
    t = time() - t0
    rec2d = linear_scale(rec_sirt_3d[0])
    r, s = calc_rmse(rec2d), calc_ssim(rec2d)
    sirt_hist.append((n_iter, t, r, s))
    if r < best_sirt['rmse']:
        best_sirt = {'rmse': r, 'ssim': s, 'rec': rec2d, 't': t, 'n': n_iter}
    print(f"   x{n_iter:4d}: RMSE={r:.2f}, SSIM={s:.4f}, {t*1000:.0f}ms")
    prev_n = n_iter

print(f"   >> 最优: SIRT x{best_sirt['n']}: RMSE={best_sirt['rmse']:.2f}, SSIM={best_sirt['ssim']:.4f}, {best_sirt['t']*1000:.0f}ms")

# ============================================================
# E. FBP + SIRT (混合)
# ============================================================
print("-" * 55)
print("E. FBP + SIRT (FBP 初始值)")
print("-" * 55)
fbs_hist = []
best_fs = {'rmse': 1e9, 'ssim': -1, 'rec': None, 't': 0, 'n': 0}
rec_fbs_3d = rec_fdk_3d.copy()
prev_n = 0
for n_iter in sirt_iters:
    t0 = time()
    rec_fbs_3d = algs.sirt(sino_3d, geo, angles, niter=n_iter - prev_n,
                           init=rec_fbs_3d, noneg=False)
    t = time() - t0
    rec2d = linear_scale(rec_fbs_3d[0])
    r, s = calc_rmse(rec2d), calc_ssim(rec2d)
    fbs_hist.append((n_iter, t, r, s))
    if r < best_fs['rmse']:
        best_fs = {'rmse': r, 'ssim': s, 'rec': rec2d, 't': t, 'n': n_iter}
    print(f"   x{n_iter:4d}: RMSE={r:.2f}, SSIM={s:.4f}, {t*1000:.0f}ms")
    prev_n = n_iter

print(f"   >> 最优: FBP+SIRT x{best_fs['n']}: RMSE={best_fs['rmse']:.2f}, SSIM={best_fs['ssim']:.4f}, {best_fs['t']*1000:.0f}ms")

# ============================================================
# 结果列表
# ============================================================
results = [
    ('Pure FBP (shepp_logan)', fbp_t, fbp_rmse, fbp_ssim, 0),
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
print("汇总对比 (最优迭代)")
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
# 等迭代次数对比 (展示混合方法的真正优势)
# ============================================================
print("\n" + "=" * 60)
print("等迭代次数对比 (混合 vs 纯 IR)")
print("=" * 60)
print(f"{'迭代数':>8s} {'Pure CGLS':>12s} {'FBP+CGLS':>12s} {'改善':>10s}  |  {'Pure SIRT':>12s} {'FBP+SIRT':>12s} {'改善':>10s}")
print("-" * 76)
# 找共同迭代次数
common_iters = set(cgls_iters) & set(sirt_iters)
for n in sorted(common_iters):
    cg = next((h for h in cgls_hist if h[0] == n), None)
    fc = next((h for h in fbc_hist if h[0] == n), None)
    sr = next((h for h in sirt_hist if h[0] == n), None)
    fs = next((h for h in fbs_hist if h[0] == n), None)
    
    cg_str = f"{cg[2]:.1f}" if cg else "-"
    fc_str = f"{fc[2]:.1f}" if fc else "-"
    sr_str = f"{sr[2]:.1f}" if sr else "-"
    fs_str = f"{fs[2]:.1f}" if fs else "-"
    
    imp_cg = f"{(cg[2]-fc[2])/cg[2]*100:+.1f}%" if cg and fc else "-"
    imp_sr = f"{(sr[2]-fs[2])/sr[2]*100:+.1f}%" if sr and fs else "-"
    
    print(f"  x{n:4d}     {cg_str:>8s}    {fc_str:>8s}   {imp_cg:>8s}  |  {sr_str:>8s}    {fs_str:>8s}   {imp_sr:>8s}")

# 额外: 展示 FBP+SIRT 在 x50/x100 的性价比
print("\n推荐配置 (性价比):")
print(f"   最快:    FBP (FDK)   → {fbp_t*1000:.0f}ms, RMSE={fbp_rmse:.1f}")
if any(h[0] == 30 for h in fbc_hist):
    f30 = next(h for h in fbc_hist if h[0] == 30)
    print(f"   均衡:    FBP+CGLS x30 → {f30[1]*1000:.0f}ms, RMSE={f30[2]:.1f}")
if any(h[0] == 100 for h in fbs_hist):
    f100 = next(h for h in fbs_hist if h[0] == 100)
    print(f"   高质量:  FBP+SIRT x100 → {f100[1]*1000:.0f}ms, RMSE={f100[2]:.1f}")
if any(h[0] == 200 for h in fbs_hist):
    f200 = next(h for h in fbs_hist if h[0] == 200)
    print(f"   最优:    FBP+SIRT x200 → {f200[1]*1000:.0f}ms, RMSE={f200[2]:.1f}")
print("\n生成可视化...")
os.makedirs("img_out", exist_ok=True)

fig = plt.figure(figsize=(22, 14))
gs = GridSpec(4, 7, figure=fig, hspace=0.3, wspace=0.3,
              width_ratios=[1, 1, 1, 1, 1, 1, 0.05])

gray_cmap = 'gray'
err_cmap = 'RdBu_r'

# 第1行: 重建对比
plot_items = [('Ground Truth', ct, None, None, None),
              ('Pure FBP', fbp_rec, fbp_rmse, fbp_ssim, fbp_t),
              ('CGLS (best)', best_cgls['rec'], best_cgls['rmse'], best_cgls['ssim'], best_cgls['t']),
              ('FBP+CGLS (best)', best_fc['rec'], best_fc['rmse'], best_fc['ssim'], best_fc['t']),
              ('SIRT (best)', best_sirt['rec'], best_sirt['rmse'], best_sirt['ssim'], best_sirt['t']),
              ('FBP+SIRT (best)', best_fs['rec'], best_fs['rmse'], best_fs['ssim'], best_fs['t'])]

for i, (title, img, rmse, ssim, t) in enumerate(plot_items):
    ax = fig.add_subplot(gs[0, i])
    # 应用软遮罩消除边缘条纹
    img_display = img * soft_mask + ct * (1 - soft_mask) if rmse is not None else img
    ax.imshow(img_display, cmap=gray_cmap, vmin=-200, vmax=600)
    tstr = title
    if rmse is not None:
        tstr += f"\nRMSE={rmse:.1f} SSIM={ssim:.4f}\n{t*1000:.0f}ms"
    ax.set_title(tstr, fontsize=9)
    ax.axis('off')

# 第2行: 误差图
err_items = [('', ct - ct),              # GT 无误差
             ('FBP Error', fbp_rec - ct),
             ('CGLS Error', best_cgls['rec'] - ct),
             ('FBP+CGLS Error', best_fc['rec'] - ct),
             ('SIRT Error', best_sirt['rec'] - ct),
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

cax = fig.add_subplot(gs[1, 6])
plt.colorbar(im, cax=cax)
cax.set_ylabel('HU Error', fontsize=8)
plt.suptitle('FBP + IR Hybrid Reconstruction (GPU: TIGRE CUDA)', fontsize=15, fontweight='bold', y=0.98)
plt.savefig("img_out/tigre_fbp_plus_ir.png", dpi=150, bbox_inches='tight')
plt.close()
print("   => img_out/tigre_fbp_plus_ir.png")

# ============================================================
# 保存总结
# ============================================================
summary = {
    'backend': 'GPU (TIGRE CUDA)',
    'config': {'N': N, 'n_angles': n_angles},
    'results': {name: {'rmse': round(r, 2), 'ssim': round(s, 4), 'time_ms': round(t*1000, 1)}
                for name, t, r, s, _ in results},
}
with open("img_out/tigre_fbp_plus_ir_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("   => img_out/tigre_fbp_plus_ir_summary.json")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
