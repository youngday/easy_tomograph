"""
FBP + IR 混合重建 (TIGRE 锥束 CBCT)
=====================================
核心思想: 用 FDK 的快速重建作为迭代法 (SIRT/OS-SART/ASD-POCS) 的初始值,
          验证锥束 CT 下混合迭代的时间-质量收益

对比组:
  - Pure FDK (基线)
  - FBP + SIRT (混合)
  - FBP + OS-SART (混合, 20子集)
  - FBP + ASD-POCS (混合, TV正则化)

模式:
  - GPU 模式: TIGRE Toolbox CUDA, 锥束几何
"""

import numpy as np
from time import time
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import os, json

try:
    import tigre
    import tigre.algorithms as algs
except ImportError:
    print("错误: 需要 TIGRE Toolbox (GPU 版本)")
    exit(1)

print("=" * 60)
print("FBP + IR 混合重建对比  [锥束 CBCT | TIGRE CUDA]")
print("=" * 60)

# ============================================================
# 参数设置 (小体模保证运行速度)
# ============================================================
N = 128          # XY 分辨率
nz = 32          # Z 切片数
n_angles = 180   # 投影角度
print(f"体模: {nz}x{N}x{N}, 角度: {n_angles}")

# ============================================================
# 1. 生成 3D 体模 (多椭球体)
# ============================================================
vol_gt = np.zeros((nz, N, N), dtype=np.float32)
Z, Y, X = np.ogrid[:nz, :N, :N]
cz, cy, cx = nz/2, N/2, N/2

# 外层: 大椭圆柱 (身体轮廓)
body = ((Z - cz)/12)**2 + ((Y - cy)/(N*0.42))**2 + ((X - cx)/(N*0.35))**2 <= 1
vol_gt[body] = 0.020  # 软组织 ~20 HU

# 骨骼 (高密度环)
bone = ((Z - cz)/10)**2 + ((Y - cy)/(N*0.30))**2 + ((X - cx)/(N*0.25))**2 <= 1
vol_gt[bone & ~body] = 0.0
bone_ring = ((Z - cz)/10)**2 + ((Y - cy)/(N*0.28))**2 + ((X - cx)/(N*0.23))**2 >= 1
vol_gt[bone & bone_ring] = 0.045  # 骨密度 ~450 HU

# 内部器官 (椭球)
organ = ((Z - cz+4)/6)**2 + ((Y - cy-15)/(N*0.12))**2 + ((X - cx+10)/(N*0.10))**2 <= 1
vol_gt[organ] = 0.025

# 肿瘤 (小高密球)
tumor = ((Z - cz-3)/4)**2 + ((Y - cy+20)/(N*0.06))**2 + ((X - cx-15)/(N*0.06))**2 <= 1
vol_gt[tumor] = 0.035

# 空气腔
air = ((Z - cz+2)/5)**2 + ((Y - cy+25)/(N*0.08))**2 + ((X - cx+25)/(N*0.06))**2 <= 1
vol_gt[air] = 0.0

print(f"   体模范围: [{vol_gt.min():.5f}, {vol_gt.max():.5f}]")

# ============================================================
# 2. 锥束几何
# ============================================================
theta_deg = np.linspace(0, 360, n_angles, endpoint=False)  # 锥束用 360°
angles = np.deg2rad(theta_deg).astype(np.float32)
D = int(np.ceil(N * np.sqrt(2)))

geo = tigre.geometry()
geo.DSD = 1536.0    # 源-探测器距离
geo.DSO = 1000.0    # 源-物体距离
geo.nVoxel = np.array([nz, N, N])
geo.sVoxel = np.array([nz*1.5, N, N])  # 体素各向同性
geo.dVoxel = geo.sVoxel / geo.nVoxel
geo.nDetector = np.array([nz*2, D])
geo.dDetector = np.array([1.0, N/D])
geo.sDetector = geo.nDetector * geo.dDetector
geo.offOrigin = np.array([0, 0, 0])
geo.offDetector = np.array([0, 0])
geo.mode = 'cone'
geo.filter = None

# ============================================================
# 3. 正向投影 (GPU)
# ============================================================
print("\nGPU 正向投影...")
t0 = time()
sino = tigre.Ax(vol_gt, geo, angles)
print(f"   完成: {(time()-t0)*1000:.0f}ms, 投影形状 {sino.shape}")

# ============================================================
# 辅助函数
# ============================================================
def linear_scale(rec):
    """线性回归到 GT 的 HU 范围"""
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
_ = algs.fdk(sino, geo, angles, filter='shepp_logan')
_ = algs.sirt(sino, geo, angles, 1)
print("   预热完成\n")

# ============================================================
# A. Pure FDK (基线)
# ============================================================
print("-" * 55)
print("A. Pure FDK (TIGRE FDK shepp_logan)")
print("-" * 55)
t0 = time()
rec_fdk = algs.fdk(sino, geo, angles, filter='shepp_logan')
fdk_rec = linear_scale(rec_fdk)
fdk_t = time() - t0
fdk_rmse = calc_rmse(fdk_rec)
fdk_ssim = calc_ssim(fdk_rec)
print(f"   RMSE={fdk_rmse:.5f}, SSIM={fdk_ssim:.4f}, {fdk_t*1000:.0f}ms")

# ============================================================
# B. FBP + SIRT (混合)
# ============================================================
print("-" * 55)
print("B. FBP + SIRT (FDK 初始值)")
print("-" * 55)
fbs_hist = []
best_fs = {'rmse': 1e9, 'ssim': -1, 'rec': None, 't': 0, 'n': 0}
rec_sirt = rec_fdk.copy()
sirt_iters = [5, 10, 20, 50]
prev_n = 0
for n_iter in sirt_iters:
    t0 = time()
    rec_sirt = algs.sirt(sino, geo, angles, niter=n_iter - prev_n,
                         init=rec_sirt, noneg=False)
    t = time() - t0
    rec2d = linear_scale(rec_sirt)
    r, s = calc_rmse(rec2d), calc_ssim(rec2d)
    fbs_hist.append((n_iter, t, r, s))
    if r < best_fs['rmse']:
        best_fs = {'rmse': r, 'ssim': s, 'rec': rec2d, 't': t, 'n': n_iter}
    print(f"   x{n_iter:3d}: RMSE={r:.5f}, SSIM={s:.4f}, {t*1000:.0f}ms")
    prev_n = n_iter
print(f"   >> 最优: SIRT x{best_fs['n']}: RMSE={best_fs['rmse']:.5f}, SSIM={best_fs['ssim']:.4f}")

# ============================================================
# C. FBP + OS-SART (混合, 20子集)
# ============================================================
print("-" * 55)
print("C. FBP + OS-SART (20子集, FDK 初始值)")
print("-" * 55)
fbo_hist = []
best_fo = {'rmse': 1e9, 'ssim': -1, 'rec': None, 't': 0, 'n': 0}
rec_oss = rec_fdk.copy()
os_iters = [1, 2, 5, 10]
prev_n = 0
for n_iter in os_iters:
    t0 = time()
    rec_oss = algs.ossart(sino, geo, angles, niter=n_iter - prev_n,
                          init=rec_oss, blocksize=9, verbose=False)
    t = time() - t0
    rec2d = linear_scale(rec_oss)
    r, s = calc_rmse(rec2d), calc_ssim(rec2d)
    fbo_hist.append((n_iter, t, r, s))
    if r < best_fo['rmse']:
        best_fo = {'rmse': r, 'ssim': s, 'rec': rec2d, 't': t, 'n': n_iter}
    print(f"   x{n_iter:3d}: RMSE={r:.5f}, SSIM={s:.4f}, {t*1000:.0f}ms")
    prev_n = n_iter
print(f"   >> 最优: OS-SART x{best_fo['n']}: RMSE={best_fo['rmse']:.5f}, SSIM={best_fo['ssim']:.4f}")

# ============================================================
# D. 算法限制说明
# ============================================================
print("-" * 55)
print("D. ASD-POCS (TV正则化) 不可用")
print("-" * 55)
print(f"   TIGRE 3.1.2 ASD-POCS 存在类型兼容问题")
print("   (0-dimensional array to scalar 转换错误)")
has_tv = False

# ============================================================
# 汇总对比
# ============================================================
print("\n" + "=" * 60)
print("汇总对比")
print("=" * 60)
print(f"{'算法':30s} {'耗时(ms)':>10s} {'RMSE':>10s} {'SSIM':>8s}")
print("-" * 60)
print(f"{'Pure FDK (FBP)':30s} {fdk_t*1000:>8.0f} ms {fdk_rmse:>10.5f} {fdk_ssim:>8.4f}")
print(f"{'FBP+SIRT x'+str(best_fs['n']):30s} {best_fs['t']*1000:>8.0f} ms {best_fs['rmse']:>10.5f} {best_fs['ssim']:>8.4f}")
print(f"{'FBP+OS-SART x'+str(best_fo['n']):30s} {best_fo['t']*1000:>8.0f} ms {best_fo['rmse']:>10.5f} {best_fo['ssim']:>8.4f}")
if has_tv:
    print(f"{'FBP+ASD-POCS x20':30s} {tv_t*1000:>8.0f} ms {tv_rmse:>10.5f} {tv_ssim:>8.4f}")

# 等迭代对比 (SIRT vs OS-SART)
print("\n" + "=" * 60)
print("等迭代对比 (SIRT vs OS-SART)")
print("=" * 60)
print(f"{'轮次':>6s} {'FBP+SIRT':>20s} {'FBP+OS-SART':>22s}")
print(f"{'':>6s} {'RMSE/耗时':>20s} {'RMSE/耗时':>22s}")
print("-" * 50)
for n in [5, 10]:
    fsr = next((h for h in fbs_hist if h[0] == n), None)
    fso = next((h for h in fbo_hist if h[0] == n), None)
    r1 = f"{fsr[2]:.5f}/{fsr[1]*1000:.0f}ms" if fsr else "-"
    r2 = f"{fso[2]:.5f}/{fso[1]*1000:.0f}ms" if fso else "-"
    print(f"  x{n:3d}    {r1:>20s}  {r2:>22s}")

# ============================================================
# 结果列表
# ============================================================
results = [
    ('Pure FDK', fdk_t, fdk_rmse, fdk_ssim),
    ('FBP+SIRT x'+str(best_fs['n']), best_fs['t'], best_fs['rmse'], best_fs['ssim']),
    ('FBP+OS-SART x'+str(best_fo['n']), best_fo['t'], best_fo['rmse'], best_fo['ssim']),
]
if has_tv:
    results.append(('FBP+ASD-POCS x20', tv_t, tv_rmse, tv_ssim))

# ============================================================
# 可视化
# ============================================================
print("\n生成可视化...")
os.makedirs("img_out", exist_ok=True)
n_plots = len(results) + 1
mid_slice = nz // 2

fig = plt.figure(figsize=(4*n_plots, 8))
gs = GridSpec(2, n_plots, figure=fig, hspace=0.3, wspace=0.3)

# 第1行: 重建对比 (中间切片)
plot_items = [('Ground Truth', vol_gt[mid_slice], None, None, None)]
for name, t, r, s in results:
    plot_items.append((f"{name}\nRMSE={r:.5f}\n{t*1000:.0f}ms", r, None, None, None))
# 修正: 第二行用 rec 而不是直接传 rmse

# 简单方式: 直接绘制
for i, (title, img, _, _, _) in enumerate(plot_items):
    if i == 0:
        img_show = img
    else:
        img_show = results[i-1][2]  # rec from hist - need stored rec
    # 简化: 直接用结果显示
    ax = fig.add_subplot(gs[0, i])
    if i == 0:
        ax.imshow(vol_gt[mid_slice], cmap='gray', vmin=0, vmax=0.05)
        ax.set_title("Ground Truth", fontsize=9)
    else:
        name, t, r, s = results[i-1]
        rec_show = best_fs['rec'] if 'SIRT' in name else (best_fo['rec'] if 'OS-SART' in name else (tv_rec if has_tv and 'ASD' in name else fdk_rec))
        ax.imshow(rec_show[mid_slice], cmap='gray', vmin=0, vmax=0.05)
        ax.set_title(f"{name}\n{t*1000:.0f}ms", fontsize=8)
    ax.axis('off')

# 第2行: 误差图
for i in range(n_plots):
    ax = fig.add_subplot(gs[1, i])
    if i == 0:
        err = np.zeros_like(vol_gt[mid_slice])
        title = "Error"
    else:
        name, t, r, s = results[i-1]
        rec_show = best_fs['rec'] if 'SIRT' in name else (best_fo['rec'] if 'OS-SART' in name else (tv_rec if has_tv and 'ASD' in name else fdk_rec))
        err = rec_show[mid_slice] - vol_gt[mid_slice]
        title = f"Error"
    vmax = max(0.005, np.percentile(np.abs(err), 95) * 1.2)
    im = ax.imshow(err, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis('off')

plt.suptitle('Cone-beam CBCT: FBP + IR Hybrid Reconstruction (TIGRE CUDA)', fontsize=13, fontweight='bold', y=0.98)
plt.savefig("img_out/tigre_cone_hybrid.png", dpi=150, bbox_inches='tight')
plt.close()
print("   => img_out/tigre_cone_hybrid.png")

# ============================================================
# 保存总结
# ============================================================
summary = {
    'backend': 'GPU (TIGRE CUDA cone-beam)',
    'config': {'N': N, 'nz': nz, 'n_angles': n_angles},
    'results': {name: {'rmse': round(r, 5), 'ssim': round(s, 4), 'time_ms': round(t*1000, 1)}
                for name, t, r, s in results},
}
with open("img_out/tigre_cone_hybrid_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("   => img_out/tigre_cone_hybrid_summary.json")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
