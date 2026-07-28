"""
ASTRA vs TIGRE 锥束 FDK 滤波器/探测器对比
============================================
不同滤波器、探测器类型对 FDK 质量的影响
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.gridspec import GridSpec

from scipy.ndimage import gaussian_filter
import tomophantom
from tomophantom import TomoP3D
import os

try:
    import tigre
    import tigre.algorithms as algs
except ImportError:
    print("TIGRE 不可用"); exit(1)
try:
    import astra
except ImportError:
    print("ASTRA 不可用"); exit(1)

N = 512; nz = 32; n_angles = 360
print(f"体模: {nz}x{N}x{N}, 角度: {n_angles}")

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

angles = np.deg2rad(np.linspace(0, 360, n_angles, endpoint=False)).astype(np.float32)
D = int(np.ceil(N * np.sqrt(2)))

def linear_scale(rec):
    mask = vol_gt > 0.001
    A = np.column_stack([rec.ravel()[mask.ravel()], np.ones(mask.sum())])
    b = vol_gt.ravel()[mask.ravel()]; coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return rec * coef[0] + coef[1]

def calc_rmse(rec):
    mask = vol_gt > 0.001; return np.sqrt(np.mean((vol_gt[mask] - rec[mask])**2))

def calc_ssim(rec):
    mask = vol_gt > 0.001
    c1, c2 = (0.01*0.05)**2, (0.03*0.05)**2
    mux, muy = vol_gt[mask].mean(), rec[mask].mean()
    sx, sy = vol_gt[mask].var(), rec[mask].var()
    sxy = np.mean((vol_gt[mask]-mux)*(rec[mask]-muy))
    return (2*mux*muy+c1)*(2*sxy+c2)/((mux**2+muy**2+c1)*(sx+sy+c2))

# ===== ASTRA FDK =====
print("\n===== 1. ASTRA FDK =====")
angles_rad = angles.copy()
DSO, DSD = 1000.0, 500.0
n_det_row, n_det_col = nz * 2, D
det_pix = 1.0

vectors = np.zeros((n_angles, 12), dtype=np.float32)
for i, th in enumerate(angles_rad):
    c, s = np.cos(th), np.sin(th)
    vectors[i, :3] = [DSO*s, -DSO*c, 0.0]
    vectors[i, 3:6] = [-DSD*s, DSD*c, 0.0]
    vectors[i, 6:9] = [det_pix*c, det_pix*s, 0.0]
    vectors[i, 9:12] = [0.0, 0.0, det_pix]

proj_geom = astra.create_proj_geom("cone_vec", n_det_row, n_det_col, vectors)
vol_geom = astra.create_vol_geom(N, N, nz)

# Forward proj
vid = astra.data3d.create("-vol", vol_geom, vol_gt.astype(np.float32))
sid = astra.data3d.create("-sino", proj_geom, 0.0)
cfg = astra.astra_dict("FP3D_CUDA")
cfg["ProjectionDataId"] = sid; cfg["VolumeDataId"] = vid
aid = astra.algorithm.create(cfg); astra.algorithm.run(aid)
sino = astra.data3d.get(sid)
astra.algorithm.delete(aid); astra.data3d.delete(sid); astra.data3d.delete(vid)

# ASTRA FDK
t0 = time()
sid_f = astra.data3d.create("-sino", proj_geom, sino)
rid_f = astra.data3d.create("-vol", vol_geom)
cfg = astra.astra_dict("FDK_CUDA")
cfg["ProjectionDataId"] = sid_f; cfg["ReconstructionDataId"] = rid_f
aid = astra.algorithm.create(cfg); astra.algorithm.run(aid)
rec_astra = astra.data3d.get(rid_f).copy()
astra.algorithm.delete(aid); astra.data3d.delete(rid_f); astra.data3d.delete(sid_f)
r = linear_scale(rec_astra)
t_astra = time() - t0
print(f"   ASTRA FDK:    {t_astra*1000:6.0f}ms  RMSE={calc_rmse(r):.5f}  SSIM={calc_ssim(r):.4f}")

# ===== TIGRE Geometry =====
print("\n===== 2. TIGRE FDK (各种滤波器) =====")
geo = tigre.geometry()
geo.DSD = DSO + DSD; geo.DSO = DSO
geo.nVoxel = np.array([nz, N, N]); geo.sVoxel = np.array([nz*1.5, N, N])
geo.dVoxel = geo.sVoxel / geo.nVoxel
geo.nDetector = np.array([nz*2, D]); geo.dDetector = np.array([1.0, N/D])
geo.sDetector = geo.nDetector * geo.dDetector
geo.offOrigin = np.array([0, 0, 0]); geo.offDetector = np.array([0, 0])
geo.mode = "cone"; geo.filter = None

sino_tigre = tigre.Ax(vol_gt, geo, angles)
print(f"   TIGRE 投影: {sino_tigre.shape}")

filters = ["shepp_logan", "ram_lak", "hamming", "hann", "cosine", "ram_lak"]
for filt in filters:
    t0 = time()
    rec = algs.fdk(sino_tigre, geo, angles, filter=filt)
    t = time() - t0
    r = linear_scale(rec)
    print(f"   TIGRE {filt:15s}: {t*1000:6.0f}ms  RMSE={calc_rmse(r):.5f}  SSIM={calc_ssim(r):.4f}")

# TIGRE FDK + Gaussian filter
print("")
for sigma in [0.3, 0.5, 1.0]:
    t0 = time()
    rec = algs.fdk(sino_tigre, geo, angles, filter="shepp_logan")
    rec_s = np.array([gaussian_filter(rec[z], sigma=sigma) for z in range(nz)])
    t = time() - t0
    r = linear_scale(rec_s)
    print(f"   TIGRE shepp+σ={sigma:.1f}:   {t*1000:6.0f}ms  RMSE={calc_rmse(r):.5f}  SSIM={calc_ssim(r):.4f}")

# ===== 探测器类型对比 =====
print("\n===== 3. TIGRE 探测器类型 =====")
for mode in ["cone", "flat"]:
    geo.mode = mode
    t0 = time()
    rec = algs.fdk(sino_tigre, geo, angles, filter="shepp_logan")
    t = time() - t0
    r = linear_scale(rec)
    print(f"   TIGRE mode={mode:6s}:     {t*1000:6.0f}ms  RMSE={calc_rmse(r):.5f}  SSIM={calc_ssim(r):.4f}")
geo.mode = "cone"

# ===== 可视化对比 =====
print("\n生成可视化...")
mid = nz // 2

# 收集各方法结果
results = {}
results["GT"] = vol_gt[mid]

# ASTRA
sino_id = astra.data3d.create("-sino", proj_geom, sino)
rid_f = astra.data3d.create("-vol", vol_geom)
cfg = astra.astra_dict("FDK_CUDA")
cfg["ProjectionDataId"] = sino_id; cfg["ReconstructionDataId"] = rid_f
aid = astra.algorithm.create(cfg); astra.algorithm.run(aid)
rec_astra = astra.data3d.get(rid_f).copy()
astra.algorithm.delete(aid); astra.data3d.delete(rid_f); astra.data3d.delete(sino_id)
r = linear_scale(rec_astra)
results["ASTRA FDK"] = (r[mid], calc_rmse(r), calc_ssim(r))

# TIGRE filters
for filt in ["shepp_logan", "hann", "hamming"]:
    rec = algs.fdk(sino_tigre, geo, angles, filter=filt)
    r = linear_scale(rec)
    results[f"TIGRE {filt}"] = (r[mid], calc_rmse(r), calc_ssim(r))

# TIGRE best (hann + gauss)
rec = algs.fdk(sino_tigre, geo, angles, filter="hann")
rec_s = np.array([gaussian_filter(rec[z], sigma=0.5) for z in range(nz)])
r = linear_scale(rec_s)
results["TIGRE hann+σ0.5"] = (r[mid], calc_rmse(r), calc_ssim(r))

n_plots = len(results) - 1
fig = plt.figure(figsize=(20, 9))
gs = GridSpec(2, n_plots, figure=fig, hspace=0.35, wspace=0.3)

keys = [k for k in results if k != "GT"]
for i, k in enumerate(keys):
    img_slice, rmse_val, ssim_val = results[k]
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img_slice, cmap='gray', vmin=0, vmax=0.035)
    ax.set_title(f"{k}\nRMSE={rmse_val:.5f}  SSIM={ssim_val:.4f}", fontsize=8)
    ax.axis('off')
    ax2 = fig.add_subplot(gs[1, i])
    e = img_slice - vol_gt[mid]
    v = max(0.005, np.percentile(np.abs(e), 95)*1.2)
    ax2.imshow(e, cmap='RdBu_r', vmin=-v, vmax=v)
    ax2.set_title(f"Error (x{v:.4f})", fontsize=8); ax2.axis('off')

plt.suptitle(f"ASTRA vs TIGRE FDK 锥束对比 ({nz}x{N}x{N}, {n_angles}角度)\nTomoPhantom Model 4",
             fontsize=13, fontweight='bold', y=0.98)
os.makedirs("img_3d_helical", exist_ok=True)
plt.savefig("img_3d_helical/fdk_compare.png", dpi=150, bbox_inches='tight')
plt.close()
print("   => img_3d_helical/fdk_compare.png")
print("\nDone!")
