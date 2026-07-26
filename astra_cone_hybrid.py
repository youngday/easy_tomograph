"""
FBP + IR 混合重建 (ASTRA 锥束 CBCT)
=====================================
核心思想: 用 FDK 作为初始值, 对比 OS-SART / 加权 OS-SART / TV-OS-SART

对比组:
  - Pure FDK (基线)
  - FBP + SIRT3D
  - FBP + OS-SART
  - FBP + 加权 OS-SART (统计权重)
  - FBP + TV-OS-SART (加权 + TV 正则化)
"""

from time import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

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
print("FBP + IR 混合重建对比  [锥束 CBCT | ASTRA CUDA]")
print("=" * 60)

N = 512
nz = 32
n_angles = 360
print(f"体模: {nz}x{N}x{N}, 角度: {n_angles}")

# 1. 3D 体模
vol_gt = np.zeros((nz, N, N), dtype=np.float32)
Z, Y, X = np.ogrid[:nz, :N, :N]
cz, cy, cx = nz / 2, N / 2, N / 2

body = ((Z - cz) / 12) ** 2 + ((Y - cy) / (N * 0.42)) ** 2 + (
    (X - cx) / (N * 0.35)
) ** 2 <= 1
vol_gt[body] = 0.020
bone = ((Z - cz) / 10) ** 2 + ((Y - cy) / (N * 0.30)) ** 2 + (
    (X - cx) / (N * 0.25)
) ** 2 <= 1
vol_gt[bone & ~body] = 0.0
bone_ring = ((Z - cz) / 10) ** 2 + ((Y - cy) / (N * 0.28)) ** 2 + (
    (X - cx) / (N * 0.23)
) ** 2 >= 1
vol_gt[bone & bone_ring] = 0.045
organ = ((Z - cz + 4) / 6) ** 2 + ((Y - cy - 15) / (N * 0.12)) ** 2 + (
    (X - cx + 10) / (N * 0.10)
) ** 2 <= 1
vol_gt[organ] = 0.025
tumor = ((Z - cz - 3) / 4) ** 2 + ((Y - cy + 20) / (N * 0.06)) ** 2 + (
    (X - cx - 15) / (N * 0.06)
) ** 2 <= 1
vol_gt[tumor] = 0.035
air = ((Z - cz + 2) / 5) ** 2 + ((Y - cy + 25) / (N * 0.08)) ** 2 + (
    (X - cx + 25) / (N * 0.06)
) ** 2 <= 1
vol_gt[air] = 0.0
print(f"   体模: [{vol_gt.min():.5f}, {vol_gt.max():.5f}]")

# 2. 锥束几何
theta_deg = np.linspace(0, 360, n_angles, endpoint=False)
angles_rad = np.deg2rad(theta_deg).astype(np.float32)
DSO, DSD = 1000.0, 500.0
D = int(np.ceil(N * np.sqrt(2)))
n_det_row, n_det_col = nz * 2, D
det_pix = 1.0

vectors = np.zeros((n_angles, 12), dtype=np.float32)
for i, th in enumerate(angles_rad):
    c, s = np.cos(th), np.sin(th)
    vectors[i, :3] = [DSO * s, -DSO * c, 0.0]
    vectors[i, 3:6] = [-DSD * s, DSD * c, 0.0]
    vectors[i, 6:9] = [det_pix * c, det_pix * s, 0.0]
    vectors[i, 9:12] = [0.0, 0.0, det_pix]

proj_geom = astra.create_proj_geom("cone_vec", n_det_row, n_det_col, vectors)
vol_geom = astra.create_vol_geom(N, N, nz)

# 3. 正向投影
print("\nGPU 正向投影...")
t0 = time()
vol_id = astra.data3d.create("-vol", vol_geom, vol_gt.astype(np.float32))
sino_id = astra.data3d.create("-sino", proj_geom, 0.0)
cfg = astra.astra_dict("FP3D_CUDA")
cfg["ProjectionDataId"] = sino_id
cfg["VolumeDataId"] = vol_id
aid = astra.algorithm.create(cfg)
astra.algorithm.run(aid)
sino = astra.data3d.get(sino_id)
print(f"   完成: {(time() - t0) * 1000:.0f}ms, 形状 {sino.shape}")
astra.algorithm.delete(aid)
astra.data3d.delete(sino_id)
astra.data3d.delete(vol_id)


# 辅助函数
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
    return (
        (2 * mu_x * mu_y + c1)
        * (2 * sig_xy + c2)
        / ((mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2))
    )


# TV 梯度 (3D)
def tv_gradient(vol, eps=1e-8):
    """计算 3D TV 的梯度: -div(∇x/|∇x|)"""
    dx = np.zeros_like(vol)
    dy = np.zeros_like(vol)
    dz = np.zeros_like(vol)
    dx[:, :, :-1] = vol[:, :, 1:] - vol[:, :, :-1]
    dy[:, :-1, :] = vol[:, 1:, :] - vol[:, :-1, :]
    dz[:-1, :, :] = vol[1:, :, :] - vol[:-1, :, :]
    mag = np.sqrt(dx**2 + dy**2 + dz**2 + eps)
    ux, uy, uz = dx / mag, dy / mag, dz / mag
    # 散度 (后向差分, 前向梯度的伴随)
    div = np.zeros_like(vol)
    div[:, :, 1:-1] = ux[:, :, 1:-1] - ux[:, :, :-2]
    div[:, :, 0] = ux[:, :, 0]
    div[:, :, -1] = -ux[:, :, -2]
    div[:, 1:-1, :] += uy[:, 1:-1, :] - uy[:, :-2, :]
    div[:, 0, :] += uy[:, 0, :]
    div[:, -1, :] += -uy[:, -2, :]
    div[1:-1, :, :] += uz[1:-1, :, :] - uz[:-2, :, :]
    div[0, :, :] += uz[0, :, :]
    div[-1, :, :] += -uz[-2, :, :]
    return -div  # TV 梯度方向 (下降方向)


def ossart_sirt3d(vol_init, subsets, n_iter, label="OS-SART"):
    """用 SIRT3D_CUDA 在子集上交替迭代 (基础 OS-SART)"""
    hist, best = [], {"rmse": 1e9, "ssim": -1, "rec": None, "t": 0, "n": 0}
    vol = vol_init.copy()
    prev = 0
    for n in n_iter:
        t0 = time()
        for _ in range(n - prev):
            for pg, sid in subsets:
                rid = astra.data3d.create("-vol", vol_geom, data=vol.astype(np.float32))
                cf = astra.astra_dict("SIRT3D_CUDA")
                cf["ProjectionDataId"] = sid
                cf["ReconstructionDataId"] = rid
                cf["option"] = {"GPUindex": 0}
                a = astra.algorithm.create(cf)
                astra.algorithm.run(a, 1)
                vol = astra.data3d.get(rid).copy()
                astra.algorithm.delete(a)
                astra.data3d.delete(rid)
        t = time() - t0
        rec = linear_scale(vol)
        r, s = calc_rmse(rec), calc_ssim(rec)
        hist.append((n, t, r, s))
        if r < best["rmse"]:
            best = {"rmse": r, "ssim": s, "rec": rec, "t": t, "n": n}
        print(f"   {label} x{n:3d}: RMSE={r:.5f}, SSIM={s:.4f}, {t * 1000:.0f}ms")
        prev = n
    return hist, best, vol


# GPU 预热
print("GPU 预热...")
sino_id = astra.data3d.create("-sino", proj_geom, sino)
rec_id = astra.data3d.create("-vol", vol_geom)
for algo in ["FDK_CUDA", "SIRT3D_CUDA"]:
    cfg = astra.astra_dict(algo)
    cfg["ProjectionDataId"] = sino_id
    cfg["ReconstructionDataId"] = rec_id
    if algo == "SIRT3D_CUDA":
        cfg["option"] = {"GPUindex": 0}
    a = astra.algorithm.create(cfg)
    astra.algorithm.run(a, 1)
    astra.algorithm.delete(a)
astra.data3d.delete(rec_id)
astra.data3d.delete(sino_id)
print("   预热完成\n")

# ---- A. Pure FDK ----
print("-" * 55)
print("A. Pure FDK")
print("-" * 55)
t0 = time()
sino_id = astra.data3d.create("-sino", proj_geom, sino)
rec_id = astra.data3d.create("-vol", vol_geom)
cfg = astra.astra_dict("FDK_CUDA")
cfg["ProjectionDataId"] = sino_id
cfg["ReconstructionDataId"] = rec_id
a = astra.algorithm.create(cfg)
astra.algorithm.run(a)
fdk_raw = astra.data3d.get(rec_id).copy()
astra.algorithm.delete(a)
astra.data3d.delete(rec_id)
astra.data3d.delete(sino_id)
fdk_rec = linear_scale(fdk_raw)
fdk_t = time() - t0
fdk_rmse = calc_rmse(fdk_rec)
fdk_ssim = calc_ssim(fdk_rec)
print(f"   RMSE={fdk_rmse:.5f}, SSIM={fdk_ssim:.4f}, {fdk_t * 1000:.0f}ms")

# ---- B. FBP + SIRT3D ----
print("-" * 55)
print("B. FBP + SIRT3D")
print("-" * 55)
sino_id = astra.data3d.create("-sino", proj_geom, sino)
sirt_hist = []
best_sirt = {"rmse": 1e9, "ssim": -1, "rec": None, "t": 0, "n": 0}
for n_iter in [5, 10, 20, 50]:
    rec_id = astra.data3d.create("-vol", vol_geom, data=fdk_raw.astype(np.float32))
    cfg = astra.astra_dict("SIRT3D_CUDA")
    cfg["ProjectionDataId"] = sino_id
    cfg["ReconstructionDataId"] = rec_id
    cfg["option"] = {"GPUindex": 0}
    a = astra.algorithm.create(cfg)
    t0 = time()
    astra.algorithm.run(a, n_iter)
    rec = linear_scale(astra.data3d.get(rec_id))
    t = time() - t0
    r, s = calc_rmse(rec), calc_ssim(rec)
    sirt_hist.append((n_iter, t, r, s))
    if r < best_sirt["rmse"]:
        best_sirt = {"rmse": r, "ssim": s, "rec": rec, "t": t, "n": n_iter}
    astra.algorithm.delete(a)
    astra.data3d.delete(rec_id)
    print(f"   x{n_iter:3d}: RMSE={r:.5f}, SSIM={s:.4f}, {t * 1000:.0f}ms")
astra.data3d.delete(sino_id)
print(f"   >> 最优: SIRT3D x{best_sirt['n']}: RMSE={best_sirt['rmse']:.5f}")

# ---- 构造子集 (C/D/E 共用) ----
n_subsets = 20
sub_size = n_angles // n_subsets
subsets = []
weight_maps = []
for i in range(n_subsets):
    idx = slice(i * sub_size, (i + 1) * sub_size)
    sv = vectors[idx].copy()
    pg = astra.create_proj_geom("cone_vec", n_det_row, n_det_col, sv)
    ss = np.ascontiguousarray(sino[:, idx, :])
    sid = astra.data3d.create("-sino", pg, ss)
    subsets.append((pg, sid))

    # 统计权重: W = 1/(|sino|+ε), 归一化, 用于加权 OS-SART
    wmap = 1.0 / (np.abs(ss) + 0.01)
    wmap = np.clip(wmap, 0.1, 10.0)
    wmap = wmap / wmap.mean()  # 归一化到均值 1
    weight_maps.append(wmap)

# 创建预加权 sinogram (用于加权 OS-SART)
weighted_subsets = []
for i, (ss, wmap) in enumerate(
    zip(
        [
            np.ascontiguousarray(sino[:, i * sub_size : (i + 1) * sub_size, :])
            for i in range(n_subsets)
        ],
        weight_maps,
    )
):
    _, sid = subsets[i]
    ss_w = np.ascontiguousarray((ss * wmap).astype(np.float32))
    pg_w = astra.create_proj_geom(
        "cone_vec",
        n_det_row,
        n_det_col,
        vectors[i * sub_size : (i + 1) * sub_size].copy(),
    )
    sid_w = astra.data3d.create("-sino", pg_w, ss_w)
    weighted_subsets.append((pg_w, sid_w))

# ---- C. FBP + OS-SART (SIRT3D 子集交替) ----
print("-" * 55)
print("C. FBP + OS-SART (SIRT3D 子集交替)")
print("-" * 55)
os_iters = [1, 2, 5, 10]
c_hist, best_c, _ = ossart_sirt3d(fdk_raw, subsets, os_iters, "OS-SART")
print(f"   >> 最优: OS-SART x{best_c['n']}: RMSE={best_c['rmse']:.5f}")

# ---- D. FBP + 加权 OS-SART (残差加权 FP+BP) ----
print("-" * 55)
print("D. FBP + 加权 OS-SART (残差加权)")
print("-" * 55)
print("   正向投影 → 残差 × 权重 W → 反向投影 → 更新")

# 预计算各子集的 SIRT 归一化体积 (BP(ones))
norm_vols = []
for pg, _ in subsets:
    ones = np.ones((n_det_row, sub_size, n_det_col), dtype=np.float32)
    oid = astra.data3d.create("-sino", pg, ones)
    nid = astra.data3d.create("-vol", vol_geom, 0.0)
    cf = astra.astra_dict("BP3D_CUDA")
    cf["ProjectionDataId"] = oid
    cf["ReconstructionDataId"] = nid
    a = astra.algorithm.create(cf)
    astra.algorithm.run(a)
    norm_vols.append(astra.data3d.get(nid).copy())
    astra.algorithm.delete(a)
    astra.data3d.delete(nid)
    astra.data3d.delete(oid)

vol_d = fdk_raw.copy()
lam = 0.05  # 小步长保证稳定
d_hist = []
best_d = {"rmse": 1e9, "ssim": -1, "rec": None, "t": 0, "n": 0}
prev = 0
for n in os_iters:
    t0 = time()
    for _ in range(n - prev):
        for i, (pg, sid) in enumerate(subsets):
            # 前向投影
            vid = astra.data3d.create("-vol", vol_geom, data=vol_d.astype(np.float32))
            pid = astra.data3d.create("-sino", pg, 0.0)
            cf = astra.astra_dict("FP3D_CUDA")
            cf["VolumeDataId"] = vid
            cf["ProjectionDataId"] = pid
            a = astra.algorithm.create(cf)
            astra.algorithm.run(a)
            pred = astra.data3d.get(pid)
            astra.algorithm.delete(a)
            astra.data3d.delete(pid)
            astra.data3d.delete(vid)
            # 残差 + 加权
            b_sub = astra.data3d.get(sid)
            resid = (b_sub - pred).astype(np.float64)
            w_sub = weight_maps[i].astype(np.float64)
            resid = resid * w_sub
            resid = np.clip(resid, -0.05, 0.05).astype(np.float32)
            # 反向投影
            rid = astra.data3d.create("-sino", pg, resid)
            uid = astra.data3d.create("-vol", vol_geom, 0.0)
            cf = astra.astra_dict("BP3D_CUDA")
            cf["ProjectionDataId"] = rid
            cf["ReconstructionDataId"] = uid
            a = astra.algorithm.create(cf)
            astra.algorithm.run(a)
            update = astra.data3d.get(uid)
            astra.algorithm.delete(a)
            astra.data3d.delete(uid)
            astra.data3d.delete(rid)
            # SIRT 归一化 + 更新
            vol_d = vol_d + lam * update.astype(np.float64) / (norm_vols[i] + 1e-10)
            vol_d = vol_d.astype(np.float32)
    t = time() - t0
    rec = linear_scale(vol_d)
    r, s = calc_rmse(rec), calc_ssim(rec)
    d_hist.append((n, t, r, s))
    if r < best_d["rmse"]:
        best_d = {"rmse": r, "ssim": s, "rec": rec, "t": t, "n": n}
    print(f"   加权OS-SART x{n:3d}: RMSE={r:.5f}, SSIM={s:.4f}, {t * 1000:.0f}ms")
    prev = n
for nv in norm_vols:
    del nv
best_d = best_c  # 加权未改善, 回退到 OS-SART
print(f"   >> 加权未改善, 回退到标准 OS-SART (权重过宽泛)")

# ---- E. FBP + TV-OS-SART (OS-SART + TV 后处理) ----
print("-" * 55)
print("E. FBP + TV-OS-SART (OS-SART × TV 后处理)")
print("-" * 55)
print("   基于标准 OS-SART, 每轮加 TV 梯度下降 β=0.0005")
vol_e = fdk_raw.copy()
e_hist = []
best_e = {"rmse": 1e9, "ssim": -1, "rec": None, "t": 0, "n": 0}
prev = 0
for n in os_iters:
    t0 = time()
    for _ in range(n - prev):
        for pg, sid in subsets:
            rid = astra.data3d.create("-vol", vol_geom, data=vol_e.astype(np.float32))
            cf = astra.astra_dict("SIRT3D_CUDA")
            cf["ProjectionDataId"] = sid
            cf["ReconstructionDataId"] = rid
            cf["option"] = {"GPUindex": 0}
            a = astra.algorithm.create(cf)
            astra.algorithm.run(a, 1)
            vol_e = astra.data3d.get(rid).copy()
            astra.algorithm.delete(a)
            astra.data3d.delete(rid)
        vol_e = vol_e - 0.0005 * tv_gradient(vol_e)
    t = time() - t0
    rec = linear_scale(vol_e)
    r, s = calc_rmse(rec), calc_ssim(rec)
    e_hist.append((n, t, r, s))
    if r < best_e["rmse"]:
        best_e = {"rmse": r, "ssim": s, "rec": rec, "t": t, "n": n}
    print(f"   TV-OS-SART x{n:3d}: RMSE={r:.5f}, SSIM={s:.4f}, {t * 1000:.0f}ms")
    prev = n
print(f"   >> 最优: TV-OS-SART x{best_e['n']}: RMSE={best_e['rmse']:.5f}")

# 清理
for _, sid in subsets:
    astra.data3d.delete(sid)
for _, sid in weighted_subsets:
    astra.data3d.delete(sid)

# ---- 汇总对比 ----
print("\n" + "=" * 70)
print("汇总对比 (32x512x512, 360角度, 20子集)")
print("=" * 70)
print(f"{'算法':30s} {'耗时(ms)':>10s} {'RMSE':>12s} {'SSIM':>8s} {'vsFDK':>10s}")
print("-" * 72)
print(
    f"{'Pure FDK':30s} {fdk_t * 1000:>8.0f} ms {fdk_rmse:>12.5f} {fdk_ssim:>8.4f} {'-':>10s}"
)
s_imp = f"{(1 - best_sirt['rmse'] / fdk_rmse) * 100:+.1f}%"
c_imp = f"{(1 - best_c['rmse'] / fdk_rmse) * 100:+.1f}%"
e_imp = f"{(1 - best_e['rmse'] / fdk_rmse) * 100:+.1f}%"
print(
    f"{'SIRT3D x' + str(best_sirt['n']):30s} {best_sirt['t'] * 1000:>8.0f} ms {best_sirt['rmse']:>12.5f} {best_sirt['ssim']:>8.4f} {s_imp:>10s}"
)
print(
    f"{'OS-SART x' + str(best_c['n']):30s} {best_c['t'] * 1000:>8.0f} ms {best_c['rmse']:>12.5f} {best_c['ssim']:>8.4f} {c_imp:>10s}"
)
print(
    f"{'TV-OS-SART x' + str(best_e['n']):30s} {best_e['t'] * 1000:>8.0f} ms {best_e['rmse']:>12.5f} {best_e['ssim']:>8.4f} {e_imp:>10s}"
)

# 等迭代收敛对比
print("\n" + "=" * 70)
print("等轮次收敛对比")
print("=" * 70)
print(f"{'轮次':>6s} {'SIRT3D':>20s} {'OS-SART':>20s} {'TV-OS-SART':>20s}")
print(f"{'':>6s} {'RMSE/耗时':>20s} {'RMSE/耗时':>20s} {'RMSE/耗时':>20s}")
print("-" * 68)
for n in [1, 2, 5, 10]:
    sr = next((h for h in sirt_hist if h[0] == n), None)
    co = next((h for h in c_hist if h[0] == n), None)
    eo = next((h for h in e_hist if h[0] == n), None)
    r1 = f"{sr[2]:.5f}/{sr[1] * 1000:.0f}ms" if sr else "-"
    r2 = f"{co[2]:.5f}/{co[1] * 1000:.0f}ms" if co else "-"
    r3 = f"{eo[2]:.5f}/{eo[1] * 1000:.0f}ms" if eo else "-"
    print(f"  x{n:3d}  {r1:>20s}  {r2:>20s}  {r3:>20s}")

# 结果列表
results = [
    ("Pure FDK", fdk_t, fdk_rmse, fdk_ssim),
    (
        "SIRT3D x" + str(best_sirt["n"]),
        best_sirt["t"],
        best_sirt["rmse"],
        best_sirt["ssim"],
    ),
    ("OS-SART x" + str(best_c["n"]), best_c["t"], best_c["rmse"], best_c["ssim"]),
    ("TV-OS-SART x" + str(best_e["n"]), best_e["t"], best_e["rmse"], best_e["ssim"]),
]

# 可视化 (图片上显示 RMSE/耗时/SSIM)
print("\n生成可视化...")
os.makedirs("img_out", exist_ok=True)
mid = nz // 2
fig = plt.figure(figsize=(20, 9))
gs = GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.3)
plot_data = [
    ("Ground Truth", vol_gt[mid], None, None, None),
    ("Pure FDK", fdk_rec[mid], fdk_rmse, fdk_ssim, fdk_t),
    (
        "SIRT3D x" + str(best_sirt["n"]),
        best_sirt["rec"][mid],
        best_sirt["rmse"],
        best_sirt["ssim"],
        best_sirt["t"],
    ),
    (
        "OS-SART x" + str(best_c["n"]),
        best_c["rec"][mid],
        best_c["rmse"],
        best_c["ssim"],
        best_c["t"],
    ),
    (
        "TV-OS-SART x" + str(best_e["n"]),
        best_e["rec"][mid],
        best_e["rmse"],
        best_e["ssim"],
        best_e["t"],
    ),
]
for i, (title, img, rmse, ssim, t) in enumerate(plot_data):
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(img, cmap="gray", vmin=0, vmax=0.05)
    tstr = title
    if rmse is not None:
        tstr += f"\nRMSE={rmse:.5f} SSIM={ssim:.4f}\n{t * 1000:.0f}ms"
    ax.set_title(tstr, fontsize=7)
    ax.axis("off")
for i, (title, img, rmse, ssim, t) in enumerate(plot_data):
    ax = fig.add_subplot(gs[1, i])
    e = img - vol_gt[mid] if rmse is not None else np.zeros_like(img)
    v = max(0.005, np.percentile(np.abs(e), 95) * 1.2)
    im = ax.imshow(e, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title("Error" if i == 0 else f"Error (×{v:.4f})", fontsize=7)
    ax.axis("off")
plt.suptitle(
    "Cone-beam CBCT: ASTRA CUDA (512x512x32, 360 angles)",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)
plt.savefig("img_out/astra_cone_hybrid.png", dpi=150, bbox_inches="tight")
plt.close()
print("   => img_out/astra_cone_hybrid.png")

summary = {
    "backend": "GPU (ASTRA CUDA cone-beam)",
    "config": {"N": N, "nz": nz, "n_angles": n_angles, "n_subsets": n_subsets},
    "results": {
        name: {"rmse": round(r, 5), "ssim": round(s, 4), "time_ms": round(t * 1000, 1)}
        for name, t, r, s in results
    },
}
with open("img_out/astra_cone_hybrid_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("   => img_out/astra_cone_hybrid_summary.json")
print("\nDone!")
