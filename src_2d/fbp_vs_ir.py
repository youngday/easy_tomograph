"""
GPU FBP vs IR 对比 (Plotly + PNG)
=================================
FBP 组: FBP_CUDA + 自实现+GPU (3种滤波器)
IR 组:  CGLS (最优迭代)
"""

from time import time

import astra
import matplotlib.pyplot as plt
import numpy as np
import tomophantom
from skimage.transform import radon
from tomophantom import TomoP2D

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

N = 512
n_angles = 360
print(f"体模: {N}x{N}, 角度: {n_angles}")

# TomoPhantom Model 4 - QRM 多椭圆体模
tp_lib = os.path.join(
    os.path.dirname(tomophantom.__file__), "phantomlib", "Phantom2DLibrary.dat"
)
ph = TomoP2D.Model(4, N, tp_lib)
ct = (ph - 0.65) * 2000 / 0.65
ct = ct.astype(np.float32)
Y, X = np.ogrid[:N, :N]
head_r = 235
circ_mask = (X - N / 2) ** 2 + (Y - N / 2) ** 2 <= head_r**2
ct[~circ_mask] = -1000

theta_deg = np.linspace(0, 180, n_angles, endpoint=False)
theta_rad = np.deg2rad(theta_deg).astype(np.float32)
sino = radon(ct, theta=theta_deg, circle=False)
D = sino.shape[0]

proj_geom = astra.create_proj_geom("parallel", 1.0, D, theta_rad)
vol_geom = astra.create_vol_geom(N, N)


def linear_scale(rec):
    rec_clip = np.clip(rec, -5000, 5000)
    mask = circ_mask
    A = np.column_stack([rec_clip.ravel()[mask.ravel()], np.ones(mask.sum())])
    b = ct.ravel()[mask.ravel()]
    coef, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return rec * coef[0] + coef[1]


def calc_rmse(rec):
    return np.sqrt(np.mean((ct[circ_mask] - rec[circ_mask]) ** 2))


# ====================================
# FBP 组
# ====================================
results = []

print("\n=== FBP 组 ===")
fbp_data = []

# GPU 预热 (消除首次调用初始化开销)
print("   GPU 预热...")
sino = np.ascontiguousarray(sino.T)

sid = astra.data2d.create("-sino", proj_geom, sino)
rid = astra.data2d.create("-vol", vol_geom)
cfg = astra.astra_dict("FBP_CUDA")
cfg["ProjectionDataId"] = sid
cfg["ReconstructionDataId"] = rid
cfg["option"] = {"FilterType": "ram-lak"}
aid = astra.algorithm.create(cfg)
astra.algorithm.run(aid)
astra.algorithm.delete(aid)
astra.data2d.delete(rid)
astra.data2d.delete(sid)

# FBP_CUDA
for filt, label in [
    ("ram-lak", "FBP_CUDA ram-lak(骨)"),
    ("shepp-logan", "FBP_CUDA shepp-logan(标准)"),
    ("hamming", "FBP_CUDA hamming(平滑)"),
]:
    t0 = time()
    sid = astra.data2d.create("-sino", proj_geom, sino)
    rid = astra.data2d.create("-vol", vol_geom)
    cfg = astra.astra_dict("FBP_CUDA")
    cfg["ProjectionDataId"] = sid
    cfg["ReconstructionDataId"] = rid
    cfg["option"] = {"FilterType": filt}
    aid = astra.algorithm.create(cfg)
    astra.algorithm.run(aid)
    rec = linear_scale(astra.data2d.get(rid))
    t = time() - t0
    r = calc_rmse(rec)
    fbp_data.append((label, t, r, rec))
    astra.algorithm.delete(aid)
    astra.data2d.delete(rid)
    astra.data2d.delete(sid)
    results.append((label, t, r, rec))
    print(f"   {label}: RMSE={r:.2f}, {t * 1000:.0f}ms")

# ====================================
# IR 组 (CGLS)
# ====================================
print("\n=== IR 组 (CGLS) ===")
# GPU 预热
sid = astra.data2d.create("-sino", proj_geom, sino)
rid = astra.data2d.create("-vol", vol_geom)
cfg = astra.astra_dict("CGLS_CUDA")
cfg["ProjectionDataId"] = sid
cfg["ReconstructionDataId"] = rid
cfg["option"] = {"GPUindex": 0}
aid = astra.algorithm.create(cfg)
astra.algorithm.run(aid, 1)
astra.algorithm.delete(aid)
astra.data2d.delete(rid)
astra.data2d.delete(sid)
print("   GPU 预热完成")

sid = astra.data2d.create("-sino", proj_geom, sino)
best_r, best_rec, best_t, best_n = 1e9, None, 0, 0
ir_data = []
for n_iter in [10, 20, 30, 50, 80, 100]:
    rid = astra.data2d.create("-vol", vol_geom)
    cfg = astra.astra_dict("CGLS_CUDA")
    cfg["ProjectionDataId"] = sid
    cfg["ReconstructionDataId"] = rid
    cfg["option"] = {"GPUindex": 0}
    aid = astra.algorithm.create(cfg)
    t0 = time()
    astra.algorithm.run(aid, n_iter)
    rec = linear_scale(astra.data2d.get(rid))
    t = time() - t0
    r = calc_rmse(rec)
    ir_data.append((n_iter, t, r, rec))
    if r < best_r:
        best_r = r
        best_rec = rec
        best_t = t
        best_n = n_iter
    astra.algorithm.delete(aid)
    astra.data2d.delete(rid)
    print(f"   x{n_iter:3d}: RMSE={r:.2f}, {t * 1000:.0f}ms")
astra.data2d.delete(sid)
results.append((f"CGLS x{best_n}", best_t, best_r, best_rec))
print(f"   >> 最优: CGLS x{best_n}: RMSE={best_r:.2f}, {best_t * 1000:.0f}ms")

# ====================================
# IR 组 (SIRT)
# ====================================
print("\n=== IR 组 (SIRT) ===")
sid = astra.data2d.create("-sino", proj_geom, sino)
best_r, best_rec, best_t, best_n = 1e9, None, 0, 0
sirt_data = []
for n_iter in [50, 100, 200, 500]:
    rid = astra.data2d.create("-vol", vol_geom)
    cfg = astra.astra_dict("SIRT_CUDA")
    cfg["ProjectionDataId"] = sid
    cfg["ReconstructionDataId"] = rid
    cfg["option"] = {"GPUindex": 0}
    aid = astra.algorithm.create(cfg)
    t0 = time()
    astra.algorithm.run(aid, n_iter)
    rec = linear_scale(astra.data2d.get(rid))
    t = time() - t0
    r = calc_rmse(rec)
    sirt_data.append((n_iter, t, r, rec))
    if r < best_r:
        best_r = r
        best_rec = rec
        best_t = t
        best_n = n_iter
    astra.algorithm.delete(aid)
    astra.data2d.delete(rid)
    print(f"   x{n_iter:4d}: RMSE={r:.2f}, {t * 1000:.0f}ms")
astra.data2d.delete(sid)
results.append((f"SIRT x{best_n}", best_t, best_r, best_rec))
print(f"   >> 最优: SIRT x{best_n}: RMSE={best_r:.2f}, {best_t * 1000:.0f}ms")

# ========== 结果 =========="
print("\n" + "=" * 55)
print(f"{'组别':>4s} {'算法':35s} {'耗时(ms)':>10s} {'RMSE':>10s}")
print("-" * 55)
for name, t, r, _ in results:
    grp = "FBP" if "FBP" in name else "IR"
    print(f"{grp:4s} {name:35s} {t * 1000:>8.0f} ms {r:>8.2f}")

# ========== Plotly 可视化 ==========
print("\n生成图表...")
os.makedirs("img_out", exist_ok=True)

gray_scale = [[0, "rgb(0,0,0)"], [1, "rgb(255,255,255)"]]
rdbu_scale = [[0, "rgb(103,0,31)"], [0.5, "rgb(255,255,255)"], [1, "rgb(0,0,100)"]]

# ===== 图1: FBP 组 =====
n_fbp = len(fbp_data)
fig1 = make_subplots(
    rows=2,
    cols=n_fbp,
    subplot_titles=[
        f"{fbp_data[i][0]}<br>RMSE={fbp_data[i][2]:.1f}<br>{fbp_data[i][1] * 1000:.0f}ms"
        for i in range(n_fbp)
    ],
    horizontal_spacing=0.01,
    vertical_spacing=0.12,
)

for i in range(n_fbp):
    name, t, r, rec = fbp_data[i]
    diff = rec - ct
    diff[~circ_mask] = 0
    md = np.abs(diff[circ_mask]).max()
    vmax = max(30, np.percentile(np.abs(diff[circ_mask]), 95) * 1.2)
    print(f"       FBP error vmax={vmax:.1f} ({name})")
    fig1.add_trace(
        go.Heatmap(z=rec, colorscale=gray_scale, zmin=-200, zmax=600, showscale=False),
        row=1,
        col=i + 1,
    )
    fig1.add_trace(
        go.Heatmap(
            z=diff, colorscale=rdbu_scale, zmin=-vmax, zmax=vmax, showscale=False
        ),
        row=2,
        col=i + 1,
    )
    fig1.update_xaxes(visible=False, row=1, col=i + 1)
    fig1.update_yaxes(visible=False, row=1, col=i + 1, autorange="reversed")
    fig1.update_xaxes(visible=False, row=2, col=i + 1)
    fig1.update_yaxes(visible=False, row=2, col=i + 1, autorange="reversed")

fig1.update_layout(
    title=dict(
        text=f"GPU FBP Group - {N}x{N} / {n_angles} angles", y=0.95, automargin=True
    ),
    height=750,
    width=280 * n_fbp,
    showlegend=False,
    margin=dict(t=120, b=40),
)
fig1.write_html("img_out/group_fbp.html")
print(f"   ✅ group_fbp.html")

# matplotlib 快速出 PNG
fig1_mpl, axes1 = plt.subplots(2, n_fbp, figsize=(4 * n_fbp, 8))
for i in range(n_fbp):
    name, t, r, rec = fbp_data[i]
    diff = rec - ct
    diff[~circ_mask] = 0
    md = np.abs(diff[circ_mask]).max()
    vmax = max(30, np.percentile(np.abs(diff[circ_mask]), 95) * 1.2)
    axes1[0, i].imshow(rec, cmap="gray", vmin=-200, vmax=600)
    axes1[1, i].imshow(diff, cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes1[0, i].axis("off")
    axes1[1, i].axis("off")
    axes1[0, i].set_title(f"{name}\nRMSE={r:.1f}\n{t * 1000:.0f}ms", fontsize=9)
fig1_mpl.suptitle(f"GPU FBP Group - {N}x{N} / {n_angles} angles", y=0.98)
plt.tight_layout()
plt.savefig("img_out/group_fbp.png", dpi=150, bbox_inches="tight")
plt.close(fig1_mpl)
print(f"   ✅ group_fbp.png (matplotlib)")

# ===== 图2: IR 组 (CGLS + SIRT) =====
fig2 = make_subplots(
    rows=2,
    cols=4,
    subplot_titles=["IR Convergence", "CGLS Recon", "CGLS Error", "SIRT Error"],
    column_widths=[0.35, 0.25, 0.2, 0.2],
    horizontal_spacing=0.08,
    vertical_spacing=0.12,
)

# 收敛曲线 (CGLS + SIRT)
fig2.add_trace(
    go.Scatter(
        x=[d[0] for d in ir_data],
        y=[d[2] for d in ir_data],
        mode="lines+markers",
        name="CGLS",
        line=dict(color="purple", width=2),
        marker=dict(size=8),
    ),
    row=1,
    col=1,
)
fig2.add_trace(
    go.Scatter(
        x=[d[0] for d in sirt_data],
        y=[d[2] for d in sirt_data],
        mode="lines+markers",
        name="SIRT",
        line=dict(color="orange", width=2),
        marker=dict(size=8),
    ),
    row=1,
    col=1,
)
fig2.update_xaxes(title_text="Iterations", row=1, col=1)
fig2.update_yaxes(title_text="RMSE", row=1, col=1)

# CGLS 最优重建
best_c = min(ir_data, key=lambda x: x[2])
fig2.add_trace(
    go.Heatmap(
        z=best_c[3], colorscale=gray_scale, zmin=-200, zmax=600, showscale=False
    ),
    row=1,
    col=2,
)
fig2.update_xaxes(visible=False, row=1, col=2)
fig2.update_yaxes(visible=False, row=1, col=2, autorange="reversed")

# CGLS 误差
best_c_diff = best_c[3] - ct
best_c_diff[~circ_mask] = 0
vmax_c = max(30, np.percentile(np.abs(best_c_diff[circ_mask]), 95) * 1.2)
print(f"       CGLS best error vmax={vmax_c:.1f}")
fig2.add_trace(
    go.Heatmap(
        z=best_c_diff, colorscale=rdbu_scale, zmin=-vmax_c, zmax=vmax_c, showscale=False
    ),
    row=1,
    col=3,
)
fig2.update_xaxes(visible=False, row=1, col=3)
fig2.update_yaxes(visible=False, row=1, col=3, autorange="reversed")

# SIRT 误差
best_s = min(sirt_data, key=lambda x: x[2])
best_s_diff = best_s[3] - ct
best_s_diff[~circ_mask] = 0
vmax_s = max(30, np.percentile(np.abs(best_s_diff[circ_mask]), 95) * 1.2)
print(f"       SIRT best error vmax={vmax_s:.1f}")
fig2.add_trace(
    go.Heatmap(
        z=best_s_diff, colorscale=rdbu_scale, zmin=-vmax_s, zmax=vmax_s, showscale=False
    ),
    row=1,
    col=4,
)
fig2.update_xaxes(visible=False, row=1, col=4)
fig2.update_yaxes(visible=False, row=1, col=4, autorange="reversed")

# 第二行: CGLS 不同迭代误差
for col, ni in enumerate([10, 30, 100]):
    for d in ir_data:
        if d[0] == ni:
            diff = d[3] - ct
            diff[~circ_mask] = 0
            vmax_d = max(30, np.percentile(np.abs(diff[circ_mask]), 95) * 1.2)
            print(f"       CGLS x{ni} error vmax={vmax_d:.1f}")
            fig2.add_trace(
                go.Heatmap(
                    z=diff,
                    colorscale=rdbu_scale,
                    zmin=-vmax_d,
                    zmax=vmax_d,
                    showscale=False,
                    name=f"CGLS x{ni} err",
                ),
                row=2,
                col=col + 1,
            )
            fig2.update_xaxes(visible=False, row=2, col=col + 1)
            fig2.update_yaxes(visible=False, row=2, col=col + 1, autorange="reversed")
            fig2.add_annotation(
                text=f"CGLS x{ni} RMSE={d[2]:.2f}",
                x=0.5,
                y=-0.15,
                showarrow=False,
                font=dict(size=10),
                xref=f"x{col + 5}",
                yref="y5",
                bgcolor="rgba(255,255,255,0.7)",
            )
            break

fig2.update_layout(
    title=dict(text="IR Group: CGLS vs SIRT Convergence + Error Maps", y=0.98),
    height=700,
    width=1100,
    showlegend=True,
    margin=dict(t=80, b=40),
    legend=dict(x=0.3, y=1.12, orientation="h"),
)
fig2.write_html("img_out/group_ir.html")
print(f"   ✅ group_ir.html")

# matplotlib 快速出 PNG
fig2_mpl = plt.figure(figsize=(12, 7))
gs = fig2_mpl.add_gridspec(
    2, 4, width_ratios=[0.35, 0.25, 0.2, 0.2], hspace=0.3, wspace=0.3
)

# 收敛曲线
ax_curve = fig2_mpl.add_subplot(gs[0, 0])
ax_curve.plot(
    [d[0] for d in ir_data],
    [d[2] for d in ir_data],
    "o-",
    color="purple",
    lw=2,
    label="CGLS",
)
ax_curve.plot(
    [d[0] for d in sirt_data],
    [d[2] for d in sirt_data],
    "s-",
    color="orange",
    lw=2,
    label="SIRT",
)
ax_curve.set_xlabel("Iterations")
ax_curve.set_ylabel("RMSE")
ax_curve.legend()
ax_curve.grid(True, alpha=0.3)
ax_curve.set_title("IR Convergence")

# CGLS 最优重建
best_c = min(ir_data, key=lambda x: x[2])
ax_cgls = fig2_mpl.add_subplot(gs[0, 1])
ax_cgls.imshow(best_c[3], cmap="gray", vmin=-200, vmax=600)
ax_cgls.axis("off")
ax_cgls.set_title(f"CGLS x{best_c[0]} Recon")

# CGLS 误差
best_c_diff = best_c[3] - ct
best_c_diff[~circ_mask] = 0
vmax_c = max(30, np.percentile(np.abs(best_c_diff[circ_mask]), 95) * 1.2)
ax_cerr = fig2_mpl.add_subplot(gs[0, 2])
im = ax_cerr.imshow(best_c_diff, cmap="RdBu", vmin=-vmax_c, vmax=vmax_c)
ax_cerr.axis("off")
ax_cerr.set_title(f"CGLS Error")

# SIRT 误差
best_s = min(sirt_data, key=lambda x: x[2])
best_s_diff = best_s[3] - ct
best_s_diff[~circ_mask] = 0
vmax_s = max(30, np.percentile(np.abs(best_s_diff[circ_mask]), 95) * 1.2)
ax_serr = fig2_mpl.add_subplot(gs[0, 3])
ax_serr.imshow(best_s_diff, cmap="RdBu", vmin=-vmax_s, vmax=vmax_s)
ax_serr.axis("off")
ax_serr.set_title(f"SIRT Error")

# 第二行: CGLS 不同迭代误差
for col, ni in enumerate([10, 30, 100]):
    for d in ir_data:
        if d[0] == ni:
            diff = d[3] - ct
            diff[~circ_mask] = 0
            vmax_d = max(30, np.percentile(np.abs(diff[circ_mask]), 95) * 1.2)
            ax = fig2_mpl.add_subplot(gs[1, col])
            ax.imshow(diff, cmap="RdBu", vmin=-vmax_d, vmax=vmax_d)
            ax.axis("off")
            ax.set_title(f"CGLS x{ni}\nRMSE={d[2]:.2f}", fontsize=10)
            break

fig2_mpl.suptitle(
    "IR Group: CGLS vs SIRT Convergence + Error Maps", y=0.98, fontsize=13
)
plt.savefig("img_out/group_ir.png", dpi=150, bbox_inches="tight")
plt.close(fig2_mpl)
print(f"   ✅ group_ir.png (matplotlib)")

# ===== 图3: 散点图 =====
fig3 = go.Figure()
color_map = {"FBP_CUDA": "red", "CGLS": "purple", "SIRT": "orange"}
for name, t, r, _ in results:
    cat = "FBP_CUDA" if "FBP_CUDA" in name else ("CGLS" if "CGLS" in name else "SIRT")
    c = color_map.get(cat, "blue")
    fig3.add_trace(
        go.Scatter(
            x=[t * 1000],
            y=[r],
            mode="markers+text",
            name=name,
            text=[name[:18]],
            textposition="top center",
            marker=dict(size=14, color=c, symbol="triangle-up"),
            hovertemplate=f"{name}<br>{t * 1000:.0f}ms<br>RMSE={r:.2f}<extra></extra>",
        )
    )
fig3.update_layout(
    title=dict(text="Speed vs Quality: FBP vs IR", y=0.95),
    xaxis_type="log",
    xaxis_title="Time (ms)",
    yaxis_title="RMSE",
    height=500,
    width=800,
    margin=dict(t=60, b=40),
)
fig3.write_html("img_out/group_scatter.html")
print(f"   ✅ group_scatter.html")

# matplotlib 快速出 PNG
fig3_mpl, ax3 = plt.subplots(figsize=(8, 5))
color_map = {"FBP_CUDA": "red", "CGLS": "purple", "SIRT": "orange"}
marker_map = {"FBP_CUDA": "^", "CGLS": "o", "SIRT": "s"}
for name, t, r, _ in results:
    cat = "FBP_CUDA" if "FBP_CUDA" in name else ("CGLS" if "CGLS" in name else "SIRT")
    ax3.scatter(t * 1000, r, c=color_map[cat], marker=marker_map[cat], s=120, zorder=5)
    ax3.annotate(
        name[:20],
        (t * 1000, r),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
    )
ax3.set_xscale("log")
ax3.set_xlabel("Time (ms)")
ax3.set_ylabel("RMSE")
ax3.set_title("Speed vs Quality: FBP vs IR")
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("img_out/group_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig3_mpl)
print(f"   ✅ group_scatter.png (matplotlib)")

print("\n全部完成 (HTML 浏览器打开)")
