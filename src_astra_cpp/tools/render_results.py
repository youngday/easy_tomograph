"""
将 C++ 重建结果渲染为与 Python 版完全同款的结果图
(对齐 src_3d_{axial,helical}/astra_cone_hybrid.py 的绘图代码):
  * 3x4 布局: GT / FDK / Hybrid / TV-OS-SART 中片 (软遮罩, gray, vmin=0 vmax=0.05)
  * 第 2 行: RdBu_r 误差图 (v = max(0.005, p95*1.2))
  * 第 3 行: z-profile (FDK orange / Hybrid purple / TV red)
  * suptitle 含时间戳, 输出文件名与 Python 版相同 (astra_cone_hybrid.png)

用法: python tools/render_results.py [axial|helical ...] [outdir 可选]
   默认渲染 axial 与 helical (C++ 二进制会在跑完后自动调用)
"""
import json
import os
import sys
from time import localtime, strftime

import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

N, nz = 512, 32
n_angles, n_subsets = 180, 10
VOX = np.prod((nz, N, N))


def load_vol(path):
    a = np.fromfile(path, dtype=np.float32)
    if a.size != VOX:
        raise ValueError(f"{path}: 大小不匹配 {a.size} != {VOX}")
    return a.reshape(nz, N, N)


def render(mode, outdir=None):
    sub = "3d_axial" if mode == "axial" else "3d_helical"
    d = outdir or f"img_{sub}/astra_cpp"
    if not os.path.isdir(d):
        print(f"跳过 {mode}: 目录不存在 {d}")
        return
    summary = json.load(open(os.path.join(d, "cpp_summary.json")))
    res = summary["results"]

    gt = None
    for p in [os.path.join(d, "cpp_gt.raw"), "src_astra_cpp/data/vol_gt.raw"]:
        if os.path.exists(p):
            gt = load_vol(p)
            break
    fdk = load_vol(os.path.join(d, "cpp_fdk.raw"))
    fdk_n = load_vol(os.path.join(d, "cpp_fdk_noisy.raw"))
    tv = load_vol(os.path.join(d, "cpp_tv.raw"))
    hyb = load_vol(os.path.join(d, "cpp_hybrid.raw"))

    # 软遮罩 (与 Python 版一致)
    Ygrid, Xgrid = np.ogrid[:N, :N]
    dist_xy = np.sqrt((Xgrid - N / 2) ** 2 + (Ygrid - N / 2) ** 2)
    body_r = N * 0.42
    soft_mask_2d = np.clip((body_r + 20 - dist_xy) / 20, 0, 1)

    mid = nz // 2
    ts = strftime("%Y-%m-%d %H:%M:%S", localtime())

    tv_key = max((k for k in res if k.startswith("TV-OS-SART")), key=lambda s: int(s.split("x")[1]))
    tv_n = int(tv_key.split("x")[1])
    cfg = summary.get("config", {})
    ep = cfg.get("epochs", {})
    n_tv = ep.get("tv_ossart", tv_n)
    n_hyb = ep.get("hybrid", 10)

    # ---- 与 Python 版 titles_upper 相同的面板 (5 列) ----
    fig = plt.figure(figsize=(30, 12) if mode == "axial" else (35, 12))
    gs = GridSpec(3, 5, figure=fig, hspace=0.45, wspace=0.3)
    panels = [
        ("Ground Truth", gt[mid] if gt is not None else np.zeros((N, N), dtype=np.float32), None, None),
        ("FDK", fdk[mid], res["Pure FDK"], None),
        ("FDK(noisy)", fdk_n[mid], res["FDK(noisy)"], None),
        (f"Hybrid IR\nOS{n_hyb}+TV{n_hyb}(β↓)+FDK10%", hyb[mid], res["Hybrid IR"], None),
        ("TV-OS-SART", tv[mid], res[tv_key], tv_n),
    ]
    for i, (title, img, r, ni) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img * soft_mask_2d, cmap="gray", vmin=0, vmax=0.05)
        if r is None:
            tstr = title
        else:
            tag = f" x{ni}" if ni else ""
            tstr = f"{title}{tag}\nRMSE={r['rmse']:.5f}  SSIM={r['ssim']:.4f}\n{r['time_ms']:.0f}ms"
        ax.set_title(tstr, fontsize=8)
        ax.axis("off")
        # 误差图
        ax2 = fig.add_subplot(gs[1, i])
        if r is not None and gt is not None:
            e = img - gt[mid]
            v = max(0.005, np.percentile(np.abs(e), 95) * 1.2)
            ax2.imshow(e * soft_mask_2d, cmap="RdBu_r", vmin=-v, vmax=v)
            ax2.set_title(f"Error  x{v:.4f}", fontsize=8)
        else:
            ax2.imshow(np.zeros_like(img), cmap="gray")
            ax2.set_title("Reference", fontsize=8)
        ax2.axis("off")

    # ---- z-profile (第 3 行) ----
    zp = summary.get("z_profile", {})
    z_coord = np.arange(nz)
    ax_z = fig.add_subplot(gs[2, :])
    for name, color in zip(["FDK", "FDK(noisy)", "Hybrid IR", "TV-OS-SART"],
                           ["orange", "brown", "purple", "red"]):
        key = name if name in zp else (tv_key if name == "TV-OS-SART" else None)
        if key and key in zp:
            ax_z.plot(z_coord, zp[key], "o-", label=name, color=color, markersize=3)
    ax_z.set_xlabel("z slice", fontsize=9)
    ax_z.set_ylabel("RMSE per slice", fontsize=9)
    ax_z.legend(fontsize=8)
    ax_z.set_title("z-profile: 沿 z 方向逐片 RMSE", fontsize=10)
    ax_z.grid(True, alpha=0.3)

    # ---- suptitle (含实际迭代轮数 + 时间戳) ----
    tr = cfg.get("target_rmse", 0.001)
    if mode == "axial":
        st = (f"ASTRA CUDA Cone-beam (32x512x512, {n_angles}角度, {n_subsets}子集)\n"
              f"+ Hybrid IR (OS-SART×{n_tv}+TV×{n_tv}+FDK混合10%, RMSE≤{tr:.3f}提前停止)\n{ts}")
    else:
        st = (f"ASTRA CUDA Helical Cone-beam (32x512x512, {n_angles}角度, pitch=16.0mm, {n_subsets}子集)\n"
              f"+ Hybrid IR (OS-SART×{n_tv}+TV×{n_tv}+FDK混合10%, RMSE≤{tr:.3f}提前停止)\n{ts}")
    plt.suptitle(st, fontsize=12, fontweight="bold", y=0.98)

    out_png = os.path.join(d, "astra_cone_hybrid.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   => {out_png}")


if __name__ == "__main__":
    args = sys.argv[1:]
    modes = [a for a in args if a in ("axial", "helical")] or ["axial", "helical"]
    outdir = next((a for a in args if a not in ("axial", "helical")), None)
    for m in modes:
        render(m, outdir)
    print("Done!")
