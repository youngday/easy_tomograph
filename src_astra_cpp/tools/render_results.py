"""
将 C++ 重建结果渲染为与 Python 版相同布局的对比图
读取 img_{axial,helical}/astra_cpp/ 下的 .raw 结果 + cpp_summary.json

用法: python tools/render_results.py [axial|helical]  (默认两者都渲染)
"""
import json
import os
import sys

import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

N, nz = 512, 32
VOX = np.prod((nz, N, N))


def load_vol(path):
    a = np.fromfile(path, dtype=np.float32)
    if a.size != VOX:
        raise ValueError(f"{path}: 大小不匹配 {a.size} != {VOX}")
    return a.reshape(nz, N, N)


def render(mode):
    sub = "3d_axial" if mode == "axial" else "3d_helical"
    d = f"img_{sub}/astra_cpp"
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
    tv = load_vol(os.path.join(d, "cpp_tv.raw"))
    hyb = load_vol(os.path.join(d, "cpp_hybrid.raw"))

    Yg, Xg = np.ogrid[:N, :N]
    dist = np.sqrt((Xg - N / 2) ** 2 + (Yg - N / 2) ** 2)
    soft = np.clip((N * 0.42 + 20 - dist) / 20, 0, 1)

    mid = nz // 2
    tv_key = max((k for k in res if k.startswith("TV-OS-SART")), key=lambda s: int(s.split("x")[1]))
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.3)
    panels = [
        ("Ground Truth", gt, None, None, None),
        ("FDK", fdk, res["Pure FDK"], None, None),
        ("Hybrid IR\nOS10+TV10(β↓)+FDK10%", hyb, res["Hybrid IR"], None, None),
        ("TV-OS-SART", tv, res[tv_key], None, None),
    ]

    for i, (title, img, r, _u, _v) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img[mid] * soft, cmap="gray", vmin=0, vmax=0.05)
        tstr = title if r is None else f"{title}\nRMSE={r['rmse']:.5f}  SSIM={r['ssim']:.4f}\n{r['time_ms']:.0f}ms"
        ax.set_title(tstr, fontsize=8)
        ax.axis("off")
        ax2 = fig.add_subplot(gs[1, i])
        if r is not None and gt is not None:
            e = img[mid] - gt[mid]
            v = max(0.005, np.percentile(np.abs(e), 95) * 1.2)
            ax2.imshow(e * soft, cmap="RdBu_r", vmin=-v, vmax=v)
            ax2.set_title(f"Error  x{v:.4f}", fontsize=8)
        else:
            ax2.imshow(np.zeros_like(img[mid]), cmap="gray")
            ax2.set_title("Reference", fontsize=8)
        ax2.axis("off")

    zp = summary.get("z_profile", {})
    axz = fig.add_subplot(gs[2, :])
    colors = ["orange", "purple", "red"]
    labels = ["FDK", "Hybrid IR", "TV-OS-SART"]
    for name, col in zip(labels, colors):
        if name in zp:
            axz.plot(np.arange(nz), zp[name], "o-", label=name, color=col, markersize=3)
    axz.set_xlabel("z slice", fontsize=9)
    axz.set_ylabel("RMSE per slice", fontsize=9)
    axz.legend(fontsize=8)
    axz.set_title("z-profile: 沿 z 方向逐片 RMSE", fontsize=10)
    axz.grid(True, alpha=0.3)

    plt.suptitle(f"ASTRA CUDA {'Helical' if mode == 'helical' else 'Cone'}-beam (C++) (32x512x512, 180角度, 10子集)\n+ Hybrid IR (OS-SART×10+TV×10+FDK混合10%)",
                 fontsize=12, fontweight="bold", y=0.98)
    out_png = os.path.join(d, "cpp_summary.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   => {out_png}")


if __name__ == "__main__":
    modes = sys.argv[1:] if len(sys.argv) > 1 else ["axial", "helical"]
    for m in modes:
        render(m)
    print("Done!")
