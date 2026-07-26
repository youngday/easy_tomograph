"""
批量查看 DICOM 文件
用法:
    python batch_view.py <目录>           # 列出目录下所有 DICOM 文件
    python batch_view.py <目录> --show    # 逐张显示所有 DICOM 图像
    python batch_view.py <目录> --info    # 显示所有文件的元数据摘要
    python batch_view.py <目录> --save    # 批量导出为 PNG
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pydicom


def find_dicom_files(directory):
    """递归查找目录下所有 DICOM 文件"""
    dicom_files = []
    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            fp = os.path.join(root, f)
            try:
                with open(fp, "rb") as fh:
                    header = fh.read(132)
                    if header[128:132] == b"DICM":
                        dicom_files.append(fp)
            except Exception:
                pass
    return dicom_files


def batch_list(directory):
    """列出目录下所有 DICOM 文件的基本信息"""
    files = find_dicom_files(directory)

    if not files:
        print(f"⚠ 目录中未找到 DICOM 文件: {directory}")
        return

    print(f"\n📂 {directory}  共 {len(files)} 个 DICOM 文件\n")
    print(f"{'#':>4}  {'文件名':40s} {'模态':6s} {'尺寸':12s} {'描述'}")
    print("-" * 90)

    for i, fp in enumerate(files):
        try:
            ds = pydicom.dcmread(fp, force=False)
            mod = ds.get((0x0008, 0x0060))
            mod = mod.value if mod else "?"
            desc = ds.get((0x0008, 0x103E))
            desc = desc.value if desc else ""
            rows = ds.get((0x0028, 0x0010))
            cols = ds.get((0x0028, 0x0011))
            size_str = f"{rows.value}x{cols.value}" if rows and cols else "?"
            fname = os.path.relpath(fp, directory)
        except Exception:
            mod, size_str, desc = "?", "?", "(read error)"
            fname = os.path.relpath(fp, directory)

        print(f"{i:4d}  {fname:40s} {mod:6s} {size_str:12s} {desc}")


def batch_info(directory):
    """显示目录下所有 DICOM 文件的元数据摘要"""
    files = find_dicom_files(directory)

    if not files:
        print(f"⚠ 未找到 DICOM 文件: {directory}")
        return

    print(f"\n📋 DICOM 元数据摘要  ({len(files)} 个文件)\n")

    keys = [
        (0x0008, 0x0060, "Modality"),
        (0x0008, 0x0070, "Manufacturer"),
        (0x0008, 0x103E, "SeriesDesc"),
        (0x0010, 0x0010, "PatientName"),
        (0x0010, 0x0020, "PatientID"),
        (0x0020, 0x000D, "StudyUID"),
        (0x0020, 0x000E, "SeriesUID"),
        (0x0028, 0x0010, "Rows"),
        (0x0028, 0x0011, "Cols"),
        (0x0028, 0x0100, "BitsAlloc"),
        (0x0028, 0x1050, "WinCenter"),
        (0x0028, 0x1051, "WinWidth"),
    ]

    # 表头
    header = f"{'文件':30s}"
    for _, _, name in keys:
        header += f"{name:15s}"
    print(header)
    print("-" * (30 + 15 * len(keys)))

    for fp in files:
        try:
            ds = pydicom.dcmread(fp, force=False)
            line = f"{os.path.basename(fp):30s}"
            for group, element, _ in keys:
                elem = ds.get((group, element))
                val = str(elem.value) if elem else "-"
                if len(val) > 14:
                    val = val[:11] + "..."
                line += f"{val:15s}"
            print(line)
        except Exception:
            print(f"{os.path.basename(fp):30s}  (read error)")


def batch_show(directory):
    """逐张显示目录下所有 DICOM 图像"""
    files = find_dicom_files(directory)

    if not files:
        print(f"⚠ 未找到 DICOM 文件: {directory}")
        return

    print(f"📸 逐张显示 {len(files)} 个 DICOM 文件 (关闭当前窗口显示下一张)")

    for fp in files:
        try:
            ds = pydicom.dcmread(fp, force=True)
            img = ds.pixel_array
            fname = os.path.basename(fp)

            plt.figure(figsize=(10, 4))
            plt.subplot(121)
            plt.imshow(img, cmap="gray")
            plt.title(f"{fname}\n{ds.SeriesDescription}")
            plt.axis("off")
            plt.colorbar(fraction=0.046)

            plt.subplot(122)
            plt.hist(img.ravel(), bins=256, color="steelblue")
            plt.title("Histogram")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")
            plt.grid(alpha=0.3)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"⚠ 无法显示 {fp}: {e}")


def batch_save(directory, out_dir="dicom_png"):
    """批量导出 DICOM 为 PNG"""
    files = find_dicom_files(directory)

    if not files:
        print(f"⚠ 未找到 DICOM 文件: {directory}")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"💾 批量导出 {len(files)} 个 DICOM → {out_dir}/")

    for fp in files:
        try:
            ds = pydicom.dcmread(fp, force=True)
            img = ds.pixel_array
            fname = os.path.splitext(os.path.basename(fp))[0] + ".png"
            out_path = os.path.join(out_dir, fname)

            # 归一化到 0-255
            img_norm = (
                (img - img.min()) / (img.max() - img.min() + 1e-8) * 255
            ).astype(np.uint8)

            plt.imsave(out_path, img_norm, cmap="gray")
            print(f"   ✅ {fname}")
        except Exception as e:
            print(f"   ⚠ {os.path.basename(fp)}: {e}")

    print(f"\n✅ 共导出 {len(files)} 张图像到 {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="批量查看 DICOM 文件")
    parser.add_argument(
        "directory", nargs="?", default=".", help="DICOM 文件目录 (默认: 当前目录)"
    )
    parser.add_argument("--show", action="store_true", help="逐张显示所有 DICOM 图像")
    parser.add_argument("--info", action="store_true", help="显示元数据摘要表格")
    parser.add_argument(
        "--save", nargs="?", const="dicom_png", help="批量导出为 PNG (可指定输出目录)"
    )
    parser.add_argument(
        "--file", type=str, default=None, help="指定单个文件 (用于 --show)"
    )

    args = parser.parse_args()

    if args.file:
        # 查看单个文件
        from read_dicom import read_and_show

        read_and_show(args.file)
    elif args.info:
        batch_info(args.directory)
    elif args.save:
        batch_save(args.directory, args.save)
    elif args.show:
        batch_show(args.directory)
    else:
        # 默认: 列出文件
        batch_list(args.directory)

        print()
        print("---")
        print("其他用法:")
        print("  --show    逐张显示图像")
        print("  --info    元数据摘要")
        print("  --save    批量导出 PNG")
        print("  --file    查看单个文件")


if __name__ == "__main__":
    main()
