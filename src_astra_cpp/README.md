# src_astra_cpp — ASTRA 锥束混合重建 (C++ 版)

将 Python 版 `src_3d_axial/astra_cone_hybrid.py` 与 `src_3d_helical/astra_cone_hybrid.py`
移植为 C++,使用 ASTRA Toolbox 的 C++ API(CUDA 加速):

- **A. Pure FDK** (hann 滤波)
- **B. TV-OS-SART** (10 子集 SIRT3D_CUDA + 各向异性 TV 去噪, β 递减)
- **C. Hybrid IR** (OS-SART×10 + TV×10 + FDK 混合 10%)

## 目录结构

```
src_astra_cpp/
├── CMakeLists.txt            # 构建脚本
├── src/
│   ├── common.h              # 公共接口
│   ├── common.cpp            # 核心流水线 (几何/TV/噪声/度量/IO)
│   ├── main_axial.cpp        # 轴向入口  → astra_axial
│   └── main_helical.cpp      # 螺旋入口  → astra_helical
├── tools/
│   ├── make_phantom.py       # 用 tomophantom 生成体模 .raw (体模无 C++ API)
│   └── render_results.py     # 将结果渲染为对比图 (matplotlib)
├── third_party/
│   └── astra/include/astra/  # ASTRA Toolbox v2.5.0 C++ 头文件 (GPLv3, 官方源码)
└── data/                     # 体模 .raw (gitignored, 由 make_phantom.py 生成)
```

## 依赖

- g++ ≥ 11 (C++17), CMake ≥ 3.16
- CUDA 运行时 (ASTRA 的 CUDA 算法需要 GPU)
- ASTRA Toolbox **v2.5.0** 的 C++ 库 `libastra.so.0` —— 由 pip 包
  `astra-toolbox==2.5.0` 自带 (`.venv/lib/python3.12/site-packages/astra/libastra.so.0`)。
  pip 包不含 C++ 头文件, 故头文件取自官方 v2.5.0 源码 (见 `third_party/`)。

> 注意: ASTRA 的 C++ 直接构造接口与 Python 接口的探测器约定不同。
> Python 传探测器**中心**; C++ 直接构造 `CConeVecProjectionGeometry3D` 期望
> 探测器**角落**(bottom-left)。`common.cpp` 的 `build_vectors()` 已做同样的
> 中心→角落转换 (与 `ConeVecProjectionGeometry3D.cpp::initializeAngles` 一致)。

## 构建

```sh
cd src_astra_cpp
cmake -S . -B build                  # 自动定位 ../.venv 下的 libastra.so.0
cmake --build build -j
# 若 libastra.so.0 在别处: cmake -S . -B build -DASTRA_LIBRARY=/path/to/libastra.so.0
```

## 运行 (在仓库根目录)

```sh
# 1. 生成体模 (一次即可, 轴向/螺旋共用)
.venv/bin/python src_astra_cpp/tools/make_phantom.py

# 2. 重建
src_astra_cpp/build/astra_axial    src_astra_cpp/data/vol_gt.raw img_3d_axial/astra_cpp
src_astra_cpp/build/astra_helical  src_astra_cpp/data/vol_gt.raw img_3d_helical/astra_cpp

# 3. 渲染对比图 (C++ 跑完会自动调用, 无需手动; 也可手动执行)
.venv/bin/python src_astra_cpp/tools/render_results.py axial helical
```

输出: `img_{axial,helical}/astra_cpp/` 下的 `astra_cone_hybrid.png`
(**与 Python 版同款 3×4 结果图**, 尺寸/布局/标题/suptitle 完全一致, C++ 跑完自动渲染)、
`cpp_*.raw` (FDK/TV/Hybrid 结果体)、`cpp_summary.json` (指标 + z-profile)、
`cpp_zprofile.csv`。(不生成中间过程的切片 PNG)

## 与 Python 版的差异

- **体模**: 由 `tools/make_phantom.py` 生成 (tomophantom 无 C++ API), 与 Python 版逐元素一致
- **噪声**: 模型相同 (泊松-高斯 + 环形伪影), 但 RNG 为 `std::mt19937`, 与 numpy
  不逐位一致 → 噪声数据的指标与 Python 版有细微差异 (同数量级)
- **干净数据 (无噪声) 的 FDK 结果与 Python 版完全一致** (同一 GPU 内核, 确定性)

## 已验证结果 (对比 Python 基线 `img_3d_axial|helical/astra_cone_hybrid_summary.json`)

| 算法 | 轴向 Python | 轴向 C++ | 螺旋 Python | 螺旋 C++ |
|---|---|---|---|---|
| Pure FDK | 0.00088 / 0.9946 | 0.00088 / 0.9946 | 0.00129 / 0.9884 | 0.00129 / 0.9884 |
| TV-OS-SART x10 | 0.00081 / 0.9954 | 0.00081 / 0.9954 | ~0.00082 / 0.9954 | 0.00082 / 0.9954 |
| Hybrid IR | 0.0009 / 0.9944 | 0.00090 / 0.9944 | 0.00091 / 0.9942 | 0.00091 / 0.9942 |

(数值为 RMSE / SSIM, 见各目录下的 `*_summary.json`。)

## 耗时 (GTX 1660, 与 Python 基线 summary.json 对比)

| 阶段 | Python | C++ (CPU TV) | C++ (GPU TV) |
|---|---|---|---|
| Pure FDK | ~301 ms | ~101 ms | **~100 ms** (3×) |
| TV-OS-SART x10 | ~5006 ms | ~4855 ms | **~4030 ms** |
| Hybrid IR | ~4762 ms | ~4527 ms | **~3500 ms** |

耗时大头是 SIRT3D_CUDA 的 200 次子集迭代 (~7.1 s)—— 两边调用的是**同一个
ASTRA CUDA 内核**, 这部分时间相同, C++ 无法再缩短; C++ 的收益来自: 去掉 Python
每迭代调用/拷贝开销 (SIRT 单次 ~36ms vs ~50ms) + **TV 改为自写 CUDA 内核**
(20 次共 ~0.2s, Python 因未装 cupy 只能 CPU numpy)。

如需自行对比 TV 速度: `src/tv_kernel.cu` 与 `src_3d_axial/tv_gpu.py` 算法等价。
