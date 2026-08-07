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
│   ├── sart_gpu.h/.cpp       # GPU 常驻 OS-SART (复刻 SIRT3D_CUDA, 消除传输开销)
│   ├── tv_gpu.h / tv_kernel.cu  # GPU TV 去噪 (CUDA 内核)
│   ├── astra_geometry.h      # 共享 cone_vec 几何构建 (中心→角落约定)
│   ├── main_axial.cpp        # 轴向入口  → astra_axial
│   └── main_helical.cpp      # 螺旋入口  → astra_helical
├── tools/
│   ├── make_phantom.py       # 用 tomophantom 生成体模 .raw (体模无 C++ API)
│   ├── make_sino_noisy.py    # 用原版 ct_noise.py 生成共享噪声 sinogram (与 Python 逐位一致)
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

# 2. 生成与 Python 版完全一致的含噪声 sinogram (numpy 噪声, 轴向/螺旋各一份)
.venv/bin/python src_astra_cpp/tools/make_sino_noisy.py both

# 3. 重建 (会自动加载共享噪声文件; 找不到时回退到内置 mt19937 噪声)
src_astra_cpp/build/astra_axial    src_astra_cpp/data/vol_gt.raw img_3d_axial/astra_cpp
src_astra_cpp/build/astra_helical  src_astra_cpp/data/vol_gt.raw img_3d_helical/astra_cpp

# 4. 渲染对比图 (C++ 跑完会自动调用, 无需手动; 也可手动执行)
.venv/bin/python src_astra_cpp/tools/render_results.py axial helical
```

输出: `img_{axial,helical}/astra_cpp/` 下的 `astra_cone_hybrid.png`
(**与 Python 版同款 3×4 结果图**, 尺寸/布局/标题/suptitle 完全一致, C++ 跑完自动渲染)、
`cpp_*.raw` (FDK/TV/Hybrid 结果体)、`cpp_summary.json` (指标 + z-profile)、
`cpp_zprofile.csv`。(不生成中间过程的切片 PNG)

## 与 Python 版的差异

- **体模**: 由 `tools/make_phantom.py` 生成 (tomophantom 无 C++ API), 与 Python 版逐元素一致
- **噪声**: 默认加载 `data/sino_noisy_{axial,helical}.raw` (由 `tools/make_sino_noisy.py`
  用**原版** `src_3d_axial/ct_noise.py` + numpy 生成) → 与 Python 版**逐位一致**,
  FDK(noisy)/TV-OS-SART/Hybrid 指标与 Python 完全一致 (0.00081/0.9954 等)。
  若噪声文件不存在, 回退到内置噪声 (模型相同, 但 `std::mt19937` 与 numpy
  不逐位一致, 指标有细微差异)
- **干净数据 (无噪声) 的 FDK 结果与 Python 版完全一致** (同一 GPU 内核, 确定性)

## 已验证结果 (对比 Python 基线 `img_3d_axial|helical/astra_cone_hybrid_summary.json`)

| 算法 | 轴向 Python | 轴向 C++ | 螺旋 Python | 螺旋 C++ |
|---|---|---|---|---|
| Pure FDK | 0.00088 / 0.9946 | 0.00088 / 0.9946 | 0.00129 / 0.9884 | 0.00129 / 0.9884 |
| TV-OS-SART x10 | 0.00081 / 0.9954 | 0.00081 / 0.9954 | ~0.00082 / 0.9954 | 0.00082 / 0.9954 |
| Hybrid IR | 0.0009 / 0.9944 | 0.00090 / 0.9944 | 0.00091 / 0.9942 | 0.00091 / 0.9942 |

(数值为 RMSE / SSIM, 见各目录下的 `*_summary.json`。)

## 耗时 (GTX 1660, 与 Python 基线 summary.json 对比)

| 阶段 | Python | C++ (CPU TV) | C++ (GPU TV) | C++ (GPU 常驻 SART) |
|---|---|---|---|---|
| Pure FDK | ~301 ms | ~101 ms | ~100 ms | **~100 ms** (3×) |
| TV-OS-SART x10 | ~5006 ms | ~4855 ms | ~4030 ms | **~2720 ms** |
| Hybrid IR | ~4762 ms | ~4527 ms | ~3500 ms | **~2620 ms** |

## 为什么能再快 30% (GPU 常驻 OS-SART, 方案 A)

旧实现每个子集迭代都经历 `CPU memcpy → run(1)(上传→内核→下载) → CPU memcpy`
(每次 ~40ms, 其中 ~12ms 是传输)。新实现 `src/sart_gpu.cpp` 用 libastra 导出的
GPU 常驻内核 (`astraCUDA3d::FP/BP` + `processVol3D` 算术) 自写 OS-SART 循环,
**精确复刻 `CudaSirtAlgorithm3D` 的更新公式**:

```
v += pixelWeight * Aᵀ·( lineWeight · (b − A·v) )
lineWeight = 1/(A·1),  pixelWeight = 1/(Aᵀ·1),  relaxation = 1
```

体积全程驻留 GPU, 每 epoch 只上传/下载一次 → 每迭代 ~40ms → ~25ms
(SIRT 200 次迭代 6.75s → 5.0s)。运算顺序/内核与 ASTRA 完全相同,
**结果与 Python 版逐位一致** (z-profile 除个别舍入边界外 max-diff = 0)。

其余收益不变: 去掉 Python 每迭代调用/拷贝开销 + TV 改为自写 CUDA 内核
(`src/tv_kernel.cu`, 20 次共 ~0.2s)。
