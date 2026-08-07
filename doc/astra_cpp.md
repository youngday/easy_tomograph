# astra_cpp

## NOTE

ASTRA 的 **Python 接口传探测器中心**,而 C++ 直接构造 `CConeVecProjectionGeometry3D` 期望探测器**角落**(bottom-left)。不转换的话重建结果差约一个探测器尺寸(最初 FDK RMSE 为 0.0079)。`build_vectors()` 已按官方 `initializeAngles()` 的逻辑做了同样的中心→角落转换。

## fdk gpu

ASTRA 的 3D FDK **只有 `CCudaFDKAlgorithm3D`** 一个实现,类型注册为 `"FDK_CUDA"`——头文件目录里**没有 CPU 版 3D FDK**(见 `third_party/astra/include/astra/`,仅 `CudaFDKAlgorithm3D.h`)
- 我们的 C++ 代码直接实例化该类,`run()` → `CompositeGeometryManager::doFDK` → `astraCUDA3d::FDK`(`fdk.cu` 的 CUDA 内核)

## optimize # 001

为什么之前"没少"

之前 C++ 的耗时构成(TV 还在 CPU 上时):

| 构成 | 耗时 | 说明 |
|---|---|---|
| SIRT 200 次子集迭代 | ~7.1s | 与 Python 调用**同一个** ASTRA CUDA 内核,时间天然相同 |
| TV 去噪 20 次 (CPU) | ~1.5s | Python 因没装 cupy 也是 CPU,但 numpy 有 SIMD 优化 |
| FDK / FP / 度量 / IO | 其余 | FDK 已 3 倍速 |

Python 每迭代还有 `data3d.store/get` + 调用开销(~50ms/次 vs C++ 35ms/次),所以 C++ 其实每轮快 28% 左右,但都被 TV(CPU)拖住了。

## 这轮优化

1. **TV 移到 GPU** — 新增 `src/tv_kernel.cu`(自写 CUDA 内核,与 `tv_gpu.py` 的 CuPy 内核同款单遍散度算法,指标完全一致):20 次 TV 从 ~1.5s 降到 **189ms**
2. **度量计算 OpenMP 并行**(linear_scale/RMSE/SSIM/z-profile,每轮迭代都要算)
3. 子集 sinogram 拷贝并行

## 优化后对比(GTX 1660,与 Python 基线 summary.json 同口径:纯算法耗时)

| 阶段 | Python | C++ (优化后) | 提升 |
|---|---|---|---|
| Pure FDK | 301 ms | 100 ms | **3.0×** |
| TV-OS-SART x10 | 5006 ms | 3905 ms | 1.28× |
| Hybrid IR | 4762 ms | 3437 ms | 1.38× |
| 合计 | ~10.1 s | **~7.4 s** | **1.35×** |

指标不变:TV-OS-SART 0.00081/0.9954、Hybrid 0.00090/0.9944,与 Python 基线一致。端到端 wall clock(含 FP/噪声/度量/输出):C++ 约 10s;Python 完整脚本含 tomophantom 生成和 matplotlib 绘图约 16-19s。

## 诚实说明剩余空间

现在耗时构成是 **SIRT 200 次迭代 ≈ 7.0s(70%)**,它是 ASTRA 的 `SIRT3D_CUDA` 内核(每次 ~35ms = 内核 ~24ms + CPU↔GPU 传输 ~11ms),Python 和 C++ 跑的是同一份 GPU 代码,无法从 C++ 侧再压缩。想再快只能:
- 换 GPU(内核时间直接按算力缩)
- 减迭代/子集数(会改变收敛结果)
- 换更快算法(如 FDK 初始化 + 更少 SIRT 轮数)

改动文件:`src/tv_kernel.cu`、`src/tv_gpu.h`、`src/common.cpp`、`CMakeLists.txt`(启用 CUDA + OpenMP)、`README.md`。构建方式不变:`cmake -S . -B build && cmake --build build -j`(目标架构默认 sm_75,换卡用 `-DCUDA_ARCHITECTURES=` 覆盖)。
