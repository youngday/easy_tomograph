# src_astra_rust — ASTRA 锥束混合重建 (Rust 版)

与 `src_astra_cpp` **算法完全一致**的 Rust 实现: GPU 内核(FP/FDK/OS-SART)仍由
ASTRA CUDA 提供, Rust 负责管线编排/度量/IO。结果与 C++/Python 版**逐位一致**。

## 架构

```
src_astra_rust/
├── Cargo.toml / build.rs      # build.rs 调 g++/nvcc 编译 shim 并链接
├── c_api/
│   ├── astra_c_api.cpp        # C shim: 仅 ASTRA 对象/GPU 内核调用 (extern "C")
│   └── tv_kernel.cu           # TV 去噪 CUDA 内核 (与 src_astra_cpp 相同)
└── src/
    ├── lib.rs
    ├── ffi.rs                 # extern "C" 声明 + 安全封装 (Geom/Sart/tv)
    ├── geometry.rs            # 锥束几何向量计算 (纯 Rust, f64)
    ├── metrics.rs             # linear_scale/RMSE/SSIM/z-profile (std 线程并行)
    ├── noise.rs               # 内置噪声回退 (MT19937, 仅共享噪声文件缺失时用)
    ├── pipeline.rs            # 完整流水线 (FDK/TV-OS-SART/Hybrid + 提前停止)
    └── bin/axial.rs, helical.rs
```

零第三方 crate(纯 std):构建不需网络。**Rust 占比**:几何向量、SART
epoch/子集循环控制、子集 sinogram 切分、FP/FDK/SART 的宿主侧数据编排
(缓冲填充/清零/读取)、TV 缓冲复用、全部度量/IO 都在 Rust;shim 只保留无法
跨 FFI 的 ASTRA C++ 对象构造与 GPU 内核调用(FP/FDK/SART 原语/TV 内核)。
FP/FDK 的 CPU 数据缓冲缓存于几何上下文,经 `astra_rs_data_ptr` 由 Rust 直接
读写,shim 的 `fp_run`/`fdk_run` 是纯算法调用。TV CUDA 内核经 extern "C" 由
Rust 直接调用。

GPU 内核无法在 Rust 中重写,故经 FFI 调用 libastra(与 C++ 相同),**结果逐位
一致**是必然且可验证的。

## 依赖

- Rust 工具链 (cargo), g++ ≥ 11 (编译 C shim), nvcc + CUDA (编译 TV 内核)
- ASTRA Toolbox v2.5.0 C++ 库 (`.venv/.../libastra.so.0`), 头文件在
  `src_astra_cpp/third_party/astra/include` (需先有 src_astra_cpp)
- 数据准备/绘图沿用 Python 辅助: `src_astra_cpp/tools/{make_phantom,make_sino_noisy,render_results}.py`

## 构建

```sh
cd src_astra_rust
cargo build --release
# 可选环境变量:
#   ASTRA_LIBRARY=/path/to/libastra.so.0   链接库位置
#   CUDA_HOME=/usr/local/cuda-xx            CUDA 目录
#   CUDA_ARCHITECTURES=sm_75                目标 GPU 架构 (默认 sm_75)
```

## 运行 (在仓库根目录)

```sh
src_astra_rust/target/release/astra_rs_axial    src_astra_cpp/data/vol_gt.raw img_3d_axial/astra_rs
src_astra_rust/target/release/astra_rs_helical  src_astra_cpp/data/vol_gt.raw img_3d_helical/astra_rs
# 可选参数: [max_epochs=10] [target_rmse=0.001]  (RMSE≤0.001 提前停止, 同 C++)
```

输出: `img_3d_{axial,helical}/astra_rs/` 下的 `cpp_*.raw`、`cpp_summary.json`、
`cpp_zprofile.csv`,以及自动渲染的 `astra_cone_hybrid.png`(复用
`src_astra_cpp/tools/render_results.py`,文件名与 C++ 一致)。

## 验证结果 (与 C++ 版对比)

| 阶段 | C++ (轴向) | Rust (轴向) | C++ (螺旋) | Rust (螺旋) |
|---|---|---|---|---|
| Pure FDK | 0.00088 / 0.9946 | 0.00088 / 0.9946 | 0.00129 / 0.9884 | 0.00129 / 0.9884 |
| FDK(noisy) | 0.00278 / 0.9438 | 0.00278 / 0.9438 | 0.00295 / 0.9361 | 0.00295 / 0.9361 |
| TV-OS-SART x6 | 0.00096 / 0.9935 | 0.00096 / 0.9935 | 0.00098 / 0.9933 | 0.00098 / 0.9933 |
| Hybrid IR (x7) | 0.00093 / 0.9940 | 0.00093 / 0.9940 | 0.00094 / 0.9938 | 0.00094 / 0.9938 |

z-profile 与 C++ 版 **max-diff = 0.00e+00** (逐位一致)。

耗时 (GTX 1660): TV-OS-SART ~1.7s, Hybrid ~2.3-2.4s — 与 C++ (~1.65s / ~2.3s) 相当
(快 ~5-8%, 去掉每轮 33MB 的 CPU 克隆后开销只剩 FFI 与 H2D/D2H 传输, 大头仍是
GPU 内核)。TV 每次 ~11ms(首次调用 ~28ms 为 CUDA 内核模块加载/预热), 与 C++
一致。
