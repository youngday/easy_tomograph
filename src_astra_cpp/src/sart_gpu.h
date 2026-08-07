#pragma once
// GPU 常驻 OS-SART (复刻 ASTRA SIRT3D_CUDA 更新公式):
//   v += pixelWeight * A^T * ( lineWeight * (b - A·v) )
//   lineWeight = 1/(A·1) (逐探测器), pixelWeight = 1/(A^T·1) (逐体素), relaxation = 1
// 体积全程驻留 GPU, 仅在每次 run() 的首尾做一次上传/下载
// → 消除旧方案每子集迭代的 CPU memcpy 与 PCIe 往返 (约 10× 传输削减)
#include <memory>
#include <string>
#include <vector>

class SARTGpu {
public:
    SARTGpu();
    ~SARTGpu();

    // sino_noisy: 全角度含噪声 sinogram, (row, angle, col) 布局; 内部按 10 子集切分
    bool init(bool helical, const std::vector<float>& sino_noisy, std::string& err);

    // vol_in: 初始重建 ([z][y][x] CPU) → 运行 n_epochs 个 epoch (每 epoch 10 次子集迭代) → vol_out
    // 允许 vol_in 与 vol_out 为同一向量
    bool run(const std::vector<float>& vol_in, int n_epochs,
             std::vector<float>& vol_out, std::string& err);

private:
    struct Impl;
    std::unique_ptr<Impl> m;
};
