#pragma once
// GPU TV 去噪 (CUDA) — 见 tv_kernel.cu (C 链接, 与 .cu 定义一致)
// v → out: out[i] = v[i] + beta * div(TV)(i), 布局 [z][y][x] (x 最快)
// 返回 0 成功; 失败返回非 0 (调用方可回退到 CPU 实现)
extern "C" int tv_denoise_cuda(const float* v, float* out, int nz, int N, float beta, float w_z, float eps);
