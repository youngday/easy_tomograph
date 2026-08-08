// ============================================================================
// GPU TV 去噪 (CUDA) — 与 Python src_3d_axial/tv_gpu.py 的 CuPy RawKernel
// 算法完全等价: 单遍计算散度 div, 输出 v + beta*div (即 v - beta*(-div))
// 体积布局: [z][y][x], x 最快 (nz * N * N 个 float)
// ============================================================================
#include <cuda_runtime.h>

#include <cstdio>

namespace {

__global__ void tv_denoise_kernel(const float* __restrict__ v,
                                  float* __restrict__ out,
                                  int nz, int N, float beta, float w_z, float eps)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nz * N * N;
    if (idx >= total) return;

    const int k = idx % N;            // x
    const int j = (idx / N) % N;      // y
    const int i = idx / (N * N);      // z

    float div = 0.0f;

    // (i,j,k) 自身 ux: div += ux[k]
    if (k < N - 1) {
        float dx = v[idx + 1] - v[idx];
        float dy = (j < N - 1) ? v[idx + N] - v[idx] : 0.0f;
        float dz = (i < nz - 1) ? v[idx + N * N] - v[idx] : 0.0f;
        float m = sqrtf(dx * dx + dy * dy + dz * dz * w_z * w_z + eps);
        div += dx / m;
    }
    // (i,j,k-1) ux: div -= ux[k-1]
    if (k > 0) {
        int p = idx - 1;
        float dx = v[p + 1] - v[p];
        float dy = (j < N - 1) ? v[p + N] - v[p] : 0.0f;
        float dz = (i < nz - 1) ? v[p + N * N] - v[p] : 0.0f;
        float m = sqrtf(dx * dx + dy * dy + dz * dz * w_z * w_z + eps);
        div -= dx / m;
    }
    // (i,j,k) 自身 uy: div += uy[j]
    if (j < N - 1) {
        float dx = (k < N - 1) ? v[idx + 1] - v[idx] : 0.0f;
        float dy = v[idx + N] - v[idx];
        float dz = (i < nz - 1) ? v[idx + N * N] - v[idx] : 0.0f;
        float m = sqrtf(dx * dx + dy * dy + dz * dz * w_z * w_z + eps);
        div += dy / m;
    }
    // (i,j-1,k) uy: div -= uy[j-1]
    if (j > 0) {
        int p = idx - N;
        float dx = (k < N - 1) ? v[p + 1] - v[p] : 0.0f;
        float dy = v[p + N] - v[p];
        float dz = (i < nz - 1) ? v[p + N * N] - v[p] : 0.0f;
        float m = sqrtf(dx * dx + dy * dy + dz * dz * w_z * w_z + eps);
        div -= dy / m;
    }
    // (i,j,k) 自身 uz: div += w_z*uz[i]
    if (i < nz - 1) {
        float dx = (k < N - 1) ? v[idx + 1] - v[idx] : 0.0f;
        float dy = (j < N - 1) ? v[idx + N] - v[idx] : 0.0f;
        float dz = v[idx + N * N] - v[idx];
        float m = sqrtf(dx * dx + dy * dy + dz * dz * w_z * w_z + eps);
        div += w_z * dz / m;
    }
    // (i-1,j,k) uz: div -= w_z*uz[i-1]
    if (i > 0) {
        int p = idx - N * N;
        float dx = (k < N - 1) ? v[p + 1] - v[p] : 0.0f;
        float dy = (j < N - 1) ? v[p + N] - v[p] : 0.0f;
        float dz = v[p + N * N] - v[p];
        float m = sqrtf(dx * dx + dy * dy + dz * dz * w_z * w_z + eps);
        div -= w_z * dz / m;
    }

    out[idx] = v[idx] + beta * div;
}

}  // namespace

// ---------------------------------------------------------------------------
// 主机侧封装: 为 volume 分配设备缓冲并执行一次 TV 去噪 (v → out)
// 返回 0 成功; 设备缓冲跨调用复用
// ---------------------------------------------------------------------------
extern "C" int tv_denoise_cuda(const float* v, float* out,
                               int nz, int N, float beta, float w_z, float eps)
{
    static float* d_v = nullptr;
    static float* d_out = nullptr;
    static size_t d_size = 0;

    const size_t bytes = (size_t)nz * N * N * sizeof(float);
    if (bytes != d_size) {
        if (d_v) cudaFree(d_v);
        if (d_out) cudaFree(d_out);
        if (cudaMalloc(&d_v, bytes) != cudaSuccess || cudaMalloc(&d_out, bytes) != cudaSuccess) {
            fprintf(stderr, "[tv_cuda] cudaMalloc 失败\n");
            return -1;
        }
        d_size = bytes;
    }

    if (cudaMemcpy(d_v, v, bytes, cudaMemcpyHostToDevice) != cudaSuccess)
        return -1;

    const int total = nz * N * N;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    tv_denoise_kernel<<<blocks, threads>>>(d_v, d_out, nz, N, beta, w_z, eps);
    if (cudaGetLastError() != cudaSuccess)
        return -1;
    if (cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        return -1;
    if (cudaDeviceSynchronize() != cudaSuccess)
        return -1;
    return 0;
}
