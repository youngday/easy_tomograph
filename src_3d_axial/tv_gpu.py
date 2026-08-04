"""
GPU TV 梯度 (CuPy RawKernel) — 与 CPU numpy 版本算法完全等价
=================================================================
各向异性 TV: z 方向权重 w_z
用法:
    from tv_gpu import tv_gradient_gpu
    grad = tv_gradient_gpu(v, w_z=1.5)   # v: (nz, N, N) float32
"""
import numpy as np

_KERNEL_SRC = r"""
extern "C" __global__ void tv_grad_kernel(
    const float* v, float* out, int nz, int N, float w_z, float eps)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nz * N * N;
    if (idx >= total) return;

    int k = idx % N;
    int j = (idx / N) % N;
    int i = idx / (N * N);

    float div = 0.0f;
    float dx, dy, dz, m;

    // (i,j,k) 自身 ux: div += ux[k]
    if (k < N-1) {
        dx = v[idx+1] - v[idx];
        dy = (j < N-1) ? v[idx+N] - v[idx] : 0.0f;
        dz = (i < nz-1) ? v[idx+N*N] - v[idx] : 0.0f;
        m = sqrtf(dx*dx + dy*dy + dz*dz*w_z*w_z + eps);
        div += dx / m;
    }
    // (i,j,k-1) ux: div -= ux[k-1]
    if (k > 0) {
        int p = idx - 1;
        dx = v[p+1] - v[p];
        dy = (j < N-1) ? v[p+N] - v[p] : 0.0f;
        dz = (i < nz-1) ? v[p+N*N] - v[p] : 0.0f;
        m = sqrtf(dx*dx + dy*dy + dz*dz*w_z*w_z + eps);
        div -= dx / m;
    }
    // (i,j,k) 自身 uy: div += uy[j]
    if (j < N-1) {
        dx = (k < N-1) ? v[idx+1] - v[idx] : 0.0f;
        dy = v[idx+N] - v[idx];
        dz = (i < nz-1) ? v[idx+N*N] - v[idx] : 0.0f;
        m = sqrtf(dx*dx + dy*dy + dz*dz*w_z*w_z + eps);
        div += dy / m;
    }
    // (i,j-1,k) uy: div -= uy[j-1]
    if (j > 0) {
        int p = idx - N;
        dx = (k < N-1) ? v[p+1] - v[p] : 0.0f;
        dy = v[p+N] - v[p];
        dz = (i < nz-1) ? v[p+N*N] - v[p] : 0.0f;
        m = sqrtf(dx*dx + dy*dy + dz*dz*w_z*w_z + eps);
        div -= dy / m;
    }
    // (i,j,k) 自身 uz: div += w_z*uz[i]
    if (i < nz-1) {
        dx = (k < N-1) ? v[idx+1] - v[idx] : 0.0f;
        dy = (j < N-1) ? v[idx+N] - v[idx] : 0.0f;
        dz = v[idx+N*N] - v[idx];
        m = sqrtf(dx*dx + dy*dy + dz*dz*w_z*w_z + eps);
        div += w_z * dz / m;
    }
    // (i-1,j,k) uz: div -= w_z*uz[i-1]
    if (i > 0) {
        int p = idx - N*N;
        dx = (k < N-1) ? v[p+1] - v[p] : 0.0f;
        dy = (j < N-1) ? v[p+N] - v[p] : 0.0f;
        dz = v[p+N*N] - v[p];
        m = sqrtf(dx*dx + dy*dy + dz*dz*w_z*w_z + eps);
        div -= w_z * dz / m;
    }

    out[idx] = -div;  // 梯度 = -div (与 CPU 版一致)
}
"""

import cupy as cp

_kernel = None
_out_buf = None


def _ensure_kernel(shape):
    global _kernel, _out_buf
    if _kernel is None:
        _kernel = cp.RawKernel(_KERNEL_SRC, "tv_grad_kernel")
        _out_buf = cp.empty(shape, dtype=cp.float32)
    return _kernel, _out_buf


def tv_gradient_gpu(v, w_z=1.5, eps=1e-8):
    """GPU 各向异性 TV 梯度, 返回 numpy 数组 (与 CPU tv_gradient 相同结果)"""
    vg = cp.ascontiguousarray(cp.asarray(v, dtype=cp.float32))
    kern, out = _ensure_kernel(vg.shape)
    total = vg.size
    threads = 256
    blocks = (total + threads - 1) // threads
    nz, N = vg.shape[0], vg.shape[1]
    kern((blocks,), (threads,), (vg, out, np.int32(nz), np.int32(N),
                                 np.float32(w_z), np.float32(eps)))
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(out)


def tv_denoise_gpu(v, beta, w_z=1.5, **kwargs):
    """GPU TV 去噪: v - β * ∇TV(v)"""
    grad = tv_gradient_gpu(v, w_z=w_z)
    return cp.asnumpy(cp.asarray(v, dtype=cp.float32) - beta * grad)


# ---- 自检: 与 CPU 版本对比 ----
def _self_test():
    from time import time as tm
    nz, N = 32, 512
    rng = np.random.default_rng(0)
    v = rng.standard_normal((nz, N, N)).astype(np.float32) * 0.01

    # CPU 参考
    def tv_gradient_cpu(v, w_z=1.5, eps=1e-8):
        dx = np.zeros_like(v); dy = np.zeros_like(v); dz = np.zeros_like(v)
        dx[:,:,:-1]=v[:,:,1:]-v[:,:,:-1]; dy[:,:-1,:]=v[:,1:,:]-v[:,:-1,:]; dz[:-1,:,:]=v[1:,:,:]-v[:-1,:,:]
        mag = np.sqrt(dx**2+dy**2+(w_z*dz)**2+eps); ux,uy,uz = dx/mag, dy/mag, w_z*dz/mag
        div = np.zeros_like(v)
        div[:,:,1:-1]=ux[:,:,1:-1]-ux[:,:,:-2]; div[:,:,0]=ux[:,:,0]; div[:,:,-1]=-ux[:,:,-2]
        div[:,1:-1,:]+=uy[:,1:-1,:]-uy[:,:-2,:]; div[:,0,:]+=uy[:,0,:]; div[:,-1,:]+=-uy[:,-2,:]
        div[1:-1,:,:]+=uz[1:-1,:,:]-uz[:-2,:,:]; div[0,:,:]+=uz[0,:,:]; div[-1,:,:]+=-uz[-2,:,:]
        return -div

    g_cpu = tv_gradient_cpu(v)
    g_gpu = cp.asnumpy(tv_gradient_gpu(v))
    err = np.abs(g_cpu - g_gpu).max()
    print(f"一致性检查: max|CPU-GPU| = {err:.2e}  {'✅ PASS' if err < 1e-4 else '❌ FAIL'}")

    # 速度
    _ = tv_gradient_gpu(v)
    t0 = tm()
    for _ in range(10): tv_gradient_gpu(v)
    t_gpu = (tm()-t0)/10
    t0 = tm()
    for _ in range(10): tv_gradient_cpu(v)
    t_cpu = (tm()-t0)/10
    print(f"CPU: {t_cpu*1000:.1f}ms  GPU: {t_gpu*1000:.1f}ms  加速: {t_cpu/t_gpu:.1f}x")


if __name__ == "__main__":
    _self_test()
