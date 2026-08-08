//! FFI 绑定: 调用 C shim (astra_c_api.cpp), 封装为安全的 Rust 接口
//! 设计: shim 只保留 ASTRA 对象/内核调用; 向量计算、SART 循环控制、TV 缓冲都在 Rust
use std::os::raw::{c_char, c_void};

// 布局常量 (与 c_api/astra_c_api.cpp 一致)
pub const N: usize = 512;
pub const NZ: usize = 32;
pub const N_ANGLES: usize = 180;
pub const N_SUBSETS: usize = 10;
pub const N_DET_ROW: usize = 64;
pub const N_DET_COL: usize = 725;
pub const NVOL: usize = NZ * N * N;
pub const NSINO: usize = N_DET_ROW * N_ANGLES * N_DET_COL;
pub const VEC_FLOATS: usize = N_ANGLES * 12; // 向量数组长度 (f64)

extern "C" {
    fn astra_rs_nvol() -> usize;
    fn astra_rs_nsino() -> usize;

    fn astra_rs_geom_create(
        vectors: *const f64,
        nv: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> *mut c_void;
    fn astra_rs_geom_free(ctx: *mut c_void);

    fn astra_rs_fp_run(ctx: *mut c_void, err: *mut c_char, err_len: usize) -> i32;
    fn astra_rs_fdk_run(ctx: *mut c_void, err: *mut c_char, err_len: usize) -> i32;
    // 宿主侧数据缓冲指针 (kind=0 sino, 1 vol), Rust 直接读写
    fn astra_rs_data_ptr(ctx: *mut c_void, kind: i32) -> *mut f32;

    fn astra_rs_sart_create(
        vectors: *const f64,
        nv: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> *mut c_void;
    fn astra_rs_sart_free(h: *mut c_void);
    // 上传第 i 个子集的 sinogram (子集切分在 Rust 端完成)
    fn astra_rs_sart_subset_upload(
        h: *mut c_void,
        i: i32,
        sino_subset: *const f32,
        n: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> i32;
    fn astra_rs_sart_upload(
        h: *mut c_void,
        vol_in: *const f32,
        err: *mut c_char,
        err_len: usize,
    ) -> i32;
    fn astra_rs_sart_subset_step(h: *mut c_void, i: i32, err: *mut c_char, err_len: usize) -> i32;
    fn astra_rs_sart_download(
        h: *mut c_void,
        vol_out: *mut f32,
        err: *mut c_char,
        err_len: usize,
    ) -> i32;

    // TV 去噪 CUDA 内核 (tv_kernel.cu, extern "C")
    fn tv_denoise_cuda(
        v: *const f32,
        out: *mut f32,
        nz: i32,
        N: i32,
        beta: f32,
        w_z: f32,
        eps: f32,
    ) -> i32;
}

fn err_buf() -> [c_char; 512] {
    [0 as c_char; 512]
}

unsafe fn take_err(buf: &[c_char]) -> String {
    let mut v: Vec<u8> = Vec::new();
    for &c in buf {
        if c == 0 {
            break;
        }
        v.push(c as u8);
    }
    String::from_utf8_lossy(&v).into_owned()
}

pub struct Geom(*mut c_void);
impl Drop for Geom {
    fn drop(&mut self) {
        unsafe { astra_rs_geom_free(self.0) }
    }
}

pub struct Sart(*mut c_void);
impl Drop for Sart {
    fn drop(&mut self) {
        unsafe { astra_rs_sart_free(self.0) }
    }
}

impl Geom {
    pub fn create(vectors: &[f64]) -> Result<Geom, String> {
        debug_assert_eq!(vectors.len(), VEC_FLOATS);
        let mut eb = err_buf();
        unsafe {
            let p =
                astra_rs_geom_create(vectors.as_ptr(), vectors.len(), eb.as_mut_ptr(), eb.len());
            if p.is_null() {
                return Err(take_err(&eb));
            }
            Ok(Geom(p))
        }
    }

    /// 正向投影: vol ([z][y][x]) → sino (row, angle, col)
    /// 数据编排 (填充/清零/读取) 在 Rust, shim 只做纯算法调用
    pub fn fp(&self, vol: &[f32], sino: &mut [f32]) -> Result<(), String> {
        debug_assert_eq!(vol.len(), NVOL);
        debug_assert_eq!(sino.len(), NSINO);
        let mut eb = err_buf();
        unsafe {
            // vol → 宿主侧 volume 缓冲
            std::slice::from_raw_parts_mut(astra_rs_data_ptr(self.0, 1), NVOL).copy_from_slice(vol);
            // sinogram 清零 (FP 内核为累加式, 与 C++ 版 memset 一致)
            std::slice::from_raw_parts_mut(astra_rs_data_ptr(self.0, 0), NSINO).fill(0.0);
            if astra_rs_fp_run(self.0, eb.as_mut_ptr(), eb.len()) != 0 {
                return Err(take_err(&eb));
            }
            // 读取 sinogram 缓冲
            sino.copy_from_slice(std::slice::from_raw_parts(
                astra_rs_data_ptr(self.0, 0),
                NSINO,
            ));
        }
        Ok(())
    }

    /// FDK (hann): sino → vol ([z][y][x])
    pub fn fdk(&self, sino: &[f32], vol: &mut [f32]) -> Result<(), String> {
        debug_assert_eq!(sino.len(), NSINO);
        debug_assert_eq!(vol.len(), NVOL);
        let mut eb = err_buf();
        unsafe {
            // sino → 宿主侧 sinogram 缓冲
            std::slice::from_raw_parts_mut(astra_rs_data_ptr(self.0, 0), NSINO)
                .copy_from_slice(sino);
            if astra_rs_fdk_run(self.0, eb.as_mut_ptr(), eb.len()) != 0 {
                return Err(take_err(&eb));
            }
            // 读取 volume 缓冲
            vol.copy_from_slice(std::slice::from_raw_parts(
                astra_rs_data_ptr(self.0, 1),
                NVOL,
            ));
        }
        Ok(())
    }
}

impl Sart {
    /// 创建 GPU 常驻 SART: 分配缓冲/预计算权重 (shim), 子集 sinogram 切分 + 上传 (Rust)
    pub fn create(vectors: &[f64], sino_noisy: &[f32]) -> Result<Sart, String> {
        debug_assert_eq!(vectors.len(), VEC_FLOATS);
        debug_assert_eq!(sino_noisy.len(), NSINO);
        let mut eb = err_buf();
        let p = unsafe {
            astra_rs_sart_create(vectors.as_ptr(), vectors.len(), eb.as_mut_ptr(), eb.len())
        };
        if p.is_null() {
            return Err(unsafe { take_err(&eb) });
        }
        let sart = Sart(p);
        // 子集切分: sino 布局 [row][angle][col], 子集 i 取角度 [i*18, (i+1)*18)
        // 子集缓冲布局 [row][sub_angle][col] (与 C++ 版 fill_subset_sino 一致)
        let sub_size = N_ANGLES / N_SUBSETS;
        let sub_len = N_DET_ROW * sub_size * N_DET_COL;
        for i in 0..N_SUBSETS {
            let mut sub = vec![0.0f32; sub_len];
            for row in 0..N_DET_ROW {
                for a in 0..sub_size {
                    let src_off = (row * N_ANGLES + i * sub_size + a) * N_DET_COL;
                    let dst_off = (row * sub_size + a) * N_DET_COL;
                    sub[dst_off..dst_off + N_DET_COL]
                        .copy_from_slice(&sino_noisy[src_off..src_off + N_DET_COL]);
                }
            }
            let mut eb2 = err_buf();
            unsafe {
                if astra_rs_sart_subset_upload(
                    p,
                    i as i32,
                    sub.as_ptr(),
                    sub.len(),
                    eb2.as_mut_ptr(),
                    eb2.len(),
                ) != 0
                {
                    return Err(take_err(&eb2));
                }
            }
        }
        Ok(sart)
    }

    fn upload(&self, vol_in: &[f32]) -> Result<(), String> {
        debug_assert_eq!(vol_in.len(), NVOL);
        let mut eb = err_buf();
        unsafe {
            if astra_rs_sart_upload(self.0, vol_in.as_ptr(), eb.as_mut_ptr(), eb.len()) != 0 {
                return Err(take_err(&eb));
            }
        }
        Ok(())
    }

    fn subset_step(&self, i: usize) -> Result<(), String> {
        let mut eb = err_buf();
        unsafe {
            if astra_rs_sart_subset_step(self.0, i as i32, eb.as_mut_ptr(), eb.len()) != 0 {
                return Err(take_err(&eb));
            }
        }
        Ok(())
    }

    fn download(&self, vol_out: &mut [f32]) -> Result<(), String> {
        debug_assert_eq!(vol_out.len(), NVOL);
        let mut eb = err_buf();
        unsafe {
            if astra_rs_sart_download(self.0, vol_out.as_mut_ptr(), eb.as_mut_ptr(), eb.len()) != 0
            {
                return Err(take_err(&eb));
            }
        }
        Ok(())
    }

    /// GPU 常驻 OS-SART: 就地运行 n_epochs (upload → 子集循环 → download), 循环控制在 Rust
    /// 就地 (vol_in == vol_out) 避免了每轮 33MB 的 CPU 拷贝
    pub fn run_inplace(&self, vol: &mut [f32], n_epochs: i32) -> Result<(), String> {
        debug_assert_eq!(vol.len(), NVOL);
        self.upload(vol)?;
        for _ in 0..n_epochs {
            for i in 0..N_SUBSETS {
                self.subset_step(i)?;
            }
        }
        self.download(vol)
    }
}

/// TV 去噪 (就地): v = v + beta*div(TV)(v) — 直接调 CUDA 内核, 缓冲在 Rust 复用
use std::cell::RefCell;
thread_local! {
    static TV_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

pub fn tv(vol: &mut [f32], beta: f32, w_z: f32) -> Result<(), String> {
    debug_assert_eq!(vol.len(), NVOL);
    TV_BUF.with(|cell| {
        let mut out = cell.borrow_mut();
        if out.len() != NVOL {
            out.resize(NVOL, 0.0);
        }
        let rc = unsafe {
            tv_denoise_cuda(
                vol.as_ptr(),
                out.as_mut_ptr(),
                NZ as i32,
                N as i32,
                beta,
                w_z,
                1e-8,
            )
        };
        if rc != 0 {
            return Err("TV CUDA 内核失败".to_string());
        }
        vol.copy_from_slice(out.as_slice());
        Ok(())
    })
}

/// 校验 shim 尺寸与 Rust 端常量一致
pub fn check_sizes() -> Result<(), String> {
    unsafe {
        if astra_rs_nvol() != NVOL || astra_rs_nsino() != NSINO {
            return Err(format!(
                "shim 尺寸不匹配: nvol={}/{} nsino={}/{}",
                astra_rs_nvol(),
                NVOL,
                astra_rs_nsino(),
                NSINO
            ));
        }
    }
    Ok(())
}
