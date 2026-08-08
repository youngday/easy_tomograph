//! FFI 绑定: 调用 C shim (astra_c_api.cpp), 封装为安全的 Rust 接口
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

extern "C" {
    fn astra_rs_nvol() -> usize;
    fn astra_rs_nsino() -> usize;

    fn astra_rs_geom_create(helical: bool, err: *mut c_char, err_len: usize) -> *mut c_void;
    fn astra_rs_geom_free(ctx: *mut c_void);

    fn astra_rs_fp(
        ctx: *mut c_void,
        vol: *const f32,
        sino_out: *mut f32,
        err: *mut c_char,
        err_len: usize,
    ) -> i32;
    fn astra_rs_fdk(
        ctx: *mut c_void,
        sino: *const f32,
        vol_out: *mut f32,
        err: *mut c_char,
        err_len: usize,
    ) -> i32;

    fn astra_rs_sart_create(
        helical: bool,
        sino_noisy: *const f32,
        n: usize,
        err: *mut c_char,
        err_len: usize,
    ) -> *mut c_void;
    fn astra_rs_sart_free(h: *mut c_void);
    fn astra_rs_sart_run(
        h: *mut c_void,
        vol_in: *const f32,
        n_epochs: i32,
        vol_out: *mut f32,
        err: *mut c_char,
        err_len: usize,
    ) -> i32;

    fn astra_rs_tv(
        vol_inout: *mut f32,
        beta: f32,
        w_z: f32,
        err: *mut c_char,
        err_len: usize,
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
    pub fn create(helical: bool) -> Result<Geom, String> {
        let mut eb = err_buf();
        unsafe {
            let p = astra_rs_geom_create(helical, eb.as_mut_ptr(), eb.len());
            if p.is_null() {
                return Err(take_err(&eb));
            }
            Ok(Geom(p))
        }
    }

    /// 正向投影: vol ([z][y][x]) → sino (row, angle, col)
    pub fn fp(&self, vol: &[f32], sino: &mut [f32]) -> Result<(), String> {
        debug_assert_eq!(vol.len(), NVOL);
        debug_assert_eq!(sino.len(), NSINO);
        let mut eb = err_buf();
        unsafe {
            if astra_rs_fp(
                self.0,
                vol.as_ptr(),
                sino.as_mut_ptr(),
                eb.as_mut_ptr(),
                eb.len(),
            ) != 0
            {
                return Err(take_err(&eb));
            }
        }
        Ok(())
    }

    /// FDK (hann): sino → vol ([z][y][x])
    pub fn fdk(&self, sino: &[f32], vol: &mut [f32]) -> Result<(), String> {
        debug_assert_eq!(sino.len(), NSINO);
        debug_assert_eq!(vol.len(), NVOL);
        let mut eb = err_buf();
        unsafe {
            if astra_rs_fdk(
                self.0,
                sino.as_ptr(),
                vol.as_mut_ptr(),
                eb.as_mut_ptr(),
                eb.len(),
            ) != 0
            {
                return Err(take_err(&eb));
            }
        }
        Ok(())
    }
}

impl Sart {
    pub fn create(helical: bool, sino_noisy: &[f32]) -> Result<Sart, String> {
        debug_assert_eq!(sino_noisy.len(), NSINO);
        let mut eb = err_buf();
        unsafe {
            let p = astra_rs_sart_create(
                helical,
                sino_noisy.as_ptr(),
                sino_noisy.len(),
                eb.as_mut_ptr(),
                eb.len(),
            );
            if p.is_null() {
                return Err(take_err(&eb));
            }
            Ok(Sart(p))
        }
    }

    /// GPU 常驻 OS-SART: vol_in → 运行 n_epochs → vol_out
    /// 调用方需保证 vol_in / vol_out 借用不重叠 (shim 先读入 GPU 再写输出)
    pub fn run(&self, vol_in: &[f32], n_epochs: i32, vol_out: &mut [f32]) -> Result<(), String> {
        debug_assert_eq!(vol_in.len(), NVOL);
        debug_assert_eq!(vol_out.len(), NVOL);
        let mut eb = err_buf();
        unsafe {
            if astra_rs_sart_run(
                self.0,
                vol_in.as_ptr(),
                n_epochs,
                vol_out.as_mut_ptr(),
                eb.as_mut_ptr(),
                eb.len(),
            ) != 0
            {
                return Err(take_err(&eb));
            }
        }
        Ok(())
    }
}

/// TV 去噪 (就地): v = v + beta*div(TV)(v)
pub fn tv(vol: &mut [f32], beta: f32, w_z: f32) -> Result<(), String> {
    debug_assert_eq!(vol.len(), NVOL);
    let mut eb = err_buf();
    unsafe {
        if astra_rs_tv(vol.as_mut_ptr(), beta, w_z, eb.as_mut_ptr(), eb.len()) != 0 {
            return Err(take_err(&eb));
        }
    }
    Ok(())
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
