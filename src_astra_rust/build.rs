// build.rs — 编译 C shim (g++ 包装 ASTRA C++/CUDA) 与 TV CUDA 内核 (nvcc),
// 打包为静态库并链接 libastra.so.0 + cudart + stdc++。
// 无第三方 crate 依赖, 全程调用系统工具链。
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn run(cmd: &mut Command) {
    let status = cmd.status().expect("无法启动编译命令");
    assert!(status.success(), "编译失败: {cmd:?}");
}

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // ---- 路径 ----
    let astra_headers = crate_dir.join("../src_astra_cpp/third_party/astra/include");
    assert!(
        astra_headers.is_dir(),
        "缺少 ASTRA 头文件: {astra_headers:?} (需先有 src_astra_cpp)"
    );
    let astra_lib = env::var("ASTRA_LIBRARY").unwrap_or_else(|_| {
        crate_dir
            .join("../.venv/lib/python3.12/site-packages/astra/libastra.so.0")
            .to_str()
            .unwrap()
            .to_string()
    });
    let astra_dir = Path::new(&astra_lib)
        .parent()
        .expect("ASTRA_LIBRARY 路径无效");
    assert!(
        Path::new(&astra_lib).exists(),
        "找不到 libastra.so.0: {astra_lib}"
    );
    let cuda_dir = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda-12.6".to_string());
    let cuda_arch = env::var("CUDA_ARCHITECTURES").unwrap_or_else(|_| "sm_75".to_string()); // GTX 1660 = Turing

    // ---- 1. nvcc 编译 TV CUDA 内核 ----
    let tv_o = out_dir.join("tv_kernel.o");
    run(Command::new("nvcc")
        .arg("-c")
        .arg(crate_dir.join("c_api/tv_kernel.cu"))
        .arg("-o")
        .arg(&tv_o)
        .arg(format!("-arch={cuda_arch}"))
        .arg("-O2"));

    // ---- 2. g++ 编译 C shim (ASTRA 是 C++ 类库, 必须 C++ 编译器) ----
    let shim_o = out_dir.join("astra_c_api.o");
    run(Command::new("g++")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-DASTRA_CUDA")
        .arg(format!("-I{}", astra_headers.display()))
        .arg(format!("-I{cuda_dir}/include")) // cuda_runtime.h
        .arg("-c")
        .arg(crate_dir.join("c_api/astra_c_api.cpp"))
        .arg("-o")
        .arg(&shim_o));

    // ---- 3. ar 打包静态库 ----
    let shim_a = out_dir.join("libastrars_shim.a");
    run(Command::new("ar")
        .arg("rcs")
        .arg(&shim_a)
        .arg(&tv_o)
        .arg(&shim_o));

    // ---- 4. 链接指令 ----
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=astrars_shim");
    println!("cargo:rustc-link-search=native={}", astra_dir.display());
    println!("cargo:rustc-link-arg=-l:libastra.so.0"); // 精确文件名 (动态库, 顺序无关)
    println!("cargo:rustc-link-search=native={cuda_dir}/lib64");
    println!("cargo:rustc-link-lib=cudart"); // TV 内核宿主代码 (cudaMemcpy 等)
    println!("cargo:rustc-link-lib=stdc++"); // shim 是 C++ 对象
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", astra_dir.display());

    println!("cargo:rerun-if-changed=c_api/astra_c_api.cpp");
    println!("cargo:rerun-if-changed=c_api/tv_kernel.cu");
    println!("cargo:rerun-if-changed=build.rs");
}
