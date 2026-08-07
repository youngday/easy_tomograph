cmake_minimum_required(VERSION 3.18)
project(astra_cpp LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 目标 GPU 架构 (GTX 1660 = Turing, sm_75)。其他显卡可用 -DCUDA_ARCHITECTURES=xx 覆盖,
# 或设为 "all" / "native"(CMake 3.23+)。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# ASTRA C++ 库 (pip 包自带 libastra.so.0, 无头文件; 头文件来自 astra-toolbox v2.5.0,
# 已打包在 third_party/astra/include). 可通过 -DASTRA_LIBRARY=... 覆盖。
set(ASTRA_LIBRARY "" CACHE FILEPATH "Path to ASTRA libastra.so.0")
if(NOT ASTRA_LIBRARY)
  set(ASTRA_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/lib/python3.12/site-packages/astra/libastra.so.0")
endif()
if(NOT EXISTS "${ASTRA_LIBRARY}")
  message(FATAL_ERROR "ASTRA library not found: ${ASTRA_LIBRARY}. Pass -DASTRA_LIBRARY=<path to libastra.so.0>")
endif()
get_filename_component(ASTRA_DIR "${ASTRA_LIBRARY}" DIRECTORY)

add_library(astracpp_common STATIC src/common.cpp src/tv_kernel.cu)
target_include_directories(astracpp_common PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/astra/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)
target_compile_definitions(astracpp_common PUBLIC ASTRA_CUDA)
target_compile_options(astracpp_common PRIVATE -O3)
target_link_libraries(astracpp_common PUBLIC CUDA::cudart)
if(OpenMP_CXX_FOUND)
  target_link_libraries(astracpp_common PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(astra_axial src/main_axial.cpp)
add_executable(astra_helical src/main_helical.cpp)
foreach(tgt astra_axial astra_helical)
  target_link_libraries(${tgt} astracpp_common "${ASTRA_LIBRARY}")
  set_target_properties(${tgt} PROPERTIES BUILD_RPATH "${ASTRA_DIR}")
endforeach()
