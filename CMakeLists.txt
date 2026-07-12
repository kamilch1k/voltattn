cmake_minimum_required(VERSION 3.18)
project(voltattn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

# Target arch. The kernel is written to the sm_70 feature set (Volta/V100).
#   V100 box (CUDA <= 12.x):  cmake -DCMAKE_CUDA_ARCHITECTURES=70 ...
#   dev box (Ada, CUDA 13.x): defaults to 89 — CUDA 13 removed sm_70 offline
#                             compilation, so Volta builds need a 12.x toolkit.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

add_executable(voltattn src/voltattn.cu)

if(MSVC)
  set(HOST_WARN -Xcompiler=/W4 -Xcompiler=/WX)
else()
  set(HOST_WARN -Xcompiler=-Wall,-Wextra,-Werror)
endif()
target_compile_options(voltattn PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --expt-relaxed-constexpr ${HOST_WARN}>
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Werror all-warnings>")

enable_testing()
add_test(NAME selftest COMMAND voltattn)
