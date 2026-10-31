#!/bin/sh
# Builds libfused_gemm.so. Needs the NVIDIA CUDA toolkit, git, and network access for the first CUTLASS download.
#   sh kernels/fused_gemm/build.sh
#   NVCC=/usr/local/cuda-12.4/bin/nvcc sh kernels/fused_gemm/build.sh
set -e
cd "$(dirname "$0")"
[ -d cutlass ] || git clone --depth 1 --branch v3.8.0 https://github.com/NVIDIA/cutlass.git cutlass
${NVCC:-nvcc} -std=c++17 -O3 -DNDEBUG --expt-relaxed-constexpr -arch=${ARCH:-sm_86} -shared -Xcompiler -fPIC -Icutlass/include \
  fused_gemm.cu -o libfused_gemm.so
echo "built $PWD/libfused_gemm.so"
