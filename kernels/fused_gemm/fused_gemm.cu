// fc1 + bias + GELU reading chosen rows, and fc2 + bias writing chosen rows, with the kernel shapes
// cuBLAS picks in the trunk (256x128 and 128x256, 3 stages, A row-major, B and D column-major, one token per column).
// The bias is C with stride 0, so every token reads the same column. Called from Python through ctypes.
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"

using bf16 = cutlass::bfloat16_t;
using RM = cutlass::layout::RowMajor;
using CM = cutlass::layout::ColumnMajor;
using Op = cutlass::arch::OpClassTensorOp;
using Arch = cutlass::arch::Sm80;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
using Mma = cutlass::gemm::GemmShape<16, 8, 16>;

template <bool GatherB, bool ScatterD, typename Epi, int M, int N, int Stages>
using Gemm = cutlass::gemm::device::GemmUniversal<
    bf16, RM, bf16, CM, bf16, CM, float, Op, Arch, cutlass::gemm::GemmShape<M, N, 32>, cutlass::gemm::GemmShape<64, 64, 32>, Mma,
    Epi, Swizzle, Stages, 8, 8, cutlass::arch::OpMultiplyAdd, cutlass::ComplexTransform::kNone, cutlass::ComplexTransform::kNone,
    false, GatherB, ScatterD>;

using EpiGelu = cutlass::epilogue::thread::LinearCombinationGELU<bf16, 8, float, float>;
using EpiLin = cutlass::epilogue::thread::LinearCombination<bf16, 8, float, float>;

template <typename G, typename P>
static int run(P params, void const *A, void const *B, void const *C, void *D, int m, int n, int k, int ldd,
               int const *gather, int const *scatter, cudaStream_t s) {
  typename G::Arguments a{cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, 1, params, A, B, C, D, 0, 0, 0, 0,
                          RM::Stride(k), CM::Stride(k), CM::Stride(0), CM::Stride(ldd), nullptr, gather, scatter};
  G g;
  auto st = G::can_implement(a);
  if (st != cutlass::Status::kSuccess) return 100 + int(st);
  if (G::get_workspace_size(a) != 0) return 2;
  st = g.initialize(a, nullptr, s);
  if (st != cutlass::Status::kSuccess) return 200 + int(st);
  st = g.run(s);
  return st == cutlass::Status::kSuccess ? 0 : 300 + int(st);
}

// shape: 0 = 256x128 x3, 1 = 128x256 x3, 2 = 128x128 x3, 3 = 128x128 x5, the tiles cuBLAS picks between
template <typename E, bool GB, bool SD, typename P>
static int pick(int shape, P p, void const *A, void const *B, void const *C, void *D, int m, int n, int k, int ldd, int const *g,
                int const *sc, cudaStream_t s) {
  switch (shape) {
    case 0: return run<Gemm<GB, SD, E, 256, 128, 3>>(p, A, B, C, D, m, n, k, ldd, g, sc, s);
    case 1: return run<Gemm<GB, SD, E, 128, 256, 3>>(p, A, B, C, D, m, n, k, ldd, g, sc, s);
    case 2: return run<Gemm<GB, SD, E, 128, 128, 3>>(p, A, B, C, D, m, n, k, ldd, g, sc, s);
    default: return run<Gemm<GB, SD, E, 128, 128, 5>>(p, A, B, C, D, m, n, k, ldd, g, sc, s);
  }
}

extern "C" {
// h[K, H] = gelu(c[idx] @ W1^T + b1), W1 is (H, C) row-major, c is (N, C), idx has K ints
int fc1_gathered(int shape, void const *W1, void const *c, void const *b1, void *h, int const *idx, int K, int H, int C, void *s) {
  using P = decltype(Gemm<true, false, EpiGelu, 256, 128, 3>::Arguments::epilogue);
  return pick<EpiGelu, true, false>(shape, P(1.f, 1.f), W1, c, b1, h, H, K, C, H, idx, nullptr, (cudaStream_t)s);
}
// The same kernels with no indexing: h = gelu(xs @ W1^T + b1) and y = h @ W2^T + b2 on compact buffers
int fc1_plain(int shape, void const *W1, void const *xs, void const *b1, void *h, int const *, int K, int H, int C, void *s) {
  using P = decltype(Gemm<false, false, EpiGelu, 256, 128, 3>::Arguments::epilogue);
  return pick<EpiGelu, false, false>(shape, P(1.f, 1.f), W1, xs, b1, h, H, K, C, H, nullptr, nullptr, (cudaStream_t)s);
}
int fc2_plain(int shape, void const *W2, void const *h, void const *b2, void *y, int const *, int K, int H, int C, void *s) {
  using P = decltype(Gemm<false, false, EpiLin, 256, 128, 3>::Arguments::epilogue);
  return pick<EpiLin, false, false>(shape, P(1.f, 1.f), W2, h, b2, y, C, K, H, C, nullptr, nullptr, (cudaStream_t)s);
}
// out[idx] = h @ W2^T + b2, W2 is (C, H) row-major, h is (K, H), out is (N, C)
int fc2_scattered(int shape, void const *W2, void const *h, void const *b2, void *out, int const *idx, int K, int H, int C, void *s) {
  using P = decltype(Gemm<false, true, EpiLin, 256, 128, 3>::Arguments::epilogue);
  return pick<EpiLin, false, true>(shape, P(1.f, 1.f), W2, h, b2, out, C, K, H, C, nullptr, idx, (cudaStream_t)s);
}
}
