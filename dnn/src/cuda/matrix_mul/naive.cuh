#pragma once
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {

template <typename AType, typename BType, typename CType, typename CompType>
void exec_gemm_naive(
        const AType* A, const BType* B, CType* C, size_t m, size_t n, size_t k,
        size_t ldA, size_t ldB, size_t ldC, bool transA, bool transB,
        cudaStream_t stream);
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
