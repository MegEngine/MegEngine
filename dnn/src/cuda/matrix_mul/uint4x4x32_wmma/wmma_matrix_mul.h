#pragma once

#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {
namespace matrix_mul {
void exec_wmma_matrix_mul_quint4_nt(
        _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
        _megdnn_workspace workspace, cudaStream_t stream);
}  // namespace matrix_mul
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
