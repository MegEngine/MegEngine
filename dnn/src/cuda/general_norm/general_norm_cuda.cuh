#pragma once
#include <cuda_runtime_api.h>

namespace megdnn {
namespace cuda {
namespace general_norm {

#define WARP_SIZE 32

template <typename T, typename T_ACC>
void forward(
        T* X_data, T* weight_data, T* bias_data, T* Y_data, T_ACC* mean_data,
        T_ACC* rstd_data, T_ACC eps, int64_t A, int64_t B, int64_t C,
        cudaStream_t stream);

template <typename T, typename T_ACC>
void backward(
        const T* dY_data, const T* X_data, const T* gamma_data, const T_ACC* mean_data,
        const T_ACC* rstd_data, T* dX_data, T* dgamma_data, T* dbeta_data, int64_t A,
        int64_t B, int64_t C, cudaStream_t stream);
}  // namespace general_norm
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
