#pragma once
#include <cuda_runtime_api.h>

namespace megdnn {
namespace cuda {
namespace group_norm {

template <typename T, typename T_ACC>
void forward(
        T* X, T* gamma, T* beta, T* Y, T_ACC* mean, T_ACC* rstd, T_ACC* tesmp_rstd,
        T_ACC eps, int group, int N, int C, int W, int imsize, cudaStream_t stream);

template <typename T, typename T_ACC>
void backward(
        const T* dY_data, const T* X_data, const T_ACC* mean_data,
        const T_ACC* rstd_data, const T* gamma_data, T* dX_data, T* dgamma_data,
        T* dbeta_data, T_ACC eps, int group, int N, int C, int imsize, T* ds, T* db,
        T* p1, T* p2, T* p3, cudaStream_t stream);

}  // namespace group_norm
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
