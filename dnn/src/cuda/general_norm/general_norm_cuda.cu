#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <cfloat>
#include "megdnn/arch.h"
#include "megdnn/dtype.h"
#include "src/cuda/cuda_shfl_compat.cuh"
#include "src/cuda/general_norm/general_norm_cuda.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace general_norm {

template <typename T, typename T_ACC = float>
__global__ void forward_kernel(
        T* X_data, T* weight_data, T* bias_data, T* Y_data, T_ACC* mean_data,
        T_ACC* rstd_data, T_ACC eps, int64_t A, int64_t B, int64_t C, cudaStream_t stream) {
    for (int64_t a = 0; a < A; ++a)
        for (int64_t c = 0; c < C; ++c) {
            T_ACC slice_sum = static_cast<T>(0.0f);
            for (int64_t b = 0; b < B; b++) {
                auto value = X_data[a * B * C + b * C + c];
                slice_sum += value;
            }
            T_ACC slice_mean = static_cast<T>(slice_sum / B);

            T_ACC slice_var = static_cast<T>(0.0f);
            for (int64_t b = 0; b < B; b++) {
                slice_var += (X_data[a * B * C + b * C + c] - slice_mean) *
                             (X_data[a * B * C + b * C + c] - slice_mean);
            }
            slice_var = slice_var / B;

            T_ACC slice_std = static_cast<T>(sqrt(slice_var + eps));
            for (int64_t b = 0; b < B; b++) {
                Y_data[a * B * C + b * C + c] =
                        (X_data[a * B * C + b * C + c] - slice_mean) / slice_std;
                if (weight_data || bias_data) {
                    Y_data[a * B * C + b * C + c] =
                            Y_data[a * B * C + b * C + c] * weight_data[b] +
                            bias_data[b];
                }
            }
            mean_data[a * C + c] = static_cast<T_ACC>(slice_mean);
            rstd_data[a * C + c] = static_cast<T_ACC>(1.0 / slice_std);
        }
}

template <typename T, typename T_ACC = float>
void forward(
        T* X_data, T* weight_data, T* bias_data, T* Y_data, T_ACC* mean_data,
        T_ACC* rstd_data, T_ACC eps, int64_t A, int64_t B, int64_t C, cudaStream_t stream) {
    printf("Gpu general forward\n");
    forward_kernel<T, T_ACC>
                <<<1, 1, 0, stream>>>(X_data, weight_data, bias_data, Y_data, mean_data, 
                rstd_data, eps, A, B, C, stream);
}

template <typename T, typename T_ACC = float>
__global__ void backward_kernel(
        const T* dY_data, const T* X_data, const T* weight_data, const T_ACC* mean_data,
        const T_ACC* rstd_data, T* dX_data, T* dweight_data, T* dbias_data, int64_t A,
        int64_t B, int64_t C, cudaStream_t stream) {
    if (dweight_data || dbias_data) {
        for (int64_t b = 0; b < B; ++b) {
            dweight_data[b] = 0;
            dbias_data[b] = 0;
        }

        for (int64_t a = 0; a < A; ++a)
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t b = 0; b < B; ++b) {
                    dweight_data[b] += (X_data[a * B * C + b * C + c] -
                                            mean_data[a * C + c]) *
                                           rstd_data[a * C + c] *
                                           dY_data[a * B * C + b * C + c];

                    dbias_data[b] += dY_data[a * B * C + b * C + c];
                }
            }
    }

    for (int64_t a = 0; a < A; ++a)
        for (int64_t c = 0; c < C; ++c) {
            T_ACC ds = static_cast<T_ACC>(0.0f);
            T_ACC db = static_cast<T_ACC>(0.0f);
            T_ACC atmp = static_cast<T_ACC>(0.0f);
            T_ACC btmp = static_cast<T_ACC>(0.0f);
            T_ACC ctmp = static_cast<T_ACC>(0.0f);

            for (int64_t b = 0; b < B; ++b) {
                auto value = X_data[a * B * C + b * C + c];
                auto dY_v = dY_data[a * B * C + b * C + c];
                auto weight_v = weight_data ? weight_data[b] : static_cast<T>(1.0f);
                db += dY_v * weight_v;
                ds += dY_v * value * weight_v;
            }

            atmp = rstd_data[a * C + c];
            btmp = (db * mean_data[a * C + c] - ds) * atmp * atmp * atmp / B;
            ctmp = -btmp * mean_data[a * C + c] - db * atmp / B;

            for (int64_t b = 0; b < B; b++) {
                auto weight_v = weight_data ? weight_data[b] : static_cast<T>(1.0f);
                dX_data[a * B * C + b * C + c] =
                        dY_data[a * B * C + b * C + c] * atmp * weight_v +
                        X_data[a * B * C + b * C + c] * btmp + ctmp;
            }
        }


}

template <typename T, typename T_ACC = float>
void backward(
        const T* dY_data, const T* X_data, const T* weight_data, const T_ACC* mean_data,
        const T_ACC* rstd_data, T* dX_data, T* dweight_data, T* dbias_data, int64_t A,
        int64_t B, int64_t C, cudaStream_t stream) {
    backward_kernel<T, T_ACC>
                <<<1, 1, 0, stream>>>(dY_data, X_data, weight_data, mean_data, rstd_data, dX_data, 
                dweight_data, dbias_data, A, B, C, stream);  
}

#define INST(T, T_ACC)                                                              \
    template void forward<T, T_ACC>(                                                \
            T*, T*, T*, T*,T_ACC*, T_ACC*, T_ACC, int64_t,int64_t,int64_t, cudaStream_t); \
    template void backward<T, T_ACC>(                                               \
            const T*, const T*, const T*, const T_ACC*, const T_ACC*,       \
            T*, T*, T*, int64_t, int64_t, int64_t, cudaStream_t);

INST(dt_float32, dt_float32)
INST(dt_float16, dt_float32)
INST(dt_bfloat16, dt_float32)
#undef INST

}  // namespace general_norm
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
