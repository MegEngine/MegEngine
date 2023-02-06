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
__global__ void general_norm_forward_kernel(
        T* X_data, T* weight_data, T* bias_data, T* Y_data, T_ACC* mean_data,
        T_ACC* rstd_data, T_ACC eps, int64_t A, int64_t B, int64_t C,
        cudaStream_t stream) {
    alignas(sizeof(double)) extern __shared__ char s_data1[];
    T_ACC* share = reinterpret_cast<T_ACC*>(&s_data1);
    int WARP_BLOCK = blockDim.x / WARP_SIZE;
    __shared__ T_ACC mean;
    __shared__ T_ACC rstd;
    size_t a = blockIdx.x / C;
    size_t c = blockIdx.x % C;
    size_t idx = a * B * C + c;

    double sum = 0;
    double sum_sqr = 0;
    for (auto b = threadIdx.x; b < B; b += blockDim.x) {
        T data = X_data[idx + b * C];
        sum += data;
        sum_sqr += data * data;
    }

#pragma unroll
    for (int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, WARP_SIZE);
        sum_sqr += __shfl_down(sum_sqr, offset, WARP_SIZE);
    }

    if (threadIdx.x % WARP_SIZE == 0) {
        share[threadIdx.x / WARP_SIZE * 2] = sum;
        share[threadIdx.x / WARP_SIZE * 2 + 1] = sum_sqr;
    }
    __syncthreads();

    if (threadIdx.x < WARP_BLOCK) {
        sum = share[threadIdx.x * 2];
        sum_sqr = share[threadIdx.x * 2 + 1];
    } else {
        sum = 0;
        sum_sqr = 0;
    }

#pragma unroll
    for (int offset = (WARP_BLOCK >> 1); offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, WARP_SIZE);
        sum_sqr += __shfl_down(sum_sqr, offset, WARP_SIZE);
    }

    if (threadIdx.x == 0) {
        mean = sum / B;
        rstd = sum_sqr / B - mean * mean;
        rstd = 1.0 / sqrt(rstd + eps);
    }
    __syncthreads();

    for (auto b = threadIdx.x; b < B; b += blockDim.x) {
        if (weight_data && bias_data)
            Y_data[idx + b * C] =
                    weight_data[b] * (X_data[idx + b * C] - mean) * rstd + bias_data[b];
        else
            Y_data[idx + b * C] = (X_data[idx + b * C] - mean) * rstd;
    }
    mean_data[a * C + c] = mean;
    rstd_data[a * C + c] = rstd;
}

template <typename T, typename T_ACC = float>
void forward(
        T* X_data, T* weight_data, T* bias_data, T* Y_data, T_ACC* mean_data,
        T_ACC* rstd_data, T_ACC eps, int64_t A, int64_t B, int64_t C,
        cudaStream_t stream) {
    size_t threads = 128;
    if (B > 1024)
        threads = 1024;
    else if (B > 512)
        threads = 512;
    else if (B > 256)
        threads = 256;

    general_norm_forward_kernel<T, T_ACC>
            <<<A * C, threads, 2 * sizeof(T_ACC) * threads / WARP_SIZE, stream>>>(
                    X_data, weight_data, bias_data, Y_data, mean_data, rstd_data, eps,
                    A, B, C, stream);
    after_kernel_launch();
}

template <typename T, typename T_ACC>
__global__ void general_norm_grad_weight_bias_kernel_CX(
        const T* data, const T_ACC* mean, const T_ACC* rstd, const T* diff, T* dweight,
        T* dbias, int64_t A, int64_t B, int64_t C) {
    __shared__ T_ACC block[2][WARP_SIZE];
    size_t b = blockIdx.x;
    T_ACC weight_sum = 0;
    T_ACC bias_sum = 0;
    for (auto a = threadIdx.y; a < A; a += blockDim.y) {
        size_t aBCbC = a * B * C + b * C;
        size_t aC = a * C;
        for (auto c = threadIdx.x; c < C; c += blockDim.x) {
            weight_sum +=
                    (data[aBCbC + c] - mean[aC + c]) * rstd[aC + c] * diff[aBCbC + c];
            bias_sum += diff[aBCbC + c];
        }
    }
    __syncthreads();
#pragma unroll
    for (int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        weight_sum += __shfl_down(weight_sum, offset, WARP_SIZE);
        bias_sum += __shfl_down(bias_sum, offset, WARP_SIZE);
    }
    if (threadIdx.x == 0) {
        block[0][threadIdx.y] = weight_sum;
        block[1][threadIdx.y] = bias_sum;
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        weight_sum = block[0][threadIdx.x];
        bias_sum = block[1][threadIdx.x];
    }
#pragma unroll
    for (int offset = (WARP_SIZE >> 1); offset > 0; offset >>= 1) {
        weight_sum += __shfl_down(weight_sum, offset, WARP_SIZE);
        bias_sum += __shfl_down(bias_sum, offset, WARP_SIZE);
    }
    if (threadIdx.y == 0 && threadIdx.x == 0) {
        dweight[b] = weight_sum;
        dbias[b] = bias_sum;
    }
}

template <typename T, typename T_ACC>
__global__ void general_norm_grad_weight_bias_kernel_C1_THREADY(
        const T* X, const T_ACC* mean, const T_ACC* rstd, const T* dY, T* dweight,
        T* dbias, int64_t A, int64_t B) {
    alignas(sizeof(double)) extern __shared__ char s_data1[];
    T_ACC* s_data_typed = reinterpret_cast<T_ACC*>(&s_data1);
    const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int unroll = 8;
    T dYs[unroll];
    T Xs[unroll];
    T_ACC* means = s_data_typed;
    T_ACC* rstds = s_data_typed + unroll * blockDim.y;
    double dg_sum = 0;
    double db_sum = 0;
    if (j < B) {
        int bcounter;
        for (bcounter = 0; bcounter < A / (blockDim.y * unroll); bcounter++) {
            int offset = (bcounter * blockDim.y + threadIdx.y) * unroll;
#pragma unroll
            for (int ii = 0; ii < unroll; ii++) {
                if (threadIdx.x == 0) {
                    means[ii * blockDim.y + threadIdx.y] = mean[offset + ii];
                    rstds[ii * blockDim.y + threadIdx.y] = rstd[offset + ii];
                }
                dYs[ii] = dY[(offset + ii) * B + j];
                Xs[ii] = X[(offset + ii) * B + j];
            }
            __syncthreads();
#pragma unroll
            for (int ii = 0; ii < unroll; ii++) {
                dg_sum += dYs[ii] * (Xs[ii] - means[ii * blockDim.y + threadIdx.y]) *
                          rstds[ii * blockDim.y + threadIdx.y];
                db_sum += dYs[ii];
            }
            __syncthreads();
        }
        int offset = (bcounter * blockDim.y + threadIdx.y) * unroll;
        for (int ii = 0; ii < 8; ii++) {
            T_ACC mean_val, rstd_val;  // we don't use smem in the tail to avoid awkward
                                       // synchronizations, perf penalty is negligible
            if ((offset + ii) < A) {
                mean_val = mean[offset + ii];
                rstd_val = rstd[offset + ii];
                dYs[0] = dY[(offset + ii) * B + j];
                Xs[0] = X[(offset + ii) * B + j];
                dg_sum += dYs[0] * (Xs[0] - mean_val) * rstd_val;
                db_sum += dYs[0];
            }
        }
        s_data_typed[threadIdx.y * blockDim.x + threadIdx.x] = dg_sum;
        s_data_typed[blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x] =
                db_sum;
        __syncthreads();
        for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
            if (threadIdx.y < offset) {
                s_data_typed[threadIdx.y * blockDim.x + threadIdx.x] +=
                        s_data_typed[(threadIdx.y + offset) * blockDim.x + threadIdx.x];
                s_data_typed
                        [blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
                         threadIdx.x] += s_data_typed
                                [blockDim.x * blockDim.y +
                                 (threadIdx.y + offset) * blockDim.x + threadIdx.x];
            }
            __syncthreads();
        }
        if (threadIdx.y == 0) {
            if (dweight) {
                dweight[j] = s_data_typed[threadIdx.x];
            }
            if (dbias) {
                dbias[j] = s_data_typed[threadIdx.x + blockDim.x * blockDim.y];
            }
        }
    }
}

template <typename T, typename T_ACC>
__global__ void general_norm_grad_weight_bias_kernel_C1_THREADX(
        const T* X, const T_ACC* mean, const T_ACC* rstd, const T* dY, T* dweight,
        T* dbias, int64_t A, int64_t B) {
    alignas(sizeof(double)) extern __shared__ char s_data1[];
    T_ACC* s_data_typed = reinterpret_cast<T_ACC*>(&s_data1);
    const int64_t j = blockIdx.x * blockDim.y + threadIdx.y;
    constexpr int unroll = 8;
    T dYs[unroll];
    T Xs[unroll];
    T_ACC* means = s_data_typed;
    T_ACC* rstds = s_data_typed + unroll * blockDim.x;
    double dg_sum = 0;
    double db_sum = 0;
    if (j < B) {
        int bcounter;
        for (bcounter = 0; bcounter < A / (blockDim.x * unroll); bcounter++) {
            int offset = (bcounter * blockDim.x + threadIdx.x) * unroll;
#pragma unroll
            for (int ii = 0; ii < unroll; ii++) {
                if (threadIdx.y == 0) {
                    means[ii * blockDim.x + threadIdx.x] = mean[offset + ii];
                    rstds[ii * blockDim.x + threadIdx.x] = rstd[offset + ii];
                }
                dYs[ii] = dY[(offset + ii) * B + j];
                Xs[ii] = X[(offset + ii) * B + j];
            }
            __syncthreads();
#pragma unroll
            for (int ii = 0; ii < unroll; ii++) {
                dg_sum += dYs[ii] * (Xs[ii] - means[ii * blockDim.x + threadIdx.x]) *
                          rstds[ii * blockDim.x + threadIdx.x];
                db_sum += dYs[ii];
            }
            __syncthreads();
        }
        int offset = (bcounter * blockDim.x + threadIdx.x) * unroll;
        for (int ii = 0; ii < 8; ii++) {
            T_ACC mean_val, rstd_val;  // we don't use smem in the tail to avoid awkward
                                       // synchronizations, perf penalty is negligible
            if ((offset + ii) < A) {
                mean_val = mean[offset + ii];
                rstd_val = rstd[offset + ii];
                dYs[0] = dY[(offset + ii) * B + j];
                Xs[0] = X[(offset + ii) * B + j];
                dg_sum += dYs[0] * (Xs[0] - mean_val) * rstd_val;
                db_sum += dYs[0];
            }
        }
        __syncthreads();
#pragma unroll
        for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            dg_sum += __shfl_down(dg_sum, offset, WARP_SIZE);
            db_sum += __shfl_down(db_sum, offset, WARP_SIZE);
        }
        if (threadIdx.x == 0) {
            if (dweight)
                dweight[j] = dg_sum;
            if (dbias)
                dbias[j] = db_sum;
        }
    }
}

template <typename T, typename T_ACC = float>
__global__ void general_norm_grad_input_kernel(
        const T* dY_data, const T* X_data, const T* weight_data, const T_ACC* mean_data,
        const T_ACC* rstd_data, T* dX_data, T* dweight_data, T* dbias_data, int64_t A,
        int64_t B, int64_t C) {
    alignas(sizeof(double)) extern __shared__ char s_data1[];
    T_ACC* share = reinterpret_cast<T_ACC*>(&s_data1);
    size_t c = blockIdx.x % C;
    size_t a = blockIdx.x / C;
    size_t idx = a * B * C + c;
    int WARP_BLOCK = blockDim.x / WARP_SIZE;
    __shared__ T_ACC atemp;
    __shared__ T_ACC btemp;
    __shared__ T_ACC ctemp;

    double db = 0;
    double ds = 0;
    for (auto b = threadIdx.x; b < B; b += blockDim.x) {
        T data = X_data[idx + b * C];
        T diff = dY_data[idx + b * C];
        T w = weight_data != nullptr ? weight_data[b] : static_cast<T>(1.0f);
        db += static_cast<T_ACC>(diff) * static_cast<T_ACC>(w);
        ds += static_cast<T_ACC>(diff) * static_cast<T_ACC>(data) *
              static_cast<T_ACC>(w);
    }

#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        db += __shfl_down(db, offset, WARP_SIZE);
        ds += __shfl_down(ds, offset, WARP_SIZE);
    }
    if (threadIdx.x % WARP_SIZE == 0) {
        share[threadIdx.x / WARP_SIZE * 2] = db;
        share[threadIdx.x / WARP_SIZE * 2 + 1] = ds;
    }
    __syncthreads();
    if (threadIdx.x < WARP_BLOCK) {
        db = share[threadIdx.x * 2];
        ds = share[threadIdx.x * 2 + 1];
    } else {
        db = 0;
        ds = 0;
    }
#pragma unroll
    for (int offset = WARP_BLOCK >> 1; offset > 0; offset >>= 1) {
        db += __shfl_down(db, offset, WARP_SIZE);
        ds += __shfl_down(ds, offset, WARP_SIZE);
    }
    if (threadIdx.x == 0) {
        atemp = rstd_data[a * C + c];
        btemp = (db * mean_data[a * C + c] - ds) * atemp * atemp * atemp / B;
        ctemp = -btemp * mean_data[a * C + c] - db * atemp / B;
    }
    __syncthreads();
    for (auto b = threadIdx.x; b < B; b += blockDim.x) {
        size_t idx = a * B * C + b * C + c;
        T w = weight_data != nullptr ? weight_data[b] : static_cast<T>(1.0f);
        dX_data[idx] = dY_data[idx] * atemp * w + X_data[idx] * btemp + ctemp;
    }
}

template <typename T, typename T_ACC = float>
void backward(
        const T* dY_data, const T* X_data, const T* weight_data, const T_ACC* mean_data,
        const T_ACC* rstd_data, T* dX_data, T* dweight_data, T* dbias_data, int64_t A,
        int64_t B, int64_t C, cudaStream_t stream) {
    if (weight_data && dweight_data && dbias_data) {
        if (C == 1) {
            dim3 threads{32, 16};
            int blocks = (B + threads.y - 1) / threads.y;
            general_norm_grad_weight_bias_kernel_C1_THREADY<T, T_ACC>
                    <<<blocks, threads, 2 * sizeof(T_ACC) * threads.x * threads.y,
                       stream>>>(
                            X_data, mean_data, rstd_data, dY_data, dweight_data,
                            dbias_data, A, B);

            // dim3 threads{32, 16};
            // int blocks = (B + threads.y - 1) / threads.y;
            // general_norm_grad_weight_bias_kernel_C1_THREADX<T, T_ACC>
            //         <<<blocks, threads, 2 * sizeof(T_ACC) * threads.x * threads.y,
            //            stream>>>(
            //                 X_data, mean_data, rstd_data, dY_data, dweight_data,
            //                 dbias_data, A, B);

            // dim3 threads{16, 32};
            // int blocks = (B + threads.x - 1) / threads.x;
            // general_norm_grad_weight_bias_kernel_C1_THREADY<T, T_ACC>
            //         <<<blocks, threads, 2 * sizeof(T_ACC) * threads.x * threads.y,
            //            stream>>>(
            //                 X_data, mean_data, rstd_data, dY_data, dweight_data,
            //                 dbias_data, A, B);
        } else {
            dim3 thread{32, 32};
            general_norm_grad_weight_bias_kernel_CX<T, T_ACC><<<B, thread, 0, stream>>>(
                    X_data, mean_data, rstd_data, dY_data, dweight_data, dbias_data, A,
                    B, C);
        }

        after_kernel_launch();
    }

    size_t threads = 64;
    if (B > 1024)
        threads = 1024;
    else if (B > 512)
        threads = 512;
    else if (B > 256)
        threads = 256;
    else if (B > 128)
        threads = 128;

    general_norm_grad_input_kernel<T, T_ACC>
            <<<A * C, threads, 2 * sizeof(T_ACC) * threads / WARP_SIZE, stream>>>(
                    dY_data, X_data, weight_data, mean_data, rstd_data, dX_data,
                    dweight_data, dbias_data, A, B, C);
    after_kernel_launch();
}

#define INST(T, T_ACC)                                                            \
    template void forward<T, T_ACC>(                                              \
            T*, T*, T*, T*, T_ACC*, T_ACC*, T_ACC, int64_t, int64_t, int64_t,     \
            cudaStream_t);                                                        \
    template void backward<T, T_ACC>(                                             \
            const T*, const T*, const T*, const T_ACC*, const T_ACC*, T*, T*, T*, \
            int64_t, int64_t, int64_t, cudaStream_t);

INST(dt_float32, dt_float32)
INST(dt_float16, dt_float32)
INST(dt_bfloat16, dt_float32)
#undef INST

}  // namespace general_norm
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
