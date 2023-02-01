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

constexpr int kCUDANumThreads = 256;
constexpr int vec_size = 4;

// warp size may be used as array length, or used in host function,
// so we define WARP_SIZE rather than using warpSize
#define WARP_SIZE 32

#if defined(__clang__)
#define __ubsan_ignore_float_divide_by_zero__ \
    __attribute__((no_sanitize("float-divide-by-zero")))
#else
#define __ubsan_ignore_float_divide_by_zero__
#endif

struct WelfordStat {
    float mean;
    float sigma2;
    float count;
    MEGDNN_HOST MEGDNN_DEVICE WelfordStat() : mean(0.f), sigma2(0.f), count(0.f) {}
    MEGDNN_HOST MEGDNN_DEVICE WelfordStat(float mean, float sigma2, float count)
            : mean(mean), sigma2(sigma2), count(count) {}
};

template <typename T, typename combine_t>
struct WelfordData {
    T mean;
    T sigma2;
    combine_t count;

    MEGDNN_HOST MEGDNN_DEVICE WelfordData() : mean(0), sigma2(0), count(0) {}

    MEGDNN_HOST MEGDNN_DEVICE WelfordData(T mean, T sigma2, combine_t count)
            : mean(mean), sigma2(sigma2), count(count) {}
};

template <typename T, typename combine_t, typename res_t>
struct WelfordOps {
public:
    using WelfordData_T = WelfordData<T, combine_t>;
    inline MEGDNN_DEVICE WelfordData_T reduce(WelfordData_T acc, T data) const {
        T delta = data - acc.mean;
        T new_mean = static_cast<T>(acc.mean + delta / (acc.count + 1));
        T new_delta = static_cast<T>(data - new_mean);
        return {
                new_mean,
                acc.sigma2 + delta * new_delta,
                combine_t(acc.count + 1),
        };
    }
    inline MEGDNN_DEVICE WelfordData_T
    combine(WelfordData_T lhs, WelfordData_T rhs) const {
        if (lhs.count != 0 && rhs.count != 0) {
            T delta = rhs.mean - lhs.mean;
            combine_t new_count = lhs.count + rhs.count;
            T nb_over_n = rhs.count / new_count;
            return {lhs.mean + delta * nb_over_n,
                    lhs.sigma2 + rhs.sigma2 + delta * delta * lhs.count * nb_over_n,
                    new_count};
        } else {
            return (lhs.count != 0) ? lhs : rhs;
        }
    }
    inline MEGDNN_DEVICE res_t
    project(WelfordData_T acc) const __ubsan_ignore_float_divide_by_zero__ {
        const auto mean = static_cast<T>(acc.mean);
        const combine_t divisor = static_cast<combine_t>(acc.count);
        const auto var = acc.sigma2 / divisor;
        res_t results(var, mean);
        return results;
    }

#if defined(__CUDACC__) || defined(__HIPCC__)
    inline MEGDNN_DEVICE WelfordData_T
    warp_shfl_down(WelfordData_T acc, int offset) const {
        return {__shfl_down(acc.mean, offset, warpSize),
                __shfl_down(acc.sigma2, offset, warpSize),
                __shfl_down(acc.count, offset, warpSize)};
    }
#endif
    MEGDNN_HOST MEGDNN_DEVICE WelfordOps() {}
};

template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_vector {
    T val[vec_size];
};

template <typename T, bool is_cuda>
using acc_type = T;

template <typename U>
MEGDNN_DEVICE WelfordStat
update_welford_stat_online(const U val, const WelfordStat& curr_sum) {
    U delta = static_cast<U>(val - curr_sum.mean);
    U new_count = static_cast<U>(curr_sum.count + 1.f);
    U new_mean = static_cast<U>(curr_sum.mean + delta * (1.f / new_count));
    return {new_mean, curr_sum.sigma2 + delta * (val - new_mean), new_count};
}

MEGDNN_DEVICE WelfordStat
combine_welford_stat(const WelfordStat lhs, const WelfordStat rhs) {
    using U = decltype(lhs.count);
    U delta = lhs.mean - rhs.mean;
    U count = rhs.count + lhs.count;
    U mean, sigma2;
    if (count > decltype(lhs.count){0}) {
        auto coef = 1.f / count;
        auto nA = rhs.count * coef;
        auto nB = lhs.count * coef;
        mean = nA * rhs.mean + nB * lhs.mean;
        sigma2 = rhs.sigma2 + lhs.sigma2 + delta * delta * rhs.count * nB;
    } else {
        mean = U(0);
        sigma2 = U(0);
    }
    return {mean, sigma2, count};
}

template <typename T>
MEGDNN_DEVICE WelfordStat
compute_stats(const T* __restrict__ X, const int slice_len, float* buf) {
    using vec_t = aligned_vector<T, vec_size>;
    using acc_t = acc_type<T, true>;
    const vec_t* X_vec = reinterpret_cast<const vec_t*>(X);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = slice_len / vec_size;
    WelfordStat w_stat(0.f, 0.f, 0.f);
    for (int i = thrx; i < n_vec_to_read; i += numx) {
        vec_t data = X_vec[i];
#pragma unroll
        for (int ii = 0; ii < vec_size; ii++) {
            w_stat = update_welford_stat_online(
                    static_cast<acc_t>(data.val[ii]), w_stat);
        }
    }
    // intra-warp reduction
#pragma unroll
    for (int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
        WelfordStat w_tmp{
                __shfl_down(w_stat.mean, offset, warpSize),
                __shfl_down(w_stat.sigma2, offset, warpSize),
                __shfl_down(w_stat.count, offset, warpSize)};
        w_stat = combine_welford_stat(w_stat, w_tmp);
    }

    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
        float* mean_sigma_buf = buf;
        float* count_buf = buf + blockDim.y;
        for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
            // upper half of warps write to shared
            if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2 * offset) {
                const int wrt_y = threadIdx.y - offset;
                mean_sigma_buf[2 * wrt_y] = w_stat.mean;
                mean_sigma_buf[2 * wrt_y + 1] = w_stat.sigma2;
                count_buf[wrt_y] = w_stat.count;
            }
            __syncthreads();

            // lower half merges
            if (threadIdx.x == 0 && threadIdx.y < offset) {
                WelfordStat w_tmp{
                        mean_sigma_buf[2 * threadIdx.y],
                        mean_sigma_buf[2 * threadIdx.y + 1], count_buf[threadIdx.y]};
                w_stat = combine_welford_stat(w_stat, w_tmp);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            mean_sigma_buf[0] = w_stat.mean;
            mean_sigma_buf[1] = w_stat.sigma2 / float(slice_len);
        }
        __syncthreads();
        return WelfordStat{mean_sigma_buf[0], mean_sigma_buf[1], 0.f};

    } else {
        return WelfordStat{
                __shfl(w_stat.mean, 0, warpSize),
                __shfl(w_stat.sigma2, 0, warpSize) / float(slice_len), 0.f};
    }
}

template <typename T, typename T_ACC>
__global__ void vectorized_general_norm_forward_affine_kernel(
        const int slice_len, T_ACC eps, const T* __restrict__ X, const T* weight,
        const T* bias, T_ACC* mean, T_ACC* rstd, T* Y) {
    // if we made smem WelfordStat type, there would be bank conflicts,
    // as one thread would have to write 3 consecutive floats
    extern __shared__ float s_data[];

    auto slice_id = blockIdx.x;
    const T* slice = X + slice_id * slice_len;
    WelfordStat slice_w_stat = compute_stats(slice, slice_len, s_data);
    using vec_t = aligned_vector<T, vec_size>;
    const vec_t* X_vec = reinterpret_cast<const vec_t*>(slice);
    vec_t* Y_vec = reinterpret_cast<vec_t*>(Y + slice_id * slice_len);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = slice_len / vec_size;
    T_ACC rstd_val = static_cast<T_ACC>(rsqrt(slice_w_stat.sigma2 + eps));

    for (int i = thrx; i < n_vec_to_read; i += numx) {
        vec_t data = X_vec[i];
        vec_t out;
        // computation is performed in T_ACC, X is cast to T_ACC and result is
        // implicitly cast to T

#pragma unroll
        for (int ii = 0; ii < vec_size; ii++) {
            out.val[ii] = static_cast<T_ACC>(weight[i * vec_size + ii]) *
                                  (rstd_val * (static_cast<T_ACC>(data.val[ii]) -
                                               slice_w_stat.mean)) +
                          static_cast<T_ACC>(bias[i * vec_size + ii]);
        }
        Y_vec[i] = out;
    }
    if (thrx == 0) {
        mean[slice_id] = slice_w_stat.mean;
        rstd[slice_id] = rstd_val;
    }
}

template <typename T, typename T_ACC>
__global__ void vectorized_general_norm_forward_kernel(
        const int slice_len, T_ACC eps, const T* __restrict__ X, const T* weight,
        const T* bias, T_ACC* mean, T_ACC* rstd, T* Y) {
    extern __shared__ float s_data[];

    auto slice_id = blockIdx.x;
    const T* slice = X + slice_id * slice_len;
    WelfordStat slice_w_stat = compute_stats(slice, slice_len, s_data);
    using vec_t = aligned_vector<T, vec_size>;
    const vec_t* X_vec = reinterpret_cast<const vec_t*>(slice);
    vec_t* Y_vec = reinterpret_cast<vec_t*>(Y + slice_id * slice_len);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const int n_vec_to_read = slice_len / vec_size;
    T_ACC rstd_val = static_cast<T_ACC>(rsqrt(slice_w_stat.sigma2 + eps));

    for (int i = thrx; i < n_vec_to_read; i += numx) {
        vec_t data = X_vec[i];
        vec_t out;

#pragma unroll
        for (int ii = 0; ii < vec_size; ii++) {
            out.val[ii] =
                    rstd_val * (static_cast<T_ACC>(data.val[ii]) - slice_w_stat.mean);
        }
        Y_vec[i] = out;
    }
    if (thrx == 0) {
        mean[slice_id] = slice_w_stat.mean;
        rstd[slice_id] = rstd_val;
    }
}

template <typename T, typename T_ACC>
void launch_vectorized_general_norm_forward_kernel(
        int64_t slice_len, int64_t slice_num, T_ACC eps, const T* X_data,
        const T* weight_data, const T* bias_data, T* Y_data, T_ACC* mean_data,
        T_ACC* rstd_data, cudaStream_t stream) {
    const int num_threads = 128;
    const dim3 threads(WARP_SIZE, num_threads / WARP_SIZE, 1);
    const dim3 blocks(slice_num);
    int nshared = threads.y > 1 ? threads.y * 3 / 2 * sizeof(T_ACC) : 0;

    if (weight_data == nullptr && bias_data == nullptr) {
        vectorized_general_norm_forward_kernel<<<blocks, threads, nshared, stream>>>(
                slice_len, eps, X_data, weight_data, bias_data, mean_data, rstd_data,
                Y_data);
    } else {
        vectorized_general_norm_forward_affine_kernel<<<
                blocks, threads, nshared, stream>>>(
                slice_len, eps, X_data, weight_data, bias_data, mean_data, rstd_data,
                Y_data);
    }
    after_kernel_launch();
}

template <typename T, class ReduceOp>
__inline__ MEGDNN_DEVICE T welford_warp_reduce(T val, const ReduceOp& op) {
#pragma unroll
    for (int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
        val = op.combine(val, op.warp_shfl_down(val, offset));
    }
    return val;
}

template <typename T, class ReduceOp>
__inline__ MEGDNN_DEVICE T
welford_block_reduce(T val, const ReduceOp& op, const T& identity_element, T* shared) {
    const int lid = threadIdx.x % warpSize;
    const int wid = threadIdx.x / warpSize;
    val = welford_warp_reduce(val, op);
    __syncthreads();
    if (lid == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lid] : identity_element;
    if (wid == 0) {
        val = welford_warp_reduce(val, op);
    }
    return val;
}

template <typename T, typename T_ACC>
__global__ void get_input_mean_and_rstd_kernel(
        int64_t slice_len, T_ACC eps, const T* X, T_ACC* mean, T_ACC* rstd) {
    using WelfordType = WelfordData<T_ACC, T_ACC>;
    using WelfordOp = WelfordOps<T_ACC, T_ACC, thrust::pair<T_ACC, T_ACC>>;

    __shared__ typename std::aligned_storage<
            sizeof(WelfordType), alignof(WelfordType)>::type val_shared[WARP_SIZE];
    WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(val_shared);

    const int64_t i = blockIdx.x;
    WelfordOp welford_op;
    WelfordType val(
            static_cast<T_ACC>(0), static_cast<T_ACC>(0), static_cast<T_ACC>(0));

    for (int64_t j = threadIdx.x; j < slice_len; j += blockDim.x) {
        const int64_t index = i * slice_len + j;
        val = welford_op.reduce(val, static_cast<T_ACC>(X[index]));
    }
    val = welford_block_reduce(
            val, welford_op,
            WelfordType(
                    static_cast<T_ACC>(0), static_cast<T_ACC>(0),
                    static_cast<T_ACC>(0)),
            val_shared_ptr);

    if (threadIdx.x == 0) {
        T_ACC slice_mean;
        T_ACC slice_sigma2;
        thrust::tie(slice_sigma2, slice_mean) = welford_op.project(val);
        mean[i] = slice_mean;
        rstd[i] = rsqrt(slice_sigma2 + eps);
    }
}

template <typename T, typename T_ACC>
__global__ void general_norm_forward_kernel(
        int64_t slice_len, const T* X, const T_ACC* mean, const T_ACC* rstd,
        const T* weight, const T* bias, T* Y) {
    const int64_t i = blockIdx.x;
    for (int64_t j = threadIdx.x; j < slice_len; j += blockDim.x) {
        const int64_t index = i * slice_len + j;
        const T_ACC weight_v =
                weight == nullptr ? T_ACC(1) : static_cast<T_ACC>(weight[j]);
        const T_ACC bias_v = bias == nullptr ? T_ACC(0) : static_cast<T_ACC>(bias[j]);
        Y[index] = (static_cast<T_ACC>(X[index]) - static_cast<T_ACC>(mean[i])) *
                           static_cast<T_ACC>(rstd[i]) * weight_v +
                   bias_v;
    }
}

template <typename T, typename T_ACC>
void forward(
        T* X, T* weight, T* bias, int64_t slice_num, int64_t slice_len, T_ACC eps, T* Y,
        T_ACC* mean, T_ACC* rstd, cudaStream_t stream) {
    auto can_vectorize = [&](const T* ptr, int alignment) {
        uint64_t addr = reinterpret_cast<uint64_t>(ptr);
        return addr % alignment == 0;
    };
    constexpr int num_vec_elems = vec_size;
    constexpr int alignment = num_vec_elems * sizeof(T);
    if ((std::is_same<T, dt_float32>::value || std::is_same<T, dt_float16>::value ||
         std::is_same<T, dt_bfloat16>::value) &&
        slice_len <= static_cast<int64_t>(1ULL << std::numeric_limits<float>::digits) &&
        slice_len % num_vec_elems == 0 && can_vectorize(X, alignment) &&
        can_vectorize(Y, alignment)) {
        launch_vectorized_general_norm_forward_kernel<T, T_ACC>(
                slice_len, slice_num, static_cast<T_ACC>(eps), X, weight, bias, Y, mean,
                rstd, stream);
        after_kernel_launch();
    } else {
        get_input_mean_and_rstd_kernel<T, T_ACC>
                <<<slice_num, 512, 0, stream>>>(slice_len, eps, X, mean, rstd);
        after_kernel_launch();
        general_norm_forward_kernel<T, T_ACC><<<slice_num, kCUDANumThreads, 0, stream>>>(
                slice_len, X, mean, rstd, weight, bias, Y);
        after_kernel_launch();
    }
}

template <typename T>
__inline__ MEGDNN_DEVICE T warp_reduce_sum(T val) {
#pragma unroll
    for (int offset = (warpSize >> 1); offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset, warpSize);
    }
    return val;
}

template <typename T>
__inline__ MEGDNN_DEVICE T block_reduce_sum(T val, T* shared) {
    const int lid = threadIdx.x % warpSize;
    const int wid = threadIdx.x / warpSize;
    val = warp_reduce_sum(val);
    __syncthreads();
    if (lid == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lid] : T(0);
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

template <typename T, typename T_ACC>
__inline__ MEGDNN_DEVICE void general_norm_grad_input_kernel_impl(
        const T* __restrict__ dY, const T* __restrict__ X,
        const T_ACC* __restrict__ mean, const T_ACC* __restrict__ rstd,
        const T* __restrict__ weight, T* dX, const int slice_len, T_ACC* buf) {
    const auto slice_id = blockIdx.x;
    const T_ACC mean_val = mean[slice_id];
    const T_ACC rstd_val = rstd[slice_id];
    T_ACC stats_x1{0}, stats_x2{0};
    constexpr int unroll = 4;
    auto l = unroll * threadIdx.x;
    const T* X_i = X + slice_id * slice_len;
    const T* dY_i = dY + slice_id * slice_len;
    T* dX_i = dX + slice_id * slice_len;
    // vectorized reads don't improve perf, so use regular unrolling

    for (; l + unroll - 1 < slice_len; l += blockDim.x * unroll) {
#pragma unroll
        for (int k = 0; k < unroll; k++) {
            T_ACC weight_val =
                    (weight != nullptr) ? static_cast<T_ACC>(weight[l + k]) : T_ACC(1);
            const T_ACC c_h = static_cast<T_ACC>(X_i[l + k]);
            const T_ACC c_loss = static_cast<T_ACC>(dY_i[l + k]);
            stats_x1 += c_loss * weight_val;
            stats_x2 += c_loss * weight_val * (c_h - mean_val) * rstd_val;
        }
    }
    for (; l < slice_len; l++) {
        T_ACC weight_val =
                (weight != nullptr) ? static_cast<T_ACC>(weight[l]) : T_ACC(1);
        const T_ACC c_h = static_cast<T_ACC>(X_i[l]);
        const T_ACC c_loss = static_cast<T_ACC>(dY_i[l]);
        stats_x1 += c_loss * weight_val;
        stats_x2 += c_loss * weight_val * (c_h - mean_val) * rstd_val;
    }

    stats_x1 = block_reduce_sum(stats_x1, buf);
    stats_x2 = block_reduce_sum(stats_x2, buf);
    if (threadIdx.x == 0) {
        buf[0] = stats_x1;
        buf[1] = stats_x2;
    }
    __syncthreads();
    stats_x1 = buf[0];
    stats_x2 = buf[1];
    T_ACC fH = slice_len;
    T_ACC term1 = (T_ACC(1) / fH) * rstd_val;

    for (int l = threadIdx.x; l < slice_len; l += blockDim.x) {
        const T_ACC x = X_i[l];
        const T_ACC dy = dY_i[l];
        T_ACC weight_val =
                (weight != nullptr) ? static_cast<T_ACC>(weight[l]) : T_ACC(1);
        T_ACC f_grad_input = fH * weight_val * dy;
        f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
        f_grad_input -= stats_x1;
        f_grad_input *= term1;
        dX_i[l] = f_grad_input;
    }
}

template <typename T, typename T_ACC>
__global__ void general_norm_grad_input_kernel(
        const T* __restrict__ dY, const T* __restrict__ X,
        const T_ACC* __restrict__ mean, const T_ACC* __restrict__ rstd,
        const T* __restrict__ weight, T* dX, const int slice_len) {
    alignas(sizeof(double)) extern __shared__ char s_data1[];
    T_ACC* buf = reinterpret_cast<T_ACC*>(&s_data1);

    general_norm_grad_input_kernel_impl(dY, X, mean, rstd, weight, dX, slice_len, buf);
}

template <typename T, typename T_ACC>
__global__ void general_norm_grad_weight_bias_simple_kernel(
        int64_t slice_num, int64_t slice_len, const T* dY, const T* X,
        const T_ACC* mean, const T_ACC* rstd, T* dweight, T* dbias) {
    const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < slice_len) {
        T_ACC sum1 = 0;
        T_ACC sum2 = 0;
        for (int64_t i = 0; i < slice_num; ++i) {
            const int64_t index = i * slice_len + j;
            sum1 += dweight == nullptr ? T_ACC(0)
                                       : static_cast<T_ACC>(dY[index]) *
                                                 (static_cast<T_ACC>(X[index]) -
                                                  static_cast<T_ACC>(mean[i])) *
                                                 static_cast<T_ACC>(rstd[i]);
            sum2 += dbias == nullptr ? T_ACC(0) : static_cast<T_ACC>(dY[index]);
        }
        if (dweight != nullptr) {
            dweight[j] = sum1;
        }
        if (dbias != nullptr) {
            dbias[j] = sum2;
        }
    }
}

template <typename T, typename T_ACC>
__global__ void general_norm_grad_weight_bias_kernel(
        int64_t slice_num, int64_t slice_len, const T* dY, const T* X,
        const T_ACC* mean, const T_ACC* rstd, T* dweight, T* dbias) {
    alignas(sizeof(double)) extern __shared__ char s_data1[];
    T_ACC* s_data_typed = reinterpret_cast<T_ACC*>(&s_data1);
    const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int unroll = 8;
    T dYs[unroll];
    T Xs[unroll];
    T_ACC* means = s_data_typed;
    T_ACC* rstds = s_data_typed + unroll * blockDim.y;
    T_ACC dg_sum = 0;
    T_ACC db_sum = 0;
    if (j < slice_len) {
        int bcounter;
        for (bcounter = 0; bcounter < slice_num / (blockDim.y * unroll); bcounter++) {
            int offset = (bcounter * blockDim.y + threadIdx.y) * unroll;
#pragma unroll
            for (int ii = 0; ii < unroll; ii++) {
                if (threadIdx.x == 0) {
                    means[ii * blockDim.y + threadIdx.y] = mean[offset + ii];
                    rstds[ii * blockDim.y + threadIdx.y] = rstd[offset + ii];
                }
                dYs[ii] = dY[(offset + ii) * slice_len + j];
                Xs[ii] = X[(offset + ii) * slice_len + j];
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
            if ((offset + ii) < slice_num) {
                mean_val = mean[offset + ii];
                rstd_val = rstd[offset + ii];
                dYs[0] = dY[(offset + ii) * slice_len + j];
                Xs[0] = X[(offset + ii) * slice_len + j];
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
void backward(
        const T* dY_data, const T* X_data, const T_ACC* mean_data,
        const T_ACC* rstd_data, const T* weight_data, int64_t slice_num,
        int64_t slice_len, T* dX_data, T* dweight_data, T* dbias_data,
        cudaStream_t stream) {
    if (dX_data != nullptr) {
        const int num_threads = 128;
        const dim3 blocks(slice_num);
        int nshared = (num_threads / WARP_SIZE) * sizeof(T_ACC);
        general_norm_grad_input_kernel<<<blocks, num_threads, nshared, stream>>>(
                dY_data, X_data, mean_data, rstd_data, weight_data, dX_data, slice_len);
        after_kernel_launch();
    }
    if (dweight_data || dbias_data) {
        if (slice_num < 512) {
            const int64_t B = (slice_len + kCUDANumThreads - 1) / kCUDANumThreads;
            general_norm_grad_weight_bias_simple_kernel<T, T_ACC>
                    <<<B, kCUDANumThreads, 0, stream>>>(
                            slice_num, slice_len, dY_data, X_data, mean_data, rstd_data,
                            dweight_data, dbias_data);
            after_kernel_launch();
        } else {
            dim3 threads{16, 32};
            int blocks = (slice_len + threads.x - 1) / threads.x;
            general_norm_grad_weight_bias_kernel<T, T_ACC>
                    <<<blocks, threads, 2 * sizeof(T_ACC) * threads.x * threads.y,
                       stream>>>(
                            slice_num, slice_len, dY_data, X_data, mean_data, rstd_data,
                            dweight_data, dbias_data);
            after_kernel_launch();
        }
    }
}

#define INST(T, T_ACC)                                                              \
    template void forward<T, T_ACC>(                                                \
            T*, T*, T*, int64_t, int64_t, T_ACC, T*, T_ACC*, T_ACC*, cudaStream_t); \
    template void backward<T, T_ACC>(                                               \
            const T*, const T*, const T_ACC*, const T_ACC*, const T*, int64_t,      \
            int64_t, T*, T*, T*, cudaStream_t);

INST(dt_float32, dt_float32)
INST(dt_float16, dt_float32)
INST(dt_bfloat16, dt_float32)
#undef INST

}  // namespace general_norm
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
