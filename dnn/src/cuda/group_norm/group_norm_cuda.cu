#include <stdio.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <cfloat>
#include "megdnn/arch.h"
#include "megdnn/basic_types.h"
#include "megdnn/dtype.h"
#include "src/cuda/cuda_shfl_compat.cuh"
#include "src/cuda/group_norm/group_norm_cuda.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace group_norm {

// warp size may be used as array length, or used in host function,
// so we define WARP_SIZE rather than using warpSize
#define WARP_SIZE 32

template <size_t kStart, size_t kEnd, bool kStop>
struct Compare {
    template <typename T>
    __host__ __device__ inline static bool Run(const T* d1, const T* d2) {
        return d1[kStart] == d2[kStart] &&
               Compare<kStart + 1, kEnd, kStart + 1 == kEnd>::Run(d1, d2);
    }
};

template <size_t kStart, size_t kEnd>
struct Compare<kStart, kEnd, true> {
    template <typename T>
    __host__ __device__ inline constexpr static bool Run(const T* d1, const T* d2) {
        return true;
    }
};

template <size_t N>
using UnrollCompare = Compare<0, N, N == 0>;

template <typename T, size_t kStart, size_t kEnd, bool kStop>
struct UnrollVarArgsAssignImpl {
    template <typename... Args>
    __host__ __device__ inline static void Run(T* d, T val, Args... args) {
        static_assert(sizeof...(args) + 1 == kEnd - kStart, "Wrong argument");
        d[kStart] = val;
        UnrollVarArgsAssignImpl<T, kStart + 1, kEnd, kStart + 1 == kEnd>::Run(
                d, args...);
    }
};

template <typename T, size_t kStart, size_t kEnd>
struct UnrollVarArgsAssignImpl<T, kStart, kEnd, true> {
    __host__ __device__ inline static void Run(T* d) {}
};

template <typename T>
struct UnrollVarArgsAssign {
    template <typename... Args>
    __host__ __device__ inline static void Run(T* d, Args... args) {
        UnrollVarArgsAssignImpl<T, 0, sizeof...(Args), sizeof...(Args) == 0>::Run(
                d, args...);
    }
};

template <typename T, size_t N>
class Array {
public:
    static constexpr size_t kSize = N;

    __host__ __device__ inline Array() {}

    template <typename... Args>
    __host__ __device__ inline explicit Array(const T& val, Args... args) {
        static_assert(N == sizeof...(Args) + 1, "Invalid argument");
        UnrollVarArgsAssign<T>::Run(data_, val, args...);
    }

    __host__ __device__ inline T& operator[](size_t i) { return *(data_ + i); }

    __host__ __device__ inline const T& operator[](size_t i) const {
        return *(data_ + i);
    }

private:
    template <typename U>
    __host__ __device__ static inline U* advance(U* ptr, size_t i) {
        return ptr + i;
    }

    T data_[N];
};

// ================================  group_norm forward ===========================

// implementation of groupnorm_forward from
// https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/group_norm_kernel.cu#L115

template <typename T>
__forceinline__ __device__ T
CudaShuffleDownSync(T val, int delta, int width = warpSize) {
    return __shfl_down(val, static_cast<unsigned>(delta), width);
}

template <>
__forceinline__ __device__ dt_float16
CudaShuffleDownSync(dt_float16 val, int delta, int width) {
    return dt_float16(__shfl_down(val, static_cast<unsigned>(delta), width));
}

template <>
__forceinline__ __device__ dt_bfloat16
CudaShuffleDownSync(dt_bfloat16 val, int delta, int width) {
    return dt_bfloat16(__shfl_down(val, static_cast<unsigned>(delta), width));
}

template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
    T val[VecSize];
};

template <typename T>
struct AddFunctor {
    inline T initial() { return static_cast<T>(0.0f); }

    __device__ __forceinline__ T operator()(const T a, const T b) const {
        return b + a;
    }
};

template <typename T, typename ReduceOp>
__device__ __forceinline__ T WarpReduce(T val, ReduceOp reducer) {
    for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
        T temp = CudaShuffleDownSync<T>(val, stride);
        val = reducer(val, temp);
    }
    return val;
}

template <typename T, typename ReduceOp>
__device__ __forceinline__ T BlockXReduce(T val, ReduceOp reducer) {
    __syncthreads();
    __shared__ T shared[64];
    int block_dim_x = blockDim.x;
    if (blockDim.x > WARP_SIZE) {
        block_dim_x = blockDim.x / WARP_SIZE;
        int lane = threadIdx.x % WARP_SIZE;
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int wid = tid / WARP_SIZE;
        int bid = threadIdx.y;
        val = WarpReduce<T, ReduceOp>(val, reducer);
        if (lane == 0) {
            shared[wid] = val;
        }
        __syncthreads();
        val = shared[bid * block_dim_x + lane];
    }

    for (int stride = 1; stride < block_dim_x; stride <<= 1) {
        T temp = CudaShuffleDownSync(val, stride);
        val = reducer(val, temp);
    }
    if (threadIdx.x == 0) {
        shared[threadIdx.y] = val;
    }
    __syncthreads();
    return shared[threadIdx.y];
}

template <typename T>
__device__ __forceinline__ void ReduceMeanAndVar(
        T* mean, T* var, T x_mean, T x_var, int size) {
    const int nc = blockIdx.x;
    x_mean = BlockXReduce<T, AddFunctor<T>>(x_mean, AddFunctor<T>());
    x_var = BlockXReduce<T, AddFunctor<T>>(x_var, AddFunctor<T>());
    __syncthreads();
    if (threadIdx.x == 0) {
        mean[nc] = static_cast<T>(x_mean / size);
        var[nc] = static_cast<T>(x_var / size);
    }
}

template <typename T, typename T_ACC, int VecSize, int Num>
__device__ __forceinline__ void ThreadReduce(
        Array<const T*, Num> arrs, int size, const int offset, T_ACC* out_mean,
        T_ACC* out_var) {
    const T* x = arrs[0];
    const T* y;
    if (Num == 2) {
        y = arrs[1];
    }
    using VecT = VectorType<T, VecSize>;
    int tid = threadIdx.x;
    if (offset > 0) {
        x -= offset;
        if (Num == 2) {
            y -= offset;
        }
        size += offset;
        if (tid >= offset) {
            if (Num == 1) {
                *out_mean += x[tid];
                *out_var += x[tid] * x[tid];
            } else if (Num == 2) {
                *out_mean += y[tid];
                *out_var += y[tid] * x[tid];
            }
        }
        size -= blockDim.x;
        x += blockDim.x;
        if (Num == 2) {
            y += blockDim.x;
        }
    }
    int remain = size % (VecSize * blockDim.x);

    T ins_x[VecSize];
    T ins_y[VecSize];
    VecT* ins_vec_x = reinterpret_cast<VecT*>(&ins_x);
    VecT* ins_vec_y = reinterpret_cast<VecT*>(&ins_y);

    // vector part
    for (; VecSize * tid < (size - remain); tid += blockDim.x) {
        *ins_vec_x = reinterpret_cast<const VecT*>(x)[tid];
        if (Num == 2) {
            *ins_vec_y = reinterpret_cast<const VecT*>(y)[tid];
        }

#pragma unroll
        for (int i = 0; i < VecSize; ++i) {
            if (Num == 1) {
                *out_mean += ins_x[i];
                *out_var += ins_x[i] * ins_x[i];
            } else if (Num == 2) {
                *out_mean += ins_y[i];
                *out_var += ins_y[i] * ins_x[i];
            }
        }
    }

    // scalar part
    tid = size - remain + threadIdx.x;
    for (; tid < size; tid += blockDim.x) {
        if (Num == 1) {
            *out_mean += x[tid];
            *out_var += x[tid] * x[tid];
        } else if (Num == 2) {
            *out_mean += y[tid];
            *out_var += y[tid] * x[tid];
        }
    }
}

template <typename T, typename T_ACC>
__global__ void ScalarGetMeanAndVar(const T* x, T_ACC* mean, T_ACC* var, int size) {
    int i = blockIdx.x;
    T_ACC x_mean = static_cast<T_ACC>(0);
    T_ACC x_var = static_cast<T_ACC>(0);
    for (int j = threadIdx.x; j < size; j += blockDim.x) {
        T val;
        val = x[i * size + j];
        x_mean += val;
        x_var += val * val;
    }
    ReduceMeanAndVar<T_ACC>(mean, var, x_mean, x_var, size);
}

template <typename T, typename T_ACC, int VecSize>
__global__ void VectorizedGetMeanAndVar(const T* x, T_ACC* mean, T_ACC* var, int size) {
    int i = blockIdx.x;
    T_ACC x_mean = static_cast<T_ACC>(0);
    T_ACC x_var = static_cast<T_ACC>(0);
    x += i * size;
    const int input_offset = ((uint64_t)x) % 16 / sizeof(T);
    Array<const T*, 1> ins;
    ins[0] = x;
    ThreadReduce<T, T_ACC, VecSize, 1>(ins, size, input_offset, &x_mean, &x_var);
    ReduceMeanAndVar<T_ACC>(mean, var, x_mean, x_var, size);
}

template <typename T, typename T_ACC>
__global__ void GroupNormForward(
        const T* x, const T_ACC* mean, const T_ACC* var, const T* scale, const T* bias,
        int N, int C, int W, int imsize, int groups, int group_size, float epsilon,
        T* y, T_ACC* real_var) {
    int gid = blockIdx.y;
    int cid = blockIdx.x;
    int bid = blockIdx.z;
    int ccid = gid * group_size + cid;
    if (ccid >= C)
        return;
    auto ng = bid * groups + gid;
    T_ACC x_mean = mean[ng];
    T_ACC x_var = var[ng];
    x_var = x_var - x_mean * x_mean;
    T_ACC var_inv = rsqrt(x_var + epsilon);
    if (cid == 0 && threadIdx.x == 0) {
        real_var[ng] = x_var;
    }
    for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
        T val;
        int index = (bid * C + ccid) * imsize + imid;
        val = x[index];
        val = (val - x_mean) * var_inv;
        if (scale != nullptr) {
            val *= scale[ccid];
        }
        if (bias != nullptr) {
            val += bias[ccid];
        }
        y[index] = val;
    }
}

template <typename T, typename T_ACC>
void forward(
        T* src, T* weight, T* bias, T* dst, T_ACC* mean, T_ACC* rstd, T_ACC* temp_rstd,
        T_ACC eps, int group, int N, int C, int W, int imsize, cudaStream_t stream) {
    auto group_size = C / group;
    int block_size = std::min(1024, imsize);
    dim3 grid(group_size, group, N);
    dim3 threads(block_size, 1, 1);
    int size = group_size * imsize;
    constexpr int vec_size = sizeof(float4) / sizeof(T);
    int max_block_size = std::min(size / vec_size, 1024);
    int block_size_temp = 1;
    while (block_size_temp < max_block_size) {
        block_size_temp *= 2;
    }
    block_size_temp = std::max(block_size_temp, WARP_SIZE);
    dim3 grids(N * group);
    dim3 blocks(block_size_temp);
    if (size < vec_size * block_size_temp) {
        ScalarGetMeanAndVar<T, T_ACC>
                <<<grids, blocks, 0, stream>>>(src, mean, temp_rstd, size);
        after_kernel_launch();
    } else {
        VectorizedGetMeanAndVar<T, T_ACC, vec_size>
                <<<grids, blocks, 0, stream>>>(src, mean, temp_rstd, size);
        after_kernel_launch();
    }
    GroupNormForward<T, T_ACC><<<grid, threads, 0, stream>>>(
            src, mean, temp_rstd, weight, bias, N, C, W, imsize, group, group_size, eps,
            dst, rstd);
    after_kernel_launch();
}

// ================================  group_norm backward ===========================

// implementation of groupnorm_backward from
// https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/phi/kernels/gpu/group_norm_grad_kernel.cu#L253

template <typename T, typename T_ACC>
__global__ void GetDsDbCUDAKernel(int imsize, const T* x, const T* dy, T* ds, T* db) {
    const int nc = blockIdx.x;
    T ds_sum = static_cast<T>(0);
    T db_sum = static_cast<T>(0);
    for (int i = threadIdx.x; i < imsize; i += blockDim.x) {
        const int index = nc * imsize + i;
        ds_sum += dy[index] * x[index];
        db_sum += dy[index];
    }
    ReduceMeanAndVar<T>(db, ds, db_sum, ds_sum, 1);
}

template <typename T, typename T_ACC>
__global__ void GetBiasGradientCUDAKernel(
        int N, int C, int group, T_ACC epsilon, const T_ACC* mean, const T_ACC* var,
        const T* ds, const T* db, T* d_scale, T* d_bias) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        const int G = group;
        const int D = C / G;
        T sum1 = static_cast<T>(0);
        T sum2 = static_cast<T>(0);
        for (int n = 0; n < N; ++n) {
            const int nc = n * C + c;
            const int ng = n * G + c / D;
            sum1 += (d_scale == nullptr)
                          ? T(0)
                          : ((ds[nc] - db[nc] * static_cast<T>(mean[ng])) *
                             static_cast<T>(rsqrt((float)(var[ng] + epsilon))));
            sum2 += (d_bias == nullptr) ? T(0) : db[nc];
        }
        if (d_scale != nullptr) {
            d_scale[c] = sum1;
        }
        if (d_bias != nullptr) {
            d_bias[c] = sum2;
        }
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
__inline__ MEGDNN_DEVICE T BlockReduceSum(T val, T* shared) {
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

template <typename T, typename T_ACC, int BlockDim>
__global__ void GetBackwardParamsCUDAKernel(
        int imsize, int groups, int group_size, T_ACC epsilon, const T_ACC* mean,
        const T_ACC* var, const T* scale, const T* ds, const T* db, T* p1, T* p2,
        T* p3) {
    const int n = blockIdx.x;
    const int g = blockIdx.y;
    const int ng = n * groups + g;
    T sum1 = static_cast<T>(0);
    T sum2 = static_cast<T>(0);
    T var_inv = static_cast<T>(rsqrt(var[ng] + epsilon));
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        const int64_t index = ng * group_size + i;
        const int64_t c = g * group_size + i;
        const T scale_v = scale == nullptr ? T(1) : static_cast<T>(scale[c]);
        sum1 += ds[index] * scale_v;
        sum2 += db[index] * scale_v;
        const T scale_c = scale == nullptr ? T(1) : static_cast<T>(scale[c]);
        p1[index] = scale_c * var_inv;
    }

    __shared__ T ds_shared[WARP_SIZE];
    __shared__ T db_shared[WARP_SIZE];
    sum1 = BlockReduceSum<T>(sum1, ds_shared);
    sum2 = BlockReduceSum<T>(sum2, db_shared);

    if (threadIdx.x == 0) {
        const T s = T(1) / static_cast<T>(group_size * imsize);
        const T x = (sum2 * static_cast<T>(mean[ng]) - sum1) * static_cast<T>(var_inv) *
                    static_cast<T>(var_inv) * static_cast<T>(var_inv) * s;
        p2[ng] = x;
        p3[ng] = -x * static_cast<T>(mean[ng]) - sum2 * static_cast<T>(var_inv) * s;
    }
}

template <typename T, typename T_ACC>
__global__ void GetXGradientCUDAKernel(
        int imsize, int C, int group_size, int groups, T* p1, T* p2, T* p3, const T* x,
        const T* dy, T* dx) {
    int cid = blockIdx.x;
    int gid = blockIdx.y;
    int bid = blockIdx.z;
    int ccid = bid * C + gid * group_size + cid;
    int ng = bid * groups + gid;
    int nc = gid * group_size + cid;
    for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
        int index = (bid * C + nc) * imsize + imid;
        dx[index] = p1[ccid] * dy[index] + p2[ng] * x[index] + p3[ng];
    }
}

template <typename T, typename T_ACC>
void backward(
        const T* dY_data, const T* X_data, const T_ACC* mean_data,
        const T_ACC* rstd_data, const T* weight_data, T* dX_data, T* dweight_data,
        T* dbias_data, T_ACC eps, int group, int N, int C, int imsize, T* ds, T* db,
        T* p1, T* p2, T* p3, cudaStream_t stream) {
    auto group_size = C / group;
    int block_size = std::min(1024, imsize);
    const int block_dims = 1024;
    dim3 grid(group_size, group, N);
    dim3 threads(block_size, 1, 1);
    const int max_num_threads = 1024;
    int max_block_size = std::min(imsize, max_num_threads);
    int block_size_temp = 1;
    while (block_size_temp < max_block_size) {
        block_size_temp *= 2;
    }
    block_size_temp = std::max(block_size_temp, WARP_SIZE);
    dim3 blocks(block_size_temp);
    GetDsDbCUDAKernel<T, T_ACC>
            <<<N * C, blocks, 0, stream>>>(imsize, X_data, dY_data, ds, db);
    after_kernel_launch();
    bool flag = weight_data != nullptr ? true : false;
    if (flag) {
        const int block = 256;
        GetBiasGradientCUDAKernel<T, T_ACC>
                <<<(C + block - 1) / block, block, 0, stream>>>(
                        N, C, group, eps, mean_data, rstd_data, ds, db, dweight_data,
                        dbias_data);
        after_kernel_launch();
    }

    GetBackwardParamsCUDAKernel<T, T_ACC, block_dims>
            <<<dim3(N, group), block_dims, 0, stream>>>(
                    imsize, group, group_size, eps, mean_data, rstd_data, weight_data,
                    ds, db, p1, p2, p3);
    after_kernel_launch();
    GetXGradientCUDAKernel<T, T_ACC><<<grid, threads, 0, stream>>>(
            imsize, C, group_size, group, p1, p2, p3, X_data, dY_data, dX_data);
    after_kernel_launch();
}

#define INST(T, T_ACC)                                                              \
    template void forward<T, T_ACC>(                                                \
            T*, T*, T*, T*, T_ACC*, T_ACC*, T_ACC*, T_ACC, int, int, int, int, int, \
            cudaStream_t);                                                          \
    template void backward<T, T_ACC>(                                               \
            const T*, const T*, const T_ACC*, const T_ACC*, const T*, T*, T*, T*,   \
            T_ACC, int, int, int, int, T*, T*, T*, T*, T*, cudaStream_t);

INST(dt_float32, dt_float32)
INST(dt_float16, dt_float32)
INST(dt_bfloat16, dt_float32)
#undef INST

}  // namespace group_norm
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
