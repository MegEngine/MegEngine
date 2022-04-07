#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <cfloat>
#include "megdnn/arch.h"
#include "megdnn/dtype.h"
#include "src/cuda/cuda_shfl_compat.cuh"
#include "src/cuda/lamb/lamb_cuda.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace lamb {

template <typename T>
struct square {
    __host__ __device__ T operator()(const T& x) const { return x * x; }
};

template <typename T, typename T_ACC>
__global__ void update_kernal_1(
        T_ACC* m_t_1, T_ACC* v_t_1, T_ACC* lamb_param, T* grad, T_ACC* m_t, T_ACC* v_t,
        T_ACC* new_param, T_ACC* rt, float beta_1, float beta_2, float step, float lr,
        float weight_decay, float eps, bool bias_correction, bool always_adapt,
        size_t total_nr_elem) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    T_ACC bc_1 = bias_correction ? 1 - pow(beta_1, step) : 1,
          bc_2 = bias_correction ? 1 - pow(beta_2, step) : 1;
    if (idx < total_nr_elem) {
        m_t[idx] = beta_1 * m_t_1[idx] + (1 - beta_1) * static_cast<T_ACC>(grad[idx]);
        v_t[idx] = beta_2 * v_t_1[idx] +
                   (1 - beta_2) * std::pow(static_cast<T_ACC>(grad[idx]), 2);
        rt[idx] = (m_t[idx] / bc_1) / (std::sqrt(v_t[idx] / bc_2) + eps);
        if (weight_decay != 0) {
            rt[idx] += lamb_param[idx] * weight_decay;
        }
    }
}

template <typename T, typename T_ACC>
__global__ void update_kernal_2(
        T_ACC* m_t_1, T_ACC* v_t_1, T_ACC* lamb_param, T* grad, T_ACC* m_t, T_ACC* v_t,
        T_ACC* new_param, T_ACC* rt, float beta_1, float beta_2, float step, float lr,
        float weight_decay, float eps, bool bias_correction, bool always_adapt,
        size_t total_nr_elem, T_ACC trust_ratio) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    T_ACC bc_1 = bias_correction ? 1 - pow(beta_1, step) : 1,
          bc_2 = bias_correction ? 1 - pow(beta_2, step) : 1;
    if (idx < total_nr_elem) {
        rt[idx] = (m_t[idx] / bc_1) / (std::sqrt(v_t[idx] / bc_2) + eps);
        if (weight_decay != 0) {
            rt[idx] += lamb_param[idx] * weight_decay;
        }
        new_param[idx] = lamb_param[idx] - lr * trust_ratio * rt[idx];
    }
}

template <typename T, typename T_ACC>
void update(
        T_ACC* m_t_1, T_ACC* v_t_1, T_ACC* lamb_param, T* grad, T_ACC* m_t, T_ACC* v_t,
        T_ACC* new_param, T_ACC* rt, float beta_1, float beta_2, float step, float lr,
        float weight_decay, float eps, bool bias_correction, bool always_adapt,
        size_t total_nr_elem, cudaStream_t stream) {
    size_t NR_BLOCKS = DIVUP(total_nr_elem, NR_THREADS);
    update_kernal_1<T, T_ACC><<<NR_BLOCKS, NR_THREADS, 0, stream>>>(
            m_t_1, v_t_1, lamb_param, grad, m_t, v_t, new_param, rt, beta_1, beta_2,
            step, lr, weight_decay, eps, bias_correction, always_adapt, total_nr_elem);
    after_kernel_launch();
    thrust::device_ptr<T_ACC> lamb_param_ptr(lamb_param);
    thrust::device_ptr<T_ACC> rt_ptr(rt);
    square<T_ACC> unary_op;
    thrust::plus<T_ACC> binary_op;
    T_ACC p_norm = std::sqrt(thrust::transform_reduce(
            lamb_param_ptr, lamb_param_ptr + total_nr_elem, unary_op, 0.f, binary_op));
    T_ACC d_norm = std::sqrt(thrust::transform_reduce(
            rt_ptr, rt_ptr + total_nr_elem, unary_op, 0.f, binary_op));
    T_ACC trust_ratio = 1;
    if ((always_adapt || weight_decay > 0) && p_norm > 0 && d_norm > 0) {
        trust_ratio = p_norm / d_norm;
    }

    update_kernal_2<T, T_ACC><<<NR_BLOCKS, NR_THREADS, 0, stream>>>(
            m_t_1, v_t_1, lamb_param, grad, m_t, v_t, new_param, rt, beta_1, beta_2,
            step, lr, weight_decay, eps, bias_correction, always_adapt, total_nr_elem,
            trust_ratio);
    after_kernel_launch();
}

#define INST(T, T_ACC)                                                                \
    template void update<T, T_ACC>(                                                   \
            T_ACC*, T_ACC*, T_ACC*, T*, T_ACC*, T_ACC*, T_ACC*, T_ACC*, float, float, \
            float, float, float, float, bool, bool, size_t, cudaStream_t);

INST(dt_float32, dt_float32)
INST(dt_float16, dt_float32)
INST(dt_bfloat16, dt_float32)
#undef INST

}  // namespace lamb
}  // namespace cuda
}  // namespace megdnn
