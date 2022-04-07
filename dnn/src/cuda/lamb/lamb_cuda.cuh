#pragma once
#include <cuda_runtime_api.h>

namespace megdnn {
namespace cuda {
namespace lamb {

template <typename T, typename T_ACC>
void update(
        T_ACC* m_t_1, T_ACC* v_t_1, T_ACC* lamb_param, T* grad, T_ACC* m_t, T_ACC* v_t,
        T_ACC* new_param, T_ACC* rt, float beta_1, float beta_2, float step, float lr,
        float weight_decay, float eps, bool bias_correction, bool always_adapt,
        size_t total_nr_elem, cudaStream_t stream);

}  // namespace lamb
}  // namespace cuda
}  // namespace megdnn
