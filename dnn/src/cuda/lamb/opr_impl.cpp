#include "src/cuda/lamb/opr_impl.h"
#include "./lamb_cuda.cuh"
#include "src/cuda/utils.h"

#include <cmath>
#include <functional>
#include <numeric>
namespace megdnn {
namespace cuda {
void LAMBUpdateImpl::exec(
        _megdnn_tensor_in m_t_1, _megdnn_tensor_in v_t_1, _megdnn_tensor_in lamb_param,
        _megdnn_tensor_in grad, _megdnn_tensor_out m_t, _megdnn_tensor_out v_t,
        _megdnn_tensor_out new_param, _megdnn_workspace workspace) {
    auto p = param();
    float beta_1 = p.beta_1;
    float beta_2 = p.beta_2;
    float step = p.step;
    float lr = p.lr;
    float weight_decay = p.weight_decay;
    float eps = p.eps;
    bool bias_correction = p.bias_correction;
    bool always_adapt = p.always_adapt;
    size_t total_elem = lamb_param.layout.total_nr_elems();
    auto stream = cuda_stream(handle());
    using namespace ::megdnn::cuda::lamb;

#define cb(DType)                                                                     \
    if (grad.layout.dtype == DType()) {                                               \
        using T = typename DTypeTrait<DType>::ctype;                                  \
        using T_ACC = float;                                                          \
        update<T, T_ACC>(                                                             \
                m_t_1.ptr<T_ACC>(), v_t_1.ptr<T_ACC>(), lamb_param.ptr<T_ACC>(),      \
                grad.ptr<T>(), m_t.ptr<T_ACC>(), v_t.ptr<T_ACC>(),                    \
                new_param.ptr<T_ACC>(), workspace.ptr<T_ACC>(), beta_1, beta_2, step, \
                lr, weight_decay, eps, bias_correction, always_adapt, total_elem,     \
                stream);                                                              \
        return;                                                                       \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
}

}  // namespace cuda
}  // namespace megdnn
