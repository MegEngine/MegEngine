#include "src/naive/lamb/opr_impl.h"
#include <cmath>
#include <functional>
#include <numeric>
#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;

namespace {
using Param = megdnn::LAMBUpdate::Param;

template <typename T, typename T_ACC = float>
void update(
        _megdnn_tensor_in m_t_1, _megdnn_tensor_in v_t_1, _megdnn_tensor_in lamb_param,
        _megdnn_tensor_in grad, _megdnn_tensor_out m_t, _megdnn_tensor_out v_t,
        _megdnn_tensor_out new_param, const Param& param) {
    float beta_1 = param.beta_1;
    float beta_2 = param.beta_2;
    float step = param.step;
    float lr = param.lr;
    float weight_decay = param.weight_decay;
    float eps = param.eps;
    bool bias_correction = param.bias_correction;
    bool always_adapt = param.always_adapt;

    size_t total_elem = lamb_param.layout.total_nr_elems();
    T_ACC mt, vt, bc_1, bc_2, rt, d_norm = 0;
    bc_1 = bias_correction ? 1 - pow(beta_1, step) : 1;
    bc_2 = bias_correction ? 1 - pow(beta_2, step) : 1;

    for (size_t i = 0; i < total_elem; i++) {
        mt = m_t.ptr<T_ACC>()[i] = beta_1 * m_t_1.ptr<T_ACC>()[i] +
                                   (1 - beta_1) * static_cast<T_ACC>(grad.ptr<T>()[i]);
        vt = v_t.ptr<T_ACC>()[i] =
                beta_2 * v_t_1.ptr<T_ACC>()[i] +
                (1 - beta_2) * std::pow(static_cast<T_ACC>(grad.ptr<T>()[i]), 2);
        rt = (mt / bc_1) / (sqrt(vt / bc_2) + eps);
        if (weight_decay != 0) {
            rt += lamb_param.ptr<T_ACC>()[i] * weight_decay;
        }
        d_norm += rt * rt;
    }
    d_norm = sqrt(d_norm);
    auto get_norm = [=](_megdnn_tensor_in norm) -> T_ACC {
        return sqrt(std::accumulate(
                norm.ptr<T_ACC>(), norm.ptr<T_ACC>() + total_elem, 0,
                [](T_ACC t1, T_ACC t2) -> T_ACC { return t1 + t2 * t2; }));
    };
    T_ACC p_norm = get_norm(lamb_param), trust_ratio = 1;
    if ((always_adapt || weight_decay > 0) && p_norm > 0 && d_norm > 0) {
        trust_ratio = p_norm / d_norm;
    }
    for (size_t i = 0; i < total_elem; i++) {
        mt = m_t.ptr<T_ACC>()[i];
        vt = v_t.ptr<T_ACC>()[i];
        rt = (mt / bc_1) / (sqrt(vt / bc_2) + eps);
        if (weight_decay != 0) {
            rt += lamb_param.ptr<T_ACC>()[i] * weight_decay;
        }
        new_param.ptr<T_ACC>()[i] = lamb_param.ptr<T_ACC>()[i] - lr * trust_ratio * rt;
    }
}

}  // namespace

namespace megdnn {
namespace naive {
void LAMBUpdateImpl::exec(
        _megdnn_tensor_in m_t_1, _megdnn_tensor_in v_t_1, _megdnn_tensor_in lamb_param,
        _megdnn_tensor_in grad, _megdnn_tensor_out m_t, _megdnn_tensor_out v_t,
        _megdnn_tensor_out new_param, _megdnn_workspace workspace) {
    check_exec(
            m_t_1.layout, v_t_1.layout, lamb_param.layout, grad.layout, m_t.layout,
            v_t.layout, new_param.layout, workspace.size);
#define cb(DType)                                                               \
    if (grad.layout.dtype == DType()) {                                         \
        MEGDNN_DISPATCH_CPU_KERN_OPR(update<typename DTypeTrait<DType>::ctype>( \
                m_t_1, v_t_1, lamb_param, grad, m_t, v_t, new_param, param())); \
        return;                                                                 \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
}
}  // namespace naive

}  // namespace megdnn
