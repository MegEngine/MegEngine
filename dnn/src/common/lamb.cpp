#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {

void LAMBUpdate::deduce_layout(
        const TensorLayout& m_t_1, const TensorLayout& v_t_1,
        const TensorLayout& lamb_param, const TensorLayout& grad, TensorLayout& m_t,
        TensorLayout& v_t, TensorLayout& new_param) {
    m_t = TensorLayout(m_t_1);
    v_t = TensorLayout(v_t_1);
    new_param = TensorLayout(lamb_param);
    MEGDNN_MARK_USED_VAR(grad);
}

void LAMBUpdate::check_exec(
        const TensorLayout& m_t_1, const TensorLayout& v_t_1,
        const TensorLayout& lamb_param, const TensorLayout& grad,
        const TensorLayout& m_t, const TensorLayout& v_t, const TensorLayout& new_param,
        size_t workspace_in_bytes) {
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(m_t_1, v_t_1, lamb_param, grad, m_t, v_t, new_param);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}
}  // namespace megdnn
