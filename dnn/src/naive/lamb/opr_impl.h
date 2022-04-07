#pragma once
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
namespace naive {

class LAMBUpdateImpl final : public LAMBUpdate {
public:
    using LAMBUpdate::LAMBUpdate;
    void exec(
            _megdnn_tensor_in m_t_1, _megdnn_tensor_in v_t_1,
            _megdnn_tensor_in lamb_param, _megdnn_tensor_in grad,
            _megdnn_tensor_out m_t, _megdnn_tensor_out v_t,
            _megdnn_tensor_out new_param, _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout& m_t_1, const TensorLayout& v_t_1,
            const TensorLayout& lamb_param, const TensorLayout& grad,
            const TensorLayout& m_t, const TensorLayout& v_t,
            const TensorLayout& new_param) override {
        MEGDNN_MARK_USED_VAR(m_t_1);
        MEGDNN_MARK_USED_VAR(v_t_1);
        MEGDNN_MARK_USED_VAR(lamb_param);
        MEGDNN_MARK_USED_VAR(grad);
        MEGDNN_MARK_USED_VAR(m_t);
        MEGDNN_MARK_USED_VAR(v_t);
        MEGDNN_MARK_USED_VAR(new_param);
        return 0;
    };
};
}  // namespace naive

}  // namespace megdnn
