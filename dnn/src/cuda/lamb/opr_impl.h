#pragma once
#include "megdnn/oprs.h"

#include "src/cuda/cudnn_wrapper.h"
namespace megdnn {
namespace cuda {
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
        return m_t.access_bytes();
    };
};
}  // namespace cuda
}  // namespace megdnn
