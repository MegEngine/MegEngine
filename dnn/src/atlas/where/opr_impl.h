#pragma once
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
namespace atlas {

class WhereForwardImpl : public WhereForward {
public:
    using WhereForward::WhereForward;
    void exec(
            _megdnn_tensor_in mask, _megdnn_tensor_in data1, _megdnn_tensor_in data2,
            _megdnn_tensor_out dst, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&) override {
        return 0;
    }
};

class WhereBackwardImpl : public WhereBackward {
public:
    using WhereBackward::WhereBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_in mask,
            _megdnn_tensor_out grad_data1, _megdnn_tensor_out grad_data2,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& diff, const TensorLayout& mask,
            const TensorLayout& grad_data1, const TensorLayout& grad_data2) override {
        MEGDNN_MARK_USED_VAR(diff);
        MEGDNN_MARK_USED_VAR(mask);
        MEGDNN_MARK_USED_VAR(grad_data1);
        MEGDNN_MARK_USED_VAR(grad_data2);
        return 0;
    }
};

}  // namespace atlas
}  // namespace megdnn