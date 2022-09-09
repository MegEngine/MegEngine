#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace cuda {

class WhereForwardImpl : public WhereForward {
public:
    using WhereForward::WhereForward;
    void exec(
            _megdnn_tensor_in mask, _megdnn_tensor_in data1, _megdnn_tensor_in data2,
            _megdnn_tensor_out dst, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& mask, const TensorLayout& data1,
            const TensorLayout& data2, const TensorLayout& dst) override {
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
        return 0;
    }
};

}  // namespace cuda
}  // namespace megdnn
