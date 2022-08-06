#pragma once
#include "megdnn/oprs.h"
#include "src/common/utils.h"
namespace megdnn {
namespace naive {

class SoftmaxForwardImpl : public SoftmaxForward {
public:
    using SoftmaxForward::SoftmaxForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override;

private:
    size_t reduce_worksize = 0;
    std::unique_ptr<megdnn::Reduce> reduce_opr;
    std::unique_ptr<megdnn::Elemwise> elemwise_opr;
};

class SoftmaxBackwardImpl : public SoftmaxBackward {
public:
    using SoftmaxBackward::SoftmaxBackward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad_x,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& diff,
            const TensorLayout&) override;

private:
    size_t reduce_worksize = 0;
    std::unique_ptr<megdnn::Reduce> reduce_opr;
    std::unique_ptr<megdnn::Elemwise> elemwise_opr;
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
