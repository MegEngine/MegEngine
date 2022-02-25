#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace cuda {

class SplitForwardImpl : public SplitForward {
public:
    using SplitForward::SplitForward;
    void exec(
            _megdnn_tensor_in src, const TensorNDArray& dsts,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayoutArray&) override;

private:
    template <typename T>
    void exec_internal(
            _megdnn_tensor_in src, const TensorNDArray& dsts,
            _megdnn_workspace workspace);
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
