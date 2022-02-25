#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace cuda {

class ConcatForwardImpl : public ConcatForward {
public:
    using ConcatForward::ConcatForward;
    void exec(
            _megdnn_in const TensorNDArray& srcs, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayoutArray&, const TensorLayout&) override;

private:
    template <typename T>
    void exec_internal(
            _megdnn_in const TensorNDArray& srcs, _megdnn_tensor_out dst,
            _megdnn_workspace workspace);
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
