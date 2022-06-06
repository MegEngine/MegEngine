#pragma once
#include "megdnn/oprs.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {
class NormForwardImpl : public NormForward {
    using Norm::Norm;

public:
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override;

protected:
    template <Mode mode>
    void dispatch_mode(
            _megdnn_tensor_inout src, _megdnn_tensor_inout dst,
            _megdnn_workspace workspace, size_t A, size_t B, size_t C,
            cudaStream_t stream);
};
}  // namespace cuda
}  // namespace megdnn
