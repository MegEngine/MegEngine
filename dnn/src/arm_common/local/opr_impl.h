#pragma once

#include "src/naive/local/opr_impl.h"

namespace megdnn {
namespace arm_common {

class LocalImpl final : public naive::LocalForwardImpl {
public:
    using naive::LocalForwardImpl::LocalForwardImpl;

    float_noncontig_batch_kern dispatch_float_noncontig_batch(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) override;

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) override;
};

}  // namespace arm_common
}  // namespace megdnn
// vim: syntax=cpp.doxygen
