#pragma once

#include "megdnn/oprs/general.h"
#include "src/common/utils.h"

namespace megdnn {
namespace cambricon {

class CondTakeImpl final : public CondTake {
    WorkspaceBundle make_bundle(const TensorLayout& data, const TensorLayout& mask);

public:
    using CondTake::CondTake;
    Output exec(
            _megdnn_tensor_in data, _megdnn_tensor_in mask, _megdnn_workspace workspace,
            DynOutMallocPolicyCall malloc_policy) override;

    size_t get_workspace_in_bytes(
            const TensorLayout& data, const TensorLayout& mask) override;
};

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
