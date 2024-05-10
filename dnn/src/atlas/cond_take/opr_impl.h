#pragma once

#include "megdnn/oprs/general.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/handle.h"
#include "src/common/utils.h"

namespace megdnn {
namespace atlas {

class CondTakeImpl final : public CondTake {
public:
    using CondTake::CondTake;
    Output exec(
            _megdnn_tensor_in data, _megdnn_tensor_in mask, _megdnn_workspace workspace,
            DynOutMallocPolicyCall malloc_policy) override;

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
};

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
