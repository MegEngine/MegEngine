#pragma once

#include "megdnn/oprs/general.h"
#include "src/common/utils.h"

namespace megdnn {
namespace cuda {
class NonZeroImpl final : public NonZero {
    WorkspaceBundle make_bundle(const TensorLayout& data);

public:
    using NonZero::NonZero;
    virtual TensorND exec(
            _megdnn_tensor_in src, _megdnn_workspace workspace,
            DynOutMallocPolicyCall malloc_policy);
    size_t get_workspace_in_bytes(const TensorLayout& data) override;
};

}  // namespace cuda
}  // namespace megdnn
