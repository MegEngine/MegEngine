#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {
class NonZeroImpl : public NonZero {
public:
    using NonZero::NonZero;
    TensorND exec(
            _megdnn_tensor_in src, _megdnn_workspace workspace,
            DynOutMallocPolicyCall malloc_policy);
    size_t get_workspace_in_bytes(const TensorLayout& src);
};
}  // namespace naive
}  // namespace megdnn