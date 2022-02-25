#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class CondTakeImpl : public CondTake {
    template <typename ctype>
    void dispatch_genidx(size_t size, dt_int32* dest, const TensorND& mask);

public:
    using CondTake::CondTake;

    size_t get_workspace_in_bytes(const TensorLayout& data) override;

    Output exec(
            _megdnn_tensor_in data, _megdnn_tensor_in mask, _megdnn_workspace workspace,
            DynOutMallocPolicyCall malloc_policy) override;
};

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
