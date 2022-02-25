#pragma once

#include "src/fallback/reduce/opr_impl.h"

namespace megdnn {
namespace arm_common {

class ReduceImpl : public fallback::ReduceImpl {
public:
    using fallback::ReduceImpl::ReduceImpl;

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
