#pragma once
#include "src/naive/repeat/opr_impl.h"

namespace megdnn {
namespace fallback {

class RepeatImpl : public naive::RepeatForwardImpl {
public:
    using RepeatForwardImpl::RepeatForwardImpl;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override;
};

}  // namespace fallback
}  // namespace megdnn
// vim: syntax=cpp.doxygen
