#pragma once
#include "src/naive/concat/opr_impl.h"

namespace megdnn {
namespace fallback {

class ConcatImpl : public naive::ConcatForwardImpl {
public:
    using ConcatForwardImpl::ConcatForwardImpl;
    void exec(
            _megdnn_in const TensorNDArray& srcs, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayoutArray& srcs, const TensorLayout&) override {
        return sizeof(size_t) * srcs.size();
    }
};

}  // namespace fallback
}  // namespace megdnn
// vim: syntax=cpp.doxygen
