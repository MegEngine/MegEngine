#pragma once
#include "src/naive/split/opr_impl.h"

namespace megdnn {
namespace fallback {

class SplitImpl : public naive::SplitForwardImpl {
public:
    using SplitForwardImpl::SplitForwardImpl;
    void exec(
            _megdnn_tensor_in src, _megdnn_out const TensorNDArray& dsts,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayoutArray& dsts) override {
        return sizeof(size_t) * dsts.size();
    }
};

}  // namespace fallback
}  // namespace megdnn
// vim: syntax=cpp.doxygen
