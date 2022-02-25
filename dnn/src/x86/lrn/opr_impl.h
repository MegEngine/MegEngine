#pragma once
#include "src/naive/lrn/opr_impl.h"

namespace megdnn {
namespace x86 {

class LRNImpl : public naive::LRNForwardImpl {
public:
    using naive::LRNForwardImpl::LRNForwardImpl;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
};

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
