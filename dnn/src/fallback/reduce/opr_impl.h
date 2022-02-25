#pragma once
#include "src/naive/reduce/opr_impl.h"

namespace megdnn {
namespace fallback {

class ReduceImpl : public naive::ReduceForwardImpl {
public:
    using ReduceForwardImpl::ReduceForwardImpl;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace) override;
    bool exec_optimized(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace);
    void exec_fallback(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace);
};

}  // namespace fallback
}  // namespace megdnn
// vim: syntax=cpp.doxygen
