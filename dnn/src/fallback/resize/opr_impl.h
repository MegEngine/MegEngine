#pragma once

#include "src/naive/resize/opr_impl.h"

namespace megdnn {
namespace fallback {

class ResizeImpl : public naive::ResizeImpl {
public:
    using naive::ResizeImpl::ResizeImpl;

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

private:
    // ctype: C type of input data type.
    template <typename ctype>
    void kern_fallback(const KernParam<ctype>& kern_param);

    template <typename ctype>
    void kern_fallback_nhwc(const KernParam<ctype>& kern_param);

    void exec_fallback(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace);

    void exec_gi(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace);
};  // class ResizeImpl

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
