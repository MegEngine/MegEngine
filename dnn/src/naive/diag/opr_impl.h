#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class DiagImpl : public Diag {
public:
    using Diag::Diag;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

private:
    template <typename ctype>
    void exec_internal(
            ctype* src, const TensorLayout& src_layout, ctype* dst,
            const TensorLayout& dst_layout, size_t input_ndim, int k);
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
