#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class EyeImpl : public Eye {
public:
    using Eye::Eye;
    void exec(_megdnn_tensor_out dst, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&) override { return 0; }

private:
    template <typename ctype>
    void exec_internal(ctype* dst, int m, int n);
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
