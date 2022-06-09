#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class CrossImpl : public Cross {
public:
    using Cross::Cross;

    void exec(
            _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

private:
    template <typename ctype>
    void exec_internal(
            ctype* A, size_t a1, size_t b1, size_t c1, ctype* B, size_t a2, size_t b2,
            size_t c2, ctype* C, size_t a3, size_t b3, size_t c3);
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}