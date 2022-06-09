#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace cuda {

class CrossImpl final : public Cross {
public:
    using Cross::Cross;
    void exec(
            _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& A, const TensorLayout& B,
            const TensorLayout& C) override {
        return 0;
    }
};

}  // namespace cuda
}  // namespace megdnn
   // vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}