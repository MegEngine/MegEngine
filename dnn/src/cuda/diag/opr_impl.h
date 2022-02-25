#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace cuda {

class DiagImpl final : public Diag {
public:
    using Diag::Diag;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override {
        return 0;
    }
};

}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
