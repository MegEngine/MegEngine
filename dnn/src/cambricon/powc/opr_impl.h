#pragma once
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
namespace cambricon {

class PowCImpl : public PowC {
    template <typename T>
    void do_exec_ct(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace);
    WorkspaceBundle make_bundle(const TensorLayout& src, const TensorLayout& dst);

public:
    using PowC::PowC;
    void do_exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, const float* exp_f,
            const int* exp_i) override {}
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
};

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
