#pragma once
#include "megdnn/oprs.h"
namespace megdnn {
namespace x86 {
class SeparableFilterImpl : public SeparableFilterForward {
public:
    using SeparableFilterForward::SeparableFilterForward;
    using Param = param::SeparableFilter;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter_x,
            _megdnn_tensor_in filter_y, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&) override {
        return 0;
    }

private:
    template <typename T>
    void separable_filter_exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter_x,
            _megdnn_tensor_in filter_y, _megdnn_tensor_out dst, const Param& param);
    void separable_filter_exec_8u(
            _megdnn_tensor_in src, _megdnn_tensor_in filter_x,
            _megdnn_tensor_in filter_y, _megdnn_tensor_out dst, const Param& param);
};

}  // namespace x86
}  // namespace megdnn
// vim: syntax=cpp.doxygen
