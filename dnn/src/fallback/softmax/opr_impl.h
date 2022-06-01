#pragma once
#include "megdnn/tensor_format.h"
#include "src/common/reduce_helper.h"
#include "src/common/utils.h"
#include "src/naive/softmax/opr_impl.h"

namespace megdnn {
namespace fallback {

class SoftmaxForwardImpl : public naive::SoftmaxForwardImpl {
public:
    using naive::SoftmaxForwardImpl::SoftmaxForwardImpl;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    bool usable(const TensorLayout& src) {
        return src.is_contiguous() && (src.dtype.enumv() == DTypeEnum::Float32) &&
               (src.format.type() == TensorFormat::Type::DEFAULT);
    }
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override {
        if (!usable(src)) {
            return naive::SoftmaxForwardImpl::get_workspace_in_bytes(src, dst);
        }

        auto axis = param().axis;
        if (axis < 0)
            axis += src.ndim;
        typedef DTypeTrait<dtype::Float32>::ctype Float32;

        size_t A, B, C;
        reduce::get_ABC(src, A, B, C, axis);
        if (C != 1) {
            return WorkspaceBundle(
                           nullptr, {A * C * sizeof(Float32), A * C * sizeof(Float32)})
                    .total_size_in_bytes();
        }

        return 0;
    }
};

}  // namespace fallback
}  // namespace megdnn
// vim: syntax=cpp.doxygen
