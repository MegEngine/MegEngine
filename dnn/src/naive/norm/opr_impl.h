#pragma once
#include "megdnn/oprs.h"
#include "src/common/reduce_helper.h"
#include "src/naive/reduce/opr_impl.h"

namespace megdnn {
namespace naive {
class NormForwardImpl : public Norm {
public:
    using Norm::Norm;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override;

protected:
    template <Mode mode>
    void dispatch_mode(
            _megdnn_tensor_in src, _megdnn_tensor_out dst, size_t, size_t, size_t);
};
}  // namespace naive
}  // namespace megdnn
