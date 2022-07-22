#include "src/naive/adaptive_pooling/opr_impl.h"

#include "src/common/opr_delegate.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

namespace megdnn {
namespace naive {
size_t AdaptivePoolingForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    auto opr = inplace_cpu_handle(2)->create_operator<PoolingForward>();
    opr->param() = deduce_pooling_param(src, dst);
    auto need_size = opr->get_workspace_in_bytes(src, dst);
    return need_size;
}
void AdaptivePoolingForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    MEGDNN_DISPATCH_CPU_KERN(static_cast<naive::HandleImpl*>(handle()), {
        auto opr = inplace_cpu_handle(2)->create_operator<PoolingForward>();
        opr->param() = deduce_pooling_param(src.layout, dst.layout);
        opr->exec(src, dst, workspace);
    });
}

void AdaptivePoolingBackwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_tensor_in diff,
        _megdnn_tensor_out grad, _megdnn_workspace workspace) {
    MEGDNN_DISPATCH_CPU_KERN(static_cast<naive::HandleImpl*>(handle()), {
        auto opr = inplace_cpu_handle(2)->create_operator<PoolingBackward>();
        opr->param() = deduce_pooling_param(src.layout, dst.layout);
        opr->exec(src, dst, diff, grad, workspace);
    });
}

size_t AdaptivePoolingBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
        const TensorLayout& grad) {
    auto opr = inplace_cpu_handle(2)->create_operator<PoolingBackward>();
    opr->param() = deduce_pooling_param(src, dst);
    return opr->get_workspace_in_bytes(src, dst, diff, grad);
}
}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
