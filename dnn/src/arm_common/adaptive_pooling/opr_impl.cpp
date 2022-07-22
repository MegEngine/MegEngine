#include "src/arm_common/adaptive_pooling/opr_impl.h"
#include "src/common/opr_delegate.h"
#include "src/naive/handle.h"
namespace megdnn {
namespace arm_common {

void AdaptivePoolingImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    auto adapt_fwd = [=]() {
        auto opr = inplace_cpu_handle()->create_operator<PoolingForward>();
        opr->param() = deduce_pooling_param(src.layout, dst.layout);
        opr->exec(src, dst, workspace);
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(adapt_fwd());
    return;
}

size_t AdaptivePoolingImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    auto opr = inplace_cpu_handle()->create_operator<PoolingForward>();
    opr->param() = deduce_pooling_param(src, dst);
    auto need_size = opr->get_workspace_in_bytes(src, dst);
    return need_size;
}

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
