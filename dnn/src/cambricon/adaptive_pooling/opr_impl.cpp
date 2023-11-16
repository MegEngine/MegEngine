#include "src/cambricon/adaptive_pooling/opr_impl.h"
#include <vector>
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/utils.h"

namespace megdnn {
namespace cambricon {

// TODO: implement adaptive pooling with cnnlAdaptivePooling
void AdaptivePoolingForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    auto opr = handle()->create_operator<PoolingForward>();
    opr->param() = deduce_pooling_param(src.layout, dst.layout);
    opr->exec(src, dst, workspace);
}

size_t AdaptivePoolingForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    auto opr = handle()->create_operator<PoolingForward>();
    opr->param() = deduce_pooling_param(src, dst);
    return opr->get_workspace_in_bytes(src, dst);
}

void AdaptivePoolingBackwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_tensor_in diff,
        _megdnn_tensor_out grad, _megdnn_workspace workspace) {
    auto opr = handle()->create_operator<PoolingBackward>();
    opr->param() = deduce_pooling_param(src.layout, dst.layout);
    opr->exec(src, dst, diff, grad, workspace);
}

size_t AdaptivePoolingBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
        const TensorLayout& grad) {
    auto opr = handle()->create_operator<PoolingBackward>();
    opr->param() = deduce_pooling_param(src, dst);
    return opr->get_workspace_in_bytes(src, dst, diff, grad);
}

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
