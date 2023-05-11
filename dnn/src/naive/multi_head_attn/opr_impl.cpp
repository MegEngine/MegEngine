#include "src/naive/multi_head_attn/opr_impl.h"
#include "megdnn/oprs/linalg.h"
#include "src/common/multi_head_attn/helper.h"
#include "src/common/utils.cuh"

namespace megdnn {
namespace naive {

void MultiHeadAttnForwardImpl::deduce_layout(MHA_FORWARD_LAYOUT_PARAM) {
    proxy_opr.deduce_layout(this->handle(), param(), MHA_FORWARD_CALL);
}

size_t MultiHeadAttnForwardImpl::get_workspace_in_bytes(
        MHA_FORWARD_LAYOUT_CONST_PARAM) {
    return proxy_opr.get_workspace_in_bytes(this->handle(), param(), MHA_FORWARD_CALL);
}

size_t MultiHeadAttnForwardImpl::get_mask_reservespace_in_bytes(
        MHA_FORWARD_LAYOUT_CONST_PARAM) {
    return proxy_opr.get_mask_reservespace_in_bytes(
            this->handle(), param(), MHA_FORWARD_CALL);
}

size_t MultiHeadAttnForwardImpl::get_othr_reservespace_in_bytes(
        MHA_FORWARD_LAYOUT_CONST_PARAM) {
    return proxy_opr.get_othr_reservespace_in_bytes(
            this->handle(), param(), MHA_FORWARD_CALL);
}

void MultiHeadAttnForwardImpl::exec(MHA_FORWARD_EXEC_PARAM) {
    check_exec(MHA_FORWARD_TENSOR_TO_LAYOUT_CALL, workspace.size);
    proxy_opr.exec(this->handle(), param(), MHA_FORWARD_CALL, workspace);
}

void MultiHeadAttnBackwardImpl::exec(MHA_BACKWARD_EXEC_PARAM) {
    check_exec(MHA_BACKWARD_TENSOR_TO_LAYOUT_CALL, workspace.size);
    proxy_opr.exec(this->handle(), param(), MHA_BACKWARD_CALL, workspace);
}
size_t MultiHeadAttnBackwardImpl::get_workspace_in_bytes(
        MHA_BACKWARD_LAYOUT_CONST_PARAM) {
    return proxy_opr.get_workspace_in_bytes(this->handle(), param(), MHA_BACKWARD_CALL);
}

}  // namespace naive
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
