#include "opr_impl.h"
#include "acl/acl_op_compiler.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_copy.h"
#include "aclnnop/aclnn_div.h"
#include "aclnnop/aclnn_exp.h"
#include "aclnnop/aclnn_log.h"
#include "aclnnop/aclnn_masked_select.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_neg.h"
#include "aclnnop/aclnn_relu.h"
#include "aclnnop/aclnn_rsub.h"
#include "aclnnop/aclnn_threshold_backward.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

using namespace megdnn;
using namespace atlas;
using Mode = param::Elemwise::Mode;

void ElemwiseForwardImpl::exec(const TensorNDArray& src, _megdnn_tensor_out dst) {
    auto handle = concrete_handle(this->handle());
    SmallVector<AclTensor> acl_inps;
    for (size_t i = 0; i < src.size(); ++i) {
        acl_inps.emplace_back(src[i]);
    }
    AclTensor acl_out(dst);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    if (m_param.mode == Mode::ADD) {
        aclnn_check(aclnnAddGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(),
                AclScalar(1.0, dst.layout.dtype).get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnAdd(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::MUL) {
        aclnn_check(aclnnMulGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnMul(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::RELU) {
        aclnn_check(aclnnReluGetWorkspaceSize(
                acl_inps[0].get(), acl_out.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnRelu(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::SUB) {
        aclnn_check(aclnnRsubGetWorkspaceSize(
                acl_inps[1].get(), acl_inps[0].get(),
                AclScalar(1.0, dst.layout.dtype).get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnRsubs(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::TRUE_DIV) {
        aclnn_check(aclnnDivGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnDiv(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::EXP) {
        aclnn_check(aclnnExpGetWorkspaceSize(
                acl_inps[0].get(), acl_out.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnExp(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::LOG) {
        aclnn_check(aclnnLogGetWorkspaceSize(
                acl_inps[0].get(), acl_out.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnLog(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::NEGATE) {
        AclTensor other(src[0]);
        aclnn_check(aclnnNegGetWorkspaceSize(
                acl_inps[0].get(), acl_out.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnNeg(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (m_param.mode == Mode::SWITCH_GT0) {
        AclScalar threshold(0.0f);
        aclnn_check(aclnnThresholdBackwardGetWorkspaceSize(
                acl_inps[1].get(), acl_inps[0].get(), threshold.get(), acl_out.get(),
                &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnNeg(ws.ptr(), ws_size, executor, handle->stream()));
    }
}
