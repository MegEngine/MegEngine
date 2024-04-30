#pragma once

#include "opr_impl.h"
#include <algorithm>
#include "aclnnop/aclnn_group_norm.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_rsqrt.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

namespace megdnn {
namespace atlas {
using Param = megdnn::GroupNorm::Param;

void GroupNormForwardImpl::exec(
        _megdnn_tensor_in data, _megdnn_tensor_in weight, _megdnn_tensor_in bias,
        _megdnn_tensor_out dst, _megdnn_tensor_out mean, _megdnn_tensor_out rstd,
        _megdnn_workspace workspace) {
    Param param = this->param();

    float eps = param.eps;
    bool affine = param.affine;
    size_t N = data.layout.shape[0];
    size_t C = data.layout.shape[1];
    size_t HxW = data.layout.shape[2] * data.layout.shape[3];
    const int64_t G = param.group;

    AclTensor acl_self(data), acl_gamma(weight), acl_beta(bias), acl_out(dst),
            acl_meanOut(mean), acl_rstdOut(rstd);

    uint64_t ws_size;
    aclOpExecutor* executor = nullptr;
    auto handle = concrete_handle(this->handle());

    aclnnGroupNormGetWorkspaceSize(
            acl_self.get(), acl_gamma.get(), acl_beta.get(), N, C, HxW, G, eps,
            acl_out.get(), acl_meanOut.get(), acl_rstdOut.get(), &ws_size, &executor);
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnGroupNorm(ws.ptr(), ws_size, executor, handle->stream()));

    aclnn_check(
            aclnnInplaceRsqrtGetWorkspaceSize(acl_rstdOut.get(), &ws_size, &executor));
    AclMem ws_2(ws_size, handle);
    aclnn_check(aclnnInplaceRsqrt(ws_2.ptr(), ws_size, executor, handle->stream()));

    float exponent_value = 4.0f;
    AclScalar exponent(exponent_value);
    aclnn_check(aclnnInplacePowTensorScalarGetWorkspaceSize(
            acl_rstdOut.get(), exponent.get(), &ws_size, &executor));
    AclMem ws_3(ws_size, handle);
    aclnn_check(aclnnInplacePowTensorScalar(
            ws_3.ptr(), ws_size, executor, handle->stream()));
}

}  // namespace atlas
}  // namespace megdnn