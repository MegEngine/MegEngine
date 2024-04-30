#pragma once

#include "opr_impl.h"
#include "acl/acl_op_compiler.h"
#include "aclnnop/aclnn_bitwise_or_tensor.h"
#include "aclnnop/aclnn_eq_tensor.h"
#include "aclnnop/aclnn_equal.h"
#include "aclnnop/aclnn_lt_tensor.h"
#include "aclnnop/aclnn_ne_tensor.h"
#include "megdnn/tensor_iter.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/handle.h"
#include "src/atlas/utils.h"
#include "src/common/elemwise_multi_type/opr_impl_helper.h"

using namespace megdnn;
using namespace atlas;

void ElemwiseMultiTypeImpl::dest_type_bool_mode(
        const ElemwiseOpParamN<2>& param, const TensorND& dst, Elemwise::Mode mode) {
    auto handle = concrete_handle(this->handle());
    SmallVector<AclTensor> acl_inps;
    for (size_t i = 0; i < 2; ++i) {
        const TensorND& inp = param[i];
        acl_inps.push_back(AclTensor(inp));
    }
    TensorND format_dst(dst.raw_ptr(), TensorLayout(param[0].layout, dst.layout.dtype));
    AclTensor acl_out(format_dst);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    if (mode == Elemwise::Mode::EQ) {
        aclnn_check(aclnnEqTensorGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnEqTensor(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (mode == Elemwise::Mode::LEQ) {
        // x== y
        AclMem out_1_mem(format_dst.layout.access_bytes(), handle);
        AclTensor out_1_tensor(out_1_mem.ptr(), format_dst.layout);

        aclnn_check(aclnnEqTensorGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), out_1_tensor.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnEqTensor(ws.ptr(), ws_size, executor, handle->stream()));

        AclMem out_2_mem(format_dst.layout.access_bytes(), handle);
        AclTensor out_2_tensor(out_2_mem.ptr(), format_dst.layout);

        // x < y
        aclnn_check(aclnnLtTensorGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), out_2_tensor.get(), &ws_size,
                &executor));
        AclMem ws_2(ws_size, handle);
        aclnn_check(aclnnLtTensor(ws_2.ptr(), ws_size, executor, handle->stream()));

        // x <=y  is  (x == y) or (x < y)
        aclnn_check(aclnnBitwiseOrTensorGetWorkspaceSize(
                out_1_tensor.get(), out_2_tensor.get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws_3(ws_size, handle);
        aclnn_check(
                aclnnBitwiseOrTensor(ws_3.ptr(), ws_size, executor, handle->stream()));
    } else if (mode == Elemwise::Mode::LT) {
        aclnn_check(aclnnLtTensorGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnLtTensor(ws.ptr(), ws_size, executor, handle->stream()));
    } else if (mode == Elemwise::Mode::NEQ) {
        aclnn_check(aclnnNeTensorGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnNeTensor(ws.ptr(), ws_size, executor, handle->stream()));
    } else {
        megdnn_assert_internal(0);
    }
}