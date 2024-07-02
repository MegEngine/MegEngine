#include "opr_impl.h"
#include "acl/acl_op_compiler.h"
#include "aclnnop/aclnn_eq_tensor.h"
#include "aclnnop/aclnn_equal.h"
#include "aclnnop/aclnn_le_tensor.h"
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
    megdnn_assert(
            param[0].layout.total_nr_elems() == param[1].layout.total_nr_elems() &&
            param[1].layout.total_nr_elems() == dst.layout.total_nr_elems());

    auto handle = concrete_handle(this->handle());
    SmallVector<AclTensor> acl_inps;
    TensorLayout formated_shape = param[0].layout;
    for (int i = 0; i < 2; i++) {
        if (!param[i].layout.is_contiguous()) {
            formated_shape = param[i].layout;
        }
    }

    for (size_t i = 0; i < 2; ++i) {
        const TensorND& inp = param[i];
        TensorLayout inp_layout = inp.layout;
        if (!inp_layout.eq_shape(formated_shape)) {
            inp_layout = inp_layout.reshape(formated_shape);
        }
        TensorND formated_inp(inp.raw_ptr(), inp_layout);
        acl_inps.push_back(formated_inp);
    }

    TensorLayout dst_layout = dst.layout;
    if (!dst_layout.eq_shape(formated_shape)) {
        dst_layout = dst_layout.reshape(formated_shape);
    }
    TensorND format_dst(dst.raw_ptr(), dst_layout);
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
        aclnn_check(aclnnLeTensorGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnLeTensor(ws.ptr(), ws_size, executor, handle->stream()));
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