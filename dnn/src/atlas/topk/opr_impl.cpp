#include "opr_impl.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_kthvalue.h"
#include "aclnnop/aclnn_topk.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

namespace megdnn {
namespace atlas {

void TopKImpl::do_find_kth_value(
        int k, _megdnn_tensor_in data, _megdnn_tensor_out values, int32_t*,
        _megdnn_workspace) {
    auto handle = concrete_handle(this->handle());
    if (k < 0) {
        k = data.layout.shape[1] + k + 1;
    }

    AclTensor acl_data(data), acl_values(values);
    auto int64_indices_layout = values.layout;
    int64_indices_layout.dtype = dtype::Complex64();
    AclMem int64_indices_mem(int64_indices_layout.span().dist_byte(), handle);
    TensorND int64_indices_tensor(int64_indices_mem.ptr(), int64_indices_layout);
    AclTensor acl_int64_indices_tensor(int64_indices_tensor, ACL_FORMAT_ND, ACL_INT64);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnn_check(aclnnKthvalueGetWorkspaceSize(
            acl_data.get(), static_cast<int64_t>(k), 1, false, acl_values.get(),
            acl_int64_indices_tensor.get(), &ws_size, &executor));
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnKthvalue(ws.ptr(), ws_size, executor, handle->stream()));
}

void TopKImpl::do_exec(
        int k, _megdnn_tensor_in data, _megdnn_tensor_out values, int32_t* indices,
        _megdnn_workspace workspace) {
    if (param().mode == Param::Mode::KTH_ONLY) {
        do_find_kth_value(k, data, values, indices, workspace);
        return;
    }
    auto handle = concrete_handle(this->handle());

    bool largest = false;
    if (k < 0) {
        largest = true;
    }
    k = static_cast<int>(values.layout.shape[1]);

    bool sorted = m_param.mode == Param::Mode::VALUE_IDX_SORTED;

    AclTensor acl_data(data), acl_values(values);
    auto int64_indices_layout = values.layout;
    int64_indices_layout.dtype = dtype::Complex64();
    AclMem int64_indices_mem(int64_indices_layout.span().dist_byte(), handle);
    TensorND int64_indices_tensor(int64_indices_mem.ptr(), int64_indices_layout);
    AclTensor acl_int64_indices_tensor(int64_indices_tensor, ACL_FORMAT_ND, ACL_INT64);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnn_check(aclnnTopkGetWorkspaceSize(
            acl_data.get(), static_cast<int64_t>(k), 1, largest, sorted,
            acl_values.get(), acl_int64_indices_tensor.get(), &ws_size, &executor));
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnTopk(ws.ptr(), ws_size, executor, handle->stream()));

    auto indices_layout = values.layout;
    indices_layout.dtype = dtype::Int32();
    TensorND indices_tensor(indices, indices_layout);
    AclTensor acl_indices_tensor(indices_tensor);

    uint64_t cast_ws_size = 0;
    aclOpExecutor* cast_executor = nullptr;
    aclnn_check(aclnnCastGetWorkspaceSize(
            acl_int64_indices_tensor.get(), ACL_INT32, acl_indices_tensor.get(),
            &cast_ws_size, &cast_executor));
    AclMem cast_ws(cast_ws_size, handle);
    aclnn_check(
            aclnnCast(cast_ws.ptr(), cast_ws_size, cast_executor, handle->stream()));
}

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
