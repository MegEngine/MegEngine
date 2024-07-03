#include "opr_impl.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_sort.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

namespace megdnn {
namespace atlas {

size_t ArgsortForwardImpl::get_workspace_in_bytes(
        const TensorLayout&, const TensorLayout&, const TensorLayout&) {
    return 0;
}

void ArgsortForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_tensor_out indices,
        _megdnn_workspace) {
    auto handle = concrete_handle(this->handle());
    AclTensor acl_src(src), acl_dst(dst), acl_indices(indices);

    TensorLayout int64_indices_layout(TensorShape(indices.layout), dtype::Complex64());
    AclMem acl_int64_indices_mem(int64_indices_layout.span().dist_byte(), handle);
    AclTensor acl_int64_indices(
            acl_int64_indices_mem.ptr(), int64_indices_layout, ACL_FORMAT_ND,
            ACL_INT64);

    int64_t dim = 1;
    bool descending = param().order == param::Argsort::Order::DESCENDING;
    bool stable = false;
    uint64_t sort_ws_size = 0;
    aclOpExecutor* sort_executor = nullptr;
    aclnn_check(aclnnSortGetWorkspaceSize(
            acl_src.get(), stable, dim, descending, acl_dst.get(),
            acl_int64_indices.get(), &sort_ws_size, &sort_executor));
    AclMem acl_sort_mem(sort_ws_size, handle);
    aclnn_check(aclnnSort(
            acl_sort_mem.ptr(), sort_ws_size, sort_executor, handle->stream()));

    uint64_t cast_ws_size = 0;
    aclOpExecutor* cast_executor = nullptr;
    aclnn_check(aclnnCastGetWorkspaceSize(
            acl_int64_indices.get(), ACL_INT32, acl_indices.get(), &cast_ws_size,
            &cast_executor));
    AclMem acl_cast_mem(cast_ws_size, handle);
    aclnn_check(aclnnCast(
            acl_cast_mem.ptr(), cast_ws_size, cast_executor, handle->stream()));
}

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
