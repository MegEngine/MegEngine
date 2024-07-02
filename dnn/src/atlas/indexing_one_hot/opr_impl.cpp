#include "./opr_impl.h"

#include "acl/acl.h"
#include "aclnnop/aclnn_gather.h"
#include "aclnnop/aclnn_scatter.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

namespace megdnn {
namespace atlas {

void IndexingOneHotForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in index, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    check_exec(src.layout, index.layout, dst.layout, workspace.size);
    AclTensor acl_src(src), acl_dst(dst);
    uint64_t ws_size;
    auto handle = concrete_handle(this->handle());
    aclOpExecutor* executor = nullptr;

    int dim = this->param().axis;
    TensorLayout index_layout = index.layout;
    index_layout.add_axis_inplace(dim, 1, index_layout.stride[dim]);
    AclTensor acl_index(index.raw_ptr(), index_layout);
    aclnn_check(aclnnGatherGetWorkspaceSize(
            acl_src.get(), dim, acl_index.get(), acl_dst.get(), &ws_size, &executor));
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnGather(ws.ptr(), ws_size, executor, handle->stream()));
}

void IndexingSetOneHotForwardImpl::exec(
        _megdnn_tensor_inout src, _megdnn_tensor_in index, _megdnn_tensor_in sub,
        _megdnn_workspace workspace) {
    check_exec(src.layout, index.layout, sub.layout, workspace.size);
    AclTensor acl_src(src), acl_sub(sub);
    uint64_t ws_size;
    auto handle = concrete_handle(this->handle());
    aclOpExecutor* executor = nullptr;

    int dim = this->param().axis;
    TensorLayout index_layout = index.layout;
    index_layout.add_axis_inplace(dim, 1, index_layout.stride[dim]);
    AclTensor acl_index(index.raw_ptr(), index_layout);
    aclnn_check(aclnnInplaceScatterGetWorkspaceSize(
            acl_src.get(), dim, acl_index.get(), acl_sub.get(), 0, &ws_size,
            &executor));
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnInplaceScatter(ws.ptr(), ws_size, executor, handle->stream()));
}

}  // namespace atlas
}  // namespace megdnn