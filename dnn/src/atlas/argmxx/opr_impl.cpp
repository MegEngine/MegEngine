#include "opr_impl.h"
#include "aclnnop/aclnn_argmax.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

using namespace megdnn;
using namespace atlas;

void ArgmaxForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    auto handle = concrete_handle(this->handle());

    AclTensor acl_src(src), acl_dst(dst);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    aclnn_check(aclnnArgMaxGetWorkspaceSize(
            acl_src.get(), static_cast<int64_t>(m_param.axis), true, acl_dst.get(),
            &ws_size, &executor));
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnArgMax(ws.ptr(), ws_size, executor, handle->stream()));
}
