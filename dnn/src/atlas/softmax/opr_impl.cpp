#include "opr_impl.h"
#include "aclnnop/aclnn_softmax.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

using namespace megdnn;
using namespace atlas;

void SoftmaxForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace) {
    auto handle = concrete_handle(this->handle());
    AclTensor acl_src(src), acl_dst(dst);
    int64_t dim = static_cast<int64_t>(param().axis);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnn_check(aclnnSoftmaxGetWorkspaceSize(
            acl_src.get(), dim, acl_dst.get(), &ws_size, &executor));
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnSoftmax(ws.ptr(), ws_size, executor, handle->stream()));
}

// vim: syntax=cpp.doxygen