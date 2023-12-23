#include "opr_impl.h"
#include "aclnnop/aclnn_cast.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

using namespace megdnn;
using namespace atlas;

void TypeCvtImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    check_exec(src.layout, dst.layout);
    AclTensor acl_src(src), acl_dst(dst);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnn_check(aclnnCastGetWorkspaceSize(
            acl_src.get(), as_acl_dtype(dst.layout.dtype), acl_dst.get(), &ws_size,
            &executor));
    auto handle = concrete_handle(this->handle());
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnCast(ws.ptr(), ws_size, executor, handle->stream()));
}

// vim: syntax=cpp.doxygen
