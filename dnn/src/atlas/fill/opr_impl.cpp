#include "opr_impl.h"
#include "aclnnop/aclnn_fill_scalar.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

using namespace megdnn;
using namespace atlas;

void FillImpl::exec(_megdnn_tensor_out dst, _megdnn_workspace workspace) {
    auto handle = concrete_handle(this->handle());
    check_exec(dst.layout, workspace.size);
    AclTensor acl_out(dst);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    AclScalar acl_val(param().value, dtype::Float32());
    aclnn_check(aclnnInplaceFillScalarGetWorkspaceSize(
            acl_out.get(), acl_val.get(), &ws_size, &executor));
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnInplaceFillScalar(ws.ptr(), ws_size, executor, handle->stream()));
}

// vim: syntax=cpp.doxygen