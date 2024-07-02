#include "opr_impl.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_normal.h"
#include "aclnnop/aclnn_uniform.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

using namespace megdnn;
using namespace atlas;

void GaussianRNGImpl::exec(_megdnn_tensor_inout dst, _megdnn_workspace) {
    float mean = m_param.mean;
    float std = m_param.std;
    float seed = m_param.seed;
    AclTensor acl_dst(dst);
    uint64_t ws_size;
    aclOpExecutor* executor = nullptr;
    aclnn_check(aclnnInplaceNormalGetWorkspaceSize(
            acl_dst.get(), mean, std, seed, 0, &ws_size, &executor));
    auto handle = concrete_handle(this->handle());
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnInplaceNormal(ws.ptr(), ws_size, executor, handle->stream()));
}

void UniformRNGImpl::exec(_megdnn_tensor_inout dst, _megdnn_workspace) {
    float seed = m_param.seed;
    AclTensor acl_dst(dst);
    uint64_t ws_size;
    aclOpExecutor* executor = nullptr;

    float from = 0.0, to = 1.0;
    int offset = 0;

    aclnn_check(aclnnInplaceUniformGetWorkspaceSize(
            acl_dst.get(), from, to, seed, offset, &ws_size, &executor));
    auto handle = concrete_handle(this->handle());
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnInplaceUniform(ws.ptr(), ws_size, executor, handle->stream()));
}
