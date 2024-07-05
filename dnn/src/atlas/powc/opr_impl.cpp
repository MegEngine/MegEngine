#include "opr_impl.h"
#include "aclnnop/aclnn_pow.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

using namespace megdnn;
using namespace atlas;

void PowCImpl::do_exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, const float* exp_f,
        const int* exp_i) {
    AclTensor acl_src(src), acl_dst(dst);
    uint64_t ws_size;
    aclOpExecutor* executor = nullptr;
    auto handle = concrete_handle(this->handle());
    float exp_num = 0.0f;
    if (exp_f)
        exp_num += *exp_f;
    if (exp_i)
        exp_num += *exp_i;
    AclScalar acl_scalar_exp_num(exp_num, dst.layout.dtype);
    aclnn_check(aclnnPowTensorScalarGetWorkspaceSize(
            acl_src.get(), acl_scalar_exp_num.get(), acl_dst.get(), &ws_size,
            &executor));
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnPowScalarTensor(ws.ptr(), ws_size, executor, handle->stream()));
}