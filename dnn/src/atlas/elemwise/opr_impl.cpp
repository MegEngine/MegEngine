#include "opr_impl.h"
#include "acl/acl_op_compiler.h"
#include "aclnnop/aclnn_add.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

using namespace megdnn;
using namespace atlas;
using Mode = param::Elemwise::Mode;

void ElemwiseForwardImpl::exec(const TensorNDArray& src, _megdnn_tensor_out dst) {
    auto handle = concrete_handle(this->handle());
    SmallVector<AclTensor> acl_inps;
    for (size_t i = 0; i < src.size(); ++i) {
        acl_inps.emplace_back(src[i]);
    }
    AclTensor acl_out(dst);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    if (m_param.mode == Mode::ADD) {
        aclnn_check(aclnnAddGetWorkspaceSize(
                acl_inps[0].get(), acl_inps[1].get(),
                AclScalar(1.0, dst.layout.dtype).get(), acl_out.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnAdd(ws.ptr(), ws_size, executor, handle->stream()));
    }
}
