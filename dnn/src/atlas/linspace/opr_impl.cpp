#include "src/atlas/linspace/opr_impl.h"
#include "aclnnop/aclnn_linspace.h"
#include "src/atlas/handle.h"
#include "src/common/utils.h"

namespace megdnn {
namespace atlas {
void LinspaceImpl::exec(_megdnn_tensor_out dst, _megdnn_workspace workspace) {
    double start = param().start, end = param().stop;
    uint32_t n = dst.layout.total_nr_elems();
    double step = (end - start) /
                  std::max(static_cast<double>(param().endpoint ? n - 1 : n), 1.0);

    auto handle = concrete_handle(this->handle());
    double stop = start + step * (n - 1);
    check_exec(dst.layout, workspace.size);
    AclTensor acl_out(dst);
    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    AclScalar acl_start(start), acl_stop(stop);
    aclnnLinspaceGetWorkspaceSize(
            acl_start.get(), acl_stop.get(), n, acl_out.get(), &ws_size, &executor);
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnLinspace(ws.ptr(), ws_size, executor, handle->stream()));
}

}  // namespace atlas
}  // namespace megdnn
   // vim: syntax=cpp.doxygen