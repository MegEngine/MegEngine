#include "hcc_detail/hcc_defs_prologue.h"

#include "./opr_impl.h"
#include "src/rocm/linspace/linspace.h.hip"
#include "src/rocm/utils.h"

namespace megdnn {
namespace rocm {

void LinspaceImpl::exec(_megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    auto stream = hip_stream(handle());
    auto n = dst.layout.total_nr_elems();
    auto step = (param().stop - param().start) /
                std::max(static_cast<double>(param().endpoint ? n - 1 : n), 1.0);
#define cb(dt)                                                     \
    if (dst.layout.dtype == dt()) {                                \
        using ctype = typename DTypeTrait<dt>::ctype;              \
        linspace::exec_internal<ctype>(                            \
                dst.ptr<ctype>(), param().start, step, n, stream); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

}  // namespace rocm
}  // namespace megdnn
