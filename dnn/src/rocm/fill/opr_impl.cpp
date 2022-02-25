#include "src/rocm/fill/opr_impl.h"
#include "hcc_detail/hcc_defs_prologue.h"

#include "src/rocm/fill/fill.h.hip"
#include "src/rocm/utils.h"

namespace megdnn {
namespace rocm {

void FillImpl::exec(_megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    auto stream = hip_stream(handle());
    auto size = dst.layout.total_nr_elems();
#define cb(DType)                                                                   \
    if (dst.layout.dtype == DType()) {                                              \
        using ctype = typename DTypeTrait<DType>::ctype;                            \
        fill::exec_internal<ctype>(                                                 \
                dst.ptr<ctype>(), static_cast<ctype>(param().value), size, stream); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

}  // namespace rocm
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
