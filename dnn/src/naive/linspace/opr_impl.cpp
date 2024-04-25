#include "src/naive/linspace/opr_impl.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

namespace megdnn {
namespace naive {

void LinspaceImpl::exec(_megdnn_tensor_out dst, _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    check_exec(dst.layout, workspace.size);
    auto n = dst.layout.total_nr_elems();
    auto start = param().start;
    auto step = (param().stop - param().start) /
                std::max(static_cast<double>(param().endpoint ? n - 1 : n), 1.0);
#define cb(DType)                                                        \
    if (dst.layout.dtype == DType()) {                                   \
        using ctype = typename DTypeTrait<DType>::ctype;                 \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                    \
                exec_internal<ctype>(dst.ptr<ctype>(), start, step, n)); \
        return;                                                          \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    megdnn_assert_internal(0);
#else
    __builtin_trap();
#endif
}

template <typename ctype>
void LinspaceImpl::exec_internal(ctype* dst, double start, double step, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<ctype>(start + i * step);
    }
}

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
