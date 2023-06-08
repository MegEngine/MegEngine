#include "src/naive/eye/opr_impl.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#include <cstring>
#include <limits>

namespace megdnn {
namespace naive {

void EyeImpl::exec(_megdnn_tensor_out dst, _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    check_exec(dst.layout, workspace.size);
    megdnn_assert(
            std::max(dst.layout.shape[0], dst.layout.shape[1]) <
            static_cast<size_t>(std::numeric_limits<int>::max()));
    int m = dst.layout.shape[0], n = dst.layout.shape[1];
#define cb(DType)                                                                   \
    if (dst.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {                     \
        using ctype = typename DTypeTrait<DType>::ctype;                            \
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<ctype>(dst.ptr<ctype>(), m, n)); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    cb(::megdnn::dtype::Bool)
#undef cb
#else
    __builtin_trap();
#endif
}

template <typename ctype>
void EyeImpl::exec_internal(ctype* dst, int m, int n) {
    memset(dst, 0, m * n * sizeof(ctype));
    //  i + k >= 0     i >= -k i >= 0
    //  i + k < n      i < n-k i < m
    int k = param().k;
    int from = std::max(-k, 0);
    int to = std::min(n - k, m);
    for (int i = from; i < to; ++i) {
        int j = i + k;
        dst[i * n + j] = 1;
    }
}

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
