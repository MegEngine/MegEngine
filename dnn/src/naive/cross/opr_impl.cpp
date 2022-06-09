#include "src/naive/cross/opr_impl.h"
#include <algorithm>
#include <numeric>
#include "src/common/utils.h"
#include "src/naive/handle.h"

namespace megdnn {
namespace naive {

template <typename ctype>
void CrossImpl::exec_internal(
        ctype* A, size_t a1, size_t b1, size_t c1, ctype* B, size_t a2, size_t b2,
        size_t c2, ctype* C, size_t a3, size_t b3, size_t c3) {
    (void)a2;
    (void)a3;

    size_t N = a1 * c1;
    for (size_t i = 0; i < N; ++i) {
        size_t ida = (i / c1) * b1 * c1 + i % c1;
        size_t idb = (i / c2) * b2 * c2 + i % c2;
        size_t idc = (i / c3) * b3 * c3 + i % c3;
        C[idc] = A[ida + c1] * B[idb + 2 * c2] - A[ida + 2 * c1] * B[idb + c2];
        C[idc + c3] = A[ida + 2 * c1] * B[idb] - A[ida] * B[idb + 2 * c2];
        C[idc + 2 * c3] = A[ida] * B[idb + c2] - A[ida + c1] * B[idb];
    }
}

void CrossImpl::exec(
        _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
        _megdnn_workspace workspace) {
    check_exec(A.layout, B.layout, C.layout, workspace.size);
    size_t a1, b1, c1, a2, b2, c2, a3, b3, c3;
    get_ABC(A.layout, a1, b1, c1, param().axisa);
    get_ABC(B.layout, a2, b2, c2, param().axisb);
    get_ABC(C.layout, a3, b3, c3, param().axisc);
#define cb(DType)                                                       \
    if (A.layout.dtype == DType()) {                                    \
        using ctype = typename DTypeTrait<DType>::ctype;                \
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<ctype>(              \
                A.ptr<ctype>(), a1, b1, c1, B.ptr<ctype>(), a2, b2, c2, \
                C.ptr<ctype>(), a3, b3, c3));                           \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}