#include "src/cuda/cross/opr_impl.h"

#include "src/cuda/cross/cross.cuh"
#include "src/cuda/utils.h"

#include <algorithm>
#include <numeric>

namespace megdnn {
namespace cuda {

void CrossImpl::exec(
        _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
        _megdnn_workspace workspace) {
    check_exec(A.layout, B.layout, C.layout, workspace.size);

    size_t a1, b1, c1, a2, b2, c2, a3, b3, c3;
    get_ABC(A.layout, a1, b1, c1, param().axisa);
    get_ABC(B.layout, a2, b2, c2, param().axisb);
    get_ABC(C.layout, a3, b3, c3, param().axisc);
#define cb(DType)                                                             \
    if (C.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {                 \
        using ctype = typename DTypeTrait<DType>::ctype;                      \
        cross::exec_internal<ctype>(                                          \
                A.ptr<ctype>(), b1 * c1, c1, B.ptr<ctype>(), b2 * c2, c2,     \
                C.ptr<ctype>(), b3 * c3, c3, a1 * c1, cuda_stream(handle())); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
}

}  // namespace cuda
}  // namespace megdnn
   // vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}