#include "src/cuda/fill/opr_impl.h"
#include "src/cuda/fill/kern.cuh"

#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void FillImpl::exec(_megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    auto stream = cuda_stream(handle());
    auto size = dst.layout.total_nr_elems();
#define cb(DType)                                                                   \
    if (dst.layout.dtype == DType()) {                                              \
        using ctype = typename DTypeTrait<DType>::ctype;                            \
        fill::exec_internal<ctype>(                                                 \
                dst.ptr<ctype>(), static_cast<ctype>(param().value), size, stream); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    cb(::megdnn::dtype::Bool)
#undef cb
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
