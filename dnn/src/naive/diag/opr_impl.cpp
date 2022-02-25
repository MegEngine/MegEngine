#include "src/naive/diag/opr_impl.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

namespace megdnn {
namespace naive {

template <typename ctype>
void DiagImpl::exec_internal(
        ctype* src, const TensorLayout& src_layout, ctype* dst,
        const TensorLayout& dst_layout, size_t input_ndim, int k) {
    if (input_ndim == 1) {
        size_t l = src_layout.shape[0];
        size_t s0 = dst_layout.stride[0];
        size_t s1 = dst_layout.stride[1];
        size_t start = (k >= 0) ? (k * s1) : (-k * s0);
        for (size_t i = 0; i < dst_layout.shape[0]; ++i)
            for (size_t j = 0; j < dst_layout.shape[1]; ++j)
                dst[i * s0 + j * s1] = 0;
        for (size_t i = 0; i < l; ++i)
            dst[start + i * (s0 + s1)] = src[i];
    } else {
        size_t l = dst_layout.shape[0];
        size_t s0 = src_layout.stride[0];
        size_t s1 = src_layout.stride[1];
        size_t start = (k >= 0) ? (k * s1) : (-k * s0);
        for (size_t i = 0; i < l; ++i)
            dst[i] = src[start + i * (s0 + s1)];
    }
}

void DiagImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
#define cb(DType)                                                           \
    if (src.layout.dtype == DType()) {                                      \
        using ctype = typename DTypeTrait<DType>::ctype;                    \
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<ctype>(                  \
                src.ptr<ctype>(), src.layout, dst.ptr<ctype>(), dst.layout, \
                src.layout.ndim, param().k));                               \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    cb(::megdnn::dtype::Bool)
#undef cb
}

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
