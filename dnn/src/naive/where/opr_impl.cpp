#include "src/naive/where/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

namespace megdnn {
namespace naive {

template <typename T>
void WhereForwardImpl::exec_internal(
        const dt_bool* __restrict mask, const T* __restrict data1,
        const T* __restrict data2, T* __restrict dst, size_t n) {
    rep(i, n) { dst[i] = mask[i] ? data1[i] : data2[i]; }
}

void WhereForwardImpl::exec(
        _megdnn_tensor_in mask, _megdnn_tensor_in data1, _megdnn_tensor_in data2,
        _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(mask.layout, data1.layout, data2.layout, dst.layout, workspace.size);
    auto n = data1.layout.total_nr_elems();
#define cb(DType)                                                            \
    if (data1.layout.dtype == DType()) {                                     \
        using ctype = typename DTypeTrait<DType>::ctype;                     \
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<ctype>(                   \
                mask.ptr<dt_bool>(), data1.ptr<ctype>(), data2.ptr<ctype>(), \
                dst.ptr<ctype>(), n));                                       \
        return;                                                              \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    cb(::megdnn::dtype::Bool)
#undef cb
            megdnn_assert_internal(0);
}

template <typename T>
void WhereBackwardImpl::exec_internal(
        const T* __restrict diff, const dt_bool* __restrict mask,
        T* __restrict grad_data1, T* __restrict grad_data2, size_t n) {
    rep(i, n) {
        grad_data1[i] = mask[i] ? diff[i] : 0;
        grad_data2[i] = mask[i] ? 0 : diff[i];
    }
}

void WhereBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in mask, _megdnn_tensor_out grad_data1,
        _megdnn_tensor_out grad_data2, _megdnn_workspace workspace) {
    check_exec(
            diff.layout, mask.layout, grad_data1.layout, grad_data2.layout,
            workspace.size);
    auto n = diff.layout.total_nr_elems();
#define cb(DType)                                                                \
    if (diff.layout.dtype == DType()) {                                          \
        using ctype = typename DTypeTrait<DType>::ctype;                         \
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<ctype>(                       \
                diff.ptr<ctype>(), mask.ptr<dt_bool>(), grad_data1.ptr<ctype>(), \
                grad_data2.ptr<ctype>(), n));                                    \
        return;                                                                  \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    cb(::megdnn::dtype::Bool)
#undef cb
            megdnn_assert_internal(0);
}

}  // namespace naive
}  // namespace megdnn
