#include "src/naive/masked_fill/opr_impl.h"
#include <cmath>
#include "megdnn/tensor_iter.h"
#include "src/common/elemwise_helper.cuh"
#include "src/common/utils.h"
#include "src/naive/handle.h"

namespace {
using namespace megdnn;
template <typename T>
void forward_impl(const ElemwiseOpParamN<3> src, const T value) {
    auto inp = tensor_iter_valonly<T>(src[0]).begin();
    auto out = tensor_iter_valonly<T>(src[1]).begin();
    auto mask = tensor_iter_valonly<bool>(src[2]).begin();
    size_t total = src[0].layout.total_nr_elems();
    for (size_t i = 0; i < total; ++i) {
        *out = *mask ? value : *inp;
        ++inp;
        ++out;
        ++mask;
    }
}
}  // namespace

namespace megdnn {
namespace naive {
void MaskedFillImpl::exec(
        _megdnn_tensor_in origin, _megdnn_tensor_in index, _megdnn_tensor_out dest) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    check_exec(origin.layout, index.layout, dest.layout);

    megdnn_assert(origin.layout.is_contiguous() && index.layout.is_contiguous());
    ElemwiseOpParamN<3> src;
    src[0] = origin;
    src[1] = dest;
    src[2] = index;
    if (src[2].layout.ndim < src[0].layout.ndim) {
        for (size_t n = src[2].layout.ndim; n < src[0].layout.ndim; n++)
            src[2].layout.add_axis_cont_inplace(n);
    }
    src[2].layout = src[2].layout.broadcast(origin.layout);

#define cb(DType)                                    \
    if (origin.layout.dtype == DType()) {            \
        using T = typename DTypeTrait<DType>::ctype; \
        auto value = static_cast<T>(param().value);  \
        forward_impl<T>(src, value);                 \
        return;                                      \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    cb(::megdnn::dtype::Bool)
#undef cb
#else
    __builtin_trap();
#endif
}
}  // namespace naive
}  // namespace megdnn
