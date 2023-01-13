#include "src/naive/fill/opr_impl.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#include <cstring>
#include <limits>

namespace megdnn {
namespace naive {

void FillImpl::exec(_megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    size_t size = dst.layout.total_nr_elems();
#define cb(DType)                                                                   \
    if (dst.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {                     \
        using ctype = typename DTypeTrait<DType>::ctype;                            \
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal<ctype>(dst.ptr<ctype>(), size)); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    cb(::megdnn::dtype::Bool)
#undef cb
}

template <typename ctype>
void FillImpl::exec_internal(ctype* dst, size_t size) {
    auto value = static_cast<ctype>(param().value);
    for (size_t i = 0; i < size; ++i) {
        dst[i] = value;
    }
}

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
