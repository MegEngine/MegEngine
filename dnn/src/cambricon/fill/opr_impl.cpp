#include "src/cambricon/utils.h"

#include <algorithm>
#include "cnnl.h"
#include "src/cambricon/fill/opr_impl.h"

namespace megdnn {
namespace cambricon {

void FillImpl::exec(_megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(dst.layout, workspace.size);
    auto size = dst.layout.total_nr_elems();
    CnnlTensorDescriptor out_desc;
    out_desc.set(&dst);

#define cb(DType)                                                         \
    if (dst.layout.dtype == DType()) {                                    \
        using ctype = typename DTypeTrait<DType>::ctype;                  \
        FillImpl::exec_internal<ctype>(dst.ptr<ctype>(), size, out_desc); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    cb(::megdnn::dtype::Bool)
#undef cb
}

template <typename ctype>
void FillImpl::exec_internal(ctype* dst, size_t size, CnnlTensorDescriptor& out_desc) {
    auto value = static_cast<ctype>(param().value);
    cnnlHandle_t h = cnnl_handle(this->handle());
    cnnl_check(cnnlFill_v3(h, CNNL_POINTER_MODE_HOST, &value, out_desc.desc(), dst));
}

}  // namespace cambricon
}  // namespace megdnn
   // vim: syntax=cpp.doxygen