#include "src/cuda/where/common.cuh"
#include "src/cuda/where/opr_impl.h"

#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void WhereBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in mask, _megdnn_tensor_out grad_data1,
        _megdnn_tensor_out grad_data2, _megdnn_workspace workspace) {
    check_exec(
            diff.layout, mask.layout, grad_data1.layout, grad_data2.layout,
            workspace.size);
    auto stream = cuda_stream(this->handle());
    auto n = diff.layout.total_nr_elems();
#define cb(DType)                                                                \
    if (diff.layout.dtype == DType()) {                                          \
        using ctype = typename DTypeTrait<DType>::ctype;                         \
        where_backward::backward_proxy<ctype>(                                   \
                diff.ptr<ctype>(), mask.ptr<dt_bool>(), grad_data1.ptr<ctype>(), \
                grad_data2.ptr<ctype>(), n, stream);                             \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    cb(::megdnn::dtype::Bool)
#undef cb
}

}  // namespace cuda
}  // namespace megdnn
