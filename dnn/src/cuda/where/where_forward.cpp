#include "src/cuda/where/common.cuh"
#include "src/cuda/where/opr_impl.h"

#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void WhereForwardImpl::exec(
        _megdnn_tensor_in mask, _megdnn_tensor_in data1, _megdnn_tensor_in data2,
        _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(mask.layout, data1.layout, data2.layout, dst.layout, workspace.size);
    auto stream = cuda_stream(this->handle());
    auto n = data1.layout.total_nr_elems();
#define cb(DType)                                                            \
    if (data1.layout.dtype == DType()) {                                     \
        using ctype = typename DTypeTrait<DType>::ctype;                     \
        where::forward_proxy<ctype>(                                         \
                mask.ptr<dt_bool>(), data1.ptr<ctype>(), data2.ptr<ctype>(), \
                dst.ptr<ctype>(), n, stream);                                \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    cb(::megdnn::dtype::Bool)
#undef cb
}

}  // namespace cuda
}  // namespace megdnn
