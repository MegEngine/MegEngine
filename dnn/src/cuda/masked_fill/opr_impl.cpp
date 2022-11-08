#include "./opr_impl.h"
#include "./kern.cuh"
#include "src/common/utils.h"
namespace megdnn {
namespace cuda {
void MaskedFillImpl::exec(
        _megdnn_tensor_in origin, _megdnn_tensor_in index, _megdnn_tensor_out dest) {
    check_exec(origin.layout, index.layout, dest.layout);

    megdnn_assert(index.layout.is_contiguous());
    uint32_t mask_stride = TensorLayout(origin.layout, origin.layout.dtype)
                                   .stride[index.layout.ndim - 1];
    ElemwiseOpParamN<1> ele_param;
    ele_param[0] = origin;
    ele_param.init_from_given_tensor();
    auto stream = cuda_stream(handle());

#define cb(DType)                                                      \
    if (origin.layout.dtype == DType()) {                              \
        using T = typename DTypeTrait<DType>::ctype;                   \
        auto value = static_cast<T>(param().value);                    \
        run_elemwise<MaskedFillScalarKernOp<T>, T, 1>(                 \
                ele_param, stream, {dest, index, value, mask_stride}); \
        return;                                                        \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    cb(::megdnn::dtype::Bool)
#undef cb
}
}  // namespace cuda
}  // namespace megdnn