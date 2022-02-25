#include "src/cuda/padding/opr_impl.h"
#include "src/common/utils.h"
#include "src/cuda/handle.h"
#include "src/cuda/padding/padding.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void PaddingForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    forward_check_exec(src.layout, dst.layout);
    SmallVector<size_t> offsets(get_offsets());
    // SamllVector can not be used as argument in cu file
    size_t param_offsets[MEGDNN_MAX_NDIM * 2] = {
            offsets[0],  offsets[1],  offsets[2],  offsets[3], offsets[4],
            offsets[5],  offsets[6],  offsets[7],  offsets[8], offsets[9],
            offsets[10], offsets[11], offsets[12], offsets[13]};
    auto stream = cuda_stream(this->handle());
#define cb(DType)                                                        \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {          \
        using ctype = typename DTypeTrait<DType>::ctype;                 \
        padding::padding_forward_proxy<ctype>(                           \
                src, dst, param_offsets, uint32_t(param().padding_mode), \
                param().padding_val, stream);                            \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
}

void PaddingBackwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    backward_check_exec(src.layout, dst.layout);
    SmallVector<size_t> offsets(get_offsets());
    // SamllVector can not be used as argument in cu file
    size_t param_offsets[MEGDNN_MAX_NDIM * 2] = {
            offsets[0],  offsets[1],  offsets[2],  offsets[3], offsets[4],
            offsets[5],  offsets[6],  offsets[7],  offsets[8], offsets[9],
            offsets[10], offsets[11], offsets[12], offsets[13]};
    auto stream = cuda_stream(this->handle());
#define cb(DType)                                                                 \
    if (src.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {                   \
        using ctype = typename DTypeTrait<DType>::ctype;                          \
        padding::padding_backward_proxy<ctype>(                                   \
                src, dst, param_offsets, uint32_t(param().padding_mode), stream); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
}

size_t PaddingForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    return 0;
}

size_t PaddingBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    return 0;
}
}  // namespace cuda
}  // namespace megdnn