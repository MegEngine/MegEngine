#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
void MaskedFill::deduce_layout(
        const TensorLayout& origin, const TensorLayout& /*index*/, TensorLayout& dest) {
    dest = TensorLayout(origin, origin.dtype);
}

void MaskedFill::check_exec(
        const TensorLayout& origin, const TensorLayout& index,
        const TensorLayout& dest) {
    megdnn_assert_contiguous(index);
    megdnn_assert_contiguous(dest);
    megdnn_assert(index.dtype == dtype::Bool());
    megdnn_assert(origin.ndim >= index.ndim);
    bool correct_index_shape = true;
    for (size_t i = 0; i < index.ndim; i++) {
        correct_index_shape = correct_index_shape && origin.shape[i] == index.shape[i];
    }
    megdnn_assert(correct_index_shape, "unsupported index shape");
    bool supported_dtype = false;

#define cb(Dtype) supported_dtype = supported_dtype || (origin.dtype == Dtype());
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
    cb(megdnn::dtype::Bool)
#undef cb

            megdnn_assert(supported_dtype, "unsupported dtype");
}

}  // namespace megdnn