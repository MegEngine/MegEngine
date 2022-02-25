#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn {

void TypeCvt::check_exec(const TensorLayout& src, const TensorLayout& dst) {
    megdnn_assert_contiguous(dst);
    megdnn_assert_eq_shape(src, dst);
    auto cat = src.dtype.category();
    megdnn_assert(
            cat == DTypeCategory::FLOAT || cat == DTypeCategory::INT ||
            cat == DTypeCategory::QUANTIZED || cat == DTypeCategory::BOOL);
    cat = dst.dtype.category();
    megdnn_assert(
            cat == DTypeCategory::FLOAT || cat == DTypeCategory::INT ||
            cat == DTypeCategory::QUANTIZED || cat == DTypeCategory::BOOL);
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
