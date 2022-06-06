#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
void NormForward::deduce_layout(const TensorLayout& src, TensorLayout& dst) {
    megdnn_assert(
            param().dim > -1 && param().dim < static_cast<dt_int32>(src.ndim),
            "dim params must be passed and cannot be -1.");

    SmallVector<size_t> shapeList;
    for (size_t i = 0; i < src.ndim; ++i) {
        if (static_cast<dt_int32>(i) != param().dim) {
            shapeList.append(1, static_cast<size_t>(src.shape[i]));
        } else {
            shapeList.append(1, static_cast<size_t>(1));
        }
    }
    dst = TensorLayout{TensorShape(shapeList), src.dtype};
    return;
}

void NormBase::check_exec(
        const TensorLayout& src, const TensorLayout& dst, size_t workspace_in_bytes) {
    megdnn_assert_eq_dtype(src, dst);

#if !MEGDNN_DISABLE_FLOAT16
    megdnn_assert(
            src.dtype.enumv() == DTypeEnum::Float16 ||
                    src.dtype.enumv() == DTypeEnum::Float32,
            "Float16 or Float32 is only supported.");
#else
    megdnn_assert(
            src.dtype.enumv() == DTypeEnum::Float32, "Float32 is only supported.");
#endif

    TensorLayout dst_expected;
    deduce_layout(src, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);

    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}
}  // namespace megdnn
