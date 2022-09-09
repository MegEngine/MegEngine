#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
void NonZero::check_exec(
        const TensorLayout& src, const TensorLayout& dst, size_t workspace_in_bytes) {
    dst.dtype.assert_is(infer_type(src.dtype));

    if (!src.is_empty())
        megdnn_assert(src.is_physical_contiguous());
    auto require_workspace_in_bytes = get_workspace_in_bytes(src);
    megdnn_assert(workspace_in_bytes >= require_workspace_in_bytes);
}

DType NonZero::infer_type(DType /*input*/) {
    return dtype::Int32();
}
};  // namespace megdnn