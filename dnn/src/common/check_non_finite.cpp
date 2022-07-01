#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {

void CheckNonFinite::check_exec(
        const TensorNDArray& srcs, const TensorND& dst, size_t workspace_in_bytes) {
    megdnn_assert_contiguous(dst.layout);
    megdnn_assert(srcs.size() > 0);
    TensorLayoutArray src_layouts;
    for (auto&& src : srcs) {
        src_layouts.push_back(src.layout);
    }
    auto required_workspace_in_bytes = get_workspace_in_bytes(src_layouts, dst.layout);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void CheckNonFinite::deduce_layout(const TensorLayoutArray&, TensorLayout& dst) {
    dst.shape[0] = 1;
    dst.ndim = 1;
    dst.dtype = dtype::Int32();
    dst.init_contiguous_stride();
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
