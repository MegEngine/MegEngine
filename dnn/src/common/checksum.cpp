#include "megdnn/oprs.h"
#include "src/common/utils.h"

using namespace megdnn;

void megdnn::ChecksumForward::check_exec(
        const TensorLayout& layout, size_t workspace_in_bytes) {
    megdnn_assert(
            layout.is_contiguous() && layout.ndim == 1 &&
                    layout.dtype == dtype::Byte() && layout.shape[0],
            "%s", layout.to_string().c_str());
    auto required_workspace_in_bytes = get_workspace_in_bytes(layout);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

// vim: syntax=cpp.doxygen
