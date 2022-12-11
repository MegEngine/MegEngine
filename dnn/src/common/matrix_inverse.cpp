#include "megdnn/oprs/linalg.h"

#include "src/common/utils.h"

using namespace megdnn;

void MatrixInverse::deduce_layout(const TensorLayout& src, TensorLayout& dst) {
    canonize_params(src, nullptr, nullptr);
    dst = src;
}

size_t MatrixInverse::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    size_t batch, n;
    canonize_params(src, &batch, &n);
    megdnn_assert(
            src.eq_layout(dst), "src and dst unequal: %s vs %s",
            src.to_string().c_str(), dst.to_string().c_str());
    return get_workspace_in_bytes(batch, n, src.dtype.size());
}

void MatrixInverse::canonize_params(
        const TensorLayout& layout, size_t* batch, size_t* n) {
    megdnn_assert(
            layout.ndim >= 2 && layout[layout.ndim - 2] == layout[layout.ndim - 1],
            "MatrixInverse: input must be batches of square matrices, but with input "
            "layout: %s",
            layout.to_string().c_str());
    if (!layout.is_empty()) {
        megdnn_assert(
                layout.is_contiguous(),
                "MatrixInverse: input must be contiguous, but with input layout: %s",
                layout.to_string().c_str());
    }
    megdnn_assert(
            DNN_FLOAT16_SELECT(layout.dtype == dtype::Float16(), false) ||
                    layout.dtype == dtype::Float32(),
            "MatrixInverse only supports f16 & f32");
    if (batch) {
        *batch = 1;
        for (size_t i = 0; i < layout.ndim - 2; ++i) {
            *batch *= layout[i];
        }
    }
    if (n) {
        *n = layout[layout.ndim - 1];
    }
}

void MatrixInverse::check_exec(
        const TensorLayout& src, const TensorLayout& dst, _megdnn_workspace workspace,
        size_t* batch, size_t* n) {
    canonize_params(src, batch, n);
    megdnn_assert(
            src.eq_layout(dst), "src and dst unequal: %s vs %s",
            src.to_string().c_str(), dst.to_string().c_str());
    megdnn_assert(
            workspace.size >= get_workspace_in_bytes(*batch, *n, src.dtype.size()));
}

// vim: syntax=cpp.doxygen
