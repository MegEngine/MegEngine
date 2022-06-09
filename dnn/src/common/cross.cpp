#include "megdnn/oprs.h"
#include "src/common/utils.h"

#include <algorithm>
#include <numeric>

namespace megdnn {

void Cross::deduce_layout(
        const TensorLayout& A, const TensorLayout& B, TensorLayout& C) {
    auto calibrated_axis = [](int ndim, int axis) {
        return axis < 0 ? (axis + ndim) : axis;
    };

    int axis_a = calibrated_axis(A.ndim, param().axisa);
    int axis_b = calibrated_axis(B.ndim, param().axisb);
    int axis_c = calibrated_axis(A.ndim, param().axisc);

    megdnn_assert(
            A[axis_a] == 3 && B[axis_b] == 3,
            "incompatible dimensions for cross product (dimension must be 3)");

    bool matched = true;
    TensorShape shp;
    if (A.ndim != B.ndim) {
        matched = false;
    } else {
        for (int i = 0, j = 0, k = 0; i < static_cast<int>(A.ndim); i++) {
            if (i == axis_a)
                continue;
            if (j == axis_b)
                ++j;
            if (A[i] != B[j]) {
                matched = false;
                break;
            }
            if (k == axis_c)
                ++k;
            shp[k++] = A[i];
            ++j;
        }
    }

    megdnn_assert(
            matched, "cross op shape mismatch: %s vs %s", A.to_string().c_str(),
            B.to_string().c_str());

    shp.ndim = A.ndim;
    shp[axis_c] = A[axis_a];
    C = TensorLayout{shp, A.dtype};
}

void Cross::check_exec(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C,
        size_t workspace_in_bytes) {
    megdnn_assert_eq_dtype(A, B);
    megdnn_assert_eq_dtype(B, C);
    TensorLayout c_expected;
    deduce_layout(A, B, c_expected);
    megdnn_assert_eq_layout(c_expected, C);

    megdnn_assert_contiguous(A);
    megdnn_assert_contiguous(B);
    megdnn_assert_contiguous(C);
    auto required_workspace_in_bytes = get_workspace_in_bytes(A, B, C);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void Cross::get_ABC(
        const TensorShape& shape, size_t& A, size_t& B, size_t& C, int32_t axis) {
    auto shape_arr = shape.shape;
    auto ndim = shape.ndim;
    if (axis < 0)
        axis += ndim;
    A = std::accumulate(shape_arr, shape_arr + axis, 1_z, SafeMultiplies<size_t>());
    B = shape_arr[axis];
    C = std::accumulate(
            shape_arr + (axis + 1), shape_arr + ndim, 1_z, SafeMultiplies<size_t>());
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen