#include "src/common/reduce_helper.h"

#include <algorithm>
#include <numeric>
#include "src/common/utils.h"

namespace megdnn {
namespace reduce {

void get_ABC(const TensorShape& shape, size_t& A, size_t& B, size_t& C, size_t axis) {
    auto shape_arr = shape.shape;
    auto ndim = shape.ndim;
    A = std::accumulate(shape_arr, shape_arr + axis, 1_z, SafeMultiplies<size_t>());
    B = shape_arr[axis];
    C = std::accumulate(
            shape_arr + (axis + 1), shape_arr + ndim, 1_z, SafeMultiplies<size_t>());
}

}  // namespace reduce
}  // namespace megdnn
