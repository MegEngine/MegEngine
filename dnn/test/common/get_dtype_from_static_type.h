#pragma once
#include "megdnn/dtype.h"

namespace megdnn {
namespace test {

template <typename T>
DType get_dtype_from_static_type() {
    return typename DTypeTrait<T>::dtype();
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
