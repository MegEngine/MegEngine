/**
 * \file dnn/test/common/get_dtype_from_static_type.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
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
