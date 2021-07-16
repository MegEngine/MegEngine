/**
 * \file dnn/test/common/fill.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/handle.h"
#include "megdnn/oprs/general.h"

#include "src/common/opr_trait.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {
namespace fill {

inline void run_fill_test(Handle* handle, DType dtype) {
    Checker<Fill> checker(handle);
    for (float value : {-1.23, 0.0, 0.001, 234.0, 2021.072}) {
        checker.set_param({value});
        checker.set_dtype(0, dtype);
        checker.exec(TensorShapeArray{{1, 1}});
        checker.exec(TensorShapeArray{{2, 3, 4}});
    }
}

} // namespace fill
} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
