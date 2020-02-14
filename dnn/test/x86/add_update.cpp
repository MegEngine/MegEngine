/**
 * \file dnn/test/x86/add_update.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/checker.h"
#include "test/common/resize.h"
#include "test/common/rng.h"
#include "test/x86/fixture.h"

namespace megdnn {
namespace test {

TEST_F(X86, ADD_UPDATE) {
    Checker<AddUpdate> checker(handle());

    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{1, 3, 5, 5}, {1, 3, 5, 5}});
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{2, 3, 4}, {2, 3, 4}});
    checker.set_param({2, -1, 3})
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{2, 3, 2}, {2, 3, 2}});
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{1, 3, 5, 5}, {1, 3, 1, 1}});
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{2, 3, 4}, {1}});
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
