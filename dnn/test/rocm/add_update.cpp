/**
 * \file dnn/test/rocm/add_update.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "test/rocm/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(ROCM, ADD_UPDATE) {
    Checker<AddUpdate> checker(handle_rocm());

    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{2, 3, 4}, {2, 3, 4}});
#if !MEGDNN_DISABLE_FLOAT16
    checker.set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .execs({{2, 3, 4}, {2, 3, 4}});
#endif
    checker.execl({{{2, 3, 4}, dtype::Float32()},
                   {{2, 3, 4}, {16, 4, 1}, dtype::Float32()}});
#if !MEGDNN_DISABLE_FLOAT16
    checker.execl({{{2, 3, 4}, dtype::Float16()},
                   {{2, 3, 4}, {16, 4, 1}, dtype::Float16()}});
#endif
    checker.execl({{{2, 3, 4}, {16, 4, 1}, dtype::Float32()},
                   {{2, 3, 4}, dtype::Float32()}});

    checker.execl({{{2, 3, 4}, dtype::Float32()}, {{1}, dtype::Float32()}});
    checker.execl(
            {{{2, 3, 4}, dtype::Float32()}, {{2, 1, 4}, dtype::Float32()}});
    checker.set_param({2, -1, 3})
            .set_dtype(0, dtype::Int32())
            .set_dtype(1, dtype::Int32())
            .execs({{2, 3, 2}, {2, 3, 2}});
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
