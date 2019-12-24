/**
 * \file dnn/test/cuda/add_update.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, ADD_UPDATE) {
    Checker<AddUpdate> checker(handle_cuda());

    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({{2, 3, 4}, {2, 3, 4}});
    checker.set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .execs({{2, 3, 4}, {2, 3, 4}});
    checker.set_dtype(0, dtype::BFloat16())
            .set_dtype(1, dtype::BFloat16())
            .execs({{2, 3, 4}, {2, 3, 4}});
    checker.execl({{{2, 3, 4}, dtype::Float32()},
                   {{2, 3, 4}, {16, 4, 1}, dtype::Float32()}});
    checker.execl({{{2, 3, 4}, dtype::Float16()},
                   {{2, 3, 4}, {16, 4, 1}, dtype::Float16()}});
    checker.execl({{{2, 3, 4}, dtype::BFloat16()},
                   {{2, 3, 4}, {16, 4, 1}, dtype::BFloat16()}});

    checker.execl({{{2, 3, 4}, {16, 4, 1}, dtype::Float32()},
                   {{2, 3, 4}, dtype::Float32()}});

    checker.execl({{{2, 3, 4}, dtype::Float32()}, {{1}, dtype::Float32()}});
    checker.execl(
            {{{2, 3, 4}, dtype::Float32()}, {{2, 1, 4}, dtype::Float32()}});
    checker.set_param({2, -1, 3})
            .set_dtype(0, dtype::Int32())
            .set_dtype(1, dtype::Int32())
            .execs({{2, 3, 2}, {2, 3, 2}});
    checker.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .execs({{2, 3, 2}, {2, 3, 2}});
    checker.set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Uint8())
            .execs({{2, 3, 2}, {2, 3, 2}});
    // test scalar
    checker.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .execs({{1}, {1}});
    checker.set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .execs({{4}, {1}});
    checker.execl({{{2, 3, 4}, dtype::Int8()},
                   {{2, 3, 4}, {16, 4, 1}, dtype::Int8()}});
    checker.execl({{{2, 3, 4}, dtype::Int8()},
                   {{1, 3, 1}, {16, 4, 1}, dtype::Int8()}});

    checker.execl({{{2, 3, 4}, {16, 4, 1}, dtype::Int8()},
                   {{2, 3, 4}, dtype::Int8()}});
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
