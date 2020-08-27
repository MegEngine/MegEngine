/**
 * \file dnn/test/rocm/reduce.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/rocm/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(ROCM, REDUCE) {
    using Mode = Reduce::Param::Mode;
    Checker<Reduce> checker(handle_rocm());
    UniformFloatRNG rng(-1.0f, 1.0f);
    checker.set_epsilon(1e-2);
    checker.set_rng(0, &rng);
    checker.set_param({Mode::SUM, 1});

    // 1-step
    checker.execs({{2, 64, 32}, {}});
    // 2-step
    checker.execs({{2, 192, 32}, {}});
    // 3-step
    checker.execs({{2, 4333, 32}, {}});
    // single reduce
    checker.execs({{2, 1, 1}, {}});
    checker.execs({{2, 1 + 1, 1}, {}});
    checker.execs({{2, 2048 + 1, 1}, {}});
    checker.execs({{2, 2048 * 2048 + 1, 1}, {}});
    checker.execs({{2, 1 + 1, 31}, {}});
    checker.execs({{2, 16 + 1, 31}, {}});
    checker.execs({{2, 16 * 16 + 1, 31}, {}});
    checker.execs({{2, 16 * 16 * 16 + 1, 31}, {}});
    checker.execs({{2, 16 * 16 * 16 * 16 + 1, 31}, {}});
    checker.execs({{2, 16 * 16 * 16 * 16 * 16 + 1, 31}, {}});
    checker.execs({{3, 256 * 256 + 1, 2}, {}});
    checker.execs({{3, 128 * 128 + 1, 3}, {}});
    checker.execs({{3, 64 * 64 + 1, 7}, {}});
    checker.execs({{3, 32 * 32 + 1, 15}, {}});
    checker.execs({{3, 512, 500}, {}});
    // very large reduce
    checker.execs({{1, 4194304, 1}, {}});

    auto check = [&](Reduce::Mode mode, DType src_dtype, DType dst_dtype,
                     Reduce::DataType data_type) {
        for (int32_t axis : {0, 1, 2, 3}) {
            if (data_type == Reduce::DataType::DEFAULT &&
                MEGDNN_FLOAT16_SELECT(src_dtype == dtype::Float16(), false)) {
                checker.set_epsilon(1e-2);
            } else {
                checker.set_epsilon(1e-3);
            }
            Reduce::Param param{mode, axis, data_type};
            auto dst_shape = TensorShape{2, 3, 100, 5};
            dst_shape[axis] = 1;
            checker.set_dtype(0, src_dtype)
                    .set_dtype(1, dst_dtype)
                    .set_param(param)
                    .execs({{2, 3, 100, 5}, dst_shape});
        }
    };
    for (auto mode : {Mode::SUM, Mode::MEAN, Mode::SUM_SQR, Mode::PRODUCT,
                      Mode::MIN, Mode::MAX}) {
        for (auto dtype : std::vector<DType>{
                     MEGDNN_INC_FLOAT16(dtype::Float16() MEGDNN_COMMA)
                             dtype::Float32(),
                     dtype::Int32()}) {
            check(mode, dtype, dtype, Reduce::DataType::DEFAULT);
        }
#if !MEGDNN_DISABLE_FLOAT16
        check(mode, dtype::Float16(), dtype::Float32(),
              Reduce::DataType::FLOAT_O32xC32);
        check(mode, dtype::Float16(), dtype::Float16(),
              Reduce::DataType::FLOAT_O16xC32);
        check(mode, dtype::Float32(), dtype::Float16(),
              Reduce::DataType::FLOAT_O16xC32);
        ASSERT_THROW(check(mode, dtype::Int32(), dtype::Float16(),
                           Reduce::DataType::FLOAT_O16xC32),
                     MegDNNError);
        ASSERT_THROW(check(mode, dtype::Float16(), dtype::Float16(),
                           Reduce::DataType::FLOAT_IO16xC32),
                     MegDNNError);
#endif
    }

#if !MEGDNN_DISABLE_FLOAT16
    {
        // very large reduce for I16CO32
        Reduce::Param param{Mode::SUM_SQR, 1,
                            Reduce::Param::DataType::FLOAT_O32xC32};
        checker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float32())
                .set_param(param)
                .execs({{1, 4194304, 1}, {1, 1, 1}});
    }
#endif
}

// vim: syntax=cpp.doxygen
