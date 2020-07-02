/**
 * \file dnn/test/cuda/relayout_format.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/cuda/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(CUDA, RELAYOUT_FORMAT) {
    Checker<RelayoutFormat> checker(handle_cuda());
    UniformIntRNG rng{-50, 50};
    param::RelayoutFormat param;
    param.mode = param::RelayoutFormat::Mode::NCHW4_CHWN4;

    checker.set_dtype(0, dtype::QuantizedS8{0.1f})
            .set_rng(0, &rng)
            .set_param(param)
            .execs({{22, 23, 24, 25, 4}, {}});
    param.mode = param::RelayoutFormat::Mode::CHWN4_NCHW4;
    checker.execs({{22, 23, 24, 25, 4}, {}});
}

TEST_F(CUDA, RELAYOUT_FORMAT_NCHW4) {
    Checker<RelayoutFormat> checker(handle_cuda());
    UniformIntRNG rng{-50, 50};
    param::RelayoutFormat param;
    param.mode = param::RelayoutFormat::Mode::NCHW_NCHW4_IC_SMALL;

    for (DType dtype :
         std::vector<DType>({dtype::QuantizedS8{0.1f}, dtype::Float32{}})) {
        checker.set_dtype(0, dtype).set_rng(0, &rng);

        checker.set_param(param).execs({{2, 4, 35, 36}, {}});
        checker.set_param(param).execs({{2, 3, 35, 36}, {}});
        checker.set_param(param).execs({{2, 1, 35, 36}, {}});
        param.mode = param::RelayoutFormat::Mode::
                NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT;
        checker.set_param(param).execs({{4, 3, 3, 3}, {}});
        checker.set_param(param).execs({{4, 4, 3, 3}, {}});
        checker.set_param(param).execs({{1, 4, 3, 3}, {}});
    }
}

// vim: syntax=cpp.doxygen
