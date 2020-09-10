/**
 * \file dnn/test/cuda/mask_conv.cpp
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
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/mask_conv.h"
#include "test/common/rng.h"

using namespace megdnn;
using namespace test;

TEST_F(CUDA, MASK_CONV) {
    mask_conv_test(handle_cuda());
}
#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, MASK_CONV_BENCHMARK) {
    mask_conv_benchmark(handle_cuda());
}
#endif

TEST_F(CUDA, MASK_PROPAGATE) {
    Checker<MaskPropagate> checker(handle_cuda());
    auto run = [&](size_t IH, size_t IW, size_t FH, size_t FW, size_t SH = 1,
                   size_t SW = 1, size_t PH = 0, size_t PW = 0, size_t DH = 1,
                   size_t DW = 1) {
        using Param = param::MaskPropagate;
        Param param(PH, PW, SH, SW, FH, FW, DH, DW);
        TensorShape src_shape({IH, IW}), dst({});
        auto rng = std::make_unique<BernoulliRNG>(0.5);
        checker.set_param(param).set_rng(0, rng.get()).execs({src_shape, dst});
#undef test
    };
#define cb(DType)                        \
    checker.set_dtype(0, DType());       \
    run(3, 3, 1, 1);                     \
    run(5, 5, 2, 3, 2, 2);               \
    run(5, 5, 3, 3, 2, 2, 1, 2);         \
    run(5, 5, 3, 3, 2, 1, 1, 2);         \
    run(5, 5, 3, 3, 1, 2, 2, 2);         \
    run(24, 23, 4, 4, 1, 1, 3, 2);       \
    run(24, 23, 4, 4, 1, 1, 3, 2, 2, 2); \
    run(24, 23, 4, 4, 1, 1, 3, 2, 2, 3); \
    run(24, 23, 4, 4, 1, 1, 3, 2, 3, 3);

    // cb(dtype::Int32)
    MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb);
#undef cb
}
