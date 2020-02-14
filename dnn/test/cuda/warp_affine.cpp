/**
 * \file dnn/test/cuda/warp_affine.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"
#include "test/common/warp_affine.h"
#include "test/common/checker.h"
#include "test/common/benchmarker.h"
#include "include/megdnn/thin/small_vector.h"

namespace megdnn {
namespace test {

// FIXME test WARP_PERSPECTIVE_CV failed here
#if 0
TEST_F(CUDA, WARP_AFFINE_CV)
{
    using namespace warp_affine;
    std::vector<TestArg> args = get_cv_args();
    Checker<WarpAffine> checker(handle_cuda());

    for (auto &&arg: args) {
        if (arg.src[3] == 2) continue;
        checker.set_param(arg.param)
            .set_epsilon(1 + 1e-3)
            .set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Uint8())
            .execs({arg.src, arg.trans, arg.dst});
    }

    for (auto &&arg: args) {
        if (arg.src[3] == 2) continue;
        checker.set_param(arg.param)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .execs({arg.src, arg.trans, arg.dst});
    }

}
#endif

TEST_F(CUDA, WARP_AFFINE) {
    //! NCHW
    for (auto dtype : std::vector<DType>{dtype::Float32()}) {
        for (auto bmode :
             {WarpAffine::BorderMode::WRAP, WarpAffine::BorderMode::REFLECT,
              WarpAffine::BorderMode::CONSTANT,
              WarpAffine::BorderMode::REPLICATE,
              WarpAffine::BorderMode::CONSTANT}) {
            Checker<WarpAffine> checker(handle_cuda());
            NormalRNG rng;
            checker.set_rng(1, &rng);
            WarpAffine::Param param;
            param.border_val = 0.3f;
            param.border_mode = bmode;
            param.imode = param::WarpAffine::InterpolationMode::LINEAR;
            param.format = param::WarpAffine::Format::NCHW;
            checker.set_param(param);
            checker.set_dtype(0, dtype);
            checker.set_dtype(1, dtype);
            checker.set_dtype(2, dtype);
            checker.execs({{2, 3, 10, 11}, {2, 2, 3}, {2, 3, 11, 12}});
            checker.execs({{22, 3, 10, 11}, {22, 2, 3}, {22, 3, 11, 12}});
        }
    }

    //! NHWC
    for (auto dtype : std::vector<DType>{dtype::Float32()}) {
        for (auto bmode :
             {WarpAffine::BorderMode::WRAP, WarpAffine::BorderMode::REFLECT,
              WarpAffine::BorderMode::CONSTANT,
              WarpAffine::BorderMode::REPLICATE,
              WarpAffine::BorderMode::CONSTANT}) {
            Checker<WarpAffine> checker(handle_cuda());
            NormalRNG rng;
            checker.set_rng(1, &rng);
            WarpAffine::Param param;
            param.format = param::WarpAffine::Format::NHWC;
            param.border_val = 0.3f;
            param.border_mode = bmode;
            param.imode = param::WarpAffine::InterpolationMode::LINEAR;
            checker.set_param(param);
            checker.set_dtype(0, dtype);
            checker.set_dtype(1, dtype);
            checker.set_dtype(2, dtype);
            checker.execs({{2, 3, 10, 11}, {2, 2, 3}, {2, 12, 11, 11}});
            checker.execs({{22, 3, 10, 12}, {22, 2, 3}, {22, 3, 11, 12}});
        }
    }
}

#if MEGDNN_WITH_BENCHMARK

TEST_F(CUDA, WARP_AFFINE_BENCHMARK) {
    const size_t RUNS = 50;
    Benchmarker<WarpAffine> benchmark(handle_cuda());
    benchmark.set_display(false);
    benchmark.set_times(RUNS);
    using Param = param::WarpAffine;
    Param param;

    auto run = [&benchmark, &param](TensorShape src, TensorShape mat,
                                    TensorShape dst) {
        benchmark.set_param(param);
        auto used = benchmark.execs({src, mat, dst});
        printf("src: %s dst: %s used: %.2f ms %.2f Gflops\n",
               src.to_string().c_str(), dst.to_string().c_str(), used,
               //! 8 mul + 3 add
               11 * dst.total_nr_elems() / (used / RUNS) / 1e6);
    };

    run({1, 100, 100, 1}, {1, 2, 3}, {1, 112, 112, 1});
    run({512, 100, 100, 1}, {512, 2, 3}, {512, 112, 112, 1});
}

#endif

}  // namespace test
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
