/**
 * \file dnn/test/cuda/cvt_color.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/checker.h"
#include "test/common/benchmarker.h"
#include "test/common/cvt_color.h"

#include "test/cuda/fixture.h"

namespace megdnn {
namespace test {

using Mode = param::CvtColor::Mode;

TEST_F(CUDA, CVTCOLOR)
{
    using namespace cvt_color;
    std::vector<TestArg> args = get_cuda_args();
    Checker<CvtColor> checker(handle_cuda());

    for (auto &&arg: args) {
        checker.set_param(arg.param)
            .set_dtype(0, arg.dtype)
            .set_dtype(1, arg.dtype)
            .execs({arg.src, {}});
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_CVTCOLOR_RGB2GRAY)
{
    using namespace cvt_color;
    using Param = param::CvtColor;

#define BENCHMARK_PARAM(benchmarker, dtype) \
        benchmarker.set_param(param); \
        benchmarker.set_dtype(0, dtype);

    auto run = [&](const TensorShapeArray& shapes, Param param) {
        auto handle_naive = create_cpu_handle(2);
        Benchmarker<CvtColor> benchmarker(handle_cuda());
        Benchmarker<CvtColor> benchmarker_naive(handle_naive.get());

        BENCHMARK_PARAM(benchmarker, dtype::Uint8());
        BENCHMARK_PARAM(benchmarker_naive, dtype::Uint8());
        for (auto&& shape : shapes) {
            printf("execute %s: current---naive\n", shape.to_string().c_str());
            benchmarker.execs({shape, {}});
            benchmarker_naive.execs({shape, {}});
        }

        BENCHMARK_PARAM(benchmarker, dtype::Float32());
        BENCHMARK_PARAM(benchmarker_naive, dtype::Float32());
        for (auto&& shape : shapes) {
            printf("execute %s: current---naive\n", shape.to_string().c_str());
            benchmarker.execs({shape, {}});
            benchmarker_naive.execs({shape, {}});
        }

    };

    Param param;
    TensorShapeArray shapes = {
        {1, 500, 512, 3},
        {2, 500, 512, 3},
    };

    param.mode = Param::Mode::RGB2GRAY;
    run(shapes, param);
#undef BENCHMARK_PARAM
}

// benchmark cvtcolor planar or semi-planar YUV to RGB, BGR or gray.
// data type: uint8
TEST_F(CUDA, BENCHMARK_CVTCOLOR_YUV2XXX_PLANAR_SEMIPLANAR_8U)
{
    using namespace cvt_color;
    using Param = param::CvtColor;
    int nrun = 10;

#define BENCHMARK_PARAM(benchmarker, dtype) \
    benchmarker.set_times(nrun);            \
    benchmarker.set_param(param);           \
    benchmarker.set_dtype(0, dtype);

    auto run = [&](const TensorShapeArray& shapes, Param param) {
        auto handle_naive = create_cpu_handle(2);
        Benchmarker<CvtColor> benchmarker(handle_cuda());
        Benchmarker<CvtColor> benchmarker_naive(handle_naive.get());

        BENCHMARK_PARAM(benchmarker, dtype::Uint8());
        BENCHMARK_PARAM(benchmarker_naive, dtype::Uint8());
        for (auto&& shape : shapes) {
            printf("execute %s\n", shape.to_string().c_str());
            printf("current: ");
            float t = benchmarker.execs({shape, {}}) / nrun;
            size_t computation;
            if (param.mode == Mode::YUV2GRAY_NV21 ||
                param.mode == Mode::YUV2GRAY_NV12 ||
                param.mode == Mode::YUV2GRAY_YU12 ||
                param.mode == Mode::YUV2GRAY_YV12) {
                computation = shape.total_nr_elems()/3*4;
            } else {
                computation = shape.total_nr_elems()*3;
            }
            printf("bandwidth: %.2f GiBPS\n",
                   (float)computation / (1<<30) / (t/1000));
            printf("naive: ");
            benchmarker_naive.execs({shape, {}});
        }
    };

    Param param;
    TensorShapeArray shapes = {
        {1, 480, 512, 1},
        {2, 480, 512, 1}
    };

#define MEGDNN_CALL_CVTCOLOR_BENCHMARKER(_mode) { \
        param.mode = _mode;                       \
        printf("\n=== run mode=" #_mode "\n");    \
        run(shapes, param);                       \
    }

    MEGDNN_CALL_CVTCOLOR_BENCHMARKER(Mode::YUV2BGR_NV21)
    MEGDNN_CALL_CVTCOLOR_BENCHMARKER(Mode::YUV2RGB_NV21)
    MEGDNN_CALL_CVTCOLOR_BENCHMARKER(Mode::YUV2BGR_NV12)
    MEGDNN_CALL_CVTCOLOR_BENCHMARKER(Mode::YUV2RGB_NV12)
    MEGDNN_CALL_CVTCOLOR_BENCHMARKER(Mode::YUV2BGR_YV12)
    MEGDNN_CALL_CVTCOLOR_BENCHMARKER(Mode::YUV2RGB_YV12)
    MEGDNN_CALL_CVTCOLOR_BENCHMARKER(Mode::YUV2BGR_YU12)
    MEGDNN_CALL_CVTCOLOR_BENCHMARKER(Mode::YUV2RGB_YU12)
    MEGDNN_CALL_CVTCOLOR_BENCHMARKER(Mode::YUV2GRAY_NV21)
    MEGDNN_CALL_CVTCOLOR_BENCHMARKER(Mode::YUV2GRAY_NV12)
    MEGDNN_CALL_CVTCOLOR_BENCHMARKER(Mode::YUV2GRAY_YV12)
    MEGDNN_CALL_CVTCOLOR_BENCHMARKER(Mode::YUV2GRAY_YU12)

#undef MEGDNN_CALL_CVTCOLOR_BENCHMARKER
#undef BENCHMARK_PARAM
}
#endif

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen
