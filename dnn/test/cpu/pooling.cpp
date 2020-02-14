/**
 * \file dnn/test/cpu/pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cpu/fixture.h"

#include "test/common/pooling.h"
#include "test/common/checker.h"
#include "test/common/benchmarker.h"

namespace megdnn {
namespace test {

TEST_F(CPU, POOLING)
{
    auto args = pooling::get_args();
    using Format = param::Pooling::Format;
    for (auto dtype: std::vector<DType>{dtype::Int8(), dtype::Float32()})
    for (Format format: {Format::NCHW, Format::NHWC})
    for (auto &&arg: args) {
        auto param = arg.param;
        auto src = arg.ishape;
        Checker<Pooling> checker(handle());
        param.format = format;
        if (param.format == Format::NHWC) {
            src = cvt_src_or_dst_nchw2nhwc(src);
        }
        checker.set_param(param)
            .set_dtype(0, dtype)
            .set_dtype(1, dtype)
            .exec(TensorShapeArray{
                src, {}});
    }
}

TEST_F(CPU, POOLING_INT)
{
    UniformIntRNG rng(0, 255);
    for (int modeflag = 0; modeflag < 2; ++modeflag) {
        param::Pooling param;
        param.mode = modeflag ? param::Pooling::Mode::AVERAGE :
            param::Pooling::Mode::MAX;
        param.window_h = param.window_w = 2;
        param.stride_h = param.stride_w = 2;
        param.pad_h = param.pad_w = 0;
        std::vector<size_t> sizes = {10, 12, 13, 15, 20, 63};
        for (size_t ih: sizes)
        for (size_t iw: sizes)
        {
            Checker<Pooling> checker(handle());
            checker.set_rng(0, &rng);
            checker.set_rng(1, &rng);
            checker.set_rng(2, &rng);
            checker.set_dtype(0, dtype::Int8());
            checker.set_dtype(1, dtype::Int8());
            checker.set_param(param).exec(TensorShapeArray{
                    {2, 3, ih, iw}, {}});
        }
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CPU, BENCHMARK_POOLING_INT)
{
    UniformIntRNG rng(0, 255);
    for (int modeflag = 0; modeflag < 2; ++modeflag) {
        param::Pooling param;
        if (modeflag) {
            param.mode = param::Pooling::Mode::MAX;
            std::cout << "mode=max" << std::endl;
        } else {
            param.mode = param::Pooling::Mode::AVERAGE;
            std::cout << "mode=avg" << std::endl;
        }
        param.window_h = param.window_w = 2;
        param.stride_h = param.stride_w = 2;
        param.pad_h = param.pad_w = 0;
        float time_int, time_float;
        {
            std::cout << "int: ";
            Benchmarker<Pooling> benchmarker(handle());
            benchmarker.set_dtype(0, dtype::Int8());
            benchmarker.set_dtype(1, dtype::Int8());
            benchmarker.set_rng(0, &rng);
            benchmarker.set_rng(1, &rng);
            time_int = benchmarker.set_param(param).exec({{2, 3, 640, 480}, {}});
        }
        {
            std::cout << "float: ";
            Benchmarker<Pooling> benchmarker(handle());
            time_float = benchmarker.set_param(param).exec({{2, 3, 640, 480}, {}});
        }
        printf("time: int=%.3fms float=%.3fms\n", time_int, time_float);
    }
}
#endif

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen


