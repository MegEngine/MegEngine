/**
 * \file dnn/test/cuda/remap.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "test/common/remap.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/fixture.h"

namespace megdnn {
namespace test {
namespace remap {

TEST_F(CUDA, REMAP_NCHW_FLOAT) {
    Checker<Remap> checker(handle_cuda());
    std::vector<TestArg> args = get_nchw_args();
    UniformFloatRNG float_rng(0, 255);
#define cb(data_type, data_rng)                                              \
    for (auto arg : args) {                                                  \
        UniformFloatRNG map_rng(                                             \
                -2, std::max(arg.map_xy.shape[2], arg.map_xy.shape[1]) + 2); \
        checker.set_dtype(0, data_type)                                      \
                .set_dtype(1, dtype::Float32())                              \
                .set_dtype(2, data_type)                                     \
                .set_rng(0, &data_rng)                                       \
                .set_rng(1, &map_rng)                                        \
                .set_rng(2, &data_rng)                                       \
                .set_param(arg.param)                                        \
                .execs({arg.src, arg.map_xy, arg.dst});                      \
    }
    cb(dtype::Float32(), float_rng);
    cb(dtype::Float16(), float_rng);
#undef cb
#define cb(data_type, data_rng)                                              \
    for (auto arg : args) {                                                  \
        UniformFloatRNG map_rng(                                             \
                -2, std::max(arg.map_xy.shape[2], arg.map_xy.shape[1]) + 2); \
        checker.set_dtype(0, data_type)                                      \
                .set_dtype(1, dtype::Float32())                              \
                .set_dtype(2, data_type)                                     \
                .set_rng(0, &data_rng)                                       \
                .set_rng(1, &map_rng)                                        \
                .set_rng(2, &data_rng)                                       \
                .set_param(arg.param)                                        \
                .set_epsilon(1e-2)                                           \
                .execs({arg.src, arg.map_xy, arg.dst});                      \
    }
    cb(dtype::BFloat16(), float_rng);
#undef cb
}

TEST_F(CUDA, REMAP_NCHW_INT) {
    Checker<Remap> checker(handle_cuda());
    std::vector<TestArg> args = get_nchw_args();
    UniformIntRNG uint8_rng(0, 255);
    UniformIntRNG int8_rng(-128, 127);

#define cb(data_type, data_rng)                                              \
    for (auto arg : args) {                                                  \
        UniformFloatRNG map_rng(                                             \
                -2, std::max(arg.map_xy.shape[2], arg.map_xy.shape[1]) + 2); \
        checker.set_dtype(0, data_type)                                      \
                .set_dtype(1, dtype::Float32())                              \
                .set_dtype(2, data_type)                                     \
                .set_rng(0, &data_rng)                                       \
                .set_rng(1, &map_rng)                                        \
                .set_rng(2, &data_rng)                                       \
                .set_epsilon(1)                                              \
                .set_param(arg.param)                                        \
                .execs({arg.src, arg.map_xy, arg.dst});                      \
    }
    cb(dtype::Int8(), int8_rng);
    cb(dtype::Uint8(), uint8_rng);
#undef cb
}

TEST_F(CUDA, REMAP_NHWC_FLOAT) {
    Checker<Remap> checker(handle_cuda());
    std::vector<TestArg> args = get_nhwc_args();
    UniformFloatRNG float_rng(0, 255);
#define cb(data_type, data_rng)                                              \
    for (auto arg : args) {                                                  \
        UniformFloatRNG map_rng(                                             \
                -2, std::max(arg.map_xy.shape[2], arg.map_xy.shape[1]) + 2); \
        checker.set_dtype(0, data_type)                                      \
                .set_dtype(1, dtype::Float32())                              \
                .set_dtype(2, data_type)                                     \
                .set_rng(0, &data_rng)                                       \
                .set_rng(1, &map_rng)                                        \
                .set_rng(2, &data_rng)                                       \
                .set_param(arg.param)                                        \
                .execs({arg.src, arg.map_xy, arg.dst});                      \
    }
    cb(dtype::Float32(), float_rng);
    cb(dtype::Float16(), float_rng);
#undef cb
#define cb(data_type, data_rng)                                              \
    for (auto arg : args) {                                                  \
        UniformFloatRNG map_rng(                                             \
                -2, std::max(arg.map_xy.shape[2], arg.map_xy.shape[1]) + 2); \
        checker.set_dtype(0, data_type)                                      \
                .set_dtype(1, dtype::Float32())                              \
                .set_dtype(2, data_type)                                     \
                .set_rng(0, &data_rng)                                       \
                .set_rng(1, &map_rng)                                        \
                .set_rng(2, &data_rng)                                       \
                .set_param(arg.param)                                        \
                .set_epsilon(1e-2)                                           \
                .execs({arg.src, arg.map_xy, arg.dst});                      \
    }
    cb(dtype::BFloat16(), float_rng);
#undef cb
}

TEST_F(CUDA, REMAP_NHWC_INT) {
    Checker<Remap> checker(handle_cuda());
    std::vector<TestArg> args = get_nhwc_args();
    UniformIntRNG uint8_rng(0, 255);
    UniformIntRNG int8_rng(-128, 127);

#define cb(data_type, data_rng)                                              \
    for (auto arg : args) {                                                  \
        UniformFloatRNG map_rng(                                             \
                -2, std::max(arg.map_xy.shape[2], arg.map_xy.shape[1]) + 2); \
        checker.set_dtype(0, data_type)                                      \
                .set_dtype(1, dtype::Float32())                              \
                .set_dtype(2, data_type)                                     \
                .set_rng(0, &data_rng)                                       \
                .set_rng(1, &map_rng)                                        \
                .set_rng(2, &data_rng)                                       \
                .set_epsilon(1)                                              \
                .set_param(arg.param)                                        \
                .execs({arg.src, arg.map_xy, arg.dst});                      \
    }
    cb(dtype::Int8(), int8_rng);
    cb(dtype::Uint8(), uint8_rng);
#undef cb
}

TEST_F(CUDA, REMAP_BACKWARD_DATA) {
    Checker<RemapBackwardData> checker(handle_cuda());
    std::vector<TestArg> args = get_nchw_args();
    UniformFloatRNG float_rng(0, 255);
#define cb(data_type, data_rng)                                              \
    for (auto arg : args) {                                                  \
        UniformFloatRNG map_rng(                                             \
                -2, std::max(arg.map_xy.shape[2], arg.map_xy.shape[1]) + 2); \
        checker.set_dtype(1, data_type)                                      \
                .set_dtype(0, dtype::Float32())                              \
                .set_dtype(2, data_type)                                     \
                .set_rng(1, &data_rng)                                       \
                .set_rng(0, &map_rng)                                        \
                .set_rng(2, &data_rng)                                       \
                .set_param(arg.param)                                        \
                .execs({arg.map_xy, arg.dst, arg.src});                      \
    }
    cb(dtype::Float32(), float_rng);
#undef cb
#define cb(data_type, data_rng)                                              \
    for (auto arg : args) {                                                  \
        UniformFloatRNG map_rng(                                             \
                -2, std::max(arg.map_xy.shape[2], arg.map_xy.shape[1]) + 2); \
        checker.set_dtype(1, data_type)                                      \
                .set_dtype(0, dtype::Float32())                              \
                .set_dtype(2, data_type)                                     \
                .set_rng(1, &data_rng)                                       \
                .set_rng(0, &map_rng)                                        \
                .set_rng(2, &data_rng)                                       \
                .set_param(arg.param)                                        \
                .set_epsilon(1e-1)                                           \
                .execs({arg.map_xy, arg.dst, arg.src});                      \
    }
    cb(dtype::BFloat16(), float_rng);
#undef cb
}

TEST_F(CUDA, REMAP_BACKWARD_MAT) {
    Checker<RemapBackwardMat> checker(handle_cuda());
    std::vector<TestArg> args = get_nchw_args();
    UniformFloatRNG float_rng(0, 255);
#define cb(data_type, data_rng)                                              \
    for (auto arg : args) {                                                  \
        UniformFloatRNG map_rng(                                             \
                -2, std::max(arg.map_xy.shape[2], arg.map_xy.shape[1]) + 2); \
        checker.set_dtype(0, data_type)                                      \
                .set_dtype(1, dtype::Float32())                              \
                .set_dtype(2, data_type)                                     \
                .set_dtype(3, dtype::Float32())                              \
                .set_rng(0, &data_rng)                                       \
                .set_rng(1, &map_rng)                                        \
                .set_rng(2, &data_rng)                                       \
                .set_rng(3, &map_rng)                                        \
                .set_param(arg.param)                                        \
                .set_epsilon(2e-2)                                           \
                .execs({arg.src, arg.map_xy, arg.dst, arg.map_xy});          \
    }
    cb(dtype::Float32(), float_rng);
#undef cb
#define cb(data_type, data_rng)                                              \
    for (auto arg : args) {                                                  \
        UniformFloatRNG map_rng(                                             \
                -2, std::max(arg.map_xy.shape[2], arg.map_xy.shape[1]) + 2); \
        checker.set_dtype(0, data_type)                                      \
                .set_dtype(1, dtype::Float32())                              \
                .set_dtype(2, data_type)                                     \
                .set_dtype(3, dtype::Float32())                              \
                .set_rng(0, &data_rng)                                       \
                .set_rng(1, &map_rng)                                        \
                .set_rng(2, &data_rng)                                       \
                .set_rng(3, &map_rng)                                        \
                .set_param(arg.param)                                        \
                .set_epsilon(1e-1)                                           \
                .execs({arg.src, arg.map_xy, arg.dst, arg.map_xy});          \
    }
    cb(dtype::BFloat16(), float_rng);
#undef cb
}

#if MEGDNN_WITH_BENCHMARK

TEST_F(CUDA, BENCHMARK_REMAP) {
    using Param = param::Remap;
    auto run = [&](const TensorShapeArray& shapes, Param param, DType dtype) {
        auto handle_cpu = create_cpu_handle(2);
        Benchmarker<Remap> benchmarker_naive(handle_cpu.get());
        CUBenchmarker<Remap> benchmarker_cuda(handle_cuda());
        UniformIntRNG rng(0, 0xff);
        UniformFloatRNG map_rng(
                -2, std::max(shapes[1].shape[1], shapes[1].shape[2]) + 2);
        benchmarker_naive.set_rng(0, &rng);
        benchmarker_cuda.set_rng(0, &rng);
        benchmarker_naive.set_rng(1, &map_rng);
        benchmarker_cuda.set_rng(1, &map_rng);
        benchmarker_naive.set_rng(2, &rng);
        benchmarker_cuda.set_rng(2, &rng);

        benchmarker_naive.set_dtype(1, dtype::Float32());
        benchmarker_cuda.set_dtype(1, dtype::Float32());
        benchmarker_naive.set_dtype(0, dtype).set_dtype(2, dtype);
        benchmarker_cuda.set_dtype(0, dtype).set_dtype(2, dtype);

        size_t RUN = 10;
        auto t1 = benchmarker_naive.set_display(false)
                          .set_times(RUN)
                          .set_param(param)
                          .execs(shapes);
        auto t2 = benchmarker_cuda.set_display(false).set_param(param).execs(
                shapes);

        int size = 0;
        if (dtype == dtype::Float32{}) {
            size = sizeof(float);
            printf("float32: ");
        } else if (dtype == dtype::Float16{}) {
            size = sizeof(dt_float16);
            printf("float16: ");
        } else if (dtype == dtype::Int8{}) {
            size = sizeof(dt_int8);
            printf("int8:    ");
        } else if (dtype == dtype::Uint8{}) {
            size = sizeof(dt_uint8);
            printf("uint8:   ");
        }
        const TensorShape map_xy = shapes[1];
        const TensorShape dst_layout = shapes[2];

        float calc_amount = (dst_layout.total_nr_elems() * (4.f + 1.f) * size +
                             map_xy.total_nr_elems() * sizeof(float)) /
                            (1024 * 1024 * 1024);
        printf("naive={%.3fms, %.3fGBPS}, "
               "cuda={%.3fms, %.3fGBPS}\n",
               t1 / RUN, calc_amount / (t1 / RUN) * 1e3, t2,
               calc_amount / t2 * 1e3);
    };
    Param param;
    param.imode = param::Remap::InterpolationMode::LINEAR;
    param.format = param::Remap::Format::NHWC;
    param.border_type = param::Remap::BorderMode::CONSTANT;
    run({{4, 200, 300, 10}, {4, 200, 300, 2}, {4, 200, 300, 10}}, param,
        dtype::Float32{});
    run({{4, 200, 300, 10}, {4, 200, 300, 2}, {4, 200, 300, 10}}, param,
        dtype::Float16{});
    run({{4, 200, 300, 10}, {4, 200, 300, 2}, {4, 200, 300, 10}}, param,
        dtype::Uint8{});
    run({{4, 200, 300, 10}, {4, 200, 300, 2}, {4, 200, 300, 10}}, param,
        dtype::Int8{});
    param.border_type = param::Remap::BorderMode::REPLICATE;
    run({{4, 200, 300, 10}, {4, 200, 300, 2}, {4, 200, 300, 10}}, param,
        dtype::Float32{});
    run({{4, 200, 300, 10}, {4, 200, 300, 2}, {4, 200, 300, 10}}, param,
        dtype::Float16{});
    run({{4, 200, 300, 10}, {4, 200, 300, 2}, {4, 200, 300, 10}}, param,
        dtype::Uint8{});
    run({{4, 200, 300, 10}, {4, 200, 300, 2}, {4, 200, 300, 10}}, param,
        dtype::Int8{});
    param.format = param::Remap::Format::NCHW;
    param.border_type = param::Remap::BorderMode::CONSTANT;
    run({{4, 10, 200, 300}, {4, 200, 300, 2}, {4, 10, 200, 300}}, param,
        dtype::Float32{});
    run({{4, 10, 200, 300}, {4, 200, 300, 2}, {4, 10, 200, 300}}, param,
        dtype::Float16{});
    run({{4, 10, 200, 300}, {4, 200, 300, 2}, {4, 10, 200, 300}}, param,
        dtype::Uint8{});
    run({{4, 10, 200, 300}, {4, 200, 300, 2}, {4, 10, 200, 300}}, param,
        dtype::Int8{});
    param.border_type = param::Remap::BorderMode::REPLICATE;
    run({{4, 10, 200, 300}, {4, 200, 300, 2}, {4, 10, 200, 300}}, param,
        dtype::Float32{});
    run({{4, 10, 200, 300}, {4, 200, 300, 2}, {4, 10, 200, 300}}, param,
        dtype::Float16{});
    run({{4, 10, 200, 300}, {4, 200, 300, 2}, {4, 10, 200, 300}}, param,
        dtype::Uint8{});
    run({{4, 10, 200, 300}, {4, 200, 300, 2}, {4, 10, 200, 300}}, param,
        dtype::Int8{});
}

#endif

}  // namespace remap
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
