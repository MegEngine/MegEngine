#include "test/common/padding.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(CUDA, PADDING) {
    std::vector<padding::TestArg> args = padding::get_args();
    Checker<Padding> checker(handle_cuda());
    UniformIntNonZeroRNG rng(1, 9);
    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_rng(0, &rng)
                .set_dtype(0, dtype::Int8())
                .set_dtype(1, dtype::Int8())
                .execs({arg.src, arg.dst});
    }
}

TEST_F(CUDA, PADDING_BACKWARD) {
    std::vector<padding::TestArg> args = padding::get_args_backward();
    Checker<PaddingBackward> checker(handle_cuda());
    UniformFloatRNG rng(1, 9);
    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_rng(0, &rng)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .execs({arg.src, arg.dst});
    }
}

TEST_F(CUDA, PADDING_REFLECT) {
    Checker<Padding> checker(handle_cuda(), false);
    param::Padding param;
    param.padding_val = 10;
    param.padding_mode = param::Padding::PaddingMode::REFLECT;
    param.front_offset_dim0 = 2;
    param.front_offset_dim1 = 0;
    param.front_offset_dim2 = 0;
    param.front_offset_dim3 = 0;
    param.front_offset_dim4 = 0;
    param.front_offset_dim5 = 0;
    param.front_offset_dim6 = 0;
    param.back_offset_dim0 = 3;
    param.back_offset_dim1 = 0;
    param.back_offset_dim2 = 0;
    param.back_offset_dim3 = 0;
    param.back_offset_dim4 = 0;
    param.back_offset_dim5 = 0;
    param.back_offset_dim6 = 0;
    checker.set_param(param).exect(
            Testcase{TensorValue({5}, dtype::Int8(), {1, 2, 3, 4, 5}), {}},
            Testcase{
                    {},
                    TensorValue({10}, dtype::Int8(), {3, 2, 1, 2, 3, 4, 5, 4, 3, 2})});
}

TEST_F(CUDA, PADDING_REFLECT2) {
    Checker<Padding> checker(handle_cuda(), false);
    param::Padding param;
    param.padding_val = 10;
    param.padding_mode = param::Padding::PaddingMode::REFLECT;
    param.front_offset_dim0 = 1;
    param.front_offset_dim1 = 2;
    param.front_offset_dim2 = 0;
    param.front_offset_dim3 = 0;
    param.front_offset_dim4 = 0;
    param.front_offset_dim5 = 0;
    param.front_offset_dim6 = 0;
    param.back_offset_dim0 = 1;
    param.back_offset_dim1 = 2;
    param.back_offset_dim2 = 0;
    param.back_offset_dim3 = 0;
    param.back_offset_dim4 = 0;
    param.back_offset_dim5 = 0;
    param.back_offset_dim6 = 0;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue({3, 3}, dtype::Int8(), {3, 5, 1, 3, 6, 1, 4, 7, 9}),
                    {}},
            Testcase{{}, TensorValue({5, 7}, dtype::Int8(), {1, 6, 3, 6, 1, 6, 3, 1, 5,
                                                             3, 5, 1, 5, 3, 1, 6, 3, 6,
                                                             1, 6, 3, 9, 7, 4, 7, 9, 7,
                                                             4, 1, 6, 3, 6, 1, 6, 3})});
}

TEST_F(CUDA, PADDING_REFLECT2_QUANTIZED) {
    Checker<Padding> checker(handle_cuda(), false);
    param::Padding param;
    param.padding_mode = param::Padding::PaddingMode::REFLECT;
    param.front_offset_dim0 = 2;
    param.front_offset_dim1 = 1;
    param.front_offset_dim2 = 0;
    param.front_offset_dim3 = 0;
    param.front_offset_dim4 = 0;
    param.front_offset_dim5 = 0;
    param.front_offset_dim6 = 0;
    param.back_offset_dim0 = 0;
    param.back_offset_dim1 = 2;
    param.back_offset_dim2 = 0;
    param.back_offset_dim3 = 0;
    param.back_offset_dim4 = 0;
    param.back_offset_dim5 = 0;
    param.back_offset_dim6 = 0;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue(
                            {3, 3}, dtype::QuantizedS8(), {1, 2, 3, 4, 5, 6, 7, 8, 9}),
                    {}},
            Testcase{{}, TensorValue({5, 6}, dtype::QuantizedS8(), {8, 7, 8, 9, 8, 7, 5,
                                                                    4, 5, 6, 5, 4, 2, 1,
                                                                    2, 3, 2, 1, 5, 4, 5,
                                                                    6, 5, 4, 8, 7, 8, 9,
                                                                    8, 7})});
}

TEST_F(CUDA, PADDING_REPLICATE) {
    Checker<Padding> checker(handle_cuda(), false);
    param::Padding param;
    param.padding_val = 10;
    param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    param.front_offset_dim0 = 1;
    param.front_offset_dim1 = 0;
    param.front_offset_dim2 = 0;
    param.front_offset_dim3 = 0;
    param.front_offset_dim4 = 0;
    param.front_offset_dim5 = 0;
    param.front_offset_dim6 = 0;
    param.back_offset_dim0 = 2;
    param.back_offset_dim1 = 0;
    param.back_offset_dim2 = 0;
    param.back_offset_dim3 = 0;
    param.back_offset_dim4 = 0;
    param.back_offset_dim5 = 0;
    param.back_offset_dim6 = 0;
    checker.set_param(param).exect(
            Testcase{TensorValue({9}, dtype::Int8(), {1, 2, 3, 4, 5, 6, 7, 8, 9}), {}},
            Testcase{
                    {},
                    TensorValue(
                            {12}, dtype::Int8(),
                            {1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9})});
}

TEST_F(CUDA, PADDING_REPLICATE2) {
    Checker<Padding> checker(handle_cuda(), false);
    param::Padding param;
    param.padding_val = 10;
    param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    param.front_offset_dim0 = 2;
    param.front_offset_dim1 = 1;
    param.front_offset_dim2 = 0;
    param.front_offset_dim3 = 0;
    param.front_offset_dim4 = 0;
    param.front_offset_dim5 = 0;
    param.front_offset_dim6 = 0;
    param.back_offset_dim0 = 0;
    param.back_offset_dim1 = 3;
    param.back_offset_dim2 = 0;
    param.back_offset_dim3 = 0;
    param.back_offset_dim4 = 0;
    param.back_offset_dim5 = 0;
    param.back_offset_dim6 = 0;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue({3, 3}, dtype::Int8(), {1, 2, 3, 4, 5, 6, 7, 8, 9}),
                    {}},
            Testcase{{}, TensorValue({5, 7}, dtype::Int8(), {1, 1, 2, 3, 3, 3, 3, 1, 1,
                                                             2, 3, 3, 3, 3, 1, 1, 2, 3,
                                                             3, 3, 3, 4, 4, 5, 6, 6, 6,
                                                             6, 7, 7, 8, 9, 9, 9, 9})});
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_PADDING_CONSTANT) {
    using Param = Padding::Param;

    auto run = [&](const TensorShapeArray& shapes, Param param) {
        CUBenchmarker<PaddingForward> benchmarker(handle_cuda());
        benchmarker.set_param(param);
        benchmarker.set_dtype(0, dtype::Int8()).set_dtype(1, dtype::Int8());

        for (auto&& shape : shapes) {
            double memaccess =
                    double(TensorLayout(shape, dtype::Int8()).span().dist_byte()) *
                    2e-6;
            auto time_ms = benchmarker.execs({shape, {}});
            printf("execute %s, time %.4f ms, %.4f GB/s\n", shape.to_string().c_str(),
                   time_ms, memaccess / time_ms);
        }
    };

    printf("mode -> constant; dtype -> int8\n");
    {
        TensorShapeArray shapes = {
                {16, 3, 736, 1280},
        };
        Param param;
        param.padding_mode = param::Padding::PaddingMode::CONSTANT;
        param.front_offset_dim1 = 1;
        run(shapes, param);
    }

    printf("mode -> replicate; dtype -> int8\n");
    {
        TensorShapeArray shapes = {
                {16, 3, 736, 1280},
        };
        Param param;
        param.padding_mode = param::Padding::PaddingMode::REPLICATE;
        param.front_offset_dim1 = 1;
        run(shapes, param);
    }
    printf("mode -> reflect; dtype -> int8\n");
    {
        TensorShapeArray shapes = {
                {16, 3, 736, 1280},
        };
        Param param;
        param.padding_mode = param::Padding::PaddingMode::REFLECT;
        param.front_offset_dim1 = 1;
        run(shapes, param);
    }
}
#endif
