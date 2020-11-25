/**
 * \file dnn/test/cuda/pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "megdnn/tensor_iter.h"
#include "test/common/checker.h"
#include "test/common/pooling.h"

#include "src/common/utils.h"
#include "test/cuda/utils.h"

// to check cudnn version
#include <cudnn.h>
#include "test/cuda/benchmark.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, POOLING_FORWARD)
{
    auto args = pooling::get_args();
    using Format = param::Pooling::Format;
    std::vector<DType> dtypes{dtype::Float16(), dtype::BFloat16(), dtype::Float32()};
    if (check_compute_capability(6, 0)) {
        // int pooling is supported only for Pascal or higher
        dtypes.push_back(dtype::Int8());
    }
    for (auto dtype: dtypes)
    for (auto format: {Format::NCHW, Format::NHWC})
    for (auto &&arg: args) {
        auto param = arg.param;
        auto src = arg.ishape;
        param.format = format;
        if (param.format == Format::NHWC) {
            src = cvt_src_or_dst_nchw2nhwc(src);
        }
        Checker<Pooling> checker(handle_cuda());
        if (dtype == dtype::Int8()) {
            // different versions of cuDNN differs in rounding behavior;
            // setting eps to 1 to allow for rounding errors.
            checker.set_epsilon(1 + 1e-3);
        } else if (dtype == dtype::BFloat16()) {
            checker.set_epsilon(2e-2);
        } else {
            checker.set_epsilon(1e-2);
        }
        checker.set_param(param)
            .set_dtype(0, dtype)
            .set_dtype(1, dtype)
            .exec(TensorShapeArray{
                src, {}});
    }

    /* add test for new Mode temporarily */
    for (auto dtype: dtypes)
    for (auto format: {Format::NCHW, Format::NHWC})
    for(auto &&arg : args) {
        auto param = arg.param;
        if(param.mode == Pooling::Mode::AVERAGE)
            param.mode = Pooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING;
        else continue;
        auto src = arg.ishape;
        param.format = format;
        if (param.format == Format::NHWC) {
            src = cvt_src_or_dst_nchw2nhwc(src);
        }
        Checker<Pooling> checker(handle_cuda());
        if (dtype == dtype::Int8()) {
            // different versions of cuDNN differs in rounding behavior;
            // setting eps to 1 to allow for rounding errors.
            checker.set_epsilon(1 + 1e-3);
        } else if (dtype == dtype::BFloat16()) {
            checker.set_epsilon(2e-2);
        }
        else {
            checker.set_epsilon(1e-2);
        }
        checker.set_param(param)
            .set_dtype(0, dtype)
            .set_dtype(1, dtype)
            .exec(TensorShapeArray{
                src, {}});
    }
}

TEST_F(CUDA, POOLING_BACKWARD)
{
    auto args = pooling::get_args();
    for (auto &&arg: args) {
        Checker<PoolingBackward> checker(handle_cuda());
        TensorLayout ilayout = TensorLayout(arg.ishape, dtype::Float32());
        TensorLayout olayout;

        auto constraint = [this,
                           arg](CheckerHelper::TensorValueArray& tensors_orig) {
            megdnn_assert(tensors_orig.size() == 4);
            auto opr = handle_cuda()->create_operator<PoolingForward>();
            opr->param() = arg.param;

            auto tensors_cuda_storage = CheckerHelper::alloc_tensors(
                    handle_cuda(),
                    {tensors_orig[0].layout, tensors_orig[1].layout}, 0);
            auto&& tensors_cuda = *tensors_cuda_storage;

            auto span = tensors_cuda[0].layout.span();
            auto dst = static_cast<dt_byte*>(tensors_cuda[0].raw_ptr) +
                       span.low_byte;
            auto src = static_cast<const dt_byte*>(tensors_orig[0].raw_ptr) +
                       span.low_byte;
            megdnn_memcpy_H2D(handle_cuda(), dst, src, span.dist_byte());

            auto workspace_size = opr->get_workspace_in_bytes(
                    tensors_cuda[0].layout, tensors_cuda[1].layout);
            auto workspace_cuda = megdnn_malloc(handle_cuda(), workspace_size);
            Workspace workspace{static_cast<dt_byte*>(workspace_cuda),
                                workspace_size};
            opr->exec(tensors_cuda[0], tensors_cuda[1], workspace);
            megdnn_free(handle_cuda(), workspace_cuda);

            span = tensors_cuda[1].layout.span();
            dst = static_cast<dt_byte*>(tensors_orig[1].raw_ptr) +
                  span.low_byte;
            src = static_cast<const dt_byte*>(tensors_cuda[1].raw_ptr) +
                  span.low_byte;
            megdnn_memcpy_D2H(handle_cuda(), dst, src, span.dist_byte());
        };

        {
            auto opr = handle_cuda()->create_operator<PoolingForward>();
            opr->param() = arg.param;
            opr->deduce_layout(ilayout, olayout);
        }
        auto set_dtype = [&checker](DType dtype)
        {
            checker.set_dtype(0, dtype).
                set_dtype(1, dtype).
                set_dtype(2, dtype).
                set_dtype(3, dtype);
        };

        checker.set_tensors_constraint(constraint);
        set_dtype(dtype::Float32());
        checker.set_param(arg.param).exec(TensorShapeArray{
                ilayout, olayout, olayout, ilayout});
        Float16PeriodicalRNG rng;
        set_dtype(dtype::Float16());
        checker
            .set_param(arg.param)
            .set_rng(0, &rng)
            .set_epsilon(1e-2)
            .exec(TensorShapeArray{
                    ilayout, olayout, olayout, ilayout});
        BFloat16PeriodicalRNG bf16_rng;
        set_dtype(dtype::BFloat16());
        checker.set_param(arg.param)
                .set_rng(0, &bf16_rng)
                .set_epsilon(1e-2)
                .exec(TensorShapeArray{ilayout, olayout, olayout, ilayout});
    }

    /* add test for new Mode temporarily */
    for(auto &&arg : args) {
        if(arg.param.mode == Pooling::Mode::AVERAGE)
            arg.param.mode = Pooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING;
        else continue;
        Checker<PoolingBackward> checker(handle_cuda());
        TensorLayout ilayout = TensorLayout(arg.ishape, dtype::Float32());
        TensorLayout olayout;

        auto constraint = [this,
                           arg](CheckerHelper::TensorValueArray& tensors_orig) {
            megdnn_assert(tensors_orig.size() == 4);
            auto opr = handle_cuda()->create_operator<PoolingForward>();
            opr->param() = arg.param;

            auto tensors_cuda_storage = CheckerHelper::alloc_tensors(
                    handle_cuda(),
                    {tensors_orig[0].layout, tensors_orig[1].layout}, 0);
            auto&& tensors_cuda = *tensors_cuda_storage;

            auto span = tensors_cuda[0].layout.span();
            auto dst = static_cast<dt_byte*>(tensors_cuda[0].raw_ptr) +
                       span.low_byte;
            auto src = static_cast<const dt_byte*>(tensors_orig[0].raw_ptr) +
                       span.low_byte;
            megdnn_memcpy_H2D(handle_cuda(), dst, src, span.dist_byte());

            auto workspace_size = opr->get_workspace_in_bytes(
                    tensors_cuda[0].layout, tensors_cuda[1].layout);
            auto workspace_cuda = megdnn_malloc(handle_cuda(), workspace_size);
            Workspace workspace{static_cast<dt_byte*>(workspace_cuda),
                                workspace_size};
            opr->exec(tensors_cuda[0], tensors_cuda[1], workspace);
            megdnn_free(handle_cuda(), workspace_cuda);

            span = tensors_cuda[1].layout.span();
            dst = static_cast<dt_byte*>(tensors_orig[1].raw_ptr) +
                  span.low_byte;
            src = static_cast<const dt_byte*>(tensors_cuda[1].raw_ptr) +
                  span.low_byte;
            megdnn_memcpy_D2H(handle_cuda(), dst, src, span.dist_byte());
        };

        {
            auto opr = handle_cuda()->create_operator<PoolingForward>();
            opr->param() = arg.param;
            opr->deduce_layout(ilayout, olayout);
        }
        auto set_dtype = [&checker](DType dtype)
        {
            checker.set_dtype(0, dtype).
                set_dtype(1, dtype).
                set_dtype(2, dtype).
                set_dtype(3, dtype);
        };

        checker.set_tensors_constraint(constraint);
        set_dtype(dtype::Float32());
        checker.set_param(arg.param).exec(TensorShapeArray{
                ilayout, olayout, olayout, ilayout});
        Float16PeriodicalRNG rng;
        set_dtype(dtype::Float16());
        checker
            .set_param(arg.param)
            .set_rng(0, &rng)
            .set_epsilon(1e-2)
            .exec(TensorShapeArray{
                    ilayout, olayout, olayout, ilayout});
        BFloat16PeriodicalRNG bf16_rng;
        set_dtype(dtype::BFloat16());
        checker.set_param(arg.param)
                .set_rng(0, &bf16_rng)
                .set_epsilon(1e-2)
                .exec(TensorShapeArray{ilayout, olayout, olayout, ilayout});
    }
}

TEST_F(CUDA, POOLING_FORWARD_NCHW4) {
    require_compute_capability(7, 5);
    using Param = param::Pooling;
    Checker<Pooling> checker(handle_cuda());
    Param param;
    checker.set_dtype(0, dtype::QuantizedS8(0.1f));
    param.format = Param::Format::NCHW4;
    checker.set_epsilon(1 + 1e-3);
    checker.set_param(param).exec({{20, 3, 50, 50, 4}, {}});
}

#if CUDNN_VERSION >= 7500
TEST_F(CUDA, POOLING_FORWARD_NCHW32) {
    require_compute_capability(7, 5);
    using Param = param::Pooling;
    Checker<Pooling> checker(handle_cuda());
    Param param;
    auto i8_min = std::numeric_limits<int8_t>().min();
    auto i8_max = std::numeric_limits<int8_t>().max();
    UniformIntRNG int_rng{i8_min, i8_max};
    checker.set_dtype(0, dtype::QuantizedS8(0.1f));
    param.format = Param::Format::NCHW32;
    checker.set_epsilon(1e-3).set_rng(0, &int_rng);
    checker.set_param(param).exec({{64, 8, 28, 28, 32}, {}});
}
#endif

TEST_F(CUDA, POOLING_FORWARD_CHWN4) {
    require_compute_capability(6, 1);
    using Param = param::Pooling;
    Checker<Pooling> checker(handle_cuda());
    Param param;
    auto i8_min = std::numeric_limits<int8_t>().min();
    auto i8_max = std::numeric_limits<int8_t>().max();
    UniformIntRNG int_rng{i8_min, i8_max};
    checker.set_dtype(0, dtype::QuantizedS8(0.1f));
    param.format = Param::Format::CHWN4;
    for (auto mode : {Param::Mode::MAX, Param::Mode::AVERAGE,
                      Param::Mode::AVERAGE_COUNT_EXCLUDE_PADDING}) {
        param.mode = mode;
        checker.set_epsilon(1e-3).set_rng(0, &int_rng);
        checker.set_param(param).exec({{8, 28, 28, 64, 4}, {}});
        checker.set_param(param).exec({{8, 28, 28, 15, 4}, {}});
        checker.set_param(param).exec({{8, 28, 28, 30, 4}, {}});
    }
}

TEST_F(CUDA, POOLING_FORWARD_INT8_NCHW4) {
    require_compute_capability(6, 1);
    using Param = param::Pooling;
    Checker<Pooling> checker(handle_cuda());
    Param param;
    auto i8_min = std::numeric_limits<int8_t>().min();
    auto i8_max = std::numeric_limits<int8_t>().max();
    UniformIntRNG int_rng{i8_min, i8_max};
    checker.set_dtype(0, dtype::QuantizedS8(0.1f));
    param.format = Param::Format::NCHW4;
    for (auto mode : {Param::Mode::MAX, Param::Mode::AVERAGE,
                      Param::Mode::AVERAGE_COUNT_EXCLUDE_PADDING}) {
        param.mode = mode;
        checker.set_epsilon(1e-3).set_rng(0, &int_rng);
        checker.set_param(param).exec({{64, 8, 28, 28, 4}, {}});
        checker.set_param(param).exec({{15, 8, 28, 28, 4}, {}});
        checker.set_param(param).exec({{30, 8, 28, 28, 4}, {}});
    }
}

TEST_F(CUDA, POOLING_FORWARD_INT8_NCHW32) {
    require_compute_capability(6, 1);
    using Param = param::Pooling;
    Checker<Pooling> checker(handle_cuda());
    Param param;
    auto i8_min = std::numeric_limits<int8_t>().min();
    auto i8_max = std::numeric_limits<int8_t>().max();
    UniformIntRNG int_rng{i8_min, i8_max};
    checker.set_dtype(0, dtype::QuantizedS8(0.1f));
    param.format = Param::Format::NCHW32;
    for (auto mode : {Param::Mode::MAX, Param::Mode::AVERAGE,
                      Param::Mode::AVERAGE_COUNT_EXCLUDE_PADDING}) {
        param.mode = mode;
        checker.set_epsilon(1e-3).set_rng(0, &int_rng);
        checker.set_param(param).exec({{64, 8, 28, 28, 32}, {}});
        checker.set_param(param).exec({{15, 8, 28, 28, 32}, {}});
        checker.set_param(param).exec({{30, 8, 28, 28, 32}, {}});
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_POOLING_CHWN4) {
    CUBenchmarker<Pooling> bencher(handle_cuda());
    size_t nr_times = 1000;
    bencher.set_times(nr_times);
    using Param = param::Pooling;
    Param param;
    auto run_bench = [&](size_t N, size_t C, size_t H, size_t W, size_t stride,
                         size_t padding, size_t window,
                         Param::Mode mode = Param::Mode::MAX) {
        param.mode = mode;
        param.pad_h = param.pad_w = padding;
        param.window_h = param.window_w = window;
        param.stride_h = param.stride_w = stride;
        param.format = Param::Format::NCHW4;
        bencher.set_dtype(0, dtype::QuantizedS8{0.1f});
        bencher.set_param(param);
        auto time_cudnn = bencher.execs({{N, C / 4, H, W, 4}, {}}) / nr_times;
        param.format = Param::Format::CHWN4;
        bencher.set_param(param);
        auto time_chwn4 = bencher.execs({{C / 4, H, W, N, 4}, {}}) / nr_times;
        auto time_nchw32 =
                bencher.execs({{N, C / 32, H, W, 32}, {}}) / nr_times;
        size_t oh = infer_conv_shape(H, window, stride, padding),
               ow = infer_conv_shape(W, window, stride, padding);
        float io = (N * C * H * W + N * C * oh * ow) * sizeof(int8_t);
        printf("time(cudnn)=%.2f ms, time(chwn4)=%.2f ms, time(nchw32)=%.2f "
               "ms, "
               "bandwidth(cudnn)=%.2f Gb/s, bandwidth(chwn4)=%.2f Gb/s, "
               "bandwidth(nchw32)=%.2f Gb/s\n",
               time_cudnn, time_chwn4, time_nchw32, io / (1e6 * time_cudnn),
               io / (1e6 * time_chwn4), io / (1e6 * time_nchw32));
    };
    run_bench(64, 64, 112, 112, 2, 1, 2);
    run_bench(256, 64, 112, 112, 2, 1, 2);
    run_bench(64, 64, 112, 112, 2, 1, 2, Param::Mode::AVERAGE);
    run_bench(256, 64, 112, 112, 2, 1, 2, Param::Mode::AVERAGE);
    run_bench(64, 64, 112, 112, 2, 1, 2,
              Param::Mode::AVERAGE_COUNT_EXCLUDE_PADDING);
    run_bench(256, 64, 112, 112, 2, 1, 2,
              Param::Mode::AVERAGE_COUNT_EXCLUDE_PADDING);
}
#endif
} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
