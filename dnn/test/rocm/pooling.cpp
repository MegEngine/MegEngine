/**
 * \file dnn/test/rocm/pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "test/rocm/fixture.h"

#include "megdnn/tensor_iter.h"
#include "test/common/checker.h"
#include "test/common/pooling.h"
#include "test/rocm/benchmarker.h"

#include "src/rocm/utils.h"
#include "src/common/utils.h"

namespace megdnn {
namespace test {

TEST_F(ROCM, POOLING_FORWARD) {
    auto args = pooling::get_args();
    using Format = param::Pooling::Format;
    std::vector<DType> dtypes{MEGDNN_INC_FLOAT16(dtype::Float16() MEGDNN_COMMA)
                                      dtype::Float32()};
    for (auto dtype : dtypes)
        for (auto format : {Format::NCHW})
            for (auto&& arg : args) {
                auto param = arg.param;
                auto src = arg.ishape;
                param.format = format;
                Checker<Pooling> checker(handle_rocm());
                checker.set_epsilon(1e-2);
                checker.set_param(param)
                        .set_dtype(0, dtype)
                        .set_dtype(1, dtype)
                        .exec(TensorShapeArray{src, {}});
            }
}

TEST_F(ROCM, POOLING_BACKWARD) {
    auto args = pooling::get_args();
    for (auto&& arg : args) {
        Checker<PoolingBackward> checker(handle_rocm());
        TensorLayout ilayout = TensorLayout(arg.ishape, dtype::Float32());
        TensorLayout olayout;

        auto constraint = [this,
                           arg](CheckerHelper::TensorValueArray& tensors_orig) {
            megdnn_assert(tensors_orig.size() == 4);
            auto opr = handle_rocm()->create_operator<PoolingForward>();
            opr->param() = arg.param;

            auto tensors_rocm_storage = CheckerHelper::alloc_tensors(
                    handle_rocm(),
                    {tensors_orig[0].layout, tensors_orig[1].layout}, 0);
            auto&& tensors_rocm = *tensors_rocm_storage;

            auto span = tensors_rocm[0].layout.span();
            auto dst = static_cast<dt_byte*>(tensors_rocm[0].raw_ptr) +
                       span.low_byte;
            auto src = static_cast<const dt_byte*>(tensors_orig[0].raw_ptr) +
                       span.low_byte;
            megdnn_memcpy_H2D(handle_rocm(), dst, src, span.dist_byte());

            auto workspace_size = opr->get_workspace_in_bytes(
                    tensors_rocm[0].layout, tensors_rocm[1].layout);
            auto workspace_rocm = megdnn_malloc(handle_rocm(), workspace_size);
            Workspace workspace{static_cast<dt_byte*>(workspace_rocm),
                                workspace_size};
            opr->exec(tensors_rocm[0], tensors_rocm[1], workspace);
            megdnn_free(handle_rocm(), workspace_rocm);

            span = tensors_rocm[1].layout.span();
            dst = static_cast<dt_byte*>(tensors_orig[1].raw_ptr) +
                  span.low_byte;
            src = static_cast<const dt_byte*>(tensors_rocm[1].raw_ptr) +
                  span.low_byte;
            megdnn_memcpy_D2H(handle_rocm(), dst, src, span.dist_byte());
        };

        {
            auto opr = handle_rocm()->create_operator<PoolingForward>();
            opr->param() = arg.param;
            opr->deduce_layout(ilayout, olayout);
        }
        auto set_dtype = [&checker](DType dtype) {
            checker.set_dtype(0, dtype)
                    .set_dtype(1, dtype)
                    .set_dtype(2, dtype)
                    .set_dtype(3, dtype);
        };

        checker.set_tensors_constraint(constraint);
        set_dtype(dtype::Float32());
        checker.set_param(arg.param).exec(
                TensorShapeArray{ilayout, olayout, olayout, ilayout});
#if !MEGDNN_DISABLE_FLOAT16
//! FIXME: MIOpen pooling backward for fp16 with bug
#if 0
        Float16PeriodicalRNG rng;
        set_dtype(dtype::Float16());
        checker.set_param(arg.param).set_rng(0, &rng).set_epsilon(1e-2).exec(
                TensorShapeArray{ilayout, olayout, olayout, ilayout});
#endif
#endif
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(ROCM, POOLING_FWD_BENCHMARK) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), true);
    auto benchmarker =
            ROCMBenchmarker<PoolingForward>(handle_rocm(), handle_naive(false));
    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW, size_t SH = 1,
                   size_t SW = 1, size_t FH = 1, size_t FW = 1, size_t PH = 0,
                   size_t PW = 0, DType dtype = dtype::Float32()) {
        benchmarker.set_dtype(0, dtype).set_dtype(1, dtype);
        benchmarker.set_display(true);
        PoolingForward::Param param;
        param.mode = param::Pooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING;
        param.stride_h = SH;
        param.stride_w = SW;
        param.pad_h = PH;
        param.pad_w = PW;
        param.window_h = FH;
        param.window_w = FW;
        benchmarker.set_param(param);
        size_t OH = infer_conv_shape(IH, FH, SH, PH);
        size_t OW = infer_conv_shape(IW, FW, SW, PW);
        // warm up
        benchmarker.execs({{N, IC, IH, IW}, {N, IC, OH, OW}});
        // do actual benchmark
        auto time_ms = benchmarker.execs({{N, IC, IH, IW}, {N, IC, OH, OW}});
        time_ms = benchmarker.execs({{N, IC, IH, IW}, {N, IC, OH, OW}});
        auto io = (double)N * IC * OH * OW * (1 + FH * FW) * dtype.size();
        auto gbps = io / (time_ms * 1e6);
        printf("io %.2fGB, flops %.3fGB/s\n", io / 1e9, gbps);
    };
    run(32, 128, 80, 64, 2, 2, 2, 2, 0, 0);
    run(32, 128, 40, 128, 2, 2, 2, 2, 0, 0);
    run(32, 224, 40, 32, 2, 2, 2, 2, 0, 0);

    run(32, 24, 160, 128, 2, 2, 4, 4, 0, 0);
    run(32, 24, 160, 128, 2, 2, 4, 4, 1, 1);
}

TEST_F(ROCM, POOLING_BWD_BENCHMARK) {
    using Mode = param::Pooling::Mode;
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), true);
    auto benchmarker = ROCMBenchmarker<PoolingBackward>(handle_rocm(),
                                                        handle_naive(false));
    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW, size_t SH = 1,
                   size_t SW = 1, size_t FH = 1, size_t FW = 1, size_t PH = 0,
                   size_t PW = 0,
                   Mode mode = Mode::AVERAGE_COUNT_EXCLUDE_PADDING,
                   DType dtype = dtype::Float32()) {
        benchmarker.set_dtype(0, dtype).set_dtype(1, dtype);
        benchmarker.set_display(true);
        PoolingForward::Param param;
        param.mode = mode;
        param.stride_h = SH;
        param.stride_w = SW;
        param.pad_h = PH;
        param.pad_w = PW;
        param.window_h = FH;
        param.window_w = FW;
        benchmarker.set_param(param);
        size_t OH = infer_conv_shape(IH, FH, SH, PH);
        size_t OW = infer_conv_shape(IW, FW, SW, PW);
        // warm up
        benchmarker.execs({{N, IC, IH, IW},
                           {N, IC, OH, OW},
                           {N, IC, OH, OW},
                           {N, IC, IH, IW}});
        // do actual benchmark
        auto time_ms = benchmarker.execs({{N, IC, IH, IW},
                                          {N, IC, OH, OW},
                                          {N, IC, OH, OW},
                                          {N, IC, IH, IW}});
        time_ms = benchmarker.execs({{N, IC, IH, IW},
                                     {N, IC, OH, OW},
                                     {N, IC, OH, OW},
                                     {N, IC, IH, IW}});
        double io = 0.;
        double gbps = 0.;
        if (mode == Mode::AVERAGE_COUNT_EXCLUDE_PADDING) {
            io = (double)N * IC * OH * OW * FH * FW * 2 * dtype.size();
            gbps = io / (time_ms * 1e6);
        } else {
            io = (double)N * IC * OH * OW * 2 * dtype.size();
            gbps = io / (time_ms * 1e6);
        }
        printf("Mode = %s, io %.2fGB, flops %.3fGB/s\n",
               mode == Mode::AVERAGE_COUNT_EXCLUDE_PADDING ? "AVERAGE" : "MAX",
               io / 1e9, gbps);
    };
    Mode mode = Mode::AVERAGE_COUNT_EXCLUDE_PADDING;
    run(32, 128, 80, 64, 2, 2, 2, 2, 0, 0, mode);
    run(32, 128, 40, 128, 2, 2, 2, 2, 0, 0, mode);
    run(32, 224, 40, 32, 2, 2, 2, 2, 0, 0, mode);

    run(32, 24, 160, 128, 2, 2, 4, 4, 0, 0, mode);
    run(32, 24, 160, 128, 2, 2, 4, 4, 1, 1, mode);

    mode = Mode::MAX;
    run(32, 128, 80, 64, 2, 2, 2, 2, 0, 0, mode);
    run(32, 128, 40, 128, 2, 2, 2, 2, 0, 0, mode);
    run(32, 224, 40, 32, 2, 2, 2, 2, 0, 0, mode);

    run(32, 24, 160, 128, 2, 2, 4, 4, 0, 0, mode);
    run(32, 24, 160, 128, 2, 2, 4, 4, 1, 1, mode);
}
#endif

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
