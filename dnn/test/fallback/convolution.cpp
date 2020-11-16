/**
 * \file dnn/test/fallback/convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "megdnn/dtype.h"
#include "test/fallback/fixture.h"

#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/convolution.h"

#include "test/common/rng.h"

using namespace megdnn;
using namespace test;

#if MEGDNN_WITH_BENCHMARK

TEST_F(FALLBACK, BENCHMARK_CONVOLUTION_MATRIX_MUL) {
    using Param = Convolution::Param;
    auto run = [&](const TensorShapeArray& shapes, Param param) {
        Benchmarker<Convolution> benchmarker_float(handle());
        size_t RUN = 50;
        auto tfloat = benchmarker_float.set_display(false)
                              .set_dtype(0, dtype::Float32{})
                              .set_dtype(1, dtype::Float32{})
                              .set_times(RUN)
                              .set_param(param)
                              .exec(shapes);
        size_t IC = shapes[1][1];
        size_t FH = shapes[1][2];
        size_t FW = shapes[1][3];
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = param;
        opr->deduce_layout({shapes[0], dtype::Float32()},
                           {shapes[1], dtype::Float32()}, dst_layout);
        printf("fp32 flops: %.3f mflops\n",
               (IC * dst_layout.total_nr_elems() * FH * FW * 2) /
                       (tfloat / RUN * 1000));
    };
    auto profile = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                       size_t stride) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = kernel / 2;
        param.pad_w = kernel / 2;
        param.pad_h = 0;
        param.pad_w = 0;
        printf("oc: %zd ic: %zd w: %zd h: %zd stride: %zd kernel_size: %zd\n",
               oc, ic, w, h, stride, kernel);

        run({{1, ic, h, w}, {oc, ic, kernel, kernel}, {}}, param);
    };

    profile(48, 128, 56, 88, 1, 1);
    profile(56, 128, 64, 80, 1, 1);
    profile(24, 3, 256, 320, 3, 2);
    profile(16, 3, 224, 352, 5, 2);
    profile(16, 3, 256, 320, 7, 2);
    profile(8, 8, 56, 88, 3, 1);
    profile(8, 8, 7, 11, 3, 1);
    profile(4, 4, 64, 80, 3, 1);
    profile(108, 108, 7, 7, 3, 1);
    profile(54, 54, 7, 7, 3, 1);
    profile(3, 3, 128, 128, 3, 1);
    profile(3, 3, 112, 112, 3, 1);
}

TEST_F(FALLBACK, BENCHMARK_CONVOLUTION_MATRIX_MUL_8832) {
    using Param = Convolution::Param;
    auto run = [&](const TensorShapeArray& shapes, Param param) {
        Benchmarker<Convolution> benchmarker_float(handle());
        size_t RUN = 50;
        auto tfloat = benchmarker_float.set_display(false)
                              .set_dtype(0, dtype::Int8{})
                              .set_dtype(1, dtype::Int8{})
                              .set_dtype(2, dtype::Int32{})
                              .set_times(RUN)
                              .set_param(param)
                              .exec(shapes);
        size_t IC = shapes[1][1];
        size_t FH = shapes[1][2];
        size_t FW = shapes[1][3];
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = param;
        opr->deduce_layout({shapes[0], dtype::Float32()},
                           {shapes[1], dtype::Float32()}, dst_layout);
        printf("fp32 flops: %.3f mflops\n",
               (IC * dst_layout.total_nr_elems() * FH * FW * 2) /
                       (tfloat / RUN * 1000));
    };
    auto profile = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                       size_t stride) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = kernel / 2;
        param.pad_w = kernel / 2;
        param.pad_h = 0;
        param.pad_w = 0;
        printf("oc: %zd ic: %zd w: %zd h: %zd stride: %zd kernel_size: %zd\n",
               oc, ic, w, h, stride, kernel);

        run({{1, ic, h, w}, {oc, ic, kernel, kernel}, {}}, param);
    };

    profile(48, 128, 56, 88, 1, 1);
    profile(56, 128, 64, 80, 3, 1);
    profile(24, 3, 256, 320, 3, 2);
}

TEST_F(FALLBACK, BENCHMARK_CONVOLUTION_MATRIX_MUL_8816) {
    using Param = Convolution::Param;
    auto run = [&](const TensorShapeArray& shapes, Param param) {
        Benchmarker<Convolution> benchmarker_float(handle());
        size_t RUN = 50;
        auto tfloat = benchmarker_float.set_display(false)
                              .set_dtype(0, dtype::Int8{})
                              .set_dtype(1, dtype::Int8{})
                              .set_dtype(2, dtype::Int16{})
                              .set_times(RUN)
                              .set_param(param)
                              .exec(shapes);
        size_t IC = shapes[1][1];
        size_t FH = shapes[1][2];
        size_t FW = shapes[1][3];
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = param;
        opr->deduce_layout({shapes[0], dtype::Float32()},
                           {shapes[1], dtype::Float32()}, dst_layout);
        printf("fp32 flops: %.3f mflops\n",
               (IC * dst_layout.total_nr_elems() * FH * FW * 2) /
                       (tfloat / RUN * 1000));
    };
    auto profile = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                       size_t stride) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = kernel / 2;
        param.pad_w = kernel / 2;
        param.pad_h = 0;
        param.pad_w = 0;
        printf("oc: %zd ic: %zd w: %zd h: %zd stride: %zd kernel_size: %zd\n",
               oc, ic, w, h, stride, kernel);

        run({{1, ic, h, w}, {oc, ic, kernel, kernel}, {}}, param);
    };

    profile(48, 128, 56, 88, 1, 1);
    profile(48, 128, 56, 88, 1, 2);
    profile(56, 128, 64, 80, 3, 1);
    profile(24, 3, 256, 320, 3, 2);
}

TEST_F(FALLBACK, BENCHMARK_CONVOLUTION_BACKWARD_DATA) {
    using Param = ConvolutionBackwardData::Param;
    auto run = [&](const TensorLayoutArray& tensors, Param param) {
        Benchmarker<ConvolutionBackwardData> benchmarker_fallback(handle());
        size_t RUN = 500;
        benchmarker_fallback.set_display(false)
                .set_dtype(0, dtype::Float32{})
                .set_dtype(1, dtype::Float32{})
                .set_times(RUN)
                .set_param(param);
        auto tmatmul = benchmarker_fallback
                               .set_before_exec_callback(
                                       AlgoChecker<ConvolutionBackwardData>(
                                               "DeconvMatmul"))
                               .exec(tensors);
        auto tdirect = benchmarker_fallback
                               .set_before_exec_callback(
                                       AlgoChecker<ConvolutionBackwardData>(
                                               "DeconvDirect"))
                               .exec(tensors);
        size_t IC = tensors[0][1];
        size_t FH = tensors[0][2];
        size_t FW = tensors[0][3];
        size_t total_flops = IC * tensors[1].total_nr_elems() * FH * FW * 2;
        printf("Direct_time: %.3f ms  Direct_flops: %.3f mflops\n", tdirect,
               total_flops / (tdirect / RUN * 1000));
        printf("Matmul_time: %.3f ms  Matmul_flops: %.3f mflops\n", tmatmul,
               total_flops / (tmatmul / RUN * 1000));
        printf("speedup: %.3f\n", tdirect / tmatmul);
    };

    auto profile = [&](size_t n, size_t ic, size_t oh, size_t ow, size_t oc,
                       size_t fh, size_t fw, size_t stride = 1,
                       size_t padding = 0) {
        Param param;
        param.pad_h = param.pad_w = padding;
        param.stride_h = param.stride_w = stride;
        printf("oc: %zd ic: %zd w: %zd h: %zd stride: %zd kernel_size: %zd\n",
               oc, ic, ow, oh, stride, fh);

        TensorLayout diff = TensorLayout{{n, oc, oh, ow}, dtype::Float32()};
        TensorLayout filter = TensorLayout{{oc, ic, fh, fw}, dtype::Float32()};
        TensorLayout grad;
        {
            auto opr = handle()->create_operator<ConvolutionBackwardData>();
            opr->param() = param;
            opr->deduce_layout(filter, diff, grad);
        }
        run(TensorLayoutArray{filter, diff, grad}, param);
    };

    profile(1, 1, 3, 3, 1, 2, 2);
    profile(1, 2, 3, 3, 2, 2, 2);
    profile(1, 4, 3, 3, 4, 2, 2);
    profile(1, 4, 3, 3, 8, 2, 2);
    profile(1, 8, 3, 3, 4, 2, 2);
    profile(1, 8, 3, 3, 8, 2, 2);
}

#endif

TEST_F(FALLBACK, CONVOLUTION_MATRIX_MUL) {
    Checker<Convolution> checker(handle());
    using Param = Convolution::Param;

    Param param;
    param.sparse = param::Convolution::Sparse::DENSE;
    auto run = [&](size_t n, size_t ic, size_t ih, size_t iw, size_t oc,
                   size_t fh, size_t fw) {
        param.pad_h = param.pad_w = 1;
        param.stride_h = param.stride_w = 1;
        checker.set_param(param);
        checker.execs({{n, ic, ih, iw}, {oc, ic, fh, fw}, {}});
    };

    run(1, 3, 128, 128, 5, 3, 3);
    run(1, 56, 128, 64, 80, 1, 1);
    run(1, 8, 8, 7, 11, 3, 1);
    run(1, 54, 54, 7, 7, 3, 1);
    run(1, 3, 3, 128, 128, 3, 1);
    run(1, 3, 3, 112, 112, 3, 1);
    run(1, 1, 1, 1, 1, 3, 3);
}

#if MEGDNN_X86
TEST_F(FALLBACK_MULTI_THREADS, CONVOLUTION_8816) {
    Checker<Convolution> checker(handle());
    using Param = Convolution::Param;
    checker.set_before_exec_callback(AlgoChecker<Convolution>(".+FB_GEMV.+"));
    auto run = [&](size_t n, size_t ic, size_t ih, size_t iw, size_t oc,
                   size_t fh, size_t fw, size_t pad, size_t stride,
                   size_t group) {
        Param param;
        param.sparse = group > 1 ? param::Convolution::Sparse::GROUP
                                 : param::Convolution::Sparse::DENSE;
        param.pad_h = param.pad_w = pad;
        param.stride_h = param.stride_w = stride;
        checker.set_param(param);
        if (group > 1) {
            checker.execl(
                    {{{n, ic, ih, iw}, dtype::Int8()},
                     {{group, oc / group, ic / group, fh, fw}, dtype::Int8()},
                     {{}, dtype::Int16()}});
        } else {
            checker.execl({{{n, ic, ih, iw}, dtype::Int8()},
                           {{oc, ic, fh, fw}, dtype::Int8()},
                           {{}, dtype::Int16()}});
        }
    };

    for (auto n : {1, 2})
        for (auto ic : {3, 4, 8, 12, 16})
            for (auto oc : {4, 8, 16, 32})
                for (auto ih : {7, 14, 15, 22})
                    for (auto iw : {7, 13, 11, 32})
                        for (auto filter : {1, 2, 3, 5, 7})
                            for (auto stride : {1, 2})
                                for (auto pad : {0, filter / 2}) {
                                    run(n, ic, ih, iw, oc, filter, filter, pad,
                                        stride, 1);
                                    if (ic == oc) {
                                        run(n, ic, ih, iw, oc, filter, filter,
                                            pad, stride, ic);
                                    }
                                }
}
#endif

TEST_F(FALLBACK, CONVOLUTION_NAIVE_ALGO_FP16) {
    Checker<Convolution> checker(handle());
    using Param = Convolution::Param;
    checker.set_before_exec_callback(
            AlgoChecker<ConvolutionForward>("NAIVE_ALGO"));
    Param param;
    param.sparse = param::Convolution::Sparse::DENSE;
    auto run = [&](size_t n, size_t ic, size_t ih, size_t iw, size_t oc,
                   size_t fh, size_t fw) {
        param.pad_h = param.pad_w = 1;
        param.stride_h = param.stride_w = 1;
        for (auto cmode :
             std::vector<Param::ComputeMode>{Param::ComputeMode::DEFAULT,
                                             Param::ComputeMode::FLOAT32}) {
            param.compute_mode = cmode;
            checker.set_param(param)
                    .set_dtype(0, dtype::Float16())
                    .set_dtype(1, dtype::Float16())
                    // Use inferred output dtype.
                    .set_dtype(2, {});
            checker.execs({{n, ic, ih, iw}, {oc, ic, fh, fw}, {}});
        }
    };

    run(1, 3, 128, 128, 5, 3, 3);
    run(1, 8, 8, 7, 11, 3, 1);
}

TEST_F(FALLBACK_MULTI_THREADS, CONVOLUTION_NAIVE_FALLBACK) {
    Checker<Convolution> checker(handle());
    using Param = Convolution::Param;
    checker.set_before_exec_callback(
            AlgoChecker<ConvolutionForward>("FALLBACK_ALGO"));
    Param param;
    auto run = [&](size_t n, size_t group, size_t ic, size_t ih, size_t iw,
                   size_t oc, size_t fh, size_t fw) {
        param.sparse = param::Convolution::Sparse::GROUP;
        param.pad_h = param.pad_w = 1;
        param.stride_h = param.stride_w = 1;
        TensorShape src{n, ic, ih, iw},
                filter{group, oc / group, ic / group, fh, fw};
        checker.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, {});
        checker.execs({src, filter, {}});
    };
    run(4, 1, 3, 21, 15, 5, 3, 3);
    run(1, 8, 56, 24, 31, 56, 1, 1);
    run(4, 8, 8, 8, 7, 8, 3, 1);
    run(8, 1, 54, 54, 7, 7, 3, 1);
    run(100, 1, 1, 1, 1, 1, 3, 3);
}

TEST_F(FALLBACK_MULTI_THREADS, CONVOLUTION_NAIVE_ALGO) {
    Checker<Convolution> checker(handle());
    using Param = Convolution::Param;
    checker.set_before_exec_callback(
            AlgoChecker<ConvolutionForward>("NAIVE_ALGO"));
    Param param;
    auto run = [&](size_t n, size_t group, size_t ic, size_t ih, size_t iw,
                   size_t oc, size_t fh, size_t fw) {
        param.sparse = param::Convolution::Sparse::GROUP;
        param.pad_h = param.pad_w = 1;
        param.stride_h = param.stride_w = 1;
        TensorShape src{n, ic, ih, iw},
                filter{group, oc / group, ic / group, fh, fw};
        checker.set_param(param).set_dtype(2, {});
        //! float32
        checker.set_dtype(0, dtype::Float32()).set_dtype(1, dtype::Float32());
        checker.execs({src, filter, {}});
        //! float16
        checker.set_dtype(0, dtype::Float16()).set_dtype(1, dtype::Float16());
        checker.execs({src, filter, {}});
        //! Qint8
        checker.set_dtype(0, dtype::QuantizedS8(3.34f))
                .set_dtype(1, dtype::QuantizedS8(0.32f));
        checker.execs({src, filter, {}});
        //! Quint8
        checker.set_dtype(0, dtype::Quantized8Asymm(3.34f,
                                                    static_cast<uint8_t>(21)))
                .set_dtype(1, dtype::Quantized8Asymm(0.32f,
                                                     static_cast<uint8_t>(15)));
        checker.execs({src, filter, {}});
    };
    run(4, 1, 3, 21, 15, 5, 3, 3);
    run(1, 8, 56, 24, 31, 56, 1, 1);
    run(4, 8, 8, 8, 7, 8, 3, 1);
    run(8, 1, 54, 54, 7, 7, 3, 1);
    run(100, 1, 1, 1, 1, 1, 3, 3);
}

TEST_F(FALLBACK, CONVOLUTION_MATRIX_MUL_SINT8) {
    Checker<Convolution> checker(handle());
    using Param = Convolution::Param;

    Param param;
    param.sparse = param::Convolution::Sparse::DENSE;
    auto run = [&](size_t n, size_t ic, size_t ih, size_t iw, size_t oc,
                   size_t fh, size_t fw) {
        param.pad_h = param.pad_w = 1;
        param.stride_h = param.stride_w = 1;
        checker.set_param(param)
                .set_dtype(0, dtype::QuantizedS8(0.2f))
                .set_dtype(1, dtype::QuantizedS8(0.2f))
                // Use inferred output dtype.
                .set_dtype(2, {});
        checker.execs({{n, ic, ih, iw}, {oc, ic, fh, fw}, {}});
    };

    run(1, 3, 128, 128, 5, 3, 3);
    run(1, 56, 128, 64, 80, 1, 1);
    run(1, 8, 8, 7, 11, 3, 1);
    run(1, 54, 54, 7, 7, 3, 1);
    run(1, 3, 3, 128, 128, 3, 1);
    run(1, 3, 3, 112, 112, 3, 1);
    run(1, 1, 1, 1, 1, 3, 3);
}

TEST_F(FALLBACK, CONVOLUTION_BACKWARD_DATA) {
    Checker<ConvolutionBackwardData> checker(handle());
    using Param = ConvolutionBackwardData::Param;

    Param param;
    auto run = [&](size_t n, size_t ic, size_t oh, size_t ow, size_t oc,
                   size_t fh, size_t fw, size_t stride, size_t padding,
                   size_t dilate = 1, size_t group = 1) {
        param.pad_h = param.pad_w = padding;
        param.stride_h = param.stride_w = stride;
        param.dilate_h = param.dilate_w = dilate;

        TensorLayout diff =
                TensorLayout{{n, oc * group, oh, ow}, dtype::Float32()};
        TensorLayout grad;
        TensorLayout filter;
        if (group == 1) {
            param.sparse = Param::Sparse::DENSE;
            filter = {{oc, ic, fh, fw}, dtype::Float32()};
        } else {
            param.sparse = Param::Sparse::GROUP;
            filter = {{group, oc, ic, fh, fw}, dtype::Float32()};
        }
        // TensorLayout grad;
        {
            auto opr = handle()->create_operator<ConvolutionBackwardData>();
            opr->param() = param;
            opr->deduce_layout(filter, diff, grad);
        }
        checker.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32());
        checker.exec(TensorLayoutArray{filter, diff, grad});
    };

    for (auto mode :
         {Param::Mode::CONVOLUTION, Param::Mode::CROSS_CORRELATION}) {
        param.mode = mode;
        run(4, 3, 10, 13, 5, 1, 1, 1, 0, 1, 1);
        run(5, 5, 24, 43, 11, 9, 3, 3, 12, 1, 2);
        run(4, 3, 10, 45, 2, 1, 1, 1, 0, 4, 3);
        run(2, 3, 9, 12, 2, 4, 6, 1, 0, 1, 2);
        run(3, 4, 17, 32, 2, 3, 2, 5, 4, 4, 3);
        run(5, 5, 24, 43, 11, 9, 3, 3, 12, 2, 2);
        run(2, 3, 20, 33, 3, 5, 7, 4, 15, 2, 3);
        run(4, 4, 6, 7, 9, 3, 2, 2, 1, 3, 2);
    }
}

TEST_F(FALLBACK, CONVOLUTION_BACKWARD_DATA_INT8_INT8_INT32) {
    Checker<ConvolutionBackwardData> checker(handle());
    using Param = ConvolutionBackwardData::Param;
    Param param;

    auto run = [&](size_t n, size_t ic, size_t oh, size_t ow, size_t oc,
                   size_t fh, size_t fw, size_t stride, size_t padding,
                   size_t dilate = 1, size_t group = 1) {
        param.pad_h = param.pad_w = padding;
        param.stride_h = param.stride_w = stride;
        param.dilate_h = param.dilate_w = dilate;

        TensorLayout diff =
                TensorLayout{{n, oc * group, oh, ow}, dtype::Int8()};
        TensorLayout grad;
        TensorLayout filter;
        if (group == 1) {
            param.sparse = Param::Sparse::DENSE;
            filter = {{oc, ic, fh, fw}, dtype::Int8()};
        } else {
            param.sparse = Param::Sparse::GROUP;
            filter = {{group, oc, ic, fh, fw}, dtype::Int8()};
        }
        // TensorLayout grad;
        {
            auto opr = handle()->create_operator<ConvolutionBackwardData>();
            opr->param() = param;
            opr->deduce_layout(filter, diff, grad);
        }
        checker.set_param(param)
                .set_dtype(0, dtype::Int8())
                .set_dtype(1, dtype::Int8())
                .set_dtype(2, dtype::Int32());
        checker.exec(TensorLayoutArray{filter, diff, grad});
    };

    for (auto mode :
         {Param::Mode::CONVOLUTION, Param::Mode::CROSS_CORRELATION}) {
        param.mode = mode;
        run(4, 3, 10, 13, 5, 1, 1, 1, 0, 1, 1);
        run(5, 5, 24, 43, 11, 9, 3, 3, 12, 1, 2);
        run(4, 3, 10, 45, 2, 1, 1, 1, 0, 4, 3);
        run(2, 3, 9, 12, 2, 4, 6, 1, 0, 1, 2);
        run(3, 4, 17, 32, 2, 3, 2, 5, 4, 4, 3);
        run(5, 5, 24, 43, 11, 9, 3, 3, 12, 2, 2);
        run(2, 3, 20, 33, 3, 5, 7, 4, 15, 2, 3);
        run(4, 4, 6, 7, 9, 3, 2, 2, 1, 3, 2);
    }
}

TEST_F(FALLBACK, CONVOLUTION_BACKWARD_DATA_SINT8) {
    Checker<ConvolutionBackwardData> checker(handle());
    using Param = ConvolutionBackwardData::Param;
    Param param;

    auto run = [&](size_t n, size_t ic, size_t oh, size_t ow, size_t oc,
                   size_t fh, size_t fw, size_t stride, size_t padding,
                   size_t dilate = 1, size_t group = 1) {
        param.pad_h = param.pad_w = padding;
        param.stride_h = param.stride_w = stride;
        param.dilate_h = param.dilate_w = dilate;

        TensorLayout diff =
                TensorLayout{{n, oc * group, oh, ow}, dtype::QuantizedS8(0.2f)};
        TensorLayout grad;
        TensorLayout filter;
        if (group == 1) {
            param.sparse = Param::Sparse::DENSE;
            filter = {{oc, ic, fh, fw}, dtype::QuantizedS8(0.2f)};
        } else {
            param.sparse = Param::Sparse::GROUP;
            filter = {{group, oc, ic, fh, fw}, dtype::QuantizedS8(0.2f)};
        }
        // TensorLayout grad;
        {
            auto opr = handle()->create_operator<ConvolutionBackwardData>();
            opr->param() = param;
            opr->deduce_layout(filter, diff, grad);
        }
        checker.set_param(param)
                .set_dtype(0, dtype::QuantizedS8(0.2f))
                .set_dtype(1, dtype::QuantizedS8(0.2f))
                .set_dtype(2, {});
        checker.exec(TensorLayoutArray{filter, diff, grad});
    };

    for (auto mode :
         {Param::Mode::CONVOLUTION, Param::Mode::CROSS_CORRELATION}) {
        param.mode = mode;
        run(4, 3, 10, 13, 5, 1, 1, 1, 0, 1, 1);
        run(5, 5, 24, 43, 11, 9, 3, 3, 12, 1, 2);
        run(4, 3, 10, 45, 2, 1, 1, 1, 0, 4, 3);
        run(2, 3, 9, 12, 2, 4, 6, 1, 0, 1, 2);
        run(3, 4, 17, 32, 2, 3, 2, 5, 4, 4, 3);
        run(5, 5, 24, 43, 11, 9, 3, 3, 12, 2, 2);
        run(2, 3, 20, 33, 3, 5, 7, 4, 15, 2, 3);
        run(4, 4, 6, 7, 9, 3, 2, 2, 1, 3, 2);
    }
}

TEST_F(FALLBACK, CONVOLUTION_BACKWARD_DATA_QUINT8) {
    Checker<ConvolutionBackwardData> checker(handle());
    using Param = ConvolutionBackwardData::Param;
    Param param;

    auto run = [&](size_t n, size_t ic, size_t oh, size_t ow, size_t oc,
                   size_t fh, size_t fw, size_t stride, size_t padding,
                   size_t dilate = 1, size_t group = 1) {
        param.pad_h = param.pad_w = padding;
        param.stride_h = param.stride_w = stride;
        param.dilate_h = param.dilate_w = dilate;

        TensorLayout diff =
                TensorLayout{{n, oc * group, oh, ow},
                             dtype::Quantized8Asymm(1.3f, (uint8_t)129)};
        TensorLayout grad;
        TensorLayout filter;
        if (group == 1) {
            param.sparse = Param::Sparse::DENSE;
            filter = {{oc, ic, fh, fw},
                      dtype::Quantized8Asymm(1.2f, (uint8_t)127)};
        } else {
            param.sparse = Param::Sparse::GROUP;
            filter = {{group, oc, ic, fh, fw},
                      dtype::Quantized8Asymm(1.2f, (uint8_t)127)};
        }
        // TensorLayout grad;
        {
            auto opr = handle()->create_operator<ConvolutionBackwardData>();
            opr->param() = param;
            opr->deduce_layout(filter, diff, grad);
        }
        NormalRNG rng(128.f);
        checker.set_param(param)
                .set_dtype(0, dtype::Quantized8Asymm(1.2f, (uint8_t)127))
                .set_dtype(1, dtype::Quantized8Asymm(1.3f, (uint8_t)129))
                .set_dtype(2, {});
        checker.set_rng(0, &rng).set_rng(1, &rng);
        checker.exec(TensorLayoutArray{filter, diff, grad});
    };

    for (auto mode :
         {Param::Mode::CONVOLUTION, Param::Mode::CROSS_CORRELATION}) {
        param.mode = mode;
        run(4, 3, 10, 13, 5, 1, 1, 1, 0, 1, 1);
        run(5, 5, 24, 43, 11, 9, 3, 3, 12, 1, 2);
        run(4, 3, 10, 45, 2, 1, 1, 1, 0, 4, 3);
        run(2, 3, 9, 12, 2, 4, 6, 1, 0, 1, 2);
        run(3, 4, 17, 32, 2, 3, 2, 5, 4, 4, 3);
        run(5, 5, 24, 43, 11, 9, 3, 3, 12, 2, 2);
        run(2, 3, 20, 33, 3, 5, 7, 4, 15, 2, 3);
        run(4, 4, 6, 7, 9, 3, 2, 2, 1, 3, 2);
    }
}

TEST_F(FALLBACK, CONVOLUTION_BACKWARD_DATA_NAIVE_ALGO) {
    Checker<ConvolutionBackwardData> checker(handle());
    checker.set_before_exec_callback(
            AlgoChecker<ConvolutionBackwardData>("DeconvNaive"));
    using Param = ConvolutionBackwardData::Param;
    Param param;

    auto run = [&](size_t n, size_t ic, size_t oh, size_t ow, size_t oc,
                   size_t fh, size_t fw, size_t stride, size_t padding,
                   size_t dilate = 1, size_t group = 1) {
        param.pad_h = param.pad_w = padding;
        param.stride_h = param.stride_w = stride;
        param.dilate_h = param.dilate_w = dilate;

        TensorLayout diff =
                TensorLayout{{n, oc * group, oh, ow}, dtype::Float32()};
        TensorLayout grad;
        TensorLayout filter;
        if (group == 1) {
            param.sparse = Param::Sparse::DENSE;
            filter = {{oc, ic, fh, fw}, dtype::Float32()};
        } else {
            param.sparse = Param::Sparse::GROUP;
            filter = {{group, oc, ic, fh, fw}, dtype::Float32()};
        }
        // TensorLayout grad;
        {
            auto opr = handle()->create_operator<ConvolutionBackwardData>();
            opr->param() = param;
            opr->deduce_layout(filter, diff, grad);
        }
        checker.set_param(param);
        checker.exec(TensorLayoutArray{filter, diff, grad});
    };

    for (auto mode :
         {Param::Mode::CONVOLUTION, Param::Mode::CROSS_CORRELATION}) {
        param.mode = mode;
        run(4, 3, 10, 13, 5, 1, 1, 1, 0, 1, 1);
        run(5, 5, 24, 43, 11, 9, 3, 3, 12, 1, 2);
        run(4, 3, 10, 45, 2, 1, 1, 1, 0, 4, 3);
        run(2, 3, 9, 12, 2, 4, 6, 1, 0, 1, 2);
        run(3, 4, 17, 32, 2, 3, 2, 5, 4, 4, 3);
        run(5, 5, 24, 43, 11, 9, 3, 3, 12, 2, 2);
        run(2, 3, 20, 33, 3, 5, 7, 4, 15, 2, 3);
        run(4, 4, 6, 7, 9, 3, 2, 2, 1, 3, 2);
    }
}

// vim: syntax=cpp.doxygen
