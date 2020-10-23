/**
 * \file dnn/test/cuda/dct.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "megdnn/oprs/nn.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/dct_ref.h"
#include "test/common/rng.h"
#include "test/cuda/fixture.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, DCT) {
    DctChannelSelectForward::Param param;
    Checker<DctChannelSelectForward> checker(handle_cuda());
    for (size_t n : {1, 3}) {
        for (size_t ic : {1, 3}) {
            for (size_t ih : {8, 16, 32, 512, 1024}) {
                for (size_t iw : {8, 16, 32, 64, 128, 256, 512, 1024}) {
                    checker.set_param(param)
                            .set_dtype(0, dtype::Uint8())
                            .set_dtype(1, dtype::Int32())
                            .set_dtype(2, dtype::Int32())
                            .execs({TensorShape{n, ic, ih, iw}, {}, {}, {}});
                }
            }
        }
    }
}

TEST_F(CUDA, DCT_QINT8) {
    DctChannelSelectForward::Param param;
    Checker<DctChannelSelectForward> checker(handle_cuda());
    param.format = Param::Format::NCHW4;
    for (size_t n : {1, 3}) {
        for (size_t ic : {1, 3}) {
            for (size_t ih : {8, 16, 32, 512, 1024}) {
                for (size_t iw : {8, 16, 32, 64, 128, 256, 512, 1024}) {
                    checker.set_param(param)
                            .set_dtype(0, dtype::Uint8())
                            .set_dtype(1, dtype::Int32())
                            .set_dtype(2, dtype::Int32())
                            .set_dtype(3, dtype::QuantizedS8(10.f))
                            .set_epsilon(1)
                            .execs({TensorShape{n, ic, ih, iw}, {}, {}, {}});
                }
            }
        }
    }
}

TEST_F(CUDA, DCT_WITH_FIX_32_MASK) {
    using Param = DctChannelSelectForward::Param;
    Param param;
    Checker<DctChannelSelectForward> checker(handle_cuda(), false);
    param.fastImpl = Param::FastImpl::FIX_32_MASK;
    auto test_case = gen_dct_case(3, 3, 1024, 768, 32, param);
    checker.set_param(param).exect(test_case->testcase_in,
                                   test_case->testcase_out);
}

TEST_F(CUDA, DCT_WITH_FIX_32_MASK_QINT8) {
    using Param = DctChannelSelectForward::Param;
    Param param;
    Checker<DctChannelSelectForward> checker(handle_cuda(), false);
    param.fastImpl = Param::FastImpl::FIX_32_MASK;
    param.format = Param::Format::NCHW4;
    auto test_case =
            gen_dct_case(3, 3, 1024, 768, 32, param, dtype::QuantizedS8(10.f));
    checker.set_param(param).set_epsilon(1).exect(test_case->testcase_in,
                                                  test_case->testcase_out);
}

TEST_F(CUDA, DCT_WITH_MASK) {
    Checker<DctChannelSelectForward> checker(handle_cuda(), false);
    DctChannelSelectForward::Param param;
    checker.set_param(param).exect(
            Testcase{TensorValue(
                             {1, 3, 8, 16}, dtype::Uint8(),
                             {109, 39,  30,  115, 71,  15,  206, 139, 221, 5,
                              18,  16,  93,  185, 99,  102, 205, 172, 191, 29,
                              185, 6,   47,  84,  0,   47,  105, 203, 251, 73,
                              196, 83,  3,   211, 32,  181, 49,  111, 114, 83,
                              148, 232, 77,  17,  35,  2,   154, 100, 41,  135,
                              141, 206, 56,  91,  137, 199, 104, 192, 75,  122,
                              78,  65,  184, 69,  91,  82,  2,   172, 194, 240,
                              49,  145, 87,  210, 97,  190, 179, 93,  125, 105,
                              181, 207, 148, 178, 133, 53,  25,  198, 238, 151,
                              14,  120, 213, 195, 145, 20,  122, 107, 217, 185,
                              65,  5,   115, 110, 82,  206, 163, 86,  2,   2,
                              44,  125, 50,  38,  41,  106, 30,  5,   151, 243,
                              238, 181, 232, 191, 161, 57,  23,  204,

                              109, 39,  30,  115, 71,  15,  206, 139, 221, 5,
                              18,  16,  93,  185, 99,  102, 205, 172, 191, 29,
                              185, 6,   47,  84,  0,   47,  105, 203, 251, 73,
                              196, 83,  3,   211, 32,  181, 49,  111, 114, 83,
                              148, 232, 77,  17,  35,  2,   154, 100, 41,  135,
                              141, 206, 56,  91,  137, 199, 104, 192, 75,  122,
                              78,  65,  184, 69,  91,  82,  2,   172, 194, 240,
                              49,  145, 87,  210, 97,  190, 179, 93,  125, 105,
                              181, 207, 148, 178, 133, 53,  25,  198, 238, 151,
                              14,  120, 213, 195, 145, 20,  122, 107, 217, 185,
                              65,  5,   115, 110, 82,  206, 163, 86,  2,   2,
                              44,  125, 50,  38,  41,  106, 30,  5,   151, 243,
                              238, 181, 232, 191, 161, 57,  23,  204,

                              109, 39,  30,  115, 71,  15,  206, 139, 221, 5,
                              18,  16,  93,  185, 99,  102, 205, 172, 191, 29,
                              185, 6,   47,  84,  0,   47,  105, 203, 251, 73,
                              196, 83,  3,   211, 32,  181, 49,  111, 114, 83,
                              148, 232, 77,  17,  35,  2,   154, 100, 41,  135,
                              141, 206, 56,  91,  137, 199, 104, 192, 75,  122,
                              78,  65,  184, 69,  91,  82,  2,   172, 194, 240,
                              49,  145, 87,  210, 97,  190, 179, 93,  125, 105,
                              181, 207, 148, 178, 133, 53,  25,  198, 238, 151,
                              14,  120, 213, 195, 145, 20,  122, 107, 217, 185,
                              65,  5,   115, 110, 82,  206, 163, 86,  2,   2,
                              44,  125, 50,  38,  41,  106, 30,  5,   151, 243,
                              238, 181, 232, 191, 161, 57,  23,  204}),
                     TensorValue({4}, dtype::Int32(), {0, 14, 22, 30}),
                     TensorValue({30}, dtype::Int32(),
                                 {8,  16, 9, 2, 3, 10, 17, 24, 32, 25,
                                  18, 11, 4, 5, 0, 1,  8,  16, 9,  2,
                                  3,  10, 0, 1, 8, 16, 9,  2,  3,  10}),
                     {}},
            Testcase{{},
                     {},
                     {},
                     TensorValue({1, 30, 1, 2}, dtype::Float32(),
                                 {-22.850792, -97.862236,  -101.043236,
                                  -4.727012,  28.275675,   -157.96654,
                                  42.1377,    45.06531,    -149.77373,
                                  24.487143,  -8.054966,   -13.990831,
                                  -6.9395194, -3.9211385,  64.79172,
                                  -12.363858, -47.875,     59.,
                                  56.271786,  -62.725567,  120.522675,
                                  16.559765,  85.74334,    112.904495,
                                  99.375,     29.499973,   2.0220923,
                                  -19.681704, 890.12494,   941.25,
                                  -7.0498576, 99.47632,    -22.850792,
                                  -97.862236, -101.043236, -4.727012,
                                  28.275675,  -157.96654,  42.1377,
                                  45.06531,   -149.77373,  24.487143,
                                  -8.054966,  -13.990831,  890.12494,
                                  941.25,     -7.0498576,  99.47632,
                                  -22.850792, -97.862236,  -101.043236,
                                  -4.727012,  28.275675,   -157.96654,
                                  42.1377,    45.06531,    -149.77373,
                                  24.487143,  -8.054966,   -13.990831})});
}

TEST_F(CUDA, DCT_WITH_MASK2) {
    Checker<DctChannelSelectForward> checker(handle_cuda(), false);
    DctChannelSelectForward::Param param;
    UniformIntRNG rng_oc(0, 3 * 64);
    for (size_t n : {1, 3}) {
        for (size_t ic : {1, 3}) {
            for (size_t ih : {8, 16, 32, 512, 1024}) {
                for (size_t iw : {8, 16, 32, 64, 128, 256, 512, 1024}) {
                    int random_oc = static_cast<int>(rng_oc.gen_single_val());
                    int max_oc = ic * 64;
                    int mask_oc = (random_oc % max_oc) + 1;
                    auto test_case =
                            gen_dct_case(n, ic, ih, iw, mask_oc, param);
                    checker.set_param(param).exect(test_case->testcase_in,
                                                   test_case->testcase_out);
                }
            }
        }
    }
}

TEST_F(CUDA, DCT_WITH_MASK2_QINT8) {
    Checker<DctChannelSelectForward> checker(handle_cuda(), false);
    DctChannelSelectForward::Param param;
    param.format = DctChannelSelectForward::Param::Format::NCHW4;

    UniformIntRNG rng_oc(0, 3 * 64);
    for (size_t n : {1, 3}) {
        for (size_t ic : {1, 3}) {
            for (size_t ih : {8, 16, 32, 512, 1024}) {
                for (size_t iw : {8, 16, 32, 64, 128, 256, 512, 1024}) {
                    int random_oc = static_cast<int>(rng_oc.gen_single_val());
                    int max_oc = ic * 64;
                    int mask_oc = (random_oc % max_oc) + 1;
                    mask_oc = (mask_oc + 3) / 4 * 4;
                    auto test_case = gen_dct_case(n, ic, ih, iw, mask_oc, param,
                                                  dtype::QuantizedS8(10.f));
                    checker.set_param(param).set_epsilon(1).exect(
                            test_case->testcase_in, test_case->testcase_out);
                }
            }
        }
    }
}
TEST_F(CUDA, DCT_WITH_MASK2_QINT8_CONSTRAINT) {
    DctChannelSelectForward::Param param;
    param.format = DctChannelSelectForward::Param::Format::NCHW4;

    Checker<DctChannelSelectForward> checker(handle_cuda(), false);
    checker.set_param(param)
            .set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Int32())
            .set_dtype(2, dtype::Int32())
            .set_dtype(3, dtype::QuantizedS8(10.f))
            .set_epsilon(1);

    UniformIntRNG rng_oc(0, 3 * 64);
    for (size_t n : {1, 3}) {
        for (size_t ic : {1, 3}) {
            for (size_t ih : {8, 16, 32, 512, 1024}) {
                for (size_t iw : {8, 16, 32, 64, 128, 256, 512, 1024}) {
                    int random_oc = static_cast<int>(rng_oc.gen_single_val());
                    int max_oc = ic * 64;
                    int mask_oc = (random_oc % max_oc) + 1;
                    mask_oc = (mask_oc + 3) / 4 * 4;
                    if (mask_oc < max_oc) {
                        checker
                                .set_tensors_constraint(gen_dct_constriant(
                                        n, ic, ih, iw, mask_oc, param))
                                .exec({TensorShape{n, ic, ih, iw},
                                       TensorShape{ic + 1},
                                       TensorShape{(size_t)mask_oc},
                                       {}});
                    } else {
                        checker.set_tensors_constraint({}).exec(
                                {TensorShape{n, ic, ih, iw}, {}, {}, {}});
                    }
                }
            }
        }
    }
}

#if MEGDNN_WITH_BENCHMARK

TEST_F(CUDA, BENCHMARK_DCT) {
    using Param = DctChannelSelectForward::Param;

    auto run = [&](const TensorShapeArray& shapes, Param param) {
        Benchmarker<DctChannelSelectForward> benchmarker(handle_cuda());
        benchmarker.set_param(param);
        benchmarker.set_dtype(0, dtype::Uint8())
                .set_dtype(1, dtype::Int32())
                .set_dtype(2, dtype::Int32());
        for (auto&& shape : shapes) {
            double computation = double(shape[0]) * shape[1] * shape[2] *
                                 shape[3] * 32.0 * 1e-6;
            auto time_ms = benchmarker.execs({shape, {}, {}, {}});
            printf("execute %s, %.4f Gops\n", shape.to_string().c_str(),
                   computation / time_ms);
        }
    };

    auto run_case = [&](const DctTestcase& testcase, Param param,
                        std::string comment = "") {
        Benchmarker<DctChannelSelectForward> benchmarker(handle_cuda());
        benchmarker.set_param(param);
        benchmarker.set_dtype(0, dtype::Uint8())
                .set_dtype(1, dtype::Int32())
                .set_dtype(2, dtype::Int32())
                .set_dtype(3, testcase.testcase_out[3].layout.dtype);

        auto src_shape = testcase.testcase_in[0].layout;
        double computation = double(src_shape[0]) * src_shape[1] *
                             src_shape[2] * src_shape[3] * 32.0 * 1e-6;
        auto time_ms = benchmarker.exect(testcase.testcase_in);
        printf("[%s] execute %s, %.4f Gops\n", comment.c_str(),
               src_shape.to_string().c_str(), computation / time_ms);
    };

    auto run_case_constraint =
            [&](const Benchmarker<DctChannelSelectForward>::TensorsConstriant&
                        constraint,
                Param param, const TensorShapeArray& shapes,
                std::string comment = "", DType output_dtype) {
                Benchmarker<DctChannelSelectForward> benchmarker(handle_cuda());
                benchmarker.set_param(param)
                        .set_dtype(0, dtype::Uint8())
                        .set_dtype(1, dtype::Int32())
                        .set_dtype(2, dtype::Int32())
                        .set_dtype(3, output_dtype)
                        .set_tensors_constraint(constraint);

                auto src_shape = shapes[0];
                double computation = double(src_shape[0]) * src_shape[1] *
                                     src_shape[2] * src_shape[3] * 32.0 * 1e-6;
                auto time_ms = benchmarker.exec(shapes);
                printf("[%s] execute %s, %.4f Gops\n", comment.c_str(),
                       src_shape.to_string().c_str(), computation / time_ms);
            };

    TensorShapeArray shapes = {
            {1, 3, 512, 512},
            {8, 3, 2176, 3840},
    };
    {
        Param param;
        run(shapes, param);
    }

    Param fix_32_param;
    fix_32_param.fastImpl = Param::FastImpl::FIX_32_MASK;
    {
        auto test_case = gen_dct_case(8, 3, 2176, 3840, 32, fix_32_param);
        run_case(*test_case, fix_32_param, "FIX_32_MASK");
    }

    {
        Param param;
        auto test_case = gen_dct_case(8, 3, 2176, 3840, 32, fix_32_param);
        run_case(*test_case, param, "MASK 32");
    }

    {
        Param fix_32_nchw4_param;
        fix_32_nchw4_param.fastImpl = Param::FastImpl::FIX_32_MASK;
        fix_32_nchw4_param.format = Param::Format::NCHW4;
        auto test_case = gen_dct_case(8, 3, 2176, 3840, 32, fix_32_nchw4_param,
                                      dtype::QuantizedS8(10.f));
        run_case(*test_case, fix_32_nchw4_param, "FIX_32_MASK QINT8");
    }

    {
        Param fix_32_nchw4_param;
        fix_32_nchw4_param.fastImpl = Param::FastImpl::FIX_32_MASK;
        fix_32_nchw4_param.format = Param::Format::NCHW4;
        auto test_case = gen_dct_case(8, 3, 2176, 3840, 32, fix_32_nchw4_param,
                                      dtype::QuantizedS8(10.f));
        fix_32_nchw4_param.fastImpl = Param::FastImpl::NONE;
        run_case(*test_case, fix_32_nchw4_param, "MASK 32 QINT8");
    }

    {
        Param fix_32_nchw4_param;
        fix_32_nchw4_param.fastImpl = Param::FastImpl::FIX_32_MASK;
        fix_32_nchw4_param.format = Param::Format::NCHW4;
        TensorShapeArray shapes = {{8, 3, 2176, 3840}, {4}, {32}, {}};
        auto constraint =
                gen_dct_constriant(8, 3, 2176, 3840, 32, fix_32_nchw4_param);
        run_case_constraint(constraint, fix_32_nchw4_param, shapes,
                            "FIX_32_MASK QINT8 Constraint",
                            dtype::QuantizedS8(10.f));
    }
}
#endif

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
