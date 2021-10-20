/**
 * \file dnn/test/cuda/relayout_format.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(CUDA, RELAYOUT_FORMAT) {
    Checker<RelayoutFormat> checker(handle_cuda());
    UniformIntRNG rng{-50, 50};
    param::RelayoutFormat param;
    param.mode = param::RelayoutFormat::Mode::NCHW4_CHWN4;

    checker.set_dtype(0, dtype::QuantizedS8{0.1f})
            .set_dtype(1, dtype::QuantizedS8{0.1f})
            .set_rng(0, &rng)
            .set_param(param)
            .execs({{22, 23, 24, 25, 4}, {}});
    param.mode = param::RelayoutFormat::Mode::CHWN4_NCHW4;
    checker.execs({{22, 23, 24, 25, 4}, {}});
}

TEST_F(CUDA, RELAYOUT_FORMAT_NCHW4_NCHW) {
    Checker<RelayoutFormat> checker(handle_cuda());
    UniformIntRNG rng{-50, 50};
    UniformIntRNG u8_rng{0, 255};
    param::RelayoutFormat param;
    param.mode = param::RelayoutFormat::Mode::NCHW4_NCHW;

    checker.set_dtype(0, dtype::QuantizedS8{0.1f})
            .set_dtype(1, dtype::QuantizedS8{0.1f})
            .set_rng(0, &rng)
            .set_param(param)
            .execs({{1, 1, 2, 2, 4}, {}});

    checker.set_dtype(0, dtype::Quantized8Asymm{1.f, 128})
            .set_dtype(1, dtype::Quantized8Asymm{1.f, 128})
            .set_rng(0, &u8_rng)
            .set_param(param)
            .execs({{1, 1, 2, 2, 4}, {}});

    checker.set_dtype(0, dtype::QuantizedS8{0.1f})
            .set_dtype(1, dtype::QuantizedS8{0.1f})
            .set_rng(0, &rng)
            .set_param(param)
            .execs({{22, 23, 24, 25, 4}, {}});

    param.oc = 90;
    checker.set_dtype(0, dtype::QuantizedS8{0.1f})
            .set_dtype(1, dtype::QuantizedS8{0.1f})
            .set_rng(0, &rng)
            .set_param(param)
            .execs({{22, 23, 24, 25, 4}, {}});

    param.oc = 16;
    param.group = 8;
    checker.set_dtype(0, dtype::QuantizedS8{0.1f})
            .set_dtype(1, dtype::QuantizedS8{0.1f})
            .set_rng(0, &rng)
            .set_param(param)
            .execs({{11, 16, 22, 33, 4}, {}});
}

TEST_F(CUDA, RELAYOUT_FORMAT_NCHW_NCHW4) {
    Checker<RelayoutFormat> checker(handle_cuda());
    UniformIntRNG rng{-50, 50};
    param::RelayoutFormat param;
    param.mode = param::RelayoutFormat::Mode::NCHW_NCHW4;

    for (size_t n : {1, 3}) {
        for (size_t c : {1, 2, 3, 4, 8, 9, 11, 16}) {
            for (size_t h : {3, 7, 12, 16, 22, 59, 83}) {
                for (size_t w : {3, 22, 63, 128, 256}) {
                    checker.set_dtype(0, dtype::QuantizedS8{1.f})
                            .set_dtype(1, dtype::QuantizedS8{1.f})
                            .set_rng(0, &rng)
                            .set_param(param)
                            .execs({{n, c, h, w}, {}});

                    checker.set_dtype(0, dtype::QuantizedS8{1.f})
                            .set_dtype(1, dtype::QuantizedS8{2.f})
                            .set_rng(0, &rng)
                            .set_param(param)
                            .execs({{n, c, h, w}, {}});

                    checker.set_dtype(0, dtype::QuantizedS32{1.f})
                            .set_dtype(1, dtype::QuantizedS32{1.f})
                            .set_rng(0, &rng)
                            .set_param(param)
                            .execs({{n, c, h, w}, {}});
                }
            }
        }
    }

    checker.set_dtype(0, dtype::QuantizedS8{1.f})
            .set_dtype(1, dtype::QuantizedS8{1.f})
            .set_rng(0, &rng)
            .set_param(param)
            .execs({{8, 3, 224, 224}, {}});

    checker.set_dtype(0, dtype::QuantizedS8{1.f})
            .set_dtype(1, dtype::QuantizedS8{1.f})
            .set_rng(0, &rng)
            .set_param(param)
            .execs({{8, 3, 600, 600}, {}});

    checker.set_dtype(0, dtype::QuantizedS8{1.f})
            .set_dtype(1, dtype::QuantizedS8{1.f})
            .set_rng(0, &rng)
            .set_param(param)
            .execs({{1, 6, 768, 1280}, {}});

    param.group = 2;
    checker.set_dtype(0, dtype::QuantizedS8{1.f})
            .set_dtype(1, dtype::QuantizedS8{1.f})
            .set_rng(0, &rng)
            .set_param(param)
            .execs({{8, 6, 300, 300}, {}});

    param.group = 3;
    checker.set_dtype(0, dtype::QuantizedS8{1.f})
            .set_dtype(1, dtype::QuantizedS8{1.f})
            .set_rng(0, &rng)
            .set_param(param)
            .execs({{8, 6, 300, 300}, {}});
}

TEST_F(CUDA, RELAYOUT_FORMAT_NCHW_NCHW4_WEIGHT) {
    Checker<RelayoutFormat> checker(handle_cuda());
    UniformIntRNG rng{-50, 50};
    param::RelayoutFormat param;
    param.mode = param::RelayoutFormat::Mode::NCHW_NCHW4_WEIGHT;

    for (size_t oc : {1, 3, 4, 16, 33}) {
        for (size_t ic : {1, 2, 3, 4, 8, 9, 11, 16, 33}) {
            for (size_t h : {3, 5, 7}) {
                for (size_t w : {3, 5, 7}) {
                    checker.set_dtype(0, dtype::QuantizedS8{1.f})
                            .set_dtype(1, dtype::QuantizedS8{1.f})
                            .set_rng(0, &rng)
                            .set_param(param)
                            .execs({{oc, ic, h, w}, {}});
                }
            }
        }
    }

    checker.set_dtype(0, dtype::QuantizedS8{1.f})
            .set_dtype(1, dtype::QuantizedS8{1.f})
            .set_rng(0, &rng)
            .set_param(param)
            .execs({{13, 13, 5, 5}, {}});

    checker.set_dtype(0, dtype::QuantizedS8{1.f})
            .set_dtype(1, dtype::QuantizedS8{1.f})
            .set_rng(0, &rng)
            .set_param(param)
            .execs({{4, 16, 16, 3, 3}, {}});

    checker.set_dtype(0, dtype::QuantizedS8{1.f})
            .set_dtype(1, dtype::QuantizedS8{1.f})
            .set_rng(0, &rng)
            .set_param(param)
            .execs({{4, 13, 11, 3, 3}, {}});
}

TEST_F(CUDA, RELAYOUT_FORMAT_NCHW_NCHW4_DEFAULT) {
    Checker<RelayoutFormat> checker(handle_cuda());
    UniformIntRNG rng{0, 50};
    param::RelayoutFormat param;
    param.mode = param::RelayoutFormat::Mode::NCHW_NCHW4;
    for (size_t n : {1, 3}) {
        for (size_t c : {1, 2, 3, 4, 8, 9, 11, 16}) {
            for (size_t h : {3, 7, 12, 16, 59, 83}) {
                for (size_t w : {3, 63, 128, 256}) {
                    checker.set_dtype(0, dtype::Quantized8Asymm{1.f, 128})
                            .set_dtype(1, dtype::QuantizedS8{1.f})
                            .set_rng(0, &rng)
                            .set_param(param)
                            .execs({{n, c, h, w}, {}});
                }
            }
        }
    }
}

TEST_F(CUDA, RELAYOUT_FORMAT_NCHW_NCHW4_U8) {
    Checker<RelayoutFormat> checker(handle_cuda());
    UniformIntRNG rng{0, 255};
    param::RelayoutFormat param;
    param.mode = param::RelayoutFormat::Mode::NCHW_NCHW4;
    for (size_t n : {1, 3}) {
        for (size_t c : {1, 2, 3, 4, 8, 9, 11, 16}) {
            for (size_t h : {3, 7, 12, 16, 59, 83}) {
                for (size_t w : {3, 13, 3 * 4, 63 * 4, 128 * 4, 256 * 4}) {
                    checker.set_dtype(0, dtype::Uint8())
                            .set_dtype(1, dtype::QuantizedS8{1.f})
                            .set_rng(0, &rng)
                            .set_param(param)
                            .execs({{n, c, h, w}, {}});

                    checker.set_dtype(0, dtype::Quantized8Asymm{1.f, 128})
                            .set_dtype(1, dtype::QuantizedS8{1.f})
                            .set_rng(0, &rng)
                            .set_param(param)
                            .execs({{n, c, h, w}, {}});

                    checker.set_dtype(0, dtype::Uint8())
                            .set_dtype(1, dtype::QuantizedS8{2.5f})
                            .set_rng(0, &rng)
                            .set_param(param)
                            .execs({{n, c, h, w}, {}});
                }
            }
        }
    }
}

TEST_F(CUDA, RELAYOUT_FORMAT_NCHW_NCHW4_IC_SMALL) {
    Checker<RelayoutFormat> checker(handle_cuda());
    UniformIntRNG rng{0, 50};
    param::RelayoutFormat param;
    param.mode = param::RelayoutFormat::Mode::NCHW_NCHW4_IC_SMALL;

    checker.set_dtype(0, dtype::QuantizedS8{1.f})
            .set_dtype(1, dtype::QuantizedS8{1.f})
            .set_rng(0, &rng)
            .set_param(param)
            .execs({{8, 3, 768, 1280}, {}});
}

TEST_F(CUDA, RELAYOUT_FORMAT_NCHW_NCHW64) {
    Checker<RelayoutFormat> checker(handle_cuda());
    UniformIntRNG s4{-8, 7};
    UniformIntRNG u4{0, 15};
    param::RelayoutFormat param;
    param.mode = param::RelayoutFormat::Mode::NCHW_NCHW64;
    for (size_t n : {1, 3}) {
        for (size_t c : {15, 64, 128}) {
            for (size_t h : {7, 14, 16, 28}) {
                for (size_t w : {2, 3, 7, 8, 16, 31}) {
                    checker.set_dtype(0, dtype::QuantizedS4{2.f})
                            .set_dtype(1, dtype::QuantizedS4{2.f})
                            .set_rng(0, &s4)
                            .set_param(param)
                            .execs({{n, c, h, w}, {}});

                    checker.set_dtype(0, dtype::Quantized4Asymm{1.2f, 8})
                            .set_dtype(1, dtype::Quantized4Asymm{1.2f, 4})
                            .set_rng(0, &u4)
                            .set_param(param)
                            .execs({{n, c, h, w}, {}});

                    checker.set_dtype(0, dtype::QuantizedS4{1.19990307f})
                            .set_dtype(1, dtype::QuantizedS4{1.f})
                            .set_rng(0, &s4)
                            .set_param(param)
                            .execs({{n, c, h, w}, {}});

                    checker.set_dtype(0, dtype::Quantized4Asymm{1.19990307f, 8})
                            .set_dtype(1, dtype::Quantized4Asymm{1.f, 4})
                            .set_rng(0, &u4)
                            .set_param(param)
                            .set_epsilon(1e-3)
                            .execs({{n, c, h, w}, {}});
                }
            }
        }
    }
}

TEST_F(CUDA, RELAYOUT_FORMAT_NCHW64_NCHW) {
    Checker<RelayoutFormat> checker(handle_cuda());
    UniformIntRNG s4{-8, 7};
    UniformIntRNG u4{0, 15};
    param::RelayoutFormat param;
    param.mode = param::RelayoutFormat::Mode::NCHW64_NCHW;
    for (size_t n : {1, 3}) {
        for (size_t c : {15, 64, 128}) {
            for (size_t h : {7, 14, 16, 28}) {
                for (size_t w : {2, 3, 4, 7, 14, 16, 17}) {
                    if (c % 64 != 0) {
                        param.oc = c;
                    } else {
                        param.oc = 0;
                    }
                    checker.set_dtype(0, dtype::QuantizedS4{2.f})
                            .set_dtype(1, dtype::QuantizedS4{2.f})
                            .set_rng(0, &s4)
                            .set_param(param)
                            .set_epsilon(1e-3)
                            .execs({{n, (c + 63) / 64, h, w, 64}, {}});

                    checker.set_dtype(0, dtype::Quantized4Asymm{1.2f, 4})
                            .set_dtype(1, dtype::Quantized4Asymm{1.2f, 8})
                            .set_rng(0, &u4)
                            .set_param(param)
                            .set_epsilon(1e-3)
                            .execs({{n, (c + 63) / 64, h, w, 64}, {}});

                    checker.set_dtype(0, dtype::QuantizedS4{1.19990307f})
                            .set_dtype(1, dtype::QuantizedS4{1.f})
                            .set_rng(0, &s4)
                            .set_param(param)
                            .set_epsilon(1e-3)
                            .execs({{n, (c + 63) / 64, h, w, 64}, {}});

                    checker.set_dtype(0, dtype::Quantized4Asymm{1.20211209f, 8})
                            .set_dtype(1, dtype::Quantized4Asymm{1.f, 4})
                            .set_rng(0, &u4)
                            .set_param(param)
                            .set_epsilon(1e-3)
                            .execs({{n, (c + 63) / 64, h, w, 64}, {}});
                }
            }
        }
    }
}

TEST_F(CUDA, RELAYOUT_FORMAT_NCHW_NHWC) {
    Checker<RelayoutFormat> checker(handle_cuda());
    UniformIntRNG s4{-8, 7};
    UniformIntRNG u4{0, 15};
    param::RelayoutFormat param;
    param.mode = param::RelayoutFormat::Mode::NCHW_NHWC;
    for (size_t n : {1, 3}) {
        for (size_t c : {8, 16}) {
            for (size_t h : {7, 14, 16, 28}) {
                for (size_t w : {2, 3, 7, 8, 16, 31}) {
                    checker.set_dtype(0, dtype::QuantizedS4{2.f})
                            .set_dtype(1, dtype::QuantizedS4{2.f})
                            .set_rng(0, &s4)
                            .set_param(param)
                            .execs({{n, c, h, w}, {}});

                    checker.set_dtype(0, dtype::Quantized4Asymm{1.2f, 8})
                            .set_dtype(1, dtype::Quantized4Asymm{1.2f, 4})
                            .set_rng(0, &u4)
                            .set_param(param)
                            .execs({{n, c, h, w}, {}});

                    checker.set_dtype(0, dtype::QuantizedS4{1.19990307f})
                            .set_dtype(1, dtype::QuantizedS4{1.f})
                            .set_rng(0, &s4)
                            .set_param(param)
                            .execs({{n, c, h, w}, {}});

                    checker.set_dtype(0, dtype::Quantized4Asymm{1.19990307f, 8})
                            .set_dtype(1, dtype::Quantized4Asymm{1.f, 4})
                            .set_rng(0, &u4)
                            .set_param(param)
                            .set_epsilon(1e-3)
                            .execs({{n, c, h, w}, {}});
                }
            }
        }
    }
    checker.execs({{1, 256, 384, 640}, {}});
}

TEST_F(CUDA, RELAYOUT_FORMAT_NHWC_NCHW) {
    Checker<RelayoutFormat> checker(handle_cuda());
    UniformIntRNG s4{-8, 7};
    UniformIntRNG u4{0, 15};
    param::RelayoutFormat param;
    param.mode = param::RelayoutFormat::Mode::NHWC_NCHW;
    for (size_t n : {1, 3}) {
        for (size_t c : {8, 16}) {
            for (size_t h : {7, 14, 16, 28}) {
                for (size_t w : {2, 3, 4, 7, 14, 16, 17}) {
                    checker.set_dtype(0, dtype::QuantizedS4{2.f})
                            .set_dtype(1, dtype::QuantizedS4{2.f})
                            .set_rng(0, &s4)
                            .set_param(param)
                            .set_epsilon(1e-3)
                            .execs({{n, h, w, c}, {}});

                    checker.set_dtype(0, dtype::Quantized4Asymm{1.2f, 4})
                            .set_dtype(1, dtype::Quantized4Asymm{1.2f, 8})
                            .set_rng(0, &u4)
                            .set_param(param)
                            .set_epsilon(1e-3)
                            .execs({{n, h, w, c}, {}});

                    checker.set_dtype(0, dtype::QuantizedS4{1.19990307f})
                            .set_dtype(1, dtype::QuantizedS4{1.f})
                            .set_rng(0, &s4)
                            .set_param(param)
                            .set_epsilon(1e-3)
                            .execs({{n, h, w, c}, {}});

                    checker.set_dtype(0, dtype::Quantized4Asymm{1.20211209f, 8})
                            .set_dtype(1, dtype::Quantized4Asymm{1.f, 4})
                            .set_rng(0, &u4)
                            .set_param(param)
                            .set_epsilon(1e-3)
                            .execs({{n, h, w, c}, {}});
                }
            }
        }
    }
    checker.execs({{1, 384, 640, 256}, {}});
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_RELAYOUT_FORMAT) {
    using Param = RelayoutFormat::Param;

    auto run = [&](const TensorShapeArray& shapes, Param param, Param default_param) {
        Benchmarker<RelayoutFormat> benchmarker(handle_cuda());
        benchmarker.set_param(param);
        benchmarker.set_dtype(0, dtype::QuantizedS8{1.f})
                .set_dtype(1, dtype::QuantizedS8{1.f});

        Benchmarker<RelayoutFormat> benchmarker_default(handle_cuda());
        benchmarker_default.set_param(default_param);
        benchmarker_default.set_dtype(0, dtype::QuantizedS8{1.f})
                .set_dtype(1, dtype::QuantizedS8{1.f});
        for (auto&& shape : shapes) {
            double memaccess = (double(shape.total_nr_elems()) +
                                double(shape[0]) * ((shape[1] + 3) / 4 * 4) * shape[2] *
                                        shape[3]) *
                               1e-6;
            auto time_ms = benchmarker.execs({shape, {}});
            if (shape[1] <= 4) {
                auto time_default_ms = benchmarker_default.execs({shape, {}});
                printf("execute %s, time %.4f ms, %.4f GB/s, default %.4f "
                       "GB/s\n",
                       shape.to_string().c_str(), time_ms, memaccess / time_ms,
                       memaccess / time_default_ms);
            } else {
                printf("execute %s, time %.4f ms, %.4f GB/s\n",
                       shape.to_string().c_str(), time_ms, memaccess / time_ms);
            }
        }
    };

    TensorShapeArray shapes = {
            {8, 1, 768, 1280}, {8, 3, 768, 1280},  {8, 3, 224, 224},
            {8, 4, 768, 1280}, {64, 3, 768, 1280},
    };
    {
        Param param;
        param.mode = param::RelayoutFormat::Mode::NCHW_NCHW4;
        Param default_param;
        default_param.mode = param::RelayoutFormat::Mode::NCHW_NCHW4_IC_SMALL;
        run(shapes, param, default_param);
    }
}

TEST_F(CUDA, BENCHMARK_RELAYOUT_FORMAT_QS4) {
    using Param = RelayoutFormat::Param;

    auto run = [&](const TensorShapeArray& shapes, Param param) {
        CUBenchmarker<RelayoutFormat> benchmarker(handle_cuda());
        benchmarker.set_param(param);
        benchmarker.set_dtype(0, dtype::QuantizedS4{1.19990307f})
                .set_dtype(1, dtype::QuantizedS4{1.19990307f});

        for (auto&& shape : shapes) {
            double memaccess = double(TensorLayout(shape, dtype::QuantizedS4{1.f})
                                              .span()
                                              .dist_byte()) *
                               2e-6;
            auto time_ms = benchmarker.execs({shape, {}});
            printf("execute %s, time %.4f ms, %.4f GB/s\n", shape.to_string().c_str(),
                   time_ms, memaccess / time_ms);
        }
    };

    printf("nchw -> nchw64\n");
    {
        TensorShapeArray shapes = {
                {1, 64, 56, 56},  {16, 64, 56, 56}, {64, 64, 56, 56},   {1, 64, 56, 55},
                {16, 64, 56, 55}, {64, 64, 56, 55}, {1, 256, 384, 640},
        };
        Param param;
        param.mode = param::RelayoutFormat::Mode::NCHW_NCHW64;
        run(shapes, param);
    }
    printf("nchw -> nhwc\n");
    {
        TensorShapeArray shapes = {
                {1, 64, 56, 56},    {16, 64, 56, 56},   {64, 64, 56, 56},
                {1, 64, 56, 55},    {16, 64, 56, 55},   {64, 64, 56, 55},
                {1, 256, 384, 640}, {16, 16, 384, 640},
        };
        Param param;
        param.mode = param::RelayoutFormat::Mode::NCHW_NHWC;
        run(shapes, param);
    }
    printf("nchw64 -> nchw\n");
    {
        TensorShapeArray shapes = {
                {64, 1, 56, 56, 64}, {1, 32, 7, 7, 64},    {16, 32, 7, 7, 64},
                {64, 32, 7, 7, 64},  {1, 4, 384, 640, 64},
        };
        Param param;
        param.mode = param::RelayoutFormat::Mode::NCHW64_NCHW;
        run(shapes, param);
    }
    printf("nhwc -> nchw\n");
    {
        TensorShapeArray shapes = {
                {64, 56, 56, 64},    {1, 7, 7, 64 * 32},    {16, 7, 7, 64 * 32},
                {64, 7, 7, 64 * 32}, {1, 384, 640, 64 * 4},
        };
        Param param;
        param.mode = param::RelayoutFormat::Mode::NHWC_NCHW;
        run(shapes, param);
    }
}

#endif

TEST_F(CUDA, RELAYOUT_FORMAT_NCHW4) {
    Checker<RelayoutFormat> checker(handle_cuda());
    UniformIntRNG rng{-50, 50};
    param::RelayoutFormat param;
    param.mode = param::RelayoutFormat::Mode::NCHW_NCHW4_IC_SMALL;

    for (DType dtype :
         std::vector<DType>({dtype::QuantizedS8{0.1f}, dtype::Float32{}})) {
        checker.set_dtype(0, dtype).set_dtype(1, dtype).set_rng(0, &rng);

        checker.set_param(param).execs({{2, 4, 35, 36}, {}});
        checker.set_param(param).execs({{2, 3, 35, 36}, {}});
        checker.set_param(param).execs({{2, 1, 35, 36}, {}});
        param.mode = param::RelayoutFormat::Mode::NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT;
        checker.set_param(param).execs({{4, 3, 3, 3}, {}});
        checker.set_param(param).execs({{4, 4, 3, 3}, {}});
        checker.set_param(param).execs({{1, 4, 3, 3}, {}});
    }
}

// vim: syntax=cpp.doxygen
