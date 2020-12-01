/**
 * \file dnn/test/cuda/relayout_format.cpp
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
#include "megdnn/oprs.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
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

TEST_F(CUDA, RELAYOUT_FORMAT_NCHW_NCHW4) {
    Checker<RelayoutFormat> checker(handle_cuda());
    UniformIntRNG rng{0, 50};
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

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_RELAYOUT_FORMAT) {
    using Param = RelayoutFormat::Param;

    auto run = [&](const TensorShapeArray& shapes, Param param,
                   Param default_param) {
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
                                double(shape[0]) * ((shape[1] + 3) / 4 * 4) *
                                        shape[2] * shape[3]) *
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
        param.mode = param::RelayoutFormat::Mode::
                NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT;
        checker.set_param(param).execs({{4, 3, 3, 3}, {}});
        checker.set_param(param).execs({{4, 4, 3, 3}, {}});
        checker.set_param(param).execs({{1, 4, 3, 3}, {}});
    }
}

// vim: syntax=cpp.doxygen
