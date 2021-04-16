/**
 * \file dnn/test/cuda/accuracy_shake.cpp
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
#include "megdnn/opr_param_defs.h"
#include "test/cuda/fixture.h"
#include "test/cuda/utils.h"
#include "test/common/rng.h"
#include "test/common/accuracy_shake_checker.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, SHAKE_CONV_BIAS_FORWARD) {
    require_compute_capability(6, 1);
    AccuracyShakeChecker<ConvBiasForward> checker(handle_cuda());
    NormalRNG default_rng;
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_rng(0, &default_rng)
            .set_rng(1, &default_rng);
    // convolution
    checker.exec({{64, 16, 32, 32}, {64, 16, 3, 3}, {}, {}, {}});
    // convbias without z
    checker.exec({{64, 16, 32, 32}, {64, 16, 3, 3}, {1, 64, 1, 1}, {}, {}});
    // convbias with z
    checker.exec({{64, 16, 32, 32},
                  {64, 16, 3, 3},
                  {1, 64, 1, 1},
                  {64, 64, 30, 30},
                  {}});
    ConvBias::Param param;
    // group
    param.sparse = ConvBias::Param::Sparse::GROUP;
    checker.set_param(param);
    checker.exec({{64, 16, 32, 32}, {2, 32, 8, 3, 3}, {}, {}, {}});
    checker.exec({{64, 16, 32, 32}, {2, 32, 8, 3, 3}, {1, 64, 1, 1}, {}, {}});
    checker.exec({{64, 16, 32, 32},
                  {2, 32, 8, 3, 3},
                  {1, 64, 1, 1},
                  {64, 64, 30, 30},
                  {}});
}

TEST_F(CUDA, SHAKE_CONV_BIAS_FORWARD_QS8_NCHW) {
    require_compute_capability(6, 1);
    AccuracyShakeChecker<ConvBiasForward> checker(handle_cuda());
    UniformIntRNG int_rng{-128, 127};

    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(3, dtype::QuantizedS8(0.25f))
            .set_dtype(4, dtype::QuantizedS8(0.25f))
            .set_rng(0, &int_rng)
            .set_rng(1, &int_rng)
            .set_rng(2, &int_rng)
            .set_rng(3, &int_rng);


    // convolution
    checker.exec({{64, 16, 32, 32}, {64, 16, 3, 3}, {}, {}, {}});
    // convbias without z
    checker.exec({{64, 16, 32, 32}, {64, 16, 3, 3}, {1, 64, 1, 1}, {}, {}});
    // convbias with z
    checker.exec({{64, 16, 32, 32},
                  {64, 16, 3, 3},
                  {1, 64, 1, 1},
                  {64, 64, 30, 30},
                  {}});
    // group
    ConvBias::Param param;
    param.sparse = ConvBias::Param::Sparse::GROUP;
    checker.set_param(param);
    checker.exec({{64, 16, 32, 32}, {2, 32, 8, 3, 3}, {}, {}, {}});
    checker.exec({{64, 16, 32, 32}, {2, 32, 8, 3, 3}, {1, 64, 1, 1}, {}, {}});
    checker.exec({{64, 16, 32, 32},
                  {2, 32, 8, 3, 3},
                  {1, 64, 1, 1},
                  {64, 64, 30, 30},
                  {}});
}

TEST_F(CUDA, SHAKE_CONV_BIAS_FORWARD_QS8_NHWC) {
    require_compute_capability(6, 1);

    UniformIntRNG int_rng{-50, 50};
    AccuracyShakeChecker<ConvBiasForward> checker(handle_cuda());
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NHWC;
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(4, dtype::QuantizedS8(60.25f))
            .set_rng(0, &int_rng)
            .set_rng(1, &int_rng)
            .set_rng(2, &int_rng)
            .set_param(param);
    checker.exec({{20, 32, 32, 4}, {24, 1, 1, 4}, {1, 1, 1, 24}, {}, {}});

    param.sparse = ConvBias::Param::Sparse::GROUP;
    checker.set_param(param).exec(
            {{20, 32, 32, 16}, {4, 4, 1, 1, 4}, {1, 1, 1, 16}, {}, {}});
}

TEST_F(CUDA, SHAKE_CONV_BIAS_FORWARD_QS8_NCHWX) {
    using Format = ConvBias::Param::Format;
    require_compute_capability(6, 1);
    AccuracyShakeChecker<ConvBiasForward> checker(handle_cuda());
    UniformIntRNG int_rng{-5, 5};
    UniformFloatRNG float_rng{-50, 50};

    checker.set_dtype(0, dtype::QuantizedS8(1.2f))
            .set_dtype(1, dtype::QuantizedS8(1.3f))
            .set_dtype(2, dtype::QuantizedS32(1.2 * 1.3f))
            .set_dtype(3, dtype::QuantizedS8(1.3f))
            .set_dtype(4, dtype::QuantizedS8(1.3f))
            .set_rng(0, &int_rng)
            .set_rng(1, &int_rng)
            .set_rng(2, &int_rng)
            .set_rng(3, &int_rng);

    auto run = [&](const TensorShapeArray& shapes, const Format& format) {
        ConvBias::Param param;
        param.format = format;
        checker.set_param(param).exec(
                {shapes[0], shapes[1], shapes[2], {}, {}});
    };

    run({{20, 2, 24, 24, 4}, {24, 2, 3, 3, 4}, {1, 6, 1, 1, 4}}, Format::NCHW4);
    run({{20, 1, 24, 24, 32}, {64, 1, 3, 3, 32}, {1, 2, 1, 1, 32}},
        Format::NCHW32);
    run({{16, 4, 23, 40, 4},
         {32, 4, 3, 3, 4},
         {1, 1, 1, 1, 32}}, Format::NCHW4_NCHW32);

    checker.set_dtype(0, dtype::QuantizedS8(1.9980618f))
            .set_dtype(1, dtype::QuantizedS8(1.9980927f))
            .set_dtype(2, dtype::Float32())
            .set_dtype(3, dtype::Float32())
            .set_dtype(4, dtype::Float32())
            .set_rng(0, &int_rng)
            .set_rng(1, &int_rng)
            .set_rng(2, &float_rng)
            .set_rng(3, &float_rng);
    run({{16, 4, 92, 160, 4}, {20, 4, 3, 3, 4}, {1, 20, 1, 1}},
        Format::NCHW4_NCHW);
}

TEST_F(CUDA, SHAKE_MATRIX_MUL_FORWARD) {
    AccuracyShakeChecker<MatrixMul> checker(handle_cuda());

    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .exec({{50, 100}, {100, 60}, {}});
}

TEST_F(CUDA, SHAKE_BATCH_CONV_BIAS_QS8) {
    require_compute_capability(6, 1);
    AccuracyShakeChecker<BatchConvBiasForward> checker(handle_cuda());
    UniformIntRNG const_rng{1, 1};
    UniformIntRNG rng{-5, 5};
    UniformIntRNG bias_rng{-50, 50};
    checker.set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_rng(3, &rng)
            .set_dtype(0, dtype::QuantizedS8{1.2f})
            .set_dtype(1, dtype::QuantizedS8{1.3f})
            .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
            .set_dtype(3, dtype::QuantizedS8{1.1f})
            .set_dtype(4, dtype::QuantizedS8{1.1f});
    param::BatchConvBias param;
    param.pad_h = 2, param.pad_w = 1;
    param.stride_h = 1, param.stride_w = 2;
    param.format = param::BatchConvBias::Format::NCHW4;
    checker.set_param(param).exec({{32, 4, 24, 24, 4},
                                    {32, 32, 4, 1, 1, 4},
                                    {1, 8, 1, 1, 4},
                                    {},
                                    {}});
}

TEST_F(CUDA, SHAKE_BATCHED_MATRIX_MUL) {
    AccuracyShakeChecker<BatchedMatrixMul> checker(handle_cuda());

    UniformIntRNG int_rng{-127, 127};
    NormalRNG default_rng;
    checker.set_dtype(0, dtype::QuantizedS8(1.2f))
            .set_dtype(1, dtype::QuantizedS8(1.3f))
            .set_dtype(2, {})
            .set_rng(0, &int_rng)
            .set_rng(1, &int_rng);

    checker.exec({{20, 424, 368}, {20, 368, 256}, {20, 424, 256}});

    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_rng(0, &default_rng)
            .set_rng(1, &default_rng);

    checker.exec({{20, 424, 368}, {20, 368, 256}, {20, 424, 256}});
}

TEST_F(CUDA, SHAKE_CONVOLUTION3D_FORWARD) {
    AccuracyShakeChecker<Convolution3DForward> checker(handle_cuda());
    NormalRNG default_rng;
    float scale = 1.0f / sqrt(5);
    UniformFloatRNG rng(scale, 2 * scale);
    param::Convolution3D param;
    param.mode = param::Convolution3D::Mode::CROSS_CORRELATION;
    param.stride_d = param.stride_h = param.stride_w = 2;
    param.pad_d = param.pad_h = param.pad_w = 0;
    param.dilate_d = param.dilate_h = param.dilate_w = 1;
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_rng(0, &default_rng)
            .set_rng(1, &default_rng)
            .set_param(param)
            .exec({{20, 5, 12, 12, 16}, {5, 5, 3, 3, 3}, {}});
}

TEST_F(CUDA, SHAKE_LOCAL_SHARE) {
    AccuracyShakeChecker<LocalShare> checker(handle_cuda());
    using Param = LocalShare::Param;
    Param param;
    param.spatial_groups_h = param.spatial_groups_w = 3;
    checker.set_param(param);
    checker.exec({{20, 16, 32, 32}, {3, 3, 16, 3, 3, 64}, {}});
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
