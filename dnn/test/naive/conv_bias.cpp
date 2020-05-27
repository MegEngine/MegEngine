/**
 * \file dnn/test/naive/conv_bias.cpp
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
#include "test/common/checker.h"
#include "test/common/workspace_wrapper.h"
#include "test/naive/fixture.h"

using namespace megdnn;
using namespace test;

namespace {
class TensorWrapper {
public:
    TensorWrapper(Handle* handle, TensorLayout layout) : m_handle(handle) {
        m_tensornd.raw_ptr = megdnn_malloc(m_handle, layout.span().dist_byte());
        m_tensornd.layout = layout;
    }
    ~TensorWrapper() { megdnn_free(m_handle, m_tensornd.raw_ptr); }

    TensorND tensornd() const { return m_tensornd; }

private:
    Handle* m_handle;
    TensorND m_tensornd;
};
}  // namespace

TEST_F(NAIVE, CONV_BIAS_QUANTIZED8x8x32) {
    Checker<ConvBias> checker(handle(), /* check_dispatch */ false);
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW;

    checker.set_param(param).exect(
            Testcase{TensorValue({1, 1, 4, 4}, dtype::QuantizedS8(0.1f),
                                 {90 - 128, 136 - 128, 85 - 128, 204 - 128,
                                  48 - 128, 9 - 128, 226 - 128, 25 - 128,
                                  118 - 128, 109 - 128, 87 - 128, 132 - 128,
                                  104 - 128, 163 - 128, 25 - 128, 90 - 128}),
                     TensorValue({3, 1, 3, 3}, dtype::QuantizedS8(0.2f),
                                 {153 - 124, 170 - 124, 102 - 124, 103 - 124,
                                  23 - 124,  213 - 124, 116 - 124, 195 - 124,
                                  191 - 124, 44 - 124,  50 - 124,  247 - 124,
                                  172 - 124, 42 - 124,  32 - 124,  233 - 124,
                                  163 - 124, 247 - 124, 120 - 124, 241 - 124,
                                  209 - 124, 83 - 124,  201 - 124, 115 - 124,
                                  32 - 124,  140 - 124, 147 - 124}),
                     TensorValue({1, 3, 1, 1}, dtype::QuantizedS32(0.02f),
                                 {0, 0, 0}),
                     TensorValue({1, 3, 2, 2}, dtype::QuantizedS32(0.3f),
                                 {1234, 0, 0, 0, 0, 0, 0, 0, 0, -234, 0, 0}),
                     {}},
            Testcase{{},
                     {},
                     {},
                     {},
                     TensorValue({1, 3, 2, 2}, dtype::QuantizedS32(0.1f * 0.2f),
                                 {37127, -22475, -15694, -1920,

                                  -12813, 4440, 18190, -13195,

                                  -9659, 12423, -5558, -4969})});
}

TEST_F(NAIVE, CONV_BIAS_QUANTIZED4x4x32) {
    Checker<ConvBias> checker(handle(), false);
    using Param = ConvBiasForward::Param;
    Param param;
    param.format = Param::Format::NCHW8;
    checker.set_param(param);
    auto GenTensorValueQuint4 = [](const TensorShape& shape,
                                   dtype::Quantized4Asymm dtype,
                                   const std::vector<int>& values) {
        TensorND tensor;
        tensor.layout = {shape, dtype};
        tensor.raw_ptr =
                static_cast<dt_byte*>(malloc(tensor.layout.span().dist_byte()));
        uint8_t* ptr = static_cast<uint8_t*>(tensor.raw_ptr);
        megdnn_assert(values.size() == tensor.layout.span().dist_elem());
        for (size_t i = 0; i < tensor.layout.span().dist_elem(); i += 2) {
            int val0 = values[i], val1 = values[i + 1];
            ptr[i / 2] = val0 | (val1 << 4);
        }
        return tensor;
    };
    checker.set_param(param).exect(
            Testcase{
                    GenTensorValueQuint4(
                            {1, 1, 4, 4, 8},
                            dtype::Quantized4Asymm(0.1f, uint8_t(8)),
                            {0,  6,  14, 5,  11, 2,  9,  9,  2,  1,  2,  11, 5,
                             0,  4,  8,  12, 15, 7,  7,  11, 0,  4,  1,  14, 9,
                             2,  0,  1,  11, 7,  13, 6,  11, 14, 4,  14, 6,  4,
                             3,  4,  2,  8,  15, 10, 6,  7,  0,  11, 13, 3,  9,
                             5,  13, 0,  5,  4,  5,  10, 5,  5,  0,  3,  13, 5,
                             4,  14, 10, 8,  3,  15, 1,  13, 5,  8,  9,  13, 10,
                             15, 13, 9,  0,  1,  11, 15, 4,  12, 11, 4,  5,  2,
                             9,  10, 9,  3,  1,  15, 10, 0,  1,  4,  6,  11, 2,
                             4,  9,  14, 6,  12, 0,  10, 13, 9,  7,  14, 14, 3,
                             14, 14, 7,  2,  4,  1,  9,  4,  7,  15, 10}),
                    GenTensorValueQuint4(
                            {8, 1, 3, 3, 8},
                            dtype::Quantized4Asymm(0.2f, uint8_t(7)),
                            {6,  8,  3,  6,  1,  9,  7,  8,  10, 0,  4,  11, 0,
                             1,  9,  8,  3,  3,  0,  9,  3,  2,  2,  2,  10, 5,
                             8,  7,  12, 10, 1,  11, 3,  1,  9,  8,  2,  15, 5,
                             0,  14, 3,  8,  15, 14, 7,  15, 4,  3,  3,  11, 9,
                             8,  4,  7,  14, 4,  6,  10, 7,  5,  5,  2,  0,  5,
                             0,  1,  10, 13, 1,  7,  12, 9,  11, 12, 7,  3,  15,
                             1,  10, 7,  8,  9,  1,  6,  8,  7,  0,  4,  12, 12,
                             11, 4,  0,  14, 1,  6,  15, 15, 4,  1,  2,  10, 9,
                             6,  0,  13, 2,  5,  8,  11, 1,  1,  2,  4,  13, 3,
                             3,  12, 11, 6,  5,  8,  11, 13, 12, 0,  13, 9,  13,
                             12, 1,  7,  10, 6,  12, 8,  13, 11, 1,  3,  5,  0,
                             10, 4,  8,  15, 13, 9,  7,  2,  14, 9,  9,  10, 7,
                             13, 0,  9,  4,  7,  10, 15, 4,  10, 10, 9,  13, 8,
                             7,  10, 9,  13, 12, 14, 8,  3,  6,  4,  8,  5,  5,
                             6,  3,  6,  6,  10, 4,  3,  0,  12, 8,  7,  3,  14,
                             7,  3,  2,  3,  7,  7,  3,  0,  8,  11, 3,  14, 1,
                             13, 10, 5,  7,  9,  15, 8,  9,  1,  3,  11, 13, 13,
                             6,  0,  6,  0,  10, 0,  1,  4,  3,  11, 3,  7,  1,
                             7,  10, 7,  2,  13, 15, 12, 0,  2,  0,  6,  15, 9,
                             13, 2,  10, 2,  1,  13, 13, 7,  7,  2,  10, 1,  12,
                             9,  5,  2,  8,  11, 13, 12, 5,  3,  1,  9,  14, 12,
                             6,  12, 12, 3,  7,  0,  8,  1,  9,  12, 2,  10, 11,
                             5,  11, 10, 10, 13, 9,  3,  1,  4,  9,  6,  2,  15,
                             8,  12, 5,  14, 0,  8,  1,  3,  2,  14, 1,  6,  4,
                             4,  10, 9,  5,  15, 8,  2,  4,  3,  11, 6,  12, 6,
                             3,  14, 5,  11, 5,  9,  15, 8,  3,  5,  3,  11, 9,
                             5,  7,  14, 9,  0,  5,  11, 9,  14, 13, 2,  1,  10,
                             6,  6,  6,  15, 0,  7,  9,  12, 6,  6,  5,  0,  14,
                             15, 9,  10, 10, 13, 7,  12, 5,  13, 2,  7,  14, 7,
                             14, 13, 0,  12, 10, 7,  4,  12, 1,  8,  7,  8,  0,
                             11, 12, 12, 4,  7,  9,  15, 1,  15, 11, 7,  6,  9,
                             0,  10, 6,  7,  5,  11, 14, 13, 14, 6,  3,  0,  3,
                             6,  10, 3,  5,  0,  7,  6,  14, 15, 8,  4,  13, 11,
                             3,  1,  5,  6,  2,  14, 1,  15, 4,  4,  4,  8,  7,
                             13, 0,  8,  14, 10, 8,  14, 7,  11, 0,  2,  15, 13,
                             15, 0,  7,  8,  15, 6,  6,  4,  2,  4,  10, 13, 10,
                             6,  1,  10, 14, 13, 6,  9,  6,  8,  11, 10, 13, 2,
                             6,  10, 0,  1,  6,  15, 7,  6,  6,  13, 9,  2,  9,
                             0,  2,  15, 15, 14, 0,  2,  13, 15, 15, 0,  7,  10,
                             10, 13, 15, 6,  13, 8,  5,  4,  12, 9,  4,  14, 8,
                             6,  13, 15, 2,  8,  10, 11, 6,  11, 0,  15, 0,  1,
                             5,  1,  14, 13, 7,  2,  6,  3,  9,  7,  6,  15, 12,
                             14, 2,  10, 12, 8,  14, 5,  12, 13, 15, 10, 9,  7,
                             7,  13, 6,  11, 13, 9,  4,  8,  9,  2,  11, 13, 8,
                             1,  0,  14, 6}),
                    TensorValue({1, 1, 1, 1, 8},
                                dtype::QuantizedS32(0.1f * 0.2f),
                                {0, 0, 0, 0, 0, 0, 0, 0}),
                    TensorValue(
                            {1, 1, 2, 2, 8}, dtype::QuantizedS32(0.3f),
                            {0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, -87, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                    {}},
            Testcase{
                    {},
                    {},
                    {},
                    {},
                    TensorValue(
                            {1, 1, 2, 2, 8}, dtype::QuantizedS32(0.1f * 0.2f),
                            {275,  -232, 55,  -123, 81,   -55,  -324,  64,
                             -104, -391, 242, -2,   -162, -150, -232,  -160,
                             -192, -72,  -52, -154, 198,  -48,  -1073, -105,
                             103,  -218, -22, 446,  -81,  90,   -152,  -126}),
            });
}

TEST_F(NAIVE, CONV_BIAS_QUANTIZED8x8x32_NCHW32) {
    Checker<ConvBias> checker(handle(), /* check_dispatch */ false);
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW32;

    size_t N = 2, IC = 32, IH = 4, IW = 4, OC = 32, PH = 1, PW = 1, SH = 1,
           SW = 1, FH = 3, FW = 3;
    auto&& conv_opr = handle()->create_operator<ConvBias>();
    conv_opr->param().format = ConvBias::Param::Format::NCHW4;
    conv_opr->param().pad_h = param.pad_h = PH;
    conv_opr->param().pad_w = param.pad_w = PW;
    conv_opr->param().stride_h = param.stride_h = SH;
    conv_opr->param().stride_w = param.stride_w = SW;
    size_t OH = infer_conv_shape(IH, FH, SH, PH);
    size_t OW = infer_conv_shape(IW, FW, SW, PW);
    auto i8_min = std::numeric_limits<int8_t>().min();
    auto i8_max = std::numeric_limits<int8_t>().max();
    UniformIntRNG int_rng{i8_min, i8_max};
    TensorLayout src_layout_4{{N, IC / 4, IH, IW, 4}, dtype::QuantizedS8(0.1f)};
    TensorWrapper src_ts_4{handle(), src_layout_4};
    int_rng.gen(src_ts_4.tensornd());

    TensorLayout filter_layout_4{{OC, IC / 4, FH, FW, 4},
                                 dtype::QuantizedS8(0.2f)};
    TensorWrapper filter_ts_4{handle(), filter_layout_4};
    int_rng.gen(filter_ts_4.tensornd());

    TensorLayout bias_layout_4{{1, OC / 4, 1, 1, 4},
                               dtype::QuantizedS32(0.02f)};
    TensorWrapper bias_ts_4{handle(), bias_layout_4};
    int_rng.gen(bias_ts_4.tensornd());

    TensorLayout dst_layout_4{{N, OC / 4, OH, OW, 4}, dtype::QuantizedS8(0.2f)};
    TensorWrapper dst_ts_4{handle(), dst_layout_4};
    TensorLayout z_layout_4{dst_layout_4, dtype::QuantizedS8(0.3f)};
    TensorWrapper z_ts_4{handle(), z_layout_4};
    int_rng.gen(z_ts_4.tensornd());

    size_t ws_size = conv_opr->get_workspace_in_bytes(
            src_layout_4, filter_layout_4, bias_layout_4, z_layout_4,
            dst_layout_4, nullptr);
    WorkspaceWrapper ws{handle(), ws_size};
    conv_opr->exec(src_ts_4.tensornd(), filter_ts_4.tensornd(),
                   bias_ts_4.tensornd(), z_ts_4.tensornd(), dst_ts_4.tensornd(),
                   nullptr, ws.workspace());

    TensorLayout src_layout_32{{N, IC / 32, IH, IW, 32},
                               dtype::QuantizedS8(0.1f)};
    TensorWrapper src_ts_32{handle(), src_layout_32};

    TensorLayout filter_layout_32{{OC, IC / 32, FH, FW, 32},
                                  dtype::QuantizedS8(0.2f)};
    TensorWrapper filter_ts_32{handle(), filter_layout_32};

    TensorLayout bias_layout_32{{1, OC / 32, 1, 1, 32},
                                dtype::QuantizedS32(0.02f)};
    TensorWrapper bias_ts_32{handle(), bias_layout_32};

    TensorLayout dst_layout_32{{N, OC / 32, OH, OW, 32},
                               dtype::QuantizedS8(0.2f)};
    TensorWrapper dst_ts_32{handle(), dst_layout_32};

    TensorLayout z_layout_32{dst_layout_32, dtype::QuantizedS8(0.3f)};
    TensorWrapper z_ts_32{handle(), z_layout_32};

    auto from_nchw4_to_nchw32 = [](const TensorND in, const TensorND out) {
        size_t n = out.layout[0], c = out.layout[1], h = out.layout[2],
               w = out.layout[3];
        if (in.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {
            int8_t* in_ptr = in.compatible_ptr<int8_t>();
            int8_t* out_ptr = out.compatible_ptr<int8_t>();
            for (size_t b = 0; b < n; b++) {
                for (size_t ch_out = 0; ch_out < c; ch_out++) {
                    for (size_t h_ = 0; h_ < h; h_++) {
                        for (size_t w_ = 0; w_ < w; w_++) {
                            for (size_t ch_in = 0; ch_in < 32; ch_in++) {
                                size_t ch = ch_out * 32 + ch_in;
                                size_t ch_out_ = ch / 4;
                                size_t ch_in_ = ch % 4;
                                *out_ptr = in_ptr[b * c * h * w * 32 +
                                                  ch_out_ * h * w * 4 +
                                                  h_ * w * 4 + w_ * 4 + ch_in_];
                                out_ptr++;
                            }
                        }
                    }
                }
            }
        }
        if (in.layout.dtype.enumv() == DTypeEnum::QuantizedS32) {
            int32_t* in_ptr = in.compatible_ptr<int32_t>();
            int32_t* out_ptr = out.compatible_ptr<int32_t>();
            for (size_t b = 0; b < n; b++) {
                for (size_t ch_out = 0; ch_out < c; ch_out++) {
                    for (size_t h_ = 0; h_ < h; h_++) {
                        for (size_t w_ = 0; w_ < w; w_++) {
                            for (size_t ch_in = 0; ch_in < 32; ch_in++) {
                                size_t ch = ch_out * 32 + ch_in;
                                size_t ch_out_ = ch / 4;
                                size_t ch_in_ = ch % 4;
                                *out_ptr = in_ptr[b * c * h * w * 32 +
                                                  ch_out_ * h * w * 4 +
                                                  h_ * w * 4 + w_ * 4 + ch_in_];
                                out_ptr++;
                            }
                        }
                    }
                }
            }
        }
    };

    from_nchw4_to_nchw32(src_ts_4.tensornd(), src_ts_32.tensornd());
    from_nchw4_to_nchw32(filter_ts_4.tensornd(), filter_ts_32.tensornd());
    from_nchw4_to_nchw32(bias_ts_4.tensornd(), bias_ts_32.tensornd());
    from_nchw4_to_nchw32(dst_ts_4.tensornd(), dst_ts_32.tensornd());
    from_nchw4_to_nchw32(z_ts_4.tensornd(), z_ts_32.tensornd());

    checker.set_param(param).exect(
            TensorNDArray{src_ts_32.tensornd(),
                          filter_ts_32.tensornd(),
                          bias_ts_32.tensornd(),
                          z_ts_32.tensornd(),
                          {}},
            TensorNDArray{{}, {}, {}, {}, dst_ts_32.tensornd()});
}

TEST_F(NAIVE, CONV_BIAS_NCHW44) {
    Checker<ConvBias> checker(handle(), /* check_dispatch */ false);
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW44;

    size_t n = 1;
    size_t ic = 4;
    size_t oc = 8;
    size_t h = 2;
    size_t w = 2;
    size_t filter_size = 3;
    size_t pad = 1;
    auto src_tensor_shape = TensorShape{n, ic / 4, h, w, 4};
    auto weight_tensor_shape =
            TensorShape{oc / 4, ic / 4, filter_size, filter_size, 4, 4};
    auto bias_tensor_shape = TensorShape{1, oc / 4, 1, 1, 4};
    param.pad_h = pad;
    param.pad_w = pad;
    UniformIntRNG rng{-127, 127};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_dtype(4, dtype::Float32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_epsilon(1e-3)
            .set_param(param)
            .execs({src_tensor_shape,
                    weight_tensor_shape,
                    bias_tensor_shape,
                    {},
                    {}});

    checker.set_dtype(0, dtype::QuantizedS8(2.f))
            .set_dtype(1, dtype::QuantizedS8(3.f))
            .set_dtype(2, dtype::QuantizedS32(6.f))
            .set_dtype(4, dtype::QuantizedS32(6.f))
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_epsilon(1e-3)
            .set_param(param)
            .execs({src_tensor_shape,
                    weight_tensor_shape,
                    bias_tensor_shape,
                    {},
                    {}});

    {
        // test normal conv
        ConvBias::Param param;
        param.format = ConvBias::Param::Format::NCHW44;
        param.sparse = ConvBias::Param::Sparse::DENSE;
        param.pad_h = 1;
        param.pad_w = 1;
        checker.set_param(param).exect(
                Testcase{TensorValue({1, 1, 2, 2, 4}, dtype::Float32(),
                                     {7, 2, 2, 1, 7, 5, 6, 3, 1, 2, 8, 3, 7, 7,
                                      6, 4}),
                         TensorValue(
                                 {1, 1, 3, 3, 4, 4}, dtype::Float32(),
                                 {3, 5, 5, 2, 0, 1, 4, 8, 3, 5, 0, 7, 1, 7, 0,
                                  7, 6, 4, 7, 7, 5, 2, 2, 4, 7, 6, 6, 3, 3, 2,
                                  2, 8, 5, 0, 4, 4, 0, 5, 1, 0, 0, 4, 8, 4, 7,
                                  7, 2, 0, 4, 8, 7, 3, 6, 2, 3, 0, 0, 6, 4, 4,
                                  1, 4, 3, 8, 8, 8, 7, 2, 2, 5, 5, 1, 3, 2, 8,
                                  1, 7, 0, 2, 7, 1, 6, 1, 5, 0, 6, 3, 0, 2, 4,
                                  1, 1, 4, 2, 7, 5, 7, 8, 4, 5, 5, 7, 0, 3, 3,
                                  2, 8, 6, 0, 1, 4, 6, 6, 6, 0, 1, 2, 4, 4, 1,
                                  1, 7, 8, 2, 5, 2, 8, 3, 8, 3, 5, 0, 6, 3, 4,
                                  3, 3, 7, 2, 8, 1, 1, 1, 4}),
                         TensorValue({1, 1, 1, 1, 4}, dtype::Float32(),
                                     {7, 2, 8, 1}),
                         TensorValue({1, 1, 2, 2, 4}, dtype::Float32(),
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0}),
                         {}},
                Testcase{
                        {},
                        {},
                        {},
                        {},
                        TensorValue({1, 1, 2, 2, 4}, dtype::Float32(),
                                    {264, 338, 309, 195, 276, 332, 390, 199,
                                     224, 268, 311, 218, 288, 311, 346, 277})});
    }

    {
        // test dw conv
        ConvBias::Param param;
        param.format = ConvBias::Param::Format::NCHW44;
        param.sparse = ConvBias::Param::Sparse::GROUP;
        param.pad_h = 1;
        param.pad_w = 1;
        checker.set_param(param).exect(
                Testcase{TensorValue({1, 1, 2, 2, 4}, dtype::Float32(),
                                     {5, 8, 3, 2, 4, 6, 1, 5, 0, 8, 2, 6, 8, 6,
                                      5, 7}),
                         TensorValue({1, 1, 1, 3, 3, 4}, dtype::Float32(),
                                     {3, 0, 3, 1, 6, 5, 7, 3, 5, 0, 0, 7,
                                      4, 6, 0, 1, 8, 2, 3, 7, 1, 0, 2, 4,
                                      7, 5, 3, 0, 6, 2, 1, 5, 8, 6, 3, 1}),
                         TensorValue({1, 1, 1, 1, 4}, dtype::Float32(),
                                     {4, 3, 5, 6}),
                         TensorValue({1, 1, 2, 2, 4}, dtype::Float32(),
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0}),
                         {}},
                Testcase{{},
                         {},
                         {},
                         {},
                         TensorValue({1, 1, 2, 2, 4}, dtype::Float32(),
                                     {112, 71, 33, 77, 104, 115, 19, 78, 62, 59,
                                      42, 117, 107, 93, 36, 78})});
    }

    {
        // test group conv
        ConvBias::Param param;
        param.format = ConvBias::Param::Format::NCHW44;
        param.sparse = ConvBias::Param::Sparse::GROUP;
        param.pad_h = 1;
        param.pad_w = 1;
        checker.set_param(param).exect(
                Testcase{TensorValue({1, 2, 2, 2, 4}, dtype::Float32(),
                                     {6, 3, 2, 7, 7, 6, 4, 5, 8, 6, 3,
                                      1, 1, 2, 8, 3, 1, 0, 6, 1, 3, 3,
                                      6, 0, 0, 5, 6, 7, 2, 2, 4, 4}),
                         TensorValue(
                                 {2, 1, 1, 3, 3, 4, 4}, dtype::Float32(),
                                 {3, 5, 5, 2, 0, 1, 4, 8, 3, 5, 0, 7, 1, 7, 0,
                                  7, 6, 4, 7, 7, 5, 2, 2, 4, 7, 6, 6, 3, 3, 2,
                                  2, 8, 5, 0, 4, 4, 0, 5, 1, 0, 0, 4, 8, 4, 7,
                                  7, 2, 0, 4, 8, 7, 3, 6, 2, 3, 0, 0, 6, 4, 4,
                                  1, 4, 3, 8, 8, 8, 7, 2, 2, 5, 5, 1, 3, 2, 8,
                                  1, 7, 0, 2, 7, 1, 6, 1, 5, 0, 6, 3, 0, 2, 4,
                                  1, 1, 4, 2, 7, 5, 7, 8, 4, 5, 5, 7, 0, 3, 3,
                                  2, 8, 6, 0, 1, 4, 6, 6, 6, 0, 1, 2, 4, 4, 1,
                                  1, 7, 8, 2, 5, 2, 8, 3, 8, 3, 5, 0, 6, 3, 4,
                                  3, 3, 7, 2, 8, 1, 1, 1, 4, 7, 4, 5, 0, 6, 8,
                                  7, 4, 8, 1, 3, 5, 3, 0, 0, 3, 7, 7, 7, 3, 8,
                                  1, 2, 0, 1, 1, 2, 1, 3, 0, 0, 1, 1, 3, 0, 5,
                                  6, 3, 0, 5, 4, 1, 4, 7, 0, 2, 1, 6, 7, 8, 0,
                                  2, 1, 6, 7, 6, 3, 2, 7, 6, 5, 1, 1, 1, 2, 4,
                                  6, 3, 3, 8, 0, 7, 1, 3, 7, 3, 2, 2, 4, 3, 5,
                                  5, 6, 3, 3, 1, 2, 3, 0, 4, 0, 3, 3, 5, 5, 5,
                                  2, 3, 1, 5, 4, 5, 8, 1, 7, 2, 1, 0, 1, 8, 2,
                                  6, 7, 8, 4, 4, 7, 8, 4, 5, 8, 1, 1, 0, 7, 8,
                                  4, 2, 2, 8, 6, 5, 2, 4, 8, 4, 0, 4, 0, 2, 1,
                                  7, 1, 6}),
                         TensorValue({1, 2, 1, 1, 4}, dtype::Float32(),
                                     {1, 8, 5, 6, 2, 8, 7, 7}),
                         TensorValue({1, 2, 2, 2, 4}, dtype::Float32(),
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                         {}},
                Testcase{
                        {},
                        {},
                        {},
                        {},
                        TensorValue({1, 2, 2, 2, 4}, dtype::Float32(),
                                    {260, 342, 244, 241, 293, 385, 362, 257,
                                     278, 301, 303, 226, 273, 306, 318, 307,
                                     180, 244, 169, 156, 210, 244, 206, 167,
                                     126, 165, 156, 207, 191, 141, 209, 172})});
    }

    {
        // test normal conv

        ConvBias::Param param;
        param.format = ConvBias::Param::Format::NCHW44;
        param.sparse = ConvBias::Param::Sparse::DENSE;
        param.pad_h = 1;
        param.pad_w = 1;

        checker.set_param(param).exect(
                Testcase{TensorValue({1, 1, 2, 2, 4}, dtype::Int8(),
                                     {7, 2, 2, 1, 7, 5, 6, 3, 1, 2, 8, 3, 7, 7,
                                      6, 4}),
                         TensorValue(
                                 {1, 1, 3, 3, 4, 4}, dtype::Int8(),
                                 {3, 5, 5, 2, 0, 1, 4, 8, 3, 5, 0, 7, 1, 7, 0,
                                  7, 6, 4, 7, 7, 5, 2, 2, 4, 7, 6, 6, 3, 3, 2,
                                  2, 8, 5, 0, 4, 4, 0, 5, 1, 0, 0, 4, 8, 4, 7,
                                  7, 2, 0, 4, 8, 7, 3, 6, 2, 3, 0, 0, 6, 4, 4,
                                  1, 4, 3, 8, 8, 8, 7, 2, 2, 5, 5, 1, 3, 2, 8,
                                  1, 7, 0, 2, 7, 1, 6, 1, 5, 0, 6, 3, 0, 2, 4,
                                  1, 1, 4, 2, 7, 5, 7, 8, 4, 5, 5, 7, 0, 3, 3,
                                  2, 8, 6, 0, 1, 4, 6, 6, 6, 0, 1, 2, 4, 4, 1,
                                  1, 7, 8, 2, 5, 2, 8, 3, 8, 3, 5, 0, 6, 3, 4,
                                  3, 3, 7, 2, 8, 1, 1, 1, 4}),
                         TensorValue({1, 1, 1, 1, 4}, dtype::Int32(),
                                     {7, 2, 8, 1}),
                         TensorValue({1, 1, 2, 2, 4}, dtype::Int32(),
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0}),
                         {}},
                Testcase{
                        {},
                        {},
                        {},
                        {},
                        TensorValue({1, 1, 2, 2, 4}, dtype::Int32(),
                                    {264, 338, 309, 195, 276, 332, 390, 199,
                                     224, 268, 311, 218, 288, 311, 346, 277})});
    }
}

TEST_F(NAIVE, CONV_BIAS_NCHW44_DOT) {
    Checker<ConvBias> checker(handle(), /* check_dispatch */ false);
    ConvBias::Param param;
    param.format = ConvBias::Param::Format::NCHW44_DOT;

    size_t n = 1;
    size_t ic = 4;
    size_t oc = 8;
    size_t h = 2;
    size_t w = 2;
    size_t filter_size = 3;
    size_t pad = 1;
    auto src_tensor_shape = TensorShape{n, ic / 4, h, w, 4};
    auto weight_tensor_shape =
            TensorShape{oc / 4, ic / 4, filter_size, filter_size, 4, 4};
    auto bias_tensor_shape = TensorShape{1, oc / 4, 1, 1, 4};
    param.pad_h = pad;
    param.pad_w = pad;
    UniformIntRNG rng{-127, 127};
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_dtype(4, dtype::Float32())
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_epsilon(1e-3)
            .set_param(param)
            .execs({src_tensor_shape,
                    weight_tensor_shape,
                    bias_tensor_shape,
                    {},
                    {}});

    checker.set_dtype(0, dtype::QuantizedS8(2.f))
            .set_dtype(1, dtype::QuantizedS8(3.f))
            .set_dtype(2, dtype::QuantizedS32(6.f))
            .set_dtype(4, dtype::QuantizedS32(6.f))
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng)
            .set_epsilon(1e-3)
            .set_param(param)
            .execs({src_tensor_shape,
                    weight_tensor_shape,
                    bias_tensor_shape,
                    {},
                    {}});

    {
        // test normal conv
        ConvBias::Param param;
        param.format = ConvBias::Param::Format::NCHW44_DOT;
        param.sparse = ConvBias::Param::Sparse::DENSE;
        param.pad_h = 1;
        param.pad_w = 1;
        checker.set_param(param).exect(
                Testcase{TensorValue({1, 1, 2, 2, 4}, dtype::Float32(),
                                     {7, 2, 2, 1, 7, 5, 6, 3, 1, 2, 8, 3, 7, 7,
                                      6, 4}),
                         TensorValue(
                                 {1, 1, 3, 3, 4, 4}, dtype::Float32(),
                                 {3, 0, 3, 1, 5, 1, 5, 7, 5, 4, 0, 0, 2, 8, 7,
                                  7, 6, 5, 7, 3, 4, 2, 6, 2, 7, 2, 6, 2, 7, 4,
                                  3, 8, 5, 0, 0, 7, 0, 5, 4, 7, 4, 1, 8, 2, 4,
                                  0, 4, 0, 4, 6, 0, 1, 8, 2, 6, 4, 7, 3, 4, 3,
                                  3, 0, 4, 8, 8, 2, 3, 7, 8, 5, 2, 0, 7, 5, 8,
                                  2, 2, 1, 1, 7, 1, 0, 2, 4, 6, 6, 4, 2, 1, 3,
                                  1, 7, 5, 0, 1, 5, 7, 5, 3, 0, 8, 7, 2, 1, 4,
                                  0, 8, 4, 5, 3, 6, 6, 6, 2, 1, 5, 6, 4, 7, 2,
                                  0, 4, 8, 8, 1, 1, 2, 3, 8, 6, 3, 1, 3, 3, 7,
                                  1, 5, 4, 2, 1, 0, 3, 8, 4}),

                         TensorValue({1, 1, 1, 1, 4}, dtype::Float32(),
                                     {7, 2, 8, 1}),
                         TensorValue({1, 1, 2, 2, 4}, dtype::Float32(),
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0}),
                         {}},
                Testcase{
                        {},
                        {},
                        {},
                        {},
                        TensorValue({1, 1, 2, 2, 4}, dtype::Float32(),
                                    {264, 338, 309, 195, 276, 332, 390, 199,
                                     224, 268, 311, 218, 288, 311, 346, 277})});
    }

    {
        // test dw conv
        ConvBias::Param param;
        param.format = ConvBias::Param::Format::NCHW44_DOT;
        param.sparse = ConvBias::Param::Sparse::GROUP;
        param.pad_h = 1;
        param.pad_w = 1;
        checker.set_param(param).exect(
                Testcase{TensorValue({1, 1, 2, 2, 4}, dtype::Float32(),
                                     {5, 8, 3, 2, 4, 6, 1, 5, 0, 8, 2, 6, 8, 6,
                                      5, 7}),
                         TensorValue({1, 1, 1, 3, 3, 4}, dtype::Float32(),
                                     {3, 0, 3, 1, 6, 5, 7, 3, 5, 0, 0, 7,
                                      4, 6, 0, 1, 8, 2, 3, 7, 1, 0, 2, 4,
                                      7, 5, 3, 0, 6, 2, 1, 5, 8, 6, 3, 1}),
                         TensorValue({1, 1, 1, 1, 4}, dtype::Float32(),
                                     {4, 3, 5, 6}),
                         TensorValue({1, 1, 2, 2, 4}, dtype::Float32(),
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0}),
                         {}},
                Testcase{{},
                         {},
                         {},
                         {},
                         TensorValue({1, 1, 2, 2, 4}, dtype::Float32(),
                                     {112, 71, 33, 77, 104, 115, 19, 78, 62, 59,
                                      42, 117, 107, 93, 36, 78})});
    }

    {
        // test group conv
        ConvBias::Param param;
        param.format = ConvBias::Param::Format::NCHW44_DOT;
        param.sparse = ConvBias::Param::Sparse::GROUP;
        param.pad_h = 1;
        param.pad_w = 1;
        checker.set_param(param).exect(
                Testcase{TensorValue({1, 2, 2, 2, 4}, dtype::Float32(),
                                     {6, 3, 2, 7, 7, 6, 4, 5, 8, 6, 3,
                                      1, 1, 2, 8, 3, 1, 0, 6, 1, 3, 3,
                                      6, 0, 0, 5, 6, 7, 2, 2, 4, 4}),
                         TensorValue(
                                 {2, 1, 1, 3, 3, 4, 4}, dtype::Float32(),
                                 {3, 0, 3, 1, 5, 1, 5, 7, 5, 4, 0, 0, 2, 8, 7,
                                  7, 6, 5, 7, 3, 4, 2, 6, 2, 7, 2, 6, 2, 7, 4,
                                  3, 8, 5, 0, 0, 7, 0, 5, 4, 7, 4, 1, 8, 2, 4,
                                  0, 4, 0, 4, 6, 0, 1, 8, 2, 6, 4, 7, 3, 4, 3,
                                  3, 0, 4, 8, 8, 2, 3, 7, 8, 5, 2, 0, 7, 5, 8,
                                  2, 2, 1, 1, 7, 1, 0, 2, 4, 6, 6, 4, 2, 1, 3,
                                  1, 7, 5, 0, 1, 5, 7, 5, 3, 0, 8, 7, 2, 1, 4,
                                  0, 8, 4, 5, 3, 6, 6, 6, 2, 1, 5, 6, 4, 7, 2,
                                  0, 4, 8, 8, 1, 1, 2, 3, 8, 6, 3, 1, 3, 3, 7,
                                  1, 5, 4, 2, 1, 0, 3, 8, 4, 7, 6, 8, 3, 4, 8,
                                  1, 0, 5, 7, 3, 0, 0, 4, 5, 3, 7, 8, 1, 3, 7,
                                  1, 1, 0, 7, 2, 2, 0, 3, 0, 1, 1, 1, 6, 4, 0,
                                  3, 3, 1, 2, 0, 0, 4, 1, 5, 5, 7, 6, 7, 1, 3,
                                  5, 8, 6, 2, 1, 0, 7, 7, 1, 2, 6, 6, 1, 2, 3,
                                  1, 2, 4, 8, 3, 2, 6, 0, 7, 4, 3, 7, 3, 3, 5,
                                  3, 0, 3, 5, 1, 4, 5, 6, 2, 0, 5, 3, 3, 3, 5,
                                  2, 4, 7, 1, 3, 5, 2, 8, 1, 8, 1, 2, 5, 1, 0,
                                  6, 7, 7, 8, 7, 8, 8, 1, 8, 4, 4, 1, 4, 4, 5,
                                  0, 2, 2, 2, 0, 1, 8, 4, 4, 7, 6, 8, 0, 1, 5,
                                  4, 2, 6}),
                         TensorValue({1, 2, 1, 1, 4}, dtype::Float32(),
                                     {1, 8, 5, 6, 2, 8, 7, 7}),
                         TensorValue({1, 2, 2, 2, 4}, dtype::Float32(),
                                     {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
                         {}},
                Testcase{
                        {},
                        {},
                        {},
                        {},
                        TensorValue({1, 2, 2, 2, 4}, dtype::Float32(),
                                    {260, 342, 244, 241, 293, 385, 362, 257,
                                     278, 301, 303, 226, 273, 306, 318, 307,
                                     180, 244, 169, 156, 210, 244, 206, 167,
                                     126, 165, 156, 207, 191, 141, 209, 172})});
    }
}
// vim: syntax=cpp.doxygen
