/**
 * \file dnn/src/fallback/conv_bias/winograd/strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/conv_bias/winograd/strategy.h"
#include "src/common/utils.h"
#include "src/common/winograd/winograd_helper.h"
#include "src/fallback/conv_bias/winograd/winograd.h"

namespace megdnn {
namespace fallback {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_2x3_1x1_f)

void winograd_2x3_1x1_f::filter(const float* filter,
                                float* filter_transform_buf,
                                float* transform_mid_buf, size_t OC, size_t IC,
                                size_t oc_start, size_t oc_end) {
    ::megdnn::winograd::StrategyHelper<float, float, float, float>::filter(
            filter, filter_transform_buf, transform_mid_buf, OC, IC, oc_start,
            oc_end, OUTPUT_BLOCK_SIZE, KERNEL_SIZE, {0, 1, -1}, filter_dtype);
}

void winograd_2x3_1x1_f::input(const float* input, float* input_transform_buf,
                               float* transform_mid_buf, size_t IH, size_t IW,
                               size_t IC, size_t PH, size_t PW,
                               size_t unit_start_idx, size_t nr_units_in_tile) {
    // OW = IW + 2 * PW - KERNEL_SIZE + 1
    auto units_w =
            div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
    rep(ic, IC) {
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            size_t nh = index / units_w;
            size_t nw = index % units_w;
            int ih_start = nh * OUTPUT_BLOCK_SIZE - PH;
            int iw_start = nw * OUTPUT_BLOCK_SIZE - PW;
            ::megdnn::winograd::StrategyHelper<float, float, float, float>::
                    input(input, input_transform_buf, transform_mid_buf,
                          ih_start, iw_start, IH, IW, IC, ic, unit_idx,
                          nr_units_in_tile, OUTPUT_BLOCK_SIZE, KERNEL_SIZE,
                          {0, 1, -1}, src_dtype);
        }
    }
}

void winograd_2x3_1x1_f::output(const float* output_transform_buf,
                                const float* bias, float* output,
                                float* transform_mid_buf, BiasMode bmode,
                                NonlineMode nonline_mode, size_t OH, size_t OW,
                                size_t oc_start, size_t oc_end,
                                size_t unit_start_idx,
                                size_t nr_units_in_tile) {
    auto units_w = div_ceil<size_t>(OW, OUTPUT_BLOCK_SIZE);
    size_t OC = oc_end - oc_start;

    for (size_t oc = oc_start; oc < oc_end; ++oc) {
        size_t oc_index = oc - oc_start;
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            auto nh = index / units_w;
            auto nw = index % units_w;
            size_t oh_start = nh * OUTPUT_BLOCK_SIZE;
            size_t ow_start = nw * OUTPUT_BLOCK_SIZE;
            ::megdnn::winograd::StrategyHelper<float, float, float, float>::
                    output(output_transform_buf, bias, output,
                           transform_mid_buf, bmode, nonline_mode, oh_start,
                           ow_start, OH, OW, OC, oc_start, oc_index, unit_idx,
                           nr_units_in_tile, OUTPUT_BLOCK_SIZE, KERNEL_SIZE,
                           {0, 1, -1}, dst_dtype);
        }
    }
}

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_2x3_4x4_f)

void winograd_2x3_4x4_f::filter(const float* filter,
                                float* filter_transform_buf,
                                float* transform_mid_buf, size_t OC, size_t IC,
                                size_t oc_start, size_t oc_end) {
    ::megdnn::winograd::StrategyHelper<
            float, float, float, float, param::ConvBias::Format::NCHW,
            param::MatrixMul::Format::MK4>::filter(filter, filter_transform_buf,
                                                   transform_mid_buf, OC, IC,
                                                   oc_start, oc_end,
                                                   OUTPUT_BLOCK_SIZE,
                                                   KERNEL_SIZE, {0, 1, -1},
                                                   filter_dtype);
}

void winograd_2x3_4x4_f::input(const float* input, float* input_transform_buf,
                               float* transform_mid_buf, size_t IH, size_t IW,
                               size_t IC, size_t PH, size_t PW,
                               size_t unit_start_idx, size_t nr_units_in_tile) {
    // OW = IW + 2 * PW - KERNEL_SIZE + 1
    auto units_w =
            div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
    rep(ic, IC) {
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            size_t nh = index / units_w;
            size_t nw = index % units_w;
            int ih_start = nh * OUTPUT_BLOCK_SIZE - PH;
            int iw_start = nw * OUTPUT_BLOCK_SIZE - PW;
            ::megdnn::winograd::StrategyHelper<
                    float, float, float, float, param::ConvBias::Format::NCHW,
                    param::MatrixMul::Format::MK4>::input(input,
                                                          input_transform_buf,
                                                          transform_mid_buf,
                                                          ih_start, iw_start,
                                                          IH, IW, IC, ic,
                                                          unit_idx,
                                                          nr_units_in_tile,
                                                          OUTPUT_BLOCK_SIZE,
                                                          KERNEL_SIZE,
                                                          {0, 1, -1},
                                                          src_dtype);
        }
    }
}

void winograd_2x3_4x4_f::output(const float* output_transform_buf,
                                const float* bias, float* output,
                                float* transform_mid_buf, BiasMode bmode,
                                NonlineMode nonline_mode, size_t OH, size_t OW,
                                size_t oc_start, size_t oc_end,
                                size_t unit_start_idx,
                                size_t nr_units_in_tile) {
    auto units_w = div_ceil<size_t>(OW, OUTPUT_BLOCK_SIZE);
    size_t OC = oc_end - oc_start;

    for (size_t oc = oc_start; oc < oc_end; ++oc) {
        size_t oc_index = oc - oc_start;
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            auto nh = index / units_w;
            auto nw = index % units_w;
            size_t oh_start = nh * OUTPUT_BLOCK_SIZE;
            size_t ow_start = nw * OUTPUT_BLOCK_SIZE;
            ::megdnn::winograd::StrategyHelper<
                    float, float, float, float, param::ConvBias::Format::NCHW,
                    param::MatrixMul::Format::MK4>::output(output_transform_buf,
                                                           bias, output,
                                                           transform_mid_buf,
                                                           bmode, nonline_mode,
                                                           oh_start, ow_start,
                                                           OH, OW, OC, oc_start,
                                                           oc_index, unit_idx,
                                                           nr_units_in_tile,
                                                           OUTPUT_BLOCK_SIZE,
                                                           KERNEL_SIZE,
                                                           {0, 1, -1},
                                                           dst_dtype);
        }
    }
}

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_2x3_1x1_qs8)

void winograd_2x3_1x1_qs8::filter(const int8_t* filter,
                                  int16_t* filter_transform_buf,
                                  int16_t* transform_mid_buf, size_t OC,
                                  size_t IC, size_t oc_start, size_t oc_end) {
    ::megdnn::winograd::StrategyHelper<int8_t, int8_t, int16_t, int>::filter(
            filter, filter_transform_buf, transform_mid_buf, OC, IC, oc_start,
            oc_end, OUTPUT_BLOCK_SIZE, KERNEL_SIZE, {0, 1, -1}, filter_dtype,
            2.0f);
}

void winograd_2x3_1x1_qs8::input(const int8_t* input,
                                 int16_t* input_transform_buf,
                                 int16_t* transform_mid_buf, size_t IH,
                                 size_t IW, size_t IC, size_t PH, size_t PW,
                                 size_t unit_start_idx,
                                 size_t nr_units_in_tile) {
    // OW = IW + 2 * PW - KERNEL_SIZE + 1
    auto units_w =
            div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
    rep(ic, IC) {
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            size_t nh = index / units_w;
            size_t nw = index % units_w;
            int ih_start = nh * OUTPUT_BLOCK_SIZE - PH;
            int iw_start = nw * OUTPUT_BLOCK_SIZE - PW;
            ::megdnn::winograd::StrategyHelper<int8_t, int8_t, int16_t, int>::
                    input(input, input_transform_buf, transform_mid_buf,
                          ih_start, iw_start, IH, IW, IC, ic, unit_idx,
                          nr_units_in_tile, OUTPUT_BLOCK_SIZE, KERNEL_SIZE,
                          {0, 1, -1}, src_dtype, 1.0f);
        }
    }
}

void winograd_2x3_1x1_qs8::output(const int* output_transform_buf,
                                  const int* bias, int8_t* output,
                                  int* transform_mid_buf, BiasMode bmode,
                                  NonlineMode nonline_mode, size_t OH,
                                  size_t OW, size_t oc_start, size_t oc_end,
                                  size_t unit_start_idx,
                                  size_t nr_units_in_tile) {
    float scale_input = src_dtype.param<dtype::QuantizedS8>().scale;
    float scale_filter = filter_dtype.param<dtype::QuantizedS8>().scale;

    auto units_w = div_ceil<size_t>(OW, OUTPUT_BLOCK_SIZE);
    size_t OC = oc_end - oc_start;

    for (size_t oc = oc_start; oc < oc_end; ++oc) {
        size_t oc_index = oc - oc_start;
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            auto nh = index / units_w;
            auto nw = index % units_w;
            size_t oh_start = nh * OUTPUT_BLOCK_SIZE;
            size_t ow_start = nw * OUTPUT_BLOCK_SIZE;
            ::megdnn::winograd::StrategyHelper<int8_t, int8_t, int16_t, int>::
                    output(output_transform_buf, bias, output,
                           transform_mid_buf, bmode, nonline_mode, oh_start,
                           ow_start, OH, OW, OC, oc_start, oc_index, unit_idx,
                           nr_units_in_tile, OUTPUT_BLOCK_SIZE, KERNEL_SIZE,
                           {0, 1, -1}, dst_dtype, scale_input * scale_filter,
                           2.0f, 1.0f);
        }
    }
}

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_2x3_8x8_qs8)

void winograd_2x3_8x8_qs8::filter(const int8_t* filter,
                                  int16_t* filter_transform_buf,
                                  int16_t* transform_mid_buf, size_t OC,
                                  size_t IC, size_t oc_start, size_t oc_end) {
    ::megdnn::winograd::StrategyHelper<
            int8_t, int8_t, int16_t, int, param::ConvBias::Format::NCHW,
            param::MatrixMul::Format::MK8>::filter(filter, filter_transform_buf,
                                                   transform_mid_buf, OC, IC,
                                                   oc_start, oc_end,
                                                   OUTPUT_BLOCK_SIZE,
                                                   KERNEL_SIZE, {0, 1, -1},
                                                   filter_dtype, 2.0f);
}

void winograd_2x3_8x8_qs8::input(const int8_t* input,
                                 int16_t* input_transform_buf,
                                 int16_t* transform_mid_buf, size_t IH,
                                 size_t IW, size_t IC, size_t PH, size_t PW,
                                 size_t unit_start_idx,
                                 size_t nr_units_in_tile) {
    // OW = IW + 2 * PW - KERNEL_SIZE + 1
    auto units_w =
            div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
    rep(ic, IC) {
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            size_t nh = index / units_w;
            size_t nw = index % units_w;
            int ih_start = nh * OUTPUT_BLOCK_SIZE - PH;
            int iw_start = nw * OUTPUT_BLOCK_SIZE - PW;
            ::megdnn::winograd::StrategyHelper<
                    int8_t, int8_t, int16_t, int, param::ConvBias::Format::NCHW,
                    param::MatrixMul::Format::MK8>::input(input,
                                                          input_transform_buf,
                                                          transform_mid_buf,
                                                          ih_start, iw_start,
                                                          IH, IW, IC, ic,
                                                          unit_idx,
                                                          nr_units_in_tile,
                                                          OUTPUT_BLOCK_SIZE,
                                                          KERNEL_SIZE,
                                                          {0, 1, -1}, src_dtype,
                                                          1.0f);
        }
    }
}

void winograd_2x3_8x8_qs8::output(const int* output_transform_buf,
                                  const int* bias, int8_t* output,
                                  int* transform_mid_buf, BiasMode bmode,
                                  NonlineMode nonline_mode, size_t OH,
                                  size_t OW, size_t oc_start, size_t oc_end,
                                  size_t unit_start_idx,
                                  size_t nr_units_in_tile) {
    float scale_input = src_dtype.param<dtype::QuantizedS8>().scale;
    float scale_filter = 0.f;
    if (filter_dtype.enumv() == DTypeEnum::QuantizedS8) {
        scale_filter = filter_dtype.param<dtype::QuantizedS8>().scale;
    } else {
        megdnn_assert(filter_dtype.enumv() == DTypeEnum::QuantizedS16);
        scale_filter = filter_dtype.param<dtype::QuantizedS16>().scale;
    }

    auto units_w = div_ceil<size_t>(OW, OUTPUT_BLOCK_SIZE);
    size_t OC = oc_end - oc_start;

    for (size_t oc = oc_start; oc < oc_end; ++oc) {
        size_t oc_index = oc - oc_start;
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            auto nh = index / units_w;
            auto nw = index % units_w;
            size_t oh_start = nh * OUTPUT_BLOCK_SIZE;
            size_t ow_start = nw * OUTPUT_BLOCK_SIZE;
            ::megdnn::winograd::StrategyHelper<
                    int8_t, int8_t, int16_t, int, param::ConvBias::Format::NCHW,
                    param::MatrixMul::Format::MK8>::output(output_transform_buf,
                                                           bias, output,
                                                           transform_mid_buf,
                                                           bmode, nonline_mode,
                                                           oh_start, ow_start,
                                                           OH, OW, OC, oc_start,
                                                           oc_index, unit_idx,
                                                           nr_units_in_tile,
                                                           OUTPUT_BLOCK_SIZE,
                                                           KERNEL_SIZE,
                                                           {0, 1, -1},
                                                           dst_dtype,
                                                           scale_input *
                                                                   scale_filter,
                                                           2.0f, 1.0f);
        }
    }
}

}  // namespace winograd
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
