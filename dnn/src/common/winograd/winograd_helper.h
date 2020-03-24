/**
 * \file dnn/src/common/winograd/winograd_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <vector>
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"

namespace megdnn {
namespace winograd {

using NonlineMode = ::megdnn::ConvBias::Param::NonlineMode;
using BiasMode = ConvBiasForward::BiasMode;
/**
 * \brief Strategy helper, contains some helper function for debug kernel
 * implementation
 *
 * \warning The layout should be NCHW
 */
template <typename ctype, typename dst_type, typename input_filter_compute_type,
          typename output_compute_type,
          param::ConvBias::Format layout = param::ConvBias::Format::NCHW,
          param::MatrixMul::Format format = param::MatrixMul::Format::DEFAULT>
class StrategyHelper {
public:
    static void filter(const ctype* filter,
                       input_filter_compute_type* filter_transform_buf,
                       input_filter_compute_type* transform_mid_buf, size_t OC,
                       size_t IC, size_t oc_start, size_t oc_end, size_t m,
                       size_t r, const std::vector<float>& interp_points,
                       DType dtype, float rescale = 1.0f);

    static void input(const ctype* input,
                      input_filter_compute_type* input_transform_buf,
                      input_filter_compute_type* transform_mid_buf,
                      int ih_start, int iw_start, size_t IH, size_t IW,
                      size_t IC, size_t ic, size_t unit_idx,
                      size_t nr_units_in_tile, size_t m, size_t r,
                      const std::vector<float>& interp_points, DType dtype,
                      float rescale = 1.0f);

    static void
    output(const output_compute_type* output_transform_buf,
           const output_compute_type* bias, dst_type* output,
           output_compute_type* transform_mid_buf, BiasMode bmode,
           NonlineMode nonline_mode, size_t oh_start, size_t ow_start,
           size_t OH, size_t OW, size_t OC, size_t oc_start, size_t oc_index,
           size_t unit_idx, size_t nr_units_in_tile, size_t m, size_t r,
           const std::vector<float>& interp_points, DType dtype,
           float input_filter_scale = 1.0f,    // input_scale * filter_scale
           float input_filter_rescale = 1.0f,  // input_rescale * filter_rescale
           float rescale = 1.0f);
};

}  // namespace winograd
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
