/**
 * \file dnn/src/cuda/cudnn_wrapper_v8.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/basic_types.h"
#include "megdnn/oprs/nn.h"
#include "src/common/utils.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wreorder"
#include "cudnn_frontend.h"
#pragma GCC diagnostic pop

namespace megdnn {
namespace cuda {
static inline std::pair<int64_t, int64_t> get_vector_count_and_dimension(
        const param::Convolution::Format format) {
    using Format = param::Convolution::Format;
    int64_t vector_count = 1;
    int64_t vector_dimension = 1;
    switch (format) {
        case Format::NCHW:
            break;
        case Format::NHWC:
            vector_dimension = 3;
            break;
        case Format::NCHW4:
            vector_count = 4;
            break;
        case Format::NCHW32:
            vector_count = 32;
            break;
        default:
            megdnn_assert(
                    false, "unsupported format (got:%u) for cudnn",
                    static_cast<uint32_t>(format));
    }
    return {vector_count, vector_dimension};
}

template <typename Opr>
cudnn_frontend::ExecutionPlan* get_heuristic_plan_from_opr(
        const Opr* opr, const TensorLayout& x, const TensorLayout& y,
        const TensorLayout& w, const TensorLayout& b, const TensorLayout& z,
        const typename Opr::CanonizedFilterMeta& fm);

void run_single_conv_with_plan(
        const cudnnHandle_t& handle, const cudnn_frontend::ExecutionPlan& plan,
        const TensorND& x, const TensorND& y, const TensorND& w,
        const Workspace& workspace);

void run_conv_bias_act_with_plan(
        const cudnnHandle_t& handle, const cudnn_frontend::ExecutionPlan& plan,
        const TensorND& x, const TensorND& y, const TensorND& w, const TensorND& b,
        const TensorND& z, const Workspace& workspace);

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
