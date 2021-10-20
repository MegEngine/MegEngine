/**
 * \file src/gopt/impl/opr_format_modifier.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once
#include "megbrain/graph.h"
#include "megbrain/opr/dnn/convolution.h"

namespace mgb {
namespace gopt {
enum class TensorFormats : uint32_t;
namespace intl {

#define FOREACH_FORMAT_AWARE_OPR(cb)                                                   \
    cb(Convolution) cb(ConvBiasForward) cb(ConvolutionBackwardData) cb(PoolingForward) \
            cb(WarpPerspective) cb(Resize)

#define FOREACH_MODIFY_OPR_FORMAT_OPR(cb) FOREACH_FORMAT_AWARE_OPR(cb) cb(Concat)

bool has_available_algo(const VarNodeArray& i, const cg::OperatorNodeBase* opr);

struct OprFormatInfo {
    opr::Convolution::Param::Format opr_format;
    struct TensorFormatsInfo {
        TensorFormats from;
        TensorFormats to;
    };
    TensorFormatsInfo tensor_formats;
};

VarNode* modify_opr_format(
        OprFormatInfo opr_format, const VarNodeArray& i,
        const cg::OperatorNodeBase* opr);

bool has_opr_format(const cg::OperatorNodeBase* opr);

bool has_opr_format_modifier(const cg::OperatorNodeBase* opr);

bool allow_aligned_layout(const cg::OperatorNodeBase* opr);

}  // namespace intl
}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen
