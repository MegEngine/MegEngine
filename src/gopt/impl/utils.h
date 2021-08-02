/**
 * \file src/gopt/impl/utils.h
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
#include "megbrain/gopt/global_layout_transform.h"

namespace mgb {
namespace gopt {

static inline const char* opr_format_to_string(
        OprTensorFormatsConfiguration::OprFormat opr_format) {
    using OprFormat = OprTensorFormatsConfiguration::OprFormat;
#define cb(_fmt)          \
    case OprFormat::_fmt: \
        return #_fmt
    switch (opr_format) {
        cb(NCHW);
        cb(NHWC);
        cb(NCHW4);
        cb(NCHW32);
        cb(NCHW64);
        cb(CHWN4);
        default:
            mgb_assert(false, "Invalid opr format(got:%u)",
                       static_cast<uint32_t>(opr_format));
    }
#undef cb
}

static inline megdnn::NamedTensorShape tensor_formats_to_named_tensor_shape(
        TensorFormats format) {
    switch (format) {
        case TensorFormats::NCHW:
            return {{"N"}, {"C"}, {"H"}, {"W"}};
        case TensorFormats::NHWC:
            return {{"N"}, {"H"}, {"W"}, {"C"}};
        case TensorFormats::NCHWc4:
            return {{"N"}, {"C//4"}, {"H"}, {"W"}, {"C%4"}};
        case TensorFormats::NCHWc8:
            return {{"N"}, {"C//8"}, {"H"}, {"W"}, {"C%8"}};
        case TensorFormats::NCHWc32:
            return {{"N"}, {"C//32"}, {"H"}, {"W"}, {"C%32"}};
        case TensorFormats::NCHWc64:
            return {{"N"}, {"C//64"}, {"H"}, {"W"}, {"C%64"}};
        case TensorFormats::CHWNc4:
            return {{"C//4"}, {"H"}, {"W"}, {"N"}, {"C%4"}};
        case TensorFormats::NHCWc4:
            return {{"N"}, {"H"}, {"C//4"}, {"W"}, {"C%4"}};
        case TensorFormats::KRSCk4:
            return {{"K//4"}, {"R"}, {"S"}, {"C"}, {"K%4"}};
        case TensorFormats::GKRSCk4:
            return {{"G"}, {"K//4"}, {"R"}, {"S"}, {"C"}, {"K%4"}};
        case TensorFormats::C1RSc4:
            return {{"C//4"}, {"C%1"}, {"R"}, {"S"}, {"C%4"}};
        case TensorFormats::KRSCk4c4:
            return {{"K//4"}, {"R"}, {"S"}, {"C//4"}, {"K%4"}, {"C%4"}};
        case TensorFormats::GKRSCk4c4:
            return {{"G"}, {"K//4"}, {"R"}, {"S"}, {"C//4"}, {"K%4"}, {"C%4"}};
        case TensorFormats::KCRSk4c4:
            return {{"K//4"}, {"C//4"}, {"R"}, {"S"}, {"K%4"}, {"C%4"}};
        case TensorFormats::GKCRSk4c4:
            return {{"G"}, {"K//4"}, {"C//4"}, {"R"}, {"S"}, {"K%4"}, {"C%4"}};
        case TensorFormats::KCRSc4k4:
            return {{"K//4"}, {"C//4"}, {"R"}, {"S"}, {"C%4"}, {"K%4"}};
        case TensorFormats::GKCRSc4k4:
            return {{"G"}, {"K//4"}, {"C//4"}, {"R"}, {"S"}, {"C%4"}, {"K%4"}};
        case TensorFormats::C11RSc4:
            return {{"C//4"}, {"C%1"}, {"C%1"}, {"R"}, {"S"}, {"C%4"}};
        case TensorFormats::KCRSc8k8:
            return {{"K//8"}, {"C//8"}, {"R"}, {"S"}, {"C%8"}, {"K%8"}};
        case TensorFormats::GKCRSc8k8:
            return {{"G"}, {"K//8"}, {"C//8"}, {"R"}, {"S"}, {"C%8"}, {"K%8"}};
        case TensorFormats::C11RSc8:
            return {{"C//8"}, {"C%1"}, {"C%1"}, {"R"}, {"S"}, {"C%8"}};
        case TensorFormats::KRSCk8:
            return {{"K//8"}, {"R"}, {"S"}, {"C"}, {"K%8"}};
        case TensorFormats::KCRSc4:
            return {{"K"}, {"C//4"}, {"R"}, {"S"}, {"C%4"}};
        case TensorFormats::GKCRSc4:
            return {{"G"}, {"K"}, {"C//4"}, {"R"}, {"S"}, {"C%4"}};
        case TensorFormats::KCRS:
            return {{"K"}, {"C"}, {"R"}, {"S"}};
        case TensorFormats::GKCRS:
            return {{"G"}, {"K"}, {"C"}, {"R"}, {"S"}};
        case TensorFormats::C11RS:
            return {{"C"}, {"C%1"}, {"C%1"}, {"R"}, {"S"}};
        default:
            mgb_throw(AssertionError, "invalid tensor formats(%u)",
                      static_cast<uint32_t>(format));
    }
}

}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen
