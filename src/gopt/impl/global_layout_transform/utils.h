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
#include "megbrain/gopt/layout_transform_context.h"

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
        cb(NCHW44);
        cb(NCHW88);
        cb(NCHW44_DOT);
        default:
            mgb_assert(
                    false, "Invalid opr format(got:%u)",
                    static_cast<uint32_t>(opr_format));
    }
#undef cb
}

static inline const char* config_id_to_string(
        OprTensorFormatsConfiguration::OprFormatConfigID config_id) {
    using OprFormatConfigID = OprTensorFormatsConfiguration::OprFormatConfigID;
#define cb(_fmt)                  \
    case OprFormatConfigID::_fmt: \
        return #_fmt
    switch (config_id) {
        cb(NCHW);
        cb(NHWC);
        cb(NCHW4);
        cb(NCHW8);
        cb(NCHW4_NCHW32);
        cb(NCHW4_NCHW);
        cb(NCHW32);
        cb(NCHW32_NCHW4);
        cb(NCHW64);
        cb(CHWN4);
        cb(NCHW44);
        cb(NCHW44_HYBRID);
        cb(NCHW88);
        cb(NCHW88_HYBRID);
        cb(NCHW44_DOT);
        cb(NCHW44_DOT_HYBRID);
        default:
            mgb_assert(
                    false, "Invalid config id(got:%u)",
                    static_cast<uint32_t>(config_id));
    }
#undef cb
}

static inline TensorFormats opr_format_to_tensor_formats(
        OprTensorFormatsConfiguration::OprFormat opr_format) {
    using OprFormat = OprTensorFormatsConfiguration::OprFormat;
    switch (opr_format) {
        case OprFormat::NCHW:
            return TensorFormats::NCHW;
        case OprFormat::NHWC:
            return TensorFormats::NHWC;
        case OprFormat::NCHW4:
            return TensorFormats::NCHWc4;
        case OprFormat::NCHW32:
            return TensorFormats::NCHWc32;
        case OprFormat::NCHW64:
            return TensorFormats::NCHWc64;
        case OprFormat::CHWN4:
            return TensorFormats::CHWNc4;
        case OprFormat::NCHW88:
            return TensorFormats::NCHWc8;
        case OprFormat::NCHW44:
            return TensorFormats::NCHWc4;
        case OprFormat::NCHW8:
            return TensorFormats::NCHWc8;
        default:
            mgb_throw(
                    AssertionError, "format(%s) is not supported",
                    opr_format_to_string(opr_format));
    };
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
        case TensorFormats::KRSC:
            return {{"K"}, {"R"}, {"S"}, {"C"}};
        case TensorFormats::KCRSc32:
            return {{"K"}, {"C//32"}, {"R"}, {"S"}, {"C%32"}};
        case TensorFormats::KCRSc64:
            return {{"K"}, {"C//64"}, {"R"}, {"S"}, {"C%64"}};
        case TensorFormats::CRSKc4:
            return {{"C//4"}, {"R"}, {"S"}, {"K"}, {"C%4"}};
        default:
            mgb_throw(
                    MegBrainError, "invalid tensor formats(%u)",
                    static_cast<uint32_t>(format));
    }
}

}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen
