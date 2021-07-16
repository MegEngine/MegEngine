/**
 * \file src/gopt/impl/reformat_manager.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/gopt/reformat_manager.h"
#include <numeric>
#include "megbrain/opr/tensor_manip.h"

using namespace mgb;
using namespace gopt;
using NamedTensorShape = megdnn::NamedTensorShape;

namespace {
NamedTensorShape tensor_formats_to_named_tensor_shape(TensorFormats format) {
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
};  // namespace

// =================== ReformatManager::ReformatKey ====================*/
std::string ReformatManager::ReformatKey::to_string() const {
    auto&& i = tensor_formats_to_named_tensor_shape(input_format);
    auto&& o = tensor_formats_to_named_tensor_shape(output_format);
    std::string input_name, output_name;

#define cb(_name)                          \
    if (input_dtype == DTypeEnum::_name) { \
        input_name = #_name;               \
    } else
    MEGDNN_FOREACH_DTYPE_NAME(cb)
    MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb) {
        mgb_throw(MegBrainError, "invalid input dtype enum(%u)",
                  static_cast<uint32_t>(input_dtype));
    }
#undef cb
#define cb(_name)                           \
    if (output_dtype == DTypeEnum::_name) { \
        output_name = #_name;               \
    } else
    MEGDNN_FOREACH_DTYPE_NAME(cb)
    MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb) {
        mgb_throw(MegBrainError, "invalid output dtype enum(%u)",
                  static_cast<uint32_t>(output_dtype));
    }
#undef cb
    return ssprintf("%s;%s;%s;%s;%s", i.to_string().c_str(),
                    o.to_string().c_str(),
                    std::to_string(static_cast<uint32_t>(attribute)).c_str(),
                    input_name.c_str(), output_name.c_str());
}

size_t ReformatManager::ReformatKey::Hash::operator()(
        const ReformatKey& key) const {
    auto enumhash = mgb::enumhash();
    size_t h = enumhash(key.input_format);
    h = mgb::hash_pair_combine(h, enumhash(key.output_format));
    h = mgb::hash_pair_combine(h, enumhash(key.attribute));
    h = mgb::hash_pair_combine(h, enumhash(key.input_dtype));
    h = mgb::hash_pair_combine(h, enumhash(key.output_dtype));
    return h;
}

bool ReformatManager::ReformatKey::Equal::operator()(
        const ReformatKey& lhs, const ReformatKey& rhs) const {
    return lhs.input_format == rhs.input_format &&
           lhs.output_format == rhs.output_format &&
           lhs.input_dtype == rhs.input_dtype &&
           lhs.output_dtype == rhs.output_dtype &&
           lhs.attribute == rhs.attribute;
}

// =================== ReformatManager ====================*/
#define FOREACH_FEATURE_TENSOR_FORMATS(cb)                                     \
    cb(NCHW) cb(NHWC) cb(NCHWc4) cb(NCHWc8) cb(NCHWc32) cb(NCHWc64) cb(CHWNc4) \
            cb(NHCWc4)
#define FOREACH_WEIGHT_TENSOR_FORMATS(cb)                                     \
    cb(KRSCk4) cb(KRSCk4c4) cb(KCRSk4c4) cb(KCRSc4k4) cb(KCRSc8k8) cb(KRSCk8) \
            cb(GKRSCk4) cb(GKRSCk4c4) cb(GKCRSc4k4) cb(GKCRSk4c4)             \
                    cb(GKCRSc8k8) cb(C11RSc4) cb(C11RSc8)
ReformatManager::ReformatManager() {
    static constexpr TensorFormats feature_tensor_formats[] = {
#define cb(_fmt) TensorFormats::_fmt,
            FOREACH_FEATURE_TENSOR_FORMATS(cb)
#undef cb
    };
    static constexpr int nr_feature_tensor_formats =
            sizeof(feature_tensor_formats) / sizeof(TensorFormats);
    for (int i = 0; i < nr_feature_tensor_formats; ++i) {
        for (int o = 0; o < nr_feature_tensor_formats; ++o) {
            if (i == o)
                continue;
            NamedTensorShape input_shape = tensor_formats_to_named_tensor_shape(
                    feature_tensor_formats[i]);
            NamedTensorShape output_shape =
                    tensor_formats_to_named_tensor_shape(
                            feature_tensor_formats[o]);
            auto impl = std::get<0>(
                    ReformatEmitter{input_shape, output_shape}.emit());
            m_cache.emplace(ReformatKey{feature_tensor_formats[i],
                                        feature_tensor_formats[o]},
                            impl);
        }
    }
    static constexpr TensorFormats default_weight_tensor_formats =
            TensorFormats::KCRS;
    static constexpr TensorFormats default_group_conv_weight_tensor_formats =
            TensorFormats::GKCRS;
    static constexpr TensorFormats default_chan_conv_weight_tensor_formats =
            TensorFormats::C11RS;
    static constexpr TensorFormats weight_tensor_formats[] = {
#define cb(_fmt) TensorFormats::_fmt,
            FOREACH_WEIGHT_TENSOR_FORMATS(cb)
#undef cb
    };
    static constexpr int nr_weight_tensor_formats =
            sizeof(weight_tensor_formats) / sizeof(TensorFormats);
    using Name = megdnn::Dimension::Name;
    for (int o = 0; o < nr_weight_tensor_formats; ++o) {
        NamedTensorShape output_shape =
                tensor_formats_to_named_tensor_shape(weight_tensor_formats[o]);
        TensorFormats input_format;
        if (output_shape[0].name() == Name::G) {
            input_format = default_group_conv_weight_tensor_formats;
        } else if (output_shape[0].name() == Name::C) {
            input_format = default_chan_conv_weight_tensor_formats;
        } else {
            mgb_assert(output_shape[0].name() == Name::K);
            input_format = default_weight_tensor_formats;
        }
        NamedTensorShape input_shape =
                tensor_formats_to_named_tensor_shape(input_format);
        auto impl =
                std::get<0>(ReformatEmitter{input_shape, output_shape}.emit());
        m_cache.emplace(ReformatKey{input_format, weight_tensor_formats[o]},
                        impl);
    }
    {
        auto i = TensorFormats::NCHW, o = TensorFormats::NCHWc4;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(vars[0],
                                             megdnn::param::RelayoutFormat::
                                                     Mode::NCHW_NCHW4_IC_SMALL)
                    .node();
        };
        m_cache.emplace(ReformatKey{i, o, Attribute::IC_SMALL}, impl);
    }
    {
        auto i = TensorFormats::KCRS, o = TensorFormats::KCRSc4k4;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0],
                           megdnn::param::RelayoutFormat::Mode::
                                   NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT)
                    .node();
        };
        m_cache.emplace(ReformatKey{i, o, Attribute::IC_SMALL}, impl);
    }
    {
        auto i = TensorFormats::NCHW, o = TensorFormats::NCHWc64;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0],
                           megdnn::param::RelayoutFormat::Mode::NCHW_NCHW64)
                    .node();
        };
        m_cache.emplace(
                ReformatKey{i, o, Attribute::DEFAULT, DTypeEnum::QuantizedS4,
                            DTypeEnum::QuantizedS4},
                impl);
        m_cache.emplace(ReformatKey{i, o, Attribute::DEFAULT,
                                    DTypeEnum::Quantized4Asymm,
                                    DTypeEnum::Quantized4Asymm},
                        impl);
    }
    {
        auto i = TensorFormats::NCHWc64, o = TensorFormats::NCHW;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0],
                           megdnn::param::RelayoutFormat::Mode::NCHW_NCHW64)
                    .node();
        };
        m_cache.emplace(
                ReformatKey{i, o, Attribute::DEFAULT, DTypeEnum::QuantizedS4,
                            DTypeEnum::QuantizedS4},
                impl);
        m_cache.emplace(ReformatKey{i, o, Attribute::DEFAULT,
                                    DTypeEnum::Quantized4Asymm,
                                    DTypeEnum::Quantized4Asymm},
                        impl);
    }
    {
        auto i = TensorFormats::NCHW, o = TensorFormats::NHWC;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0],
                           megdnn::param::RelayoutFormat::Mode::NCHW_NHWC)
                    .node();
        };
        m_cache.emplace(
                ReformatKey{i, o, Attribute::DEFAULT, DTypeEnum::QuantizedS4,
                            DTypeEnum::QuantizedS4},
                impl);
        m_cache.emplace(ReformatKey{i, o, Attribute::DEFAULT,
                                    DTypeEnum::Quantized4Asymm,
                                    DTypeEnum::Quantized4Asymm},
                        impl);
    }
    {
        auto i = TensorFormats::NHWC, o = TensorFormats::NCHW;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0],
                           megdnn::param::RelayoutFormat::Mode::NCHW_NHWC)
                    .node();
        };
        m_cache.emplace(
                ReformatKey{i, o, Attribute::DEFAULT, DTypeEnum::QuantizedS4,
                            DTypeEnum::QuantizedS4},
                impl);
        m_cache.emplace(ReformatKey{i, o, Attribute::DEFAULT,
                                    DTypeEnum::Quantized4Asymm,
                                    DTypeEnum::Quantized4Asymm},
                        impl);
    }
    // nhcw4
    {
        auto i = TensorFormats::KCRS, o = TensorFormats::KRSCk4;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(vars[0],
                                             megdnn::param::RelayoutFormat::
                                                     Mode::INTER_WEIGHT_DENSEI)
                    .node();
        };
        m_cache.emplace(ReformatKey{i, o, Attribute::IMAGE2D}, impl);
    }
    {
        auto i = TensorFormats::KCRS, o = TensorFormats::GKRSCk4;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(vars[0],
                                             megdnn::param::RelayoutFormat::
                                                     Mode::INTER_WEIGHT_GROUPI)
                    .node();
        };
        m_cache.emplace(ReformatKey{i, o, Attribute::IMAGE2D}, impl);
    }
    {
        auto i = TensorFormats::KCRS, o = TensorFormats::C1RSc4;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(vars[0],
                                             megdnn::param::RelayoutFormat::
                                                     Mode::INTER_WEIGHT_CHANI)
                    .node();
        };
        m_cache.emplace(ReformatKey{i, o, Attribute::IMAGE2D}, impl);
    }
    {
        auto i = TensorFormats::NCHW, o = TensorFormats::NHCWc4;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0],
                           megdnn::param::RelayoutFormat::Mode::NCHW_NHWCD4I)
                    .node();
        };
        m_cache.emplace(ReformatKey{i, o, Attribute::IMAGE2D}, impl);
    }
    {
        auto i = TensorFormats::NHCWc4, o = TensorFormats::NCHW;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0],
                           megdnn::param::RelayoutFormat::Mode::NCHW_NHWCD4I)
                    .node();
        };
        m_cache.emplace(ReformatKey{i, o, Attribute::IMAGE2D}, impl);
    }
    // nhcw4-dot
    {
        auto i = TensorFormats::KCRS, o = TensorFormats::KRSCk4c4;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0], megdnn::param::RelayoutFormat::Mode::
                                            INTER_WEIGHT_DENSEI_DOT)
                    .node();
        };
        m_cache.emplace(
                ReformatKey{i, o, Attribute::IMAGE2D, DTypeEnum::QuantizedS8,
                            DTypeEnum::QuantizedS8},
                impl);
        m_cache.emplace(ReformatKey{i, o, Attribute::IMAGE2D,
                                    DTypeEnum::Quantized8Asymm,
                                    DTypeEnum::Quantized8Asymm},
                        impl);
    }
    {
        auto i = TensorFormats::GKCRS, o = TensorFormats::GKRSCk4c4;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0], megdnn::param::RelayoutFormat::Mode::
                                            INTER_WEIGHT_GROUPI_DOT)
                    .node();
        };
        m_cache.emplace(
                ReformatKey{i, o, Attribute::IMAGE2D, DTypeEnum::QuantizedS8,
                            DTypeEnum::QuantizedS8},
                impl);
        m_cache.emplace(ReformatKey{i, o, Attribute::IMAGE2D,
                                    DTypeEnum::Quantized8Asymm,
                                    DTypeEnum::Quantized8Asymm},
                        impl);
    }
}
#undef FOREACH_FEATURE_TENSOR_FORMATS
#undef FOREACH_WEIGHT_TENSOR_FORMATS

const ReformatManager::ReformatImpl& ReformatManager::get(
        const ReformatKey& key) const {
    MGB_TRY {
        auto&& impl = m_cache.at(key);
        return impl;
    }
    MGB_CATCH(std::exception & exc, {
        mgb_log_error(
                "cannot find ReformatImpl for ReformatKey(%s), extra "
                "message(%s)",
                key.to_string().c_str(), exc.what());
        throw;
    })
}

const ReformatManager& ReformatManager::instance() {
    static ReformatManager* inst = nullptr;
    if (inst == nullptr) {
        inst = new ReformatManager();
    }
    return *inst;
}
// vim: syntax=cpp.doxygen
