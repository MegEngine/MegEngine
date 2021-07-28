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
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/utils/arith_helper.h"

using namespace mgb;
using namespace gopt;
using NamedTensorShape = megdnn::NamedTensorShape;
using Dimension = megdnn::Dimension;

namespace {
int gcd(const int& p, const int& q) {
    int x = p, y = q;
    while (y != 0) {
        if (x < y) {
            y = (y % x);
        } else {
            x = (x % y);
            std::swap(x, y);
        }
    }
    return x;
}

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

ReformatManager::ReformatKey&
ReformatManager::ReformatKey::deduce_reformat_dtype_enum(const DType& dt) {
    static const ThinHashSet<std::pair<TensorFormats, TensorFormats>> set = {
            {TensorFormats::NCHW, TensorFormats::NCHWc64},
            {TensorFormats::NCHWc64, TensorFormats::NCHW},
            {TensorFormats::NCHW, TensorFormats::NHWC},
            {TensorFormats::NHWC, TensorFormats::NCHW}};
    if (set.count({input_format, output_format}) > 0 &&
        (dt.enumv() == DTypeEnum::QuantizedS4 ||
         dt.enumv() == DTypeEnum::Quantized4Asymm)) {
        input_dtype = output_dtype = dt.enumv();
    }
    return *this;
}

// =================== ReformatManager ====================*/
ReformatManager::ReformatManager() {
    using Attribute = ReformatKey::Attribute;
    {
        auto i = TensorFormats::NCHWc4, o = TensorFormats::CHWNc4;
        auto&& impl1 = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0],
                           megdnn::param::RelayoutFormat::Mode::NCHW4_CHWN4)
                    .node();
        };
        m_cache.emplace(ReformatKey{i, o}, impl1);
        auto&& impl2 = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0],
                           megdnn::param::RelayoutFormat::Mode::CHWN4_NCHW4)
                    .node();
        };
        m_cache.emplace(ReformatKey{o, i}, impl2);
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
        auto i = TensorFormats::KCRS, o = TensorFormats::KCRSc4;
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
                           megdnn::param::RelayoutFormat::Mode::NCHW64_NCHW)
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
                           megdnn::param::RelayoutFormat::Mode::NHWC_NCHW)
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

ReformatManager::ReformatImpl ReformatManager::get(
        const ReformatKey& key) const {
    using Attribute = ReformatKey::Attribute;
    MGB_TRY {
        auto find = m_cache.find(key);
        if (find != m_cache.end()) {
            auto rst = find->second;
            return rst;
        }
        mgb_assert(key.attribute == Attribute::DEFAULT);
        auto&& i = key.input_format;
        auto&& o = key.output_format;
        auto ishp = tensor_formats_to_named_tensor_shape(i);
        auto oshp = tensor_formats_to_named_tensor_shape(o);
        auto builder = std::get<0>(ReformatEmitter{ishp, oshp}.emit());
        return builder;
    }
    MGB_CATCH(std::exception & exc, {
        mgb_log_error(
                "cannot find ReformatImpl for ReformatKey(%s), extra "
                "message(%s)",
                key.to_string().c_str(), exc.what());
        throw;
    })
}

ReformatManager::ReformatImpl ReformatManager::auto_aligned_reformat_featrue(
        const VarNode* orig_var, TensorFormats orig_format,
        const ReformatKey& key) const {
    NamedTensorShape input_shape =
            tensor_formats_to_named_tensor_shape(key.input_format);
    NamedTensorShape output_shape =
            tensor_formats_to_named_tensor_shape(key.output_format);
    size_t input_alignment, output_alignment;
    size_t input_channel_idx, output_channel_idx;
    for (size_t i = 0; i < input_shape.ndim; ++i) {
        if (input_shape[i].name() == Dimension::Name::C &&
            input_shape[i].extent() == Dimension::UNDETERMINED_EXTENT) {
            input_channel_idx = i;
            input_alignment = input_shape[i].stride();
            break;
        }
    }
    for (size_t i = 0; i < output_shape.ndim; ++i) {
        if (output_shape[i].name() == Dimension::Name::C &&
            output_shape[i].extent() == Dimension::UNDETERMINED_EXTENT) {
            output_channel_idx = i;
            output_alignment = output_shape[i].stride();
            break;
        }
    }
    NamedTensorShape orig_shape =
            tensor_formats_to_named_tensor_shape(orig_format);
    size_t orig_channel = 0;
    for (size_t i = 0; i < orig_shape.ndim; ++i) {
        if (orig_shape[i].name() == Dimension::Name::C &&
            orig_shape[i].extent() == Dimension::UNDETERMINED_EXTENT) {
            orig_channel = orig_var->shape()[i] * orig_shape[i].stride();
            break;
        }
    }
    mgb_assert(orig_channel > 0,
               "incompatible NamedTensorShape for feature(got:%s)",
               orig_shape.to_string().c_str());
    size_t aligned_in_channel =
            divup(orig_channel, input_alignment) * input_alignment;
    size_t aligned_out_channel =
            divup(orig_channel, output_alignment) * output_alignment;
   size_t common_alignment = input_alignment * output_alignment /
                              gcd(input_alignment, output_alignment);
    size_t aligned_channel =
            divup(orig_channel, common_alignment) * common_alignment;
    auto builder = [key, aligned_channel, aligned_in_channel,
                    aligned_out_channel, input_shape, input_channel_idx,
                    output_shape,
                    output_channel_idx](const VarNodeArray& vars) {
        VarNode *x, *cur;
        x = cur = vars[0];
        if (aligned_channel > aligned_in_channel) {
            auto padding_shape = input_shape;
            auto&& dim = padding_shape[input_channel_idx];
            size_t const_extent =
                    (aligned_channel - aligned_in_channel) / dim.stride();
            padding_shape[input_channel_idx] =
                    Dimension(dim.name(), dim.stride(), const_extent);
            auto make_shape = std::get<0>(
                    MakeShapeEmitter{input_shape, padding_shape}.emit());
            auto padding_shp_var = make_shape({x});
            auto padding = std::get<0>(
                    PaddingEmitter{const_extent, input_channel_idx}.emit());
            cur = padding({cur, padding_shp_var});
        }
        cur = ReformatManager::instance().get(key)({cur});
        if (aligned_channel > aligned_out_channel) {
            auto&& dim = output_shape[output_channel_idx];
            size_t const_extent = aligned_out_channel / dim.stride();
            auto sub = std::get<0>(
                    SubtensorEmitter{const_extent, output_channel_idx}.emit());
            cur = sub({cur});
        }
        return cur;
    };
    return builder;
}

ReformatManager::ReformatImpl ReformatManager::auto_aligned_reformat_weight(
        const VarNode* orig_var, const ReformatKey& key,
        const AlignmentDesc& extra_alignment) const {
    size_t in_channels = 0, out_channels = 0;
    size_t input_channel_idx, output_channel_idx;
    Dimension::Name out_channel_name;
    auto input_shape = tensor_formats_to_named_tensor_shape(key.input_format);
    for (size_t i = 0; i < input_shape.ndim; ++i) {
        if (input_shape[i].name() == Dimension::Name::C &&
            input_shape[i].extent() == Dimension::UNDETERMINED_EXTENT) {
            in_channels = orig_var->shape()[i];
            input_channel_idx = i;
            mgb_assert(input_shape[i].stride() == 1,
                       "unsupport weight format(got:%s)",
                       input_shape.to_string().c_str());
        } else if ((input_shape[i].name() == Dimension::Name::K ||
                    input_shape[i].name() == Dimension::Name::N) &&
                   input_shape[i].extent() == Dimension::UNDETERMINED_EXTENT) {
            out_channels = orig_var->shape()[i];
            out_channel_name = input_shape[i].name();
            output_channel_idx = i;
            mgb_assert(input_shape[i].stride() == 1,
                       "unsupport weight format(got:%s)",
                       input_shape.to_string().c_str());
        }
    }
    size_t in_channel_alignment, out_channel_alignment = 1;
    auto output_shape = tensor_formats_to_named_tensor_shape(key.output_format);
    for (size_t i = 0; i < output_shape.ndim; ++i) {
        if (output_shape[i].name() == Dimension::Name::C &&
            output_shape[i].extent() == Dimension::UNDETERMINED_EXTENT) {
            in_channel_alignment = output_shape[i].stride();
        } else if (output_shape[i].name() == out_channel_name &&
                   output_shape[i].extent() == Dimension::UNDETERMINED_EXTENT) {
            out_channel_alignment = output_shape[i].stride();
        }
    }
    size_t aligned_in_channel =
            divup(in_channels, in_channel_alignment) * in_channel_alignment;
    if (extra_alignment.name == out_channel_name) {
        out_channel_alignment =
                extra_alignment.alignment * out_channel_alignment /
                gcd(extra_alignment.alignment, out_channel_alignment);
    }
    size_t aligned_out_channel =
            divup(out_channels, out_channel_alignment) * out_channel_alignment;
    auto builder = [key, input_shape, in_channels, input_channel_idx,
                    aligned_in_channel, out_channels, output_channel_idx,
                    aligned_out_channel](const VarNodeArray& vars) {
        VarNode *x, *cur;
        x = cur = vars[0];
        if (aligned_in_channel > in_channels) {
            auto padding_shape = input_shape;
            auto&& dim = padding_shape[input_channel_idx];
            size_t const_extent =
                    (aligned_in_channel - in_channels) / dim.stride();
            padding_shape[input_channel_idx] =
                    Dimension(dim.name(), dim.stride(), const_extent);
            auto make_shape = std::get<0>(
                    MakeShapeEmitter{input_shape, padding_shape}.emit());
            auto padding_shp_var = make_shape({x});
            auto padding = std::get<0>(
                    PaddingEmitter{const_extent, input_channel_idx}.emit());
            cur = padding({cur, padding_shp_var});
        }
        if (aligned_out_channel > out_channels) {
            auto padding_shape = input_shape;
            auto&& dim = padding_shape[output_channel_idx];
            size_t const_extent =
                    (aligned_out_channel - out_channels) / dim.stride();
            padding_shape[output_channel_idx] =
                    Dimension(dim.name(), dim.stride(), const_extent);
            auto make_shape = std::get<0>(
                    MakeShapeEmitter{input_shape, padding_shape}.emit());
            auto padding_shp_var = make_shape({cur});
            auto padding = std::get<0>(
                    PaddingEmitter{const_extent, output_channel_idx}.emit());
            cur = padding({cur, padding_shp_var});
        }
        cur = ReformatManager::instance().get(key)({cur});
        return cur;
    };
    return builder;
}

const ReformatManager& ReformatManager::instance() {
    static ReformatManager inst;
    return inst;
}
// vim: syntax=cpp.doxygen
