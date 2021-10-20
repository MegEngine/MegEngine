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
#include "./utils.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/utils/arith_helper.h"

using namespace mgb;
using namespace gopt;
using NamedTensorShape = megdnn::NamedTensorShape;
using Dimension = megdnn::Dimension;

namespace {
static inline int gcd(const int& p, const int& q) {
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

static inline size_t extra_alignment(
        ReformatManager::ReformatKey::Attribute attr, TensorFormats target_formats,
        DType dt, size_t channel_alignment) {
    using Attribute = ReformatManager::ReformatKey::Attribute;
    if (attr & Attribute::AUTO_PADDING_NHWC) {
        constexpr size_t alignment_in_bits = 32;
        size_t dtype_bits = dt.is_low_bit() ? dt.low_bit() : dt.size(1) * 8;
        size_t extra_alignment =
                alignment_in_bits >= dtype_bits ? alignment_in_bits / dtype_bits : 1;
        if (target_formats == TensorFormats::NHWC ||
            target_formats == TensorFormats::KRSC)
            channel_alignment = extra_alignment * channel_alignment /
                                gcd(channel_alignment, extra_alignment);
        return channel_alignment;
    }
    return channel_alignment;
}

static inline std::tuple<size_t, size_t> extra_alignment(
        const ReformatManager::ReformatKey& key, DType dt,
        size_t input_channel_alignment, size_t output_channel_alignment) {
    using Attribute = ReformatManager::ReformatKey::Attribute;
    if (key.attribute & Attribute::AUTO_PADDING_NHWC) {
        constexpr size_t alignment_in_bits = 32;
        size_t dtype_bits = dt.is_low_bit() ? dt.low_bit() : dt.size(1) * 8;
        size_t extra_alignment =
                alignment_in_bits >= dtype_bits ? alignment_in_bits / dtype_bits : 1;
        if (key.input_format == TensorFormats::NHWC ||
            key.input_format == TensorFormats::KRSC)
            input_channel_alignment = input_channel_alignment * extra_alignment /
                                      gcd(input_channel_alignment, extra_alignment);
        if (key.output_format == TensorFormats::NHWC ||
            key.output_format == TensorFormats::KRSC)
            output_channel_alignment = output_channel_alignment * extra_alignment /
                                       gcd(output_channel_alignment, extra_alignment);
        return std::make_tuple(input_channel_alignment, output_channel_alignment);
    }
    return std::make_tuple(input_channel_alignment, output_channel_alignment);
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
        mgb_throw(
                MegBrainError, "invalid input dtype enum(%u)",
                static_cast<uint32_t>(input_dtype));
    }
#undef cb
#define cb(_name)                           \
    if (output_dtype == DTypeEnum::_name) { \
        output_name = #_name;               \
    } else
    MEGDNN_FOREACH_DTYPE_NAME(cb)
    MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb) {
        mgb_throw(
                MegBrainError, "invalid output dtype enum(%u)",
                static_cast<uint32_t>(output_dtype));
    }
#undef cb
    return ssprintf(
            "%s;%s;%s;%s;%s", i.to_string().c_str(), o.to_string().c_str(),
            std::to_string(static_cast<uint32_t>(attribute)).c_str(),
            input_name.c_str(), output_name.c_str());
}

size_t ReformatManager::ReformatKey::Hash::operator()(const ReformatKey& key) const {
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
           lhs.input_dtype == rhs.input_dtype && lhs.output_dtype == rhs.output_dtype &&
           lhs.attribute == rhs.attribute;
}

// =================== ReformatManager ====================*/
ReformatManager::ReformatManager() {
    using Attribute = ReformatKey::Attribute;
    {
        auto i = TensorFormats::NCHWc4, o = TensorFormats::CHWNc4;
        auto&& impl1 = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0], megdnn::param::RelayoutFormat::Mode::NCHW4_CHWN4)
                    .node();
        };
        m_cache.emplace(ReformatKey{i, o}, impl1);
        auto&& impl2 = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0], megdnn::param::RelayoutFormat::Mode::CHWN4_NCHW4)
                    .node();
        };
        m_cache.emplace(ReformatKey{o, i}, impl2);
    }
    {
        auto i = TensorFormats::NCHW, o = TensorFormats::NCHWc4;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0],
                           megdnn::param::RelayoutFormat::Mode::NCHW_NCHW4_IC_SMALL)
                    .node();
        };
        m_cache.emplace(ReformatKey{i, o, Attribute::IC_SMALL}, impl);
    }
    {
        auto i = TensorFormats::KCRS, o = TensorFormats::KCRSc4;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0], megdnn::param::RelayoutFormat::Mode::
                                            NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT)
                    .node();
        };
        m_cache.emplace(ReformatKey{i, o, Attribute::IC_SMALL}, impl);
    }
    {
        auto i = TensorFormats::NCHW, o = TensorFormats::NCHWc64;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0], megdnn::param::RelayoutFormat::Mode::NCHW_NCHW64)
                    .node();
        };
        m_cache.emplace(
                ReformatKey{
                        i, o, Attribute::DEFAULT, DTypeEnum::QuantizedS4,
                        DTypeEnum::QuantizedS4},
                impl);
        m_cache.emplace(
                ReformatKey{
                        i, o, Attribute::DEFAULT, DTypeEnum::Quantized4Asymm,
                        DTypeEnum::Quantized4Asymm},
                impl);
    }
    {
        auto i = TensorFormats::NCHWc64, o = TensorFormats::NCHW;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0], megdnn::param::RelayoutFormat::Mode::NCHW64_NCHW)
                    .node();
        };
        m_cache.emplace(
                ReformatKey{
                        i, o, Attribute::DEFAULT, DTypeEnum::QuantizedS4,
                        DTypeEnum::QuantizedS4},
                impl);
        m_cache.emplace(
                ReformatKey{
                        i, o, Attribute::DEFAULT, DTypeEnum::Quantized4Asymm,
                        DTypeEnum::Quantized4Asymm},
                impl);
    }
    {
        auto i = TensorFormats::NCHW, o = TensorFormats::NHWC;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0], megdnn::param::RelayoutFormat::Mode::NCHW_NHWC)
                    .node();
        };
        m_cache.emplace(
                ReformatKey{
                        i, o, Attribute::DEFAULT, DTypeEnum::QuantizedS4,
                        DTypeEnum::QuantizedS4},
                impl);
        m_cache.emplace(
                ReformatKey{
                        i, o, Attribute::DEFAULT, DTypeEnum::Quantized4Asymm,
                        DTypeEnum::Quantized4Asymm},
                impl);
    }
    {
        auto i = TensorFormats::NHWC, o = TensorFormats::NCHW;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0], megdnn::param::RelayoutFormat::Mode::NHWC_NCHW)
                    .node();
        };
        m_cache.emplace(
                ReformatKey{
                        i, o, Attribute::DEFAULT, DTypeEnum::QuantizedS4,
                        DTypeEnum::QuantizedS4},
                impl);
        m_cache.emplace(
                ReformatKey{
                        i, o, Attribute::DEFAULT, DTypeEnum::Quantized4Asymm,
                        DTypeEnum::Quantized4Asymm},
                impl);
    }
    // nhcw4
    {
        auto i = TensorFormats::KCRS, o = TensorFormats::KRSCk4;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0],
                           megdnn::param::RelayoutFormat::Mode::INTER_WEIGHT_DENSEI)
                    .node();
        };
        m_cache.emplace(ReformatKey{i, o, Attribute::IMAGE2D}, impl);
    }
    {
        auto i = TensorFormats::KCRS, o = TensorFormats::GKRSCk4;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0],
                           megdnn::param::RelayoutFormat::Mode::INTER_WEIGHT_GROUPI)
                    .node();
        };
        m_cache.emplace(ReformatKey{i, o, Attribute::IMAGE2D}, impl);
    }
    {
        auto i = TensorFormats::KCRS, o = TensorFormats::C1RSc4;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0],
                           megdnn::param::RelayoutFormat::Mode::INTER_WEIGHT_CHANI)
                    .node();
        };
        m_cache.emplace(ReformatKey{i, o, Attribute::IMAGE2D}, impl);
    }
    {
        auto i = TensorFormats::NCHW, o = TensorFormats::NHCWc4;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0], megdnn::param::RelayoutFormat::Mode::NCHW_NHWCD4I)
                    .node();
        };
        m_cache.emplace(ReformatKey{i, o, Attribute::IMAGE2D}, impl);
    }
    {
        auto i = TensorFormats::NHCWc4, o = TensorFormats::NCHW;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0], megdnn::param::RelayoutFormat::Mode::NCHW_NHWCD4I)
                    .node();
        };
        m_cache.emplace(ReformatKey{i, o, Attribute::IMAGE2D}, impl);
    }
    // nhcw4-dot
    {
        auto i = TensorFormats::KCRS, o = TensorFormats::KRSCk4c4;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0],
                           megdnn::param::RelayoutFormat::Mode::INTER_WEIGHT_DENSEI_DOT)
                    .node();
        };
        m_cache.emplace(
                ReformatKey{
                        i, o, Attribute::IMAGE2D, DTypeEnum::QuantizedS8,
                        DTypeEnum::QuantizedS8},
                impl);
        m_cache.emplace(
                ReformatKey{
                        i, o, Attribute::IMAGE2D, DTypeEnum::Quantized8Asymm,
                        DTypeEnum::Quantized8Asymm},
                impl);
    }
    {
        auto i = TensorFormats::GKCRS, o = TensorFormats::GKRSCk4c4;
        auto&& impl = [](const VarNodeArray& vars) {
            return opr::RelayoutFormat::make(
                           vars[0],
                           megdnn::param::RelayoutFormat::Mode::INTER_WEIGHT_GROUPI_DOT)
                    .node();
        };
        m_cache.emplace(
                ReformatKey{
                        i, o, Attribute::IMAGE2D, DTypeEnum::QuantizedS8,
                        DTypeEnum::QuantizedS8},
                impl);
        m_cache.emplace(
                ReformatKey{
                        i, o, Attribute::IMAGE2D, DTypeEnum::Quantized8Asymm,
                        DTypeEnum::Quantized8Asymm},
                impl);
    }
}

ReformatManager::ReformatImpl ReformatManager::get(const ReformatKey& key) const {
    using Attribute = ReformatKey::Attribute;
    MGB_TRY {
        {
            auto find = m_cache.find(key);
            if (find != m_cache.end()) {
                auto rst = find->second;
                return rst;
            }
        }
        if (key.attribute == Attribute::AUTO_PADDING_NHWC) {
            auto key_ = key;
            key_.attribute = Attribute::DEFAULT;
            auto find = m_cache.find(key_);
            if (find != m_cache.end()) {
                auto rst = find->second;
                return rst;
            }
        }
        mgb_assert(
                !(key.attribute & Attribute::IMAGE2D) &&
                !(key.attribute & Attribute::IC_SMALL));
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
    size_t input_alignment = 0;
    size_t output_alignment = 0;
    size_t input_channel_idx = input_shape.ndim, output_channel_idx = input_shape.ndim;
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
    mgb_assert(
            input_channel_idx < input_shape.ndim &&
                    output_channel_idx < input_shape.ndim,
            "invalid channel idx(in_channel:%zu, out_channel:%zu, shp:%s)",
            input_channel_idx, output_channel_idx, input_shape.to_string().c_str());
    mgb_assert(
            input_alignment > 0 && output_alignment > 0,
            "invalid alignment(in_channel:%zu, out_channel:%zu, shp:%s)",
            input_alignment, output_alignment, input_shape.to_string().c_str());
    std::tie(input_alignment, output_alignment) =
            extra_alignment(key, orig_var->dtype(), input_alignment, output_alignment);
    NamedTensorShape orig_shape = tensor_formats_to_named_tensor_shape(orig_format);
    size_t orig_channel = 0;
    mgb_assert(
            orig_var->shape().ndim == orig_shape.ndim,
            "incompatible NamedTensorShape for "
            "feature(var:%s;shape:%s)",
            cg::dump_var_info({const_cast<VarNode*>(orig_var)}).c_str(),
            orig_shape.to_string().c_str());
    for (size_t i = 0; i < orig_shape.ndim; ++i) {
        if (orig_shape[i].name() == Dimension::Name::C &&
            orig_shape[i].extent() == Dimension::UNDETERMINED_EXTENT) {
            orig_channel = orig_var->shape()[i] * orig_shape[i].stride();
            break;
        }
    }
    mgb_assert(
            orig_channel > 0,
            "incompatible NamedTensorShape for "
            "feature(var:%s;shape:%s)",
            cg::dump_var_info({const_cast<VarNode*>(orig_var)}).c_str(),
            orig_shape.to_string().c_str());
    size_t aligned_in_channel = divup(orig_channel, input_alignment) * input_alignment;
    size_t aligned_out_channel =
            divup(orig_channel, output_alignment) * output_alignment;
    size_t common_alignment =
            input_alignment * output_alignment / gcd(input_alignment, output_alignment);
    size_t aligned_channel = divup(orig_channel, common_alignment) * common_alignment;
    auto builder = [key, aligned_channel, aligned_in_channel, aligned_out_channel,
                    input_shape, input_channel_idx, output_shape,
                    output_channel_idx](const VarNodeArray& vars) {
        VarNode *x, *cur;
        x = cur = vars[0];
        if (aligned_channel > aligned_in_channel) {
            auto padding_shape = input_shape;
            auto&& dim = padding_shape[input_channel_idx];
            size_t const_extent = (aligned_channel - aligned_in_channel) / dim.stride();
            padding_shape[input_channel_idx] =
                    Dimension(dim.name(), dim.stride(), const_extent);
            auto make_shape =
                    std::get<0>(MakeShapeEmitter{input_shape, padding_shape}.emit());
            auto padding_shp_var = make_shape({x});
            auto padding = std::get<0>(
                    PaddingEmitter{padding_shape, const_extent, input_channel_idx}
                            .emit());
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
    Dimension::Name out_channel_name = Dimension::Name::C;
    auto input_shape = tensor_formats_to_named_tensor_shape(key.input_format);
    size_t input_channel_idx = input_shape.ndim, output_channel_idx = input_shape.ndim;
    for (size_t i = 0; i < input_shape.ndim; ++i) {
        if (input_shape[i].name() == Dimension::Name::C &&
            input_shape[i].extent() == Dimension::UNDETERMINED_EXTENT) {
            in_channels = orig_var->shape()[i] * input_shape[i].stride();
            input_channel_idx = i;
            mgb_assert(
                    input_shape[i].stride() == 1, "unsupport weight format(got:%s)",
                    input_shape.to_string().c_str());
        } else if (
                (input_shape[i].name() == Dimension::Name::K ||
                 input_shape[i].name() == Dimension::Name::N) &&
                input_shape[i].extent() == Dimension::UNDETERMINED_EXTENT) {
            out_channels = orig_var->shape()[i];
            out_channel_name = input_shape[i].name();
            output_channel_idx = i;
            mgb_assert(
                    input_shape[i].stride() == 1, "unsupport weight format(got:%s)",
                    input_shape.to_string().c_str());
        }
    }
    /* \notes: FIXME this is a hack. Since the layout of weight in channelwise
     * convolution does not have output channel dimension, so we mannually modify the
     * out_channel_name, out_channel_idx to bypass the following assertion statements. */
    bool is_channelwise = key.input_format == TensorFormats::C11RS;
    if (is_channelwise) {
        out_channel_name = Dimension::Name::K;
        out_channels = in_channels;
        output_channel_idx = input_channel_idx;
    }
    mgb_assert(
            out_channel_name == Dimension::Name::K ||
                    out_channel_name == Dimension::Name::N,
            "invalid out channel(shp:%s)", input_shape.to_string().c_str());
    mgb_assert(
            (input_channel_idx < input_shape.ndim &&
             output_channel_idx < input_shape.ndim) ||
                    (is_channelwise && output_channel_idx == input_channel_idx),
            "invalid channel idx(in_channel:%zu, out_channel:%zu, shp:%s)",
            input_channel_idx, output_channel_idx, input_shape.to_string().c_str());
    size_t in_channel_alignment = 0, out_channel_alignment = 0;
    auto output_shape = tensor_formats_to_named_tensor_shape(key.output_format);
    for (size_t i = 0; i < output_shape.ndim; ++i) {
        if (output_shape[i].name() == Dimension::Name::C &&
            output_shape[i].extent() == Dimension::UNDETERMINED_EXTENT) {
            in_channel_alignment = output_shape[i].stride();
        } else if (
                output_shape[i].name() == out_channel_name &&
                output_shape[i].extent() == Dimension::UNDETERMINED_EXTENT) {
            out_channel_alignment = output_shape[i].stride();
        }
    }
    /* \notes: FIXME this is a hack. Since the layout of weight in channelwise
     * convolution does not have output channel dimension, so we mannually modify the
     * out_channel_alignment to bypass the following assertion statements. */
    if (is_channelwise) {
        mgb_assert(out_channel_alignment == 0);
        out_channel_alignment = 1;
    }
    mgb_assert(
            in_channel_alignment > 0 && out_channel_alignment > 0,
            "invalid alignment(in_channel:%zu, out_channel:%zu, shp:%s)",
            in_channel_alignment, out_channel_alignment,
            output_shape.to_string().c_str());
    in_channel_alignment = ::extra_alignment(
            key.attribute, key.output_format, orig_var->dtype(), in_channel_alignment);
    out_channel_alignment = ::extra_alignment(
            key.attribute, key.output_format, orig_var->dtype(), out_channel_alignment);
    size_t aligned_in_channel =
            divup(in_channels, in_channel_alignment) * in_channel_alignment;
    if (extra_alignment.name == out_channel_name) {
        out_channel_alignment = extra_alignment.alignment * out_channel_alignment /
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
            size_t const_extent = (aligned_in_channel - in_channels) / dim.stride();
            padding_shape[input_channel_idx] =
                    Dimension(dim.name(), dim.stride(), const_extent);
            auto make_shape =
                    std::get<0>(MakeShapeEmitter{input_shape, padding_shape}.emit());
            auto padding_shp_var = make_shape({x});
            auto padding = std::get<0>(
                    PaddingEmitter{padding_shape, const_extent, input_channel_idx}
                            .emit());
            cur = padding({cur, padding_shp_var});
        }
        if (aligned_out_channel > out_channels) {
            auto padding_shape = input_shape;
            auto&& dim = padding_shape[output_channel_idx];
            size_t const_extent = (aligned_out_channel - out_channels) / dim.stride();
            padding_shape[output_channel_idx] =
                    Dimension(dim.name(), dim.stride(), const_extent);
            auto make_shape =
                    std::get<0>(MakeShapeEmitter{input_shape, padding_shape}.emit());
            auto padding_shp_var = make_shape({cur});
            auto padding = std::get<0>(
                    PaddingEmitter{padding_shape, const_extent, output_channel_idx}
                            .emit());
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

TensorShape ReformatManager::try_make_tensor_shape(
        const VarNode* var, TensorFormats orig_formats, TensorFormats target_formats,
        ReformatKey::Attribute extra_attribute, bool allow_aligned) {
    using Dimension = megdnn::Dimension;
    static constexpr uint32_t UNDETERMINED_EXTENT = Dimension::UNDETERMINED_EXTENT;
    auto orig_shape = tensor_formats_to_named_tensor_shape(orig_formats);
    auto target_shape = tensor_formats_to_named_tensor_shape(target_formats);

    TensorShape oshp = var->shape();
    mgb_assert(
            oshp.is_scalar() || oshp.ndim == orig_shape.ndim,
            "orig shape of var node is not compatible with tensor "
            "formats(var:%s;shp:%s;fmt:%s)",
            var->cname(), oshp.to_string().c_str(), orig_shape.to_string().c_str());
    if (oshp.is_scalar())
        return oshp;
    TensorShape tshp;
    ThinHashMap<Dimension::Name, int> name2dominant;
    for (size_t i = 0; i < orig_shape.ndim; ++i) {
        auto name = orig_shape[i].name();
        if (orig_shape[i].extent() == UNDETERMINED_EXTENT) {
            auto insert = name2dominant.insert(std::make_pair(name, i));
            mgb_assert(insert.second);
        }
    }

    tshp.ndim = target_shape.ndim;
    for (size_t i = 0; i < target_shape.ndim; ++i) {
        auto name = target_shape[i].name();
        if (target_shape[i].extent() == UNDETERMINED_EXTENT) {
            int idx = name2dominant.at(name);
            bool mul = orig_shape[idx] < target_shape[i];
            size_t factor = mul ? (target_shape[i] / orig_shape[idx]).extent()
                                : (orig_shape[idx] / target_shape[i]).extent();
            if (mul)
                tshp[i] = oshp[idx] * factor;
            else {
                if (allow_aligned)
                    tshp[i] = divup(oshp[idx], factor);
                else if (!(oshp[idx] % factor)) {
                    tshp[i] = oshp[idx] / factor;
                } else {
                    return TensorShape{};
                }
            }

            /// hack for nhwc auto padding
            if (name == Dimension::Name::C) {
                size_t channel_alignment = target_shape[i].stride();
                size_t channels = tshp[i] * channel_alignment;
                size_t new_channel_alignment = extra_alignment(
                        extra_attribute, target_formats, var->dtype(),
                        channel_alignment);
                tshp[i] = divup(channels, new_channel_alignment) *
                          new_channel_alignment / channel_alignment;
            }
        } else {
            tshp[i] = target_shape[i].extent();
        }
    }
    return tshp;
}

TensorShape ReformatManager::make_aligned_tensor_shape(
        const VarNode* var, TensorFormats orig_formats, TensorFormats target_formats,
        ReformatKey::Attribute extra_attribute) {
    auto tshp = ReformatManager::try_make_tensor_shape(
            var, orig_formats, target_formats, extra_attribute);
    mgb_assert(tshp.ndim);
    return tshp;
}

TensorShape ReformatManager::make_aligned_weight_shape(
        const VarNode* var, TensorFormats orig_formats, TensorFormats target_formats,
        TensorFormats extra_formats, ReformatKey::Attribute extra_attribute) {
    auto tshp = make_aligned_tensor_shape(
            var, orig_formats, target_formats, extra_attribute);
    auto extra_shape = tensor_formats_to_named_tensor_shape(extra_formats);
    using Dimension = megdnn::Dimension;
    static constexpr uint32_t UNDETERMINED_EXTENT = Dimension::UNDETERMINED_EXTENT;
    size_t out_channel_alignment = 1;
    for (size_t i = 0; i < extra_shape.ndim; ++i) {
        auto name = extra_shape[i].name();
        if (name == Dimension::Name::C &&
            extra_shape[i].extent() == UNDETERMINED_EXTENT) {
            out_channel_alignment = extra_shape[i].stride();
            out_channel_alignment = extra_alignment(
                    extra_attribute, target_formats, var->dtype(),
                    out_channel_alignment);
        }
    }

    auto target_shape = tensor_formats_to_named_tensor_shape(target_formats);
    for (size_t i = 0; i < target_shape.ndim; ++i) {
        auto name = target_shape[i].name();
        if ((name == Dimension::Name::K || name == Dimension::Name::N) &&
            target_shape[i].extent() == UNDETERMINED_EXTENT) {
            size_t out_channels = tshp[i] * target_shape[i].stride();
            tshp[i] = divup(out_channels, out_channel_alignment) *
                      out_channel_alignment / target_shape[i].stride();
        }
    }
    return tshp;
}

ReformatManager::AlignmentDesc ReformatManager::make_aligned_desc(
        TensorFormats weight_format, TensorFormats out_feature_format) {
    using Name = Dimension::Name;
    auto weight_shape = tensor_formats_to_named_tensor_shape(weight_format);
    auto out_shape = tensor_formats_to_named_tensor_shape(out_feature_format);
    size_t out_channel_alignment = 1;
    for (size_t i = 0; i < out_shape.ndim; ++i) {
        auto name = out_shape[i].name();
        auto extent = out_shape[i].extent();
        if ((name == Name::C || name == Name::K) &&
            extent == Dimension::UNDETERMINED_EXTENT) {
            out_channel_alignment = out_shape[i].stride();
            break;
        }
    }
    Name out_channel_name = Name::N;
    for (size_t i = 0; i < weight_shape.ndim; ++i) {
        auto name = weight_shape[i].name();
        auto extent = weight_shape[i].extent();
        if ((name == Name::N || name == Name::K) &&
            extent == Dimension::UNDETERMINED_EXTENT) {
            out_channel_name = name;
        }
    }
    return AlignmentDesc{out_channel_name, out_channel_alignment};
}

// vim: syntax=cpp.doxygen
