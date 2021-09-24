/**
 * \file src/gopt/impl/opr_tensor_formats_config.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./utils.h"
#include "megbrain/gopt/layout_transform_context.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"

#include "midout.h"
MIDOUT_DECL(megbrain_opr_tensor_formats_config)
#define MIDOUT_B(...) MIDOUT_BEGIN(megbrain_opr_tensor_formats_config, __VA_ARGS__) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using namespace mgb;
using namespace cg;
using namespace gopt;
using OprFormat = opr::ConvBias::Param::Format;

namespace {
template <typename Opr>
struct ConvParamTrait;

#define INST(_conv, _weight_idx, _bias_idx, _has_bias) \
    template <>                                        \
    struct ConvParamTrait<opr::_conv> {                \
        static constexpr int weight_idx = _weight_idx; \
        static constexpr int bias_idx = _bias_idx;     \
        static constexpr bool has_bias = _has_bias;    \
    }
INST(ConvBias, 1, 2, true);
INST(ConvolutionForward, 1, 0, false);
INST(ConvolutionBackwardData, 0, 0, false);

template <typename Opr, size_t weight_idx = ConvParamTrait<Opr>::weight_idx>
static bool is_channel_wise_conv(const OperatorNodeBase* opr) {
    MGB_MARK_USED_VAR(ConvParamTrait<Opr>::has_bias);
    MGB_MARK_USED_VAR(ConvParamTrait<Opr>::bias_idx);
    auto&& conv = opr->cast_final_safe<Opr>();
    auto format = conv.param().format;
    auto weight = opr->input(weight_idx);
    auto weight_shp = weight->shape();
    if (conv.param().sparse == Opr::Param::Sparse::DENSE)
        return false;
    size_t ocpg, icpg;
    if (format == Opr::Param::Format::NCHW) {
        ocpg = weight_shp[1], icpg = weight_shp[2];
        return ocpg == 1 && icpg == 1;
    }
    return false;
}

template <OprFormat opr_format_>
struct OprSingleInOutTensorFormatsDispatcherImpl;

template <>
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormat::NCHW> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW;
        config.input_dtypes = {opr->input(0)->dtype().enumv()};
        config.input_tensor_types = {TensorType::FEATURE};
        config.output_dtypes = {opr->output(0)->dtype().enumv()};
        config.input_tensor_formats = {TensorFormats::NCHW};
        config.output_tensor_formats = {TensorFormats::NCHW};
        return config;
    }
};

template <>
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormat::NCHW44> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(
            const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW44;
        bool available = true;
        available &= opr->input(0)->dtype().enumv() == DTypeEnum::Float32;
        config.input_dtypes = {opr->input(0)->dtype().enumv()};
        config.input_tensor_types = {TensorType::FEATURE};
        config.output_dtypes = {opr->output(0)->dtype().enumv()};
        config.input_tensor_formats = {TensorFormats::NCHWc4};
        config.output_tensor_formats = {TensorFormats::NCHWc4};
        if (!available)
            return None;
        return config;
    }
};

#if !MEGDNN_DISABLE_FLOAT16
template <>
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormat::NCHW88> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(
            const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW88;
        bool available = true;
        available &= opr->input(0)->dtype().enumv() == DTypeEnum::Float16;
        config.input_dtypes = {opr->input(0)->dtype().enumv()};
        config.input_tensor_types = {TensorType::FEATURE};
        config.output_dtypes = {opr->output(0)->dtype().enumv()};
        config.input_tensor_formats = {TensorFormats::NCHWc8};
        config.output_tensor_formats = {TensorFormats::NCHWc8};
        if (!available)
            return None;
        return config;
    }
};
#endif

template <>
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormat::NCHW4> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW4;
        bool available = true;
        available &= opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        config.input_dtypes = {opr->input(0)->dtype().enumv()};
        config.input_tensor_types = {TensorType::FEATURE};
        available &= opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        config.output_dtypes = {opr->output(0)->dtype().enumv()};
        config.input_tensor_formats = {TensorFormats::NCHWc4};
        config.output_tensor_formats = {TensorFormats::NCHWc4};
        if (available)
            return config;
        return None;
    }
};

template <>
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormat::CHWN4> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::CHWN4;
        bool available = true;
        available &= opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        config.input_dtypes = {opr->input(0)->dtype().enumv()};
        config.input_tensor_types = {TensorType::FEATURE};
        available &= opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        config.output_dtypes = {opr->output(0)->dtype().enumv()};
        config.input_tensor_formats = {TensorFormats::CHWNc4};
        config.output_tensor_formats = {TensorFormats::CHWNc4};
        if (available)
            return config;
        return None;
    }
};

template <>
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormat::NCHW32> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW32;
        bool available = true;
        available &= opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        config.input_dtypes = {opr->input(0)->dtype().enumv()};
        config.input_tensor_types = {TensorType::FEATURE};
        available &= opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        config.output_dtypes = {opr->output(0)->dtype().enumv()};
        config.input_tensor_formats = {TensorFormats::NCHWc32};
        config.output_tensor_formats = {TensorFormats::NCHWc32};
        if (available)
            return config;
        return None;
    }
};

template <>
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormat::NHWC> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NHWC;
        bool available = true;
        available &= opr->input(0)->dtype().enumv() == DTypeEnum::Quantized4Asymm ||
                     opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS4;
        config.input_dtypes = {opr->input(0)->dtype().enumv()};
        config.input_tensor_types = {TensorType::FEATURE};
        available &= opr->output(0)->dtype().enumv() == opr->input(0)->dtype().enumv();
        config.output_dtypes = {opr->output(0)->dtype().enumv()};
        config.input_tensor_formats = {TensorFormats::NHWC};
        config.output_tensor_formats = {TensorFormats::NHWC};
        if (available)
            return config;
        return None;
    }
};

template <>
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormat::NCHW64> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW64;
        bool available = true;
        available &= opr->input(0)->dtype().enumv() == DTypeEnum::Quantized4Asymm ||
                     opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS4;
        config.input_dtypes = {opr->input(0)->dtype().enumv()};
        config.input_tensor_types = {TensorType::FEATURE};
        available &= opr->output(0)->dtype().enumv() == opr->input(0)->dtype().enumv();
        config.output_dtypes = {opr->output(0)->dtype().enumv()};
        config.input_tensor_formats = {TensorFormats::NCHWc64};
        config.output_tensor_formats = {TensorFormats::NCHWc64};
        if (available)
            return config;
        return None;
    }
};

template <typename Opr, OprFormat opr_format_>
struct ConvTensorFormatsDispatcherImpl;

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormat::NCHW> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW;
        // setup dtypes
        for (size_t i = 0; i < opr->input().size(); ++i) {
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        // setup tensor formats
        if (conv.param().sparse == Opr::Param::Sparse::DENSE) {
            config.input_tensor_formats = {
                    TensorFormats::NCHW, TensorFormats::KCRS, TensorFormats::NCHW,
                    TensorFormats::NCHW};
        } else {
            mgb_assert(conv.param().sparse == Opr::Param::Sparse::GROUP);
            if (is_channel_wise_conv<Opr>(opr)) {
                config.input_tensor_formats = {
                        TensorFormats::NCHW, TensorFormats::C11RS, TensorFormats::NCHW,
                        TensorFormats::NCHW};
            } else {
                config.input_tensor_formats = {
                        TensorFormats::NCHW, TensorFormats::GKCRS, TensorFormats::NCHW,
                        TensorFormats::NCHW};
            }
        }
        config.output_tensor_formats = {TensorFormats::NCHW};
        return config;
    }
};

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormat::NHWC> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NHWC;
        bool available = true;
        for (size_t i = 0; i < opr->input().size(); ++i) {
            if (i == 2)
                available &= opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS32;
            else {
                bool i4_config =
                        opr->input(i)->dtype().enumv() == DTypeEnum::Quantized4Asymm ||
                        opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS4;
                bool i8_config =
                        opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS8;
                available &= (i4_config || i8_config);
            }
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        bool i4_config =
                opr->output(0)->dtype().enumv() == DTypeEnum::Quantized4Asymm ||
                opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS4;
        bool i8_config = opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        available &= (i4_config || i8_config);
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        available &= conv.param().sparse == Opr::Param::Sparse::DENSE;
        config.input_tensor_formats = {
                TensorFormats::NHWC, TensorFormats::NHWC, TensorFormats::NHWC,
                TensorFormats::NHWC};
        config.output_tensor_formats = {TensorFormats::NHWC};
        if (available)
            return config;
        return None;
    }
};

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormat::NCHW4> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW4;
        bool available = true;
        // setup dtypes
        for (size_t i = 0; i < opr->input().size(); ++i) {
            if (i == 2)
                available &= opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS32;
            else
                available &= opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS8;
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        available &= opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        // setup tensor formats
        if (conv.param().sparse == Opr::Param::Sparse::DENSE) {
            config.input_tensor_formats = {
                    TensorFormats::NCHWc4, TensorFormats::NCHWc4, TensorFormats::NCHWc4,
                    TensorFormats::NCHWc4};
        } else {
            mgb_assert(conv.param().sparse == Opr::Param::Sparse::GROUP);
            if (is_channel_wise_conv<Opr>(opr)) {
                config.input_tensor_formats = {
                        TensorFormats::NCHWc4, TensorFormats::C11RSc4,
                        TensorFormats::NCHWc4, TensorFormats::NCHWc4};
            } else {
                config.input_tensor_formats = {
                        TensorFormats::NCHWc4, TensorFormats::GKCRSc4,
                        TensorFormats::NCHWc4, TensorFormats::NCHWc4};
            }
        }
        config.output_tensor_formats = {TensorFormats::NCHWc4};
        if (available)
            return config;
        return None;
    }
};

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormat::NCHW32> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW32;
        bool available = true;
        for (size_t i = 0; i < opr->input().size(); ++i) {
            if (i == 2)
                available &= opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS32;
            else
                available &= opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS8;
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        available &= opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        available &= conv.param().sparse == Opr::Param::Sparse::DENSE;
        config.input_tensor_formats = {
                TensorFormats::NCHWc32, TensorFormats::NCHWc32, TensorFormats::NCHWc32,
                TensorFormats::NCHWc32};
        config.output_tensor_formats = {TensorFormats::NCHWc32};
        if (available)
            return config;
        return None;
    }
};

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormat::NCHW64> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW64;
        bool available = true;
        for (size_t i = 0; i < opr->input().size(); ++i) {
            if (i == 2)
                available &= opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS32;
            else
                available &=
                        opr->input(i)->dtype().enumv() == DTypeEnum::Quantized4Asymm ||
                        opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS4;
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        available &= opr->output(0)->dtype().enumv() == DTypeEnum::Quantized4Asymm ||
                     opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS4;
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        available &= conv.param().sparse == Opr::Param::Sparse::DENSE;
        config.input_tensor_formats = {
                TensorFormats::NCHWc64, TensorFormats::NCHWc64, TensorFormats::NCHWc64,
                TensorFormats::NCHWc64};
        config.output_tensor_formats = {TensorFormats::NCHWc64};
        if (available)
            return config;
        return None;
    }
};

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormat::CHWN4> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::CHWN4;
        bool available = true;
        for (size_t i = 0; i < opr->input().size(); ++i) {
            if (i == 2)
                available &= opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS32;
            else
                available &= opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS8;
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        available &= opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        available &= conv.param().sparse == Opr::Param::Sparse::DENSE;
        config.input_tensor_formats = {
                TensorFormats::CHWNc4, TensorFormats::CHWNc4, TensorFormats::CHWNc4,
                TensorFormats::CHWNc4};
        config.output_tensor_formats = {TensorFormats::CHWNc4};
        if (available)
            return config;
        return None;
    }
};

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormat::NCHW44> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(
            const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW44;
        bool available = true;
        // setup dtypes
        for (size_t i = 0; i < opr->input().size(); ++i) {
            available &= opr->input(i)->dtype().enumv() == DTypeEnum::Float32;
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type =
                    i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        available &= opr->output(0)->dtype().enumv() == DTypeEnum::Float32;
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        // setup tensor formats
        if (conv.param().sparse == Opr::Param::Sparse::DENSE) {
            config.input_tensor_formats = {
                    TensorFormats::NCHWc4, TensorFormats::KCRSc4k4,
                    TensorFormats::NCHWc4, TensorFormats::NCHWc4};
        } else {
            mgb_assert(conv.param().sparse == Opr::Param::Sparse::GROUP);
            if (is_channel_wise_conv<Opr>(opr)) {
                config.input_tensor_formats = {
                        TensorFormats::NCHWc4, TensorFormats::C11RSc4,
                        TensorFormats::NCHWc4, TensorFormats::NCHWc4};
            } else {
                config.input_tensor_formats = {
                        TensorFormats::NCHWc4, TensorFormats::GKCRSc4k4,
                        TensorFormats::NCHWc4, TensorFormats::NCHWc4};
            }
        }
        config.output_tensor_formats = {TensorFormats::NCHWc4};
        if (!available)
            return None;
        return config;
    }
};

#if !MEGDNN_DISABLE_FLOAT16
template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormat::NCHW88> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(
            const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW88;
        bool available = true;
        // setup dtypes
        for (size_t i = 0; i < opr->input().size(); ++i) {
            available &= opr->input(i)->dtype().enumv() == DTypeEnum::Float16;
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type =
                    i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        available &= opr->output(0)->dtype().enumv() == DTypeEnum::Float16;
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        // setup tensor formats
        if (conv.param().sparse == Opr::Param::Sparse::DENSE) {
            config.input_tensor_formats = {
                    TensorFormats::NCHWc8, TensorFormats::KCRSc8k8,
                    TensorFormats::NCHWc8, TensorFormats::NCHWc8};
        } else {
            mgb_assert(conv.param().sparse == Opr::Param::Sparse::GROUP);
            if (is_channel_wise_conv<Opr>(opr)) {
                config.input_tensor_formats = {
                        TensorFormats::NCHWc8, TensorFormats::C11RSc8,
                        TensorFormats::NCHWc8, TensorFormats::NCHWc8};
            } else {
                config.input_tensor_formats = {
                        TensorFormats::NCHWc8, TensorFormats::GKCRSc8k8,
                        TensorFormats::NCHWc8, TensorFormats::NCHWc8};
            }
        }
        config.output_tensor_formats = {TensorFormats::NCHWc8};
        if (!available)
            return None;
        return config;
    }
};
#endif

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormat::NCHW44_DOT> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(
            const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW44_DOT;
        bool available = true;
        // setup dtypes
        for (size_t i = 0; i < opr->input().size(); ++i) {
            if (i == 2) {
                available &= opr->input(i)->dtype().enumv() ==
                             DTypeEnum::QuantizedS32;
            } else {
                available &= opr->input(i)->dtype().enumv() ==
                                     DTypeEnum::QuantizedS8 ||
                             opr->input(i)->dtype().enumv() ==
                                     DTypeEnum::Quantized8Asymm;
            }
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type =
                    i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        available &=
                opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8 ||
                opr->output(0)->dtype().enumv() == DTypeEnum::Quantized8Asymm;
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        // setup tensor formats
        if (conv.param().sparse == Opr::Param::Sparse::DENSE) {
            config.input_tensor_formats = {
                    TensorFormats::NCHWc4, TensorFormats::KCRSk4c4,
                    TensorFormats::NCHWc4, TensorFormats::NCHWc4};
        } else {
            mgb_assert(conv.param().sparse == Opr::Param::Sparse::GROUP);
            if (is_channel_wise_conv<Opr>(opr)) {
                available = false;
            } else {
                config.input_tensor_formats = {
                        TensorFormats::NCHWc4, TensorFormats::GKCRSk4c4,
                        TensorFormats::NCHWc4, TensorFormats::NCHWc4};
            }
        }
        config.output_tensor_formats = {TensorFormats::NCHWc4};
        if (!available)
            return None;
        return config;
    }
};

template <>
struct ConvTensorFormatsDispatcherImpl<opr::ConvolutionBackwardData, OprFormat::NCHW> {
    using Opr = opr::ConvolutionBackwardData;
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW;
        // setup dtypes
        for (size_t i = 0; i < opr->input().size(); ++i) {
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 0 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        // setup tensor formats
        if (conv.param().sparse == Opr::Param::Sparse::DENSE) {
            config.input_tensor_formats = {
                    TensorFormats::NCHW, TensorFormats::NCHW, TensorFormats::NCHW,
                    TensorFormats::NCHW};
        } else {
            mgb_assert(conv.param().sparse == Opr::Param::Sparse::GROUP);
            if (is_channel_wise_conv<Opr>(opr)) {
                config.input_tensor_formats = {
                        TensorFormats::C11RS, TensorFormats::NCHW, TensorFormats::NCHW,
                        TensorFormats::NCHW};
            } else {
                config.input_tensor_formats = {
                        TensorFormats::GKCRS, TensorFormats::NCHW, TensorFormats::NCHW,
                        TensorFormats::NCHW};
            }
        }
        config.output_tensor_formats = {TensorFormats::NCHW};
        return config;
    }
};

template <>
struct ConvTensorFormatsDispatcherImpl<opr::ConvolutionBackwardData, OprFormat::NCHW4> {
    using Opr = opr::ConvolutionBackwardData;
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW4;
        bool available = true;
        for (size_t i = 0; i < opr->input().size(); ++i) {
            available &= opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS8;
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 0 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        available &= opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        available &= conv.param().sparse == opr::ConvBias::Param::Sparse::DENSE;
        config.input_tensor_formats = {
                TensorFormats::NCHWc4, TensorFormats::NCHWc4, TensorFormats::NCHWc4,
                TensorFormats::NCHWc4};
        config.output_tensor_formats = {TensorFormats::NCHWc4};
        if (available)
            return config;
        return None;
    }
};

template <>
struct ConvTensorFormatsDispatcherImpl<opr::ConvolutionBackwardData, OprFormat::NHWC> {
    using Opr = opr::ConvolutionBackwardData;
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NHWC;
        bool available = true;
        for (size_t i = 0; i < opr->input().size(); ++i) {
            available &= opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS8;
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 0 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        available &= opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        available &= conv.param().sparse == opr::ConvBias::Param::Sparse::DENSE;
        config.input_tensor_formats = {
                TensorFormats::NHWC, TensorFormats::NHWC, TensorFormats::NHWC,
                TensorFormats::NHWC};
        config.output_tensor_formats = {TensorFormats::NHWC};
        if (available)
            return config;
        return None;
    }
};

struct StaticData {
    struct KeyHash {
        size_t operator()(const std::pair<Typeinfo*, OprFormat>& val) const {
            size_t h1 = mgb::hash<Typeinfo*>(val.first);
            size_t h2 = std::hash<uint32_t>()(static_cast<uint32_t>(val.second));
            return mgb::hash_pair_combine(h1, h2);
        }
    };
    using OprTensorFormatsDispatcher =
            OprTensorFormatsConfiguration::OprTensorFormatsDispatcher;
    std::unordered_map<
            std::pair<Typeinfo*, OprFormat>, OprTensorFormatsDispatcher, KeyHash>
            typefmt2dispatcher;
    StaticData();
};

StaticData::StaticData() {
#define OPR_TENSOR_FORMATS_CONFIG_REG(_Opr, _fmt)                   \
    typefmt2dispatcher[{opr::_Opr::typeinfo(), OprFormat::_fmt}] =  \
            [](const OperatorNodeBase* opr) {                       \
                MIDOUT_B(opr::_Opr, midout_iv(OprFormat::_fmt))     \
                return ConvTensorFormatsDispatcherImpl<             \
                        opr::_Opr, OprFormat::_fmt>::dispatch(opr); \
                MIDOUT_E                                            \
            }

#define OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(_Opr, _fmt)    \
    typefmt2dispatcher[{opr::_Opr::typeinfo(), OprFormat::_fmt}] = \
            [](const OperatorNodeBase* opr) {                      \
                MIDOUT_B(opr::_Opr, midout_iv(OprFormat::_fmt))    \
                return OprSingleInOutTensorFormatsDispatcherImpl<  \
                        OprFormat::_fmt>::dispatch(opr);           \
                MIDOUT_E                                           \
            }

    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NHWC);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW4);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, CHWN4);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW32);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW64);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW44);
#if !MEGDNN_DISABLE_FLOAT16
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW88);
#endif
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW44_DOT);

    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionForward, NCHW);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionForward, NCHW4);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionForward, NCHW44);
#if !MEGDNN_DISABLE_FLOAT16
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionForward, NCHW88);
#endif
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionForward, NCHW44_DOT);

    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionBackwardData, NCHW);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionBackwardData, NHWC);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionBackwardData, NCHW4);

    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(WarpPerspectiveForward, NCHW);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(WarpPerspectiveForward, NHWC);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(WarpPerspectiveForward, NCHW4);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(WarpPerspectiveForward, NCHW64);

    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, NCHW);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, NHWC);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, NCHW4);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, CHWN4);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, NCHW32);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, NCHW64);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, NCHW44);
#if !MEGDNN_DISABLE_FLOAT16
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, NCHW88);
#endif

    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(ResizeForward, NCHW);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(ResizeForward, NCHW44);
#if !MEGDNN_DISABLE_FLOAT16 
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(ResizeForward, NCHW88);
#endif

#undef OPR_TENSOR_FORMATS_CONFIG_REG
#undef OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG
}

StaticData& static_data() {
    static StaticData inst;
    return inst;
}
}  // namespace

OprTensorFormatsConfiguration::OprTensorFormatsDispatcher*
OprTensorFormatsConfiguration::find_dispatcher_by_type_format(
        Typeinfo* type, OprFormat opr_format) {
    auto&& typefmt2dispatcher = static_data().typefmt2dispatcher;
    auto iter = typefmt2dispatcher.find(std::make_pair(type, opr_format));
    mgb_assert(
            iter != typefmt2dispatcher.end(),
            "cannot find OprTensorFormatsDispatcher for opr type(%s) and "
            "opr format(%s)",
            type->name, opr_format_to_string(opr_format));
    return &iter->second;
}

// vim: syntax=cpp.doxygen
