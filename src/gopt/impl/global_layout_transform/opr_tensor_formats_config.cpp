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
using OprFormat = OprTensorFormatsConfiguration::OprFormat;
using OprFormatConfigID = OprTensorFormatsConfiguration::OprFormatConfigID;

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
    } else {
        mgb_assert(false, "invalid opr format(%s)", opr_format_to_string(format));
    }
    return false;
}

template <OprFormatConfigID config_id>
struct OprSingleInOutTensorFormatsDispatcherImpl;

template <>
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormatConfigID::NCHW> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW;
        config.config_id = OprFormatConfigID::NCHW;
        config.input_dtypes = {opr->input(0)->dtype().enumv()};
        config.input_tensor_types = {TensorType::FEATURE};
        config.output_dtypes = {opr->output(0)->dtype().enumv()};
        config.input_tensor_formats = {TensorFormats::NCHW};
        config.output_tensor_formats = {TensorFormats::NCHW};
        return config;
    }
};

/* \remark: Here, maybe we needn't check data type of input and output tensors. Because
 * algo available checker will skip the configuration that has no underlying impls. */
template <>
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormatConfigID::NCHW44> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW44;
        config.config_id = OprFormatConfigID::NCHW44;
        bool f32_config = opr->input(0)->dtype().enumv() == DTypeEnum::Float32;
        bool i8_config = opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        bool available = f32_config || i8_config;
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

template <>
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormatConfigID::NCHW88> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW88;
        config.config_id = OprFormatConfigID::NCHW88;
        bool available = opr->input(0)->dtype().enumv() == DTypeEnum::Float32;
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

template <>
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormatConfigID::NCHW4> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW4;
        config.config_id = OprFormatConfigID::NCHW4;
        bool available = opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
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
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormatConfigID::CHWN4> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::CHWN4;
        config.config_id = OprFormatConfigID::CHWN4;
        bool available = opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
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
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormatConfigID::NCHW32> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW32;
        config.config_id = OprFormatConfigID::NCHW32;
        bool available = opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
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
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormatConfigID::NHWC> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NHWC;
        config.config_id = OprFormatConfigID::NHWC;
        bool f16_config = DNN_FLOAT16_SELECT(
                (opr->input(0)->dtype().enumv() == DTypeEnum::Float16), true);
        bool i4_config = opr->input(0)->dtype().enumv() == DTypeEnum::Quantized4Asymm ||
                         opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS4;
        bool available = f16_config || i4_config;
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
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormatConfigID::NCHW64> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW64;
        config.config_id = OprFormatConfigID::NCHW64;
        bool available = opr->input(0)->dtype().enumv() == DTypeEnum::Quantized4Asymm ||
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

template <>
struct OprSingleInOutTensorFormatsDispatcherImpl<OprFormatConfigID::NHWCD4> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NHWCD4;
        config.config_id = OprFormatConfigID::NHWCD4;
        bool available =
                opr->input(0)->dtype().enumv() == DTypeEnum::Float32 ||
                DNN_FLOAT16_SELECT(
                        (opr->input(0)->dtype().enumv() == DTypeEnum::Float16), true) ||
                opr->input(0)->dtype().enumv() == DTypeEnum::Int8 ||
                opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS8;
        config.input_dtypes = {opr->input(0)->dtype().enumv()};
        config.input_tensor_types = {TensorType::FEATURE};
        config.output_dtypes = {opr->output(0)->dtype().enumv()};
        config.input_tensor_formats = {TensorFormats::NHCWc4};
        config.output_tensor_formats = {TensorFormats::NHCWc4};
        if (available)
            return config;
        return None;
    }
};

template <typename Opr, OprFormatConfigID config_id>
struct ConvTensorFormatsDispatcherImpl;

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormatConfigID::NCHW> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW;
        config.config_id = OprFormatConfigID::NCHW;
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
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormatConfigID::NHWC> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NHWC;
        config.config_id = OprFormatConfigID::NHWC;
        auto check_dtype = [](const DType& dt) {
            bool f16_config =
                    DNN_FLOAT16_SELECT((dt.enumv() == DTypeEnum::Float16), true);
            bool i4_config = dt.enumv() == DTypeEnum::Quantized4Asymm ||
                             dt.enumv() == DTypeEnum::QuantizedS4;
            bool i8_config = dt.enumv() == DTypeEnum::QuantizedS8;
            return f16_config || i4_config || i8_config;
        };
        bool available = true;
        for (size_t i = 0; i < opr->input().size(); ++i) {
            if (i == 2) {
                available &=
                        opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS32 ||
                        DNN_FLOAT16_SELECT(
                                opr->input(i)->dtype().enumv() == DTypeEnum::Float16,
                                true);
            } else {
                available &= check_dtype(opr->input(i)->dtype());
            }
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        available &= check_dtype(opr->output(0)->dtype());
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        available &= conv.param().sparse == Opr::Param::Sparse::DENSE;
        config.input_tensor_formats = {
                TensorFormats::NHWC, TensorFormats::KRSC, TensorFormats::NHWC,
                TensorFormats::NHWC};
        config.output_tensor_formats = {TensorFormats::NHWC};
        if (available)
            return config;
        return None;
    }
};

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormatConfigID::NCHW4> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW4;
        config.config_id = OprFormatConfigID::NCHW4;
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
                    TensorFormats::NCHWc4, TensorFormats::KCRSc4, TensorFormats::NCHWc4,
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
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormatConfigID::NCHW4_NCHW32> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW4_NCHW32;
        config.config_id = OprFormatConfigID::NCHW4_NCHW32;
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
                TensorFormats::NCHWc4, TensorFormats::KCRSc4, TensorFormats::NCHWc32,
                TensorFormats::NCHWc32};
        config.output_tensor_formats = {TensorFormats::NCHWc32};
        if (available)
            return config;
        return None;
    }
};

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormatConfigID::NCHW4_NCHW> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW4_NCHW;
        config.config_id = OprFormatConfigID::NCHW4_NCHW;
        bool available = true;
        for (size_t i = 0; i < opr->input().size(); ++i) {
            if (i >= 2)
                available &= opr->input(i)->dtype().enumv() == DTypeEnum::Float32;
            else
                available &= opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS8;
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        available &= opr->output(0)->dtype().enumv() == DTypeEnum::Float32;
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        available &= conv.param().sparse == Opr::Param::Sparse::DENSE;
        config.input_tensor_formats = {
                TensorFormats::NCHWc4, TensorFormats::KCRSc4, TensorFormats::NCHW,
                TensorFormats::NCHW};
        config.output_tensor_formats = {TensorFormats::NCHW};
        if (available)
            return config;
        return None;
    }
};

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormatConfigID::NCHW32> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW32;
        config.config_id = OprFormatConfigID::NCHW32;
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
                TensorFormats::NCHWc32, TensorFormats::KCRSc32, TensorFormats::NCHWc32,
                TensorFormats::NCHWc32};
        config.output_tensor_formats = {TensorFormats::NCHWc32};
        if (available)
            return config;
        return None;
    }
};

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormatConfigID::NCHW32_NCHW4> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW32_NCHW4;
        config.config_id = OprFormatConfigID::NCHW32_NCHW4;
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
                TensorFormats::NCHWc32, TensorFormats::KCRSc32, TensorFormats::NCHWc4,
                TensorFormats::NCHWc4};
        config.output_tensor_formats = {TensorFormats::NCHWc4};
        if (available)
            return config;
        return None;
    }
};

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormatConfigID::NCHW64> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW64;
        config.config_id = OprFormatConfigID::NCHW64;
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
                TensorFormats::NCHWc64, TensorFormats::KCRSc64, TensorFormats::NCHWc64,
                TensorFormats::NCHWc64};
        config.output_tensor_formats = {TensorFormats::NCHWc64};
        if (available)
            return config;
        return None;
    }
};

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormatConfigID::CHWN4> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::CHWN4;
        config.config_id = OprFormatConfigID::CHWN4;
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
                TensorFormats::CHWNc4, TensorFormats::CRSKc4, TensorFormats::CHWNc4,
                TensorFormats::CHWNc4};
        config.output_tensor_formats = {TensorFormats::CHWNc4};
        if (available)
            return config;
        return None;
    }
};

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormatConfigID::NCHW44> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW44;
        config.config_id = OprFormatConfigID::NCHW44;
        bool available = true;
        auto check_dtype = [](DType dt, bool is_bias) {
            bool f32_config = dt.enumv() == DTypeEnum::Float32;
            auto i8_dtype = DTypeEnum::QuantizedS8;
            if (is_bias)
                i8_dtype = DTypeEnum::QuantizedS32;
            bool i8_config = dt.enumv() == i8_dtype;
            return f32_config || i8_config;
        };
        // setup dtypes
        for (size_t i = 0; i < opr->input().size(); ++i) {
            bool is_bias =
                    ConvParamTrait<Opr>::has_bias && i == ConvParamTrait<Opr>::bias_idx;
            available &= check_dtype(opr->input(i)->dtype(), is_bias);
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        available &= check_dtype(opr->output(0)->dtype(), false);
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

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormatConfigID::NCHW44_HYBRID> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW44;
        config.config_id = OprFormatConfigID::NCHW44_HYBRID;
        bool available = true;
        auto check_dtype = [](DType dt, bool is_bias) {
            bool f32_config = dt.enumv() == DTypeEnum::Float32;
            auto i8_dtype = DTypeEnum::QuantizedS8;
            if (is_bias)
                i8_dtype = DTypeEnum::QuantizedS32;
            bool i8_config = dt.enumv() == i8_dtype;
            return f32_config || i8_config;
        };
        // setup dtypes
        for (size_t i = 0; i < opr->input().size(); ++i) {
            bool is_bias =
                    ConvParamTrait<Opr>::has_bias && i == ConvParamTrait<Opr>::bias_idx;
            available &= check_dtype(opr->input(i)->dtype(), is_bias);
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        // FIXME: hack for nchw nchw44 hybrid mode
        static_assert(
                std::is_same<Opr, opr::ConvolutionForward>::value ||
                        std::is_same<Opr, opr::ConvBiasForward>::value,
                "nchw44 hybrid only support conv or conv_bias opr");
        size_t in_channel = opr->input(0)->shape()[1];
        available &= in_channel <= 4_z && check_dtype(opr->output(0)->dtype(), false);
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        available &= conv.param().sparse == Opr::Param::Sparse::DENSE;
        config.input_tensor_formats = {
                TensorFormats::NCHW, TensorFormats::KRSCk4, TensorFormats::NCHWc4,
                TensorFormats::NCHWc4};
        config.output_tensor_formats = {TensorFormats::NCHWc4};
        if (!available)
            return None;
        return config;
    }
};

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormatConfigID::NCHW88> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW88;
        config.config_id = OprFormatConfigID::NCHW88;
        bool available = true;
        // setup dtypes
        for (size_t i = 0; i < opr->input().size(); ++i) {
            available &= opr->input(i)->dtype().enumv() == DTypeEnum::Float32;
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        available &= opr->output(0)->dtype().enumv() == DTypeEnum::Float32;
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

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormatConfigID::NCHW88_HYBRID> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW88;
        config.config_id = OprFormatConfigID::NCHW88_HYBRID;
        bool available = true;
        // setup dtypes
        for (size_t i = 0; i < opr->input().size(); ++i) {
            available &= opr->input(i)->dtype().enumv() == DTypeEnum::Float32;
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        // FIXME: hack for nchw nchw88 hybrid mode
        static_assert(
                std::is_same<Opr, opr::ConvolutionForward>::value ||
                        std::is_same<Opr, opr::ConvBiasForward>::value,
                "nchw nchw88 hybrid mode only support conv or conv_bias opr");
        size_t in_channel = opr->input(0)->shape()[1];
        available &= in_channel <= 8_z &&
                     opr->output(0)->dtype().enumv() == DTypeEnum::Float32;
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        available &= conv.param().sparse == Opr::Param::Sparse::DENSE;
        // setup tensor formats
        config.input_tensor_formats = {
                TensorFormats::NCHW, TensorFormats::KRSCk8, TensorFormats::NCHWc8,
                TensorFormats::NCHWc8};
        config.output_tensor_formats = {TensorFormats::NCHWc8};
        if (!available)
            return None;
        return config;
    }
};

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormatConfigID::NCHW44_DOT> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW44_DOT;
        config.config_id = OprFormatConfigID::NCHW44_DOT;
        bool available = true;
        // setup dtypes
        for (size_t i = 0; i < opr->input().size(); ++i) {
            if (i == 2) {
                available &= opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS32;
            } else {
                available &=
                        opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS8 ||
                        opr->input(i)->dtype().enumv() == DTypeEnum::Quantized8Asymm;
            }
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        available &= opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8 ||
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

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormatConfigID::NCHW44_DOT_HYBRID> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW44_DOT;
        config.config_id = OprFormatConfigID::NCHW44_DOT_HYBRID;
        bool available = true;
        // setup dtypes
        for (size_t i = 0; i < opr->input().size(); ++i) {
            if (i == 2) {
                available &= opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS32;
            } else {
                available &=
                        opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS8 ||
                        opr->input(i)->dtype().enumv() == DTypeEnum::Quantized8Asymm;
            }
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        available &= opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8 ||
                     opr->output(0)->dtype().enumv() == DTypeEnum::Quantized8Asymm;
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        available &= conv.param().sparse == Opr::Param::Sparse::DENSE;
        // FIXME: hack for nchw nchw44 dot hybrid mode
        static_assert(
                std::is_same<Opr, opr::ConvolutionForward>::value ||
                        std::is_same<Opr, opr::ConvBiasForward>::value,
                "nchw44 dot hybrid only support conv or conv_bias opr");
        size_t in_channel = opr->input(0)->shape()[1];
        available &= in_channel <= 4_z;
        // setup tensor formats
        config.input_tensor_formats = {
                TensorFormats::NCHW, TensorFormats::KRSCk4, TensorFormats::NCHWc4,
                TensorFormats::NCHWc4};
        config.output_tensor_formats = {TensorFormats::NCHWc4};
        if (!available)
            return None;
        return config;
    }
};

template <typename Opr>
struct ConvTensorFormatsDispatcherImpl<Opr, OprFormatConfigID::NHWCD4> {
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NHWCD4;
        config.config_id = OprFormatConfigID::NHWCD4;
        for (size_t i = 0; i < opr->input().size(); ++i) {
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 1 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        if (conv.param().sparse == Opr::Param::Sparse::DENSE) {
            if (opr->input(1)->dtype().enumv() == DTypeEnum::QuantizedS8 ||
                opr->input(1)->dtype().enumv() == DTypeEnum::Quantized8Asymm) {
                config.input_tensor_formats = {
                        TensorFormats::NHCWc4, TensorFormats::KRSCk4c4,
                        TensorFormats::NHCWc4, TensorFormats::NHCWc4};
            } else {
                config.input_tensor_formats = {
                        TensorFormats::NHCWc4, TensorFormats::KRSCk4,
                        TensorFormats::NHCWc4, TensorFormats::NHCWc4};
            }
        } else {
            mgb_assert(conv.param().sparse == Opr::Param::Sparse::GROUP);
            if (is_channel_wise_conv<Opr>(opr)) {
                config.input_tensor_formats = {
                        TensorFormats::NHCWc4, TensorFormats::C1RSc4,
                        TensorFormats::NHCWc4, TensorFormats::NHCWc4};
            } else {
                if (opr->input(1)->dtype().enumv() == DTypeEnum::QuantizedS8 ||
                    opr->input(1)->dtype().enumv() == DTypeEnum::Quantized8Asymm) {
                    config.input_tensor_formats = {
                            TensorFormats::NHCWc4, TensorFormats::GKRSCk4c4,
                            TensorFormats::NHCWc4, TensorFormats::NHCWc4};
                } else {
                    config.input_tensor_formats = {
                            TensorFormats::NHCWc4, TensorFormats::GKRSCk4,
                            TensorFormats::NHCWc4, TensorFormats::NHCWc4};
                }
            }
        }
        config.output_tensor_formats = {TensorFormats::NHCWc4};
        return config;
    }
};

template <>
struct ConvTensorFormatsDispatcherImpl<
        opr::ConvolutionBackwardData, OprFormatConfigID::NCHW> {
    using Opr = opr::ConvolutionBackwardData;
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW;
        config.config_id = OprFormatConfigID::NCHW;
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
                    TensorFormats::KCRS, TensorFormats::NCHW, TensorFormats::NCHW,
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
struct ConvTensorFormatsDispatcherImpl<
        opr::ConvolutionBackwardData, OprFormatConfigID::NCHW4> {
    using Opr = opr::ConvolutionBackwardData;
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NCHW4;
        config.config_id = OprFormatConfigID::NCHW4;
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
                TensorFormats::KCRSc4, TensorFormats::NCHWc4, TensorFormats::NCHWc4,
                TensorFormats::NCHWc4};
        config.output_tensor_formats = {TensorFormats::NCHWc4};
        if (available)
            return config;
        return None;
    }
};

template <>
struct ConvTensorFormatsDispatcherImpl<
        opr::ConvolutionBackwardData, OprFormatConfigID::NHWC> {
    using Opr = opr::ConvolutionBackwardData;
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NHWC;
        config.config_id = OprFormatConfigID::NHWC;
        bool available = true;
        for (size_t i = 0; i < opr->input().size(); ++i) {
            available &=
                    opr->input(i)->dtype().enumv() == DTypeEnum::QuantizedS8 ||
                    DNN_FLOAT16_SELECT(
                            opr->input(i)->dtype().enumv() == DTypeEnum::Float16, true);
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 0 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        available &=
                opr->output(0)->dtype().enumv() == DTypeEnum::QuantizedS8 ||
                DNN_FLOAT16_SELECT(
                        opr->output(0)->dtype().enumv() == DTypeEnum::Float16, true);
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        available &= conv.param().sparse == opr::ConvBias::Param::Sparse::DENSE;
        config.input_tensor_formats = {
                TensorFormats::KRSC, TensorFormats::NHWC, TensorFormats::NHWC,
                TensorFormats::NHWC};
        config.output_tensor_formats = {TensorFormats::NHWC};
        if (available)
            return config;
        return None;
    }
};

template <>
struct ConvTensorFormatsDispatcherImpl<
        opr::ConvolutionBackwardData, OprFormatConfigID::NHWCD4> {
    using Opr = opr::ConvolutionBackwardData;
    static Maybe<OprTensorFormatsConfiguration> dispatch(const OperatorNodeBase* opr) {
        const auto& conv = opr->cast_final_safe<Opr>();
        OprTensorFormatsConfiguration config;
        config.typeinfo = opr->dyn_typeinfo();
        config.opr_format = OprFormat::NHWCD4;
        config.config_id = OprFormatConfigID::NHWCD4;
        for (size_t i = 0; i < opr->input().size(); ++i) {
            config.input_dtypes.emplace_back(opr->input(i)->dtype().enumv());
            TensorType tensor_type = i == 0 ? TensorType::WEIGHT : TensorType::FEATURE;
            config.input_tensor_types.emplace_back(tensor_type);
        }
        config.output_dtypes.emplace_back(opr->output(0)->dtype().enumv());
        if (conv.param().sparse == Opr::Param::Sparse::DENSE) {
            if (opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS8 ||
                opr->input(0)->dtype().enumv() == DTypeEnum::Quantized8Asymm) {
                config.input_tensor_formats = {
                        TensorFormats::KRSCk4c4, TensorFormats::NHCWc4,
                        TensorFormats::NHCWc4, TensorFormats::NHCWc4};
            } else {
                config.input_tensor_formats = {
                        TensorFormats::KRSCk4, TensorFormats::NHCWc4,
                        TensorFormats::NHCWc4, TensorFormats::NHCWc4};
            }
        } else {
            mgb_assert(conv.param().sparse == Opr::Param::Sparse::GROUP);
            if (is_channel_wise_conv<Opr>(opr)) {
                config.input_tensor_formats = {
                        TensorFormats::C1RSc4, TensorFormats::NHCWc4,
                        TensorFormats::NHCWc4, TensorFormats::NHCWc4};
            } else {
                if (opr->input(0)->dtype().enumv() == DTypeEnum::QuantizedS8 ||
                    opr->input(0)->dtype().enumv() == DTypeEnum::Quantized8Asymm) {
                    config.input_tensor_formats = {
                            TensorFormats::GKRSCk4c4, TensorFormats::NHCWc4,
                            TensorFormats::NHCWc4, TensorFormats::NHCWc4};
                } else {
                    config.input_tensor_formats = {
                            TensorFormats::GKRSCk4, TensorFormats::NHCWc4,
                            TensorFormats::NHCWc4, TensorFormats::NHCWc4};
                }
            }
        }
        config.output_tensor_formats = {TensorFormats::NHCWc4};
        return config;
    }
};

struct StaticData {
    struct KeyHash {
        size_t operator()(const std::pair<Typeinfo*, OprFormatConfigID>& val) const {
            size_t h1 = mgb::hash<Typeinfo*>(val.first);
            size_t h2 = std::hash<uint32_t>()(static_cast<uint32_t>(val.second));
            return mgb::hash_pair_combine(h1, h2);
        }
    };
    using OprTensorFormatsDispatcher =
            OprTensorFormatsConfiguration::OprTensorFormatsDispatcher;
    std::unordered_map<
            std::pair<Typeinfo*, OprFormatConfigID>, OprTensorFormatsDispatcher,
            KeyHash>
            typefmt2dispatcher;
    StaticData();
};

StaticData::StaticData() {
#define OPR_TENSOR_FORMATS_CONFIG_REG(_Opr, _fmt)                           \
    typefmt2dispatcher[{opr::_Opr::typeinfo(), OprFormatConfigID::_fmt}] =  \
            [](const OperatorNodeBase* opr) {                               \
                MIDOUT_B(opr::_Opr, midout_iv(OprFormatConfigID::_fmt))     \
                return ConvTensorFormatsDispatcherImpl<                     \
                        opr::_Opr, OprFormatConfigID::_fmt>::dispatch(opr); \
                MIDOUT_E                                                    \
            }

#define OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(_Opr, _fmt)            \
    typefmt2dispatcher[{opr::_Opr::typeinfo(), OprFormatConfigID::_fmt}] = \
            [](const OperatorNodeBase* opr) {                              \
                MIDOUT_B(opr::_Opr, midout_iv(OprFormatConfigID::_fmt))    \
                return OprSingleInOutTensorFormatsDispatcherImpl<          \
                        OprFormatConfigID::_fmt>::dispatch(opr);           \
                MIDOUT_E                                                   \
            }

    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NHWC);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW4);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, CHWN4);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW32);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW32_NCHW4);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW4_NCHW32);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW64);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW44);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW88);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW88_HYBRID);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW44_DOT);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW44_HYBRID);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NCHW44_DOT_HYBRID);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvBias, NHWCD4);

    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionForward, NCHW);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionForward, NHWC);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionForward, NCHW4);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionForward, NCHW44);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionForward, NCHW88);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionForward, NCHW88_HYBRID);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionForward, NCHW44_DOT);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionForward, NCHW44_HYBRID);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionForward, NCHW44_DOT_HYBRID);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionForward, NHWCD4);

    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionBackwardData, NCHW);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionBackwardData, NHWC);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionBackwardData, NCHW4);
    OPR_TENSOR_FORMATS_CONFIG_REG(ConvolutionBackwardData, NHWCD4);

    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(WarpPerspectiveForward, NCHW);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(WarpPerspectiveForward, NHWC);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(WarpPerspectiveForward, NCHW4);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(WarpPerspectiveForward, NCHW64);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(WarpPerspectiveForward, NHWCD4);

    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, NCHW);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, NHWC);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, NCHW4);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, CHWN4);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, NCHW32);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, NCHW64);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, NCHW44);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, NCHW88);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(PoolingForward, NHWCD4);

    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(ResizeForward, NCHW);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(ResizeForward, NCHW44);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(ResizeForward, NCHW88);
    OPR_SINGLE_IN_OUT_TENSOR_FORMATS_CONFIG_REG(ResizeForward, NHWCD4);

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
        Typeinfo* type, OprFormatConfigID config_id) {
    auto&& typefmt2dispatcher = static_data().typefmt2dispatcher;
    auto iter = typefmt2dispatcher.find(std::make_pair(type, config_id));
    mgb_assert(
            iter != typefmt2dispatcher.end(),
            "cannot find OprTensorFormatsDispatcher for opr type(%s) and "
            "opr format configuration id(%s)",
            type->name, config_id_to_string(config_id));
    return &iter->second;
}

// vim: syntax=cpp.doxygen
