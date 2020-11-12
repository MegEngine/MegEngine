/**
 * \file dnn/src/common/conv_bias.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/common/conv_bias.h"
#include "megdnn/oprs/nn.h"
#include "src/common/utils.h"
#include "src/common/opr_delegate.h"

namespace megdnn {

void ConvBiasForward::deduce_dtype(DType src, DType filter, DType /* bias */,
                                   DType /* z */, DType& dst) {
    check_or_deduce_dtype_fwd(src, filter, dst);
}

void ConvBiasForward::deduce_layout(const TensorLayout& src,
                                    const TensorLayout& filter,
                                    const TensorLayout& /* bias */,
                                    const TensorLayout& /* z */,
                                    TensorLayout& dst) {
    deduce_layout_fwd(src, filter, dst);
}

ConvBiasForward::CanonizedFilterMeta ConvBiasForward::check_exec(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& bias, const TensorLayout& z,
        const TensorLayout& dst, size_t workspace_in_bytes,
        const PreprocessedFilter* preprocessed_filter) {
    if ((param().format == param::ConvBias::Format::NCHW_WINOGRAD ||
         param().format == param::ConvBias::Format::NCHW88_WINOGRAD ||
         param().format == param::ConvBias::Format::NCHW44_WINOGRAD) &&
        src.dtype.category() == DTypeCategory::QUANTIZED) {
        megdnn_assert(filter.dtype.enumv() == DTypeEnum::QuantizedS16 ||
                      //!int8 winogradf23_44 using float,QuantizedS32 take the scale
                      filter.dtype.enumv() == DTypeEnum::QuantizedS32);
        megdnn_assert(src.dtype.enumv() == DTypeEnum::QuantizedS8 ||
                      src.dtype.enumv() == DTypeEnum::Quantized8Asymm);
    } else {
        megdnn_assert(src.dtype.enumv() == filter.dtype.enumv());
    }
    if (src.dtype.enumv() == DTypeEnum::QuantizedS8) {
        if (bias.dtype.enumv() == DTypeEnum::QuantizedS32) {
            float scale_src = src.dtype.param<dtype::QuantizedS8>().scale;
            float scale_filter = 0.f;
            if (param().format == param::ConvBias::Format::NCHW_WINOGRAD ||
                param().format == param::ConvBias::Format::NCHW88_WINOGRAD ||
                param().format == param::ConvBias::Format::NCHW44_WINOGRAD) {
                if (filter.dtype.enumv() == DTypeEnum::QuantizedS32) {
                    //! int8 winogradf23_44 using float,QuantizedS32 take the
                    //! scale
                    scale_filter =
                            filter.dtype.param<dtype::QuantizedS32>().scale;
                } else {
                    scale_filter =
                            filter.dtype.param<dtype::QuantizedS16>().scale;
                }
            } else {
                scale_filter = filter.dtype.param<dtype::QuantizedS8>().scale;
            }
            float scale_bias = bias.dtype.param<dtype::QuantizedS32>().scale;
            megdnn_assert(
                    std::abs(scale_src * scale_filter - scale_bias) < 1e-6,
                    "scale_src: %f scale_filter: %f scale_bias: %f", scale_src,
                    scale_filter, scale_bias);
        } else {
            megdnn_assert(bias.dtype.enumv() == DTypeEnum::Float32);
        }
    } else if (src.dtype.enumv() == DTypeEnum::Quantized8Asymm) {
        if (bias.dtype.enumv() == DTypeEnum::QuantizedS32) {
            float scale_src = src.dtype.param<dtype::Quantized8Asymm>().scale;
            float scale_filter = 0.f;
            if (param().format == param::ConvBias::Format::NCHW_WINOGRAD ||
                param().format == param::ConvBias::Format::NCHW88_WINOGRAD ||
                param().format == param::ConvBias::Format::NCHW44_WINOGRAD) {
                scale_filter = filter.dtype.param<dtype::QuantizedS16>().scale;
            } else {
                scale_filter =
                        filter.dtype.param<dtype::Quantized8Asymm>().scale;
            }
            float scale_bias = bias.dtype.param<dtype::QuantizedS32>().scale;
            megdnn_assert(
                    std::abs(scale_src * scale_filter - scale_bias) < 1e-6,
                    "scale_src: %f scale_filter: %f scale_bias: %f", scale_src,
                    scale_filter, scale_bias);
        } else {
            megdnn_assert(bias.dtype.enumv() == DTypeEnum::Float32);
        }
    }

    auto ret = check_layout_fwd(src, filter, dst);
    megdnn_assert_contiguous(bias);
    auto required_workspace_in_bytes = get_workspace_in_bytes(
            src, filter, bias, z, dst, preprocessed_filter);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes,
                  "worksapce have size of %zu, but need %zu",
                  workspace_in_bytes, required_workspace_in_bytes);
    if (bias.ndim != 0) {
        //! bias.layout == dst.layout failed, no assert information
        auto check_eq = [](const TensorLayout& bias, const TensorLayout& dst) {
            if (dst.dtype.category() == DTypeCategory::QUANTIZED) {
                return bias.eq_shape(dst);
            } else {
                return bias.eq_layout(dst);
            }
        };
        if (check_eq(bias, dst))
            return ret;
        if (param().format == param::ConvBias::Format::NCHW ||
            param().format == param::ConvBias::Format::NCHW_WINOGRAD ||
            param().format == param::ConvBias::Format::NCHW4_NCHW) {
            megdnn_assert(bias.shape[0] == 1);
            megdnn_assert(bias.shape[1] == dst.shape[1], "bias:%s, dst:%s",
                          bias.to_string().c_str(), dst.to_string().c_str());
            megdnn_assert(bias.shape[2] == 1);
            megdnn_assert(bias.shape[3] == 1);
        } else if (param().format == param::ConvBias::Format::NHWC) {
            megdnn_assert(bias.shape[0] == 1);
            megdnn_assert(bias.shape[1] == 1);
            megdnn_assert(bias.shape[2] == 1);
            megdnn_assert(bias.shape[3] == dst.shape[3], "bias:%s, dst:%s",
                          bias.to_string().c_str(), dst.to_string().c_str());
        } else if (param().format == param::ConvBias::Format::NCHW4 ||
                   param().format == param::ConvBias::Format::NCHW44 ||
                   param().format == param::ConvBias::Format::NCHW44_DOT ||
                   param().format == param::ConvBias::Format::NCHW44_WINOGRAD ||
                   param().format == param::ConvBias::Format::NCHW32_NCHW4) {
            megdnn_assert(bias.shape[0] == 1);
            megdnn_assert(bias.shape[1] == dst.shape[1], "bias:%s, dst:%s",
                          bias.to_string().c_str(), dst.to_string().c_str());
            megdnn_assert(bias.shape[2] == 1);
            megdnn_assert(bias.shape[3] == 1);
            megdnn_assert(bias.shape[4] == 4);
        } else if (param().format == param::ConvBias::Format::NCHW8 ||
                   param().format == param::ConvBias::Format::NCHW88 ||
                   param().format == param::ConvBias::Format::NCHW88_WINOGRAD) {
            megdnn_assert(bias.shape[0] == 1);
            megdnn_assert(bias.shape[1] == dst.shape[1], "bias:%s, dst:%s",
                          bias.to_string().c_str(), dst.to_string().c_str());
            megdnn_assert(bias.shape[2] == 1);
            megdnn_assert(bias.shape[3] == 1);
            megdnn_assert(bias.shape[4] == 8);
        } else if (param().format == param::ConvBias::Format::NCHW32 ||
                   param().format == param::ConvBias::Format::NCHW4_NCHW32) {
            megdnn_assert(bias.shape[0] == 1);
            megdnn_assert(bias.shape[1] == dst.shape[1], "bias:%s, dst:%s",
                          bias.to_string().c_str(), dst.to_string().c_str());
            megdnn_assert(bias.shape[2] == 1);
            megdnn_assert(bias.shape[3] == 1);
            megdnn_assert(bias.shape[4] == 32);
        } else if (param().format == param::ConvBias::Format::CHWN4) {
            megdnn_assert(bias.shape[0] == dst.shape[0], "bias:%s, dst:%s",
                          bias.to_string().c_str(), dst.to_string().c_str());
            megdnn_assert(bias.shape[1] == 1);
            megdnn_assert(bias.shape[2] == 1);
            megdnn_assert(bias.shape[3] == 1);
            megdnn_assert(bias.shape[4] == 4);
        } else {
            megdnn_assert(param().format == param::ConvBias::Format::NHWCD4);
            megdnn_assert(bias.shape[0] == 1);
            megdnn_assert(bias.shape[1] == 1);
            megdnn_assert(bias.shape[2] == dst.shape[2], "bias:%s, dst:%s",
                          bias.to_string().c_str(), dst.to_string().c_str());
            megdnn_assert(bias.shape[3] == 1);
            megdnn_assert(bias.shape[4] == 4);
        }
    }

    if (z.ndim != 0) {
        megdnn_assert(param().format != param::ConvBias::Format::NCHW_WINOGRAD);
        megdnn_assert(param().format !=
                      param::ConvBias::Format::NCHW88_WINOGRAD);
        megdnn_assert(param().format !=
                      param::ConvBias::Format::NCHW44_WINOGRAD);
        megdnn_assert(param().format != param::ConvBias::Format::NCHW4_NCHW32);
        megdnn_assert(param().format != param::ConvBias::Format::NCHW32_NCHW4);
        megdnn_assert(z.dtype.enumv() == dst.dtype.enumv());
        megdnn_assert(z.eq_shape(dst));
    }
    return ret;
}
/*!
 * \brief deduce the origin filter layout and param after winograd transformed
 */
void ConvBiasForward::deduce_winograd_origin_layout_and_param(
        const Param::Format format, const size_t output_block_size,
        const TensorLayout& src_layout,
        const TensorLayout& winograd_filter_layout, TensorLayout& origin_layout,
        Param& origin_param) {
    if (format == megdnn::param::ConvBias::Format::NCHW88_WINOGRAD ||
        format == megdnn::param::ConvBias::Format::NCHW44_WINOGRAD ||
        format == megdnn::param::ConvBias::Format::NCHW_WINOGRAD) {
        //! change NCHWxx_WINOGRAD to NCHWxx
        size_t OC = 0;
        size_t IC = 0;
        size_t GROUP = 1;
        size_t FH = winograd_filter_layout[1] - output_block_size + 1;

        //! {alpha, alpha, IC, OC}
        if (winograd_filter_layout.ndim == 4) {
            OC = winograd_filter_layout[3];
            IC = winograd_filter_layout[2];
        }
        //! {group, alpha, alpha, IC, OC}
        else if (winograd_filter_layout.ndim == 5) {
            OC = winograd_filter_layout[4];
            IC = winograd_filter_layout[3];
            GROUP = winograd_filter_layout[0];
        }
        //! {alpha, alpha, OC/f, IC/f, f, f}
        else if (winograd_filter_layout.ndim == 6) {
            OC = winograd_filter_layout[2] * winograd_filter_layout[5];
            IC = winograd_filter_layout[3] * winograd_filter_layout[4];
        }
        //! {group, alpha, alpha, OC/f, IC/f, f, f}
        else if (winograd_filter_layout.ndim == 7) {
            OC = winograd_filter_layout[3] * winograd_filter_layout[6];
            IC = winograd_filter_layout[4] * winograd_filter_layout[5];
            GROUP = winograd_filter_layout[0];
        }
        auto origin_data_type = winograd_filter_layout.dtype;
        if (src_layout.dtype.enumv() == DTypeEnum::QuantizedS8) {
            if (origin_data_type.enumv() == DTypeEnum::QuantizedS16) {
                float scale =
                        origin_data_type.param<dtype::QuantizedS16>().scale;
                origin_data_type = megdnn::dtype::QuantizedS8(scale);
            } else {
                //! In order to braing the sacle of filter, the transformed
                //! qint8 winograd filter computing with float dtype is Qint32
                megdnn_assert(origin_data_type.enumv() ==
                              DTypeEnum::QuantizedS32);
                float scale =
                        origin_data_type.param<dtype::QuantizedS32>().scale;
                origin_data_type = megdnn::dtype::QuantizedS8(scale);
            }
        }

        if (GROUP == 1) {
            if (format == megdnn::param::ConvBias::Format::NCHW_WINOGRAD) {
                origin_layout =
                        TensorLayout({OC, IC, FH, FH}, origin_data_type);
            } else if (format ==
                       megdnn::param::ConvBias::Format::NCHW44_WINOGRAD) {
                origin_layout = TensorLayout({OC / 4, IC / 4, FH, FH, 4, 4},
                                             origin_data_type);
            } else {
                megdnn_assert(format ==
                              megdnn::param::ConvBias::Format::NCHW88_WINOGRAD);
                origin_layout = TensorLayout({OC / 8, IC / 8, FH, FH, 8, 8},
                                             origin_data_type);
            }
        } else {
            if (format == megdnn::param::ConvBias::Format::NCHW_WINOGRAD) {
                origin_layout =
                        TensorLayout({GROUP, OC, IC, FH, FH}, origin_data_type);
            } else if (format ==
                       megdnn::param::ConvBias::Format::NCHW44_WINOGRAD) {
                origin_layout =
                        TensorLayout({GROUP, OC / 4, IC / 4, FH, FH, 4, 4},
                                     origin_data_type);
            } else {
                megdnn_assert(format ==
                              megdnn::param::ConvBias::Format::NCHW88_WINOGRAD);
                origin_layout =
                        TensorLayout({GROUP, OC / 8, IC / 8, FH, FH, 8, 8},
                                     origin_data_type);
            }
        }
        origin_param.output_block_size = 0;
        if (format == megdnn::param::ConvBias::Format::NCHW_WINOGRAD) {
            origin_param.format = megdnn::param::ConvBias::Format::NCHW;
        } else if (format == megdnn::param::ConvBias::Format::NCHW44_WINOGRAD) {
            origin_param.format = megdnn::param::ConvBias::Format::NCHW44;
        } else {
            megdnn_assert(format ==
                          megdnn::param::ConvBias::Format::NCHW88_WINOGRAD);
            origin_param.format = megdnn::param::ConvBias::Format::NCHW88;
        }
    }
}

template <typename T>
struct NCHWParamTrait;

template <typename T>
struct NCHW44ParamTrait;

std::string ConvBias::WinogradParam::to_string() const {
    return ssprintf("%u:%u:%u", channel_block_size, output_block_size,
                    tile_size);
}

template <typename T>
std::string ConvBias::algo_name(const std::string& base, const T& p,
                                param::ConvBias::Format format) {
    if (format == param::ConvBias::Format::NCHW) {
        return ssprintf("%s:%s:%s", NCHWParamTrait<T>::category.c_str(),
                        base.c_str(), p.to_string().c_str());
    } else if (format == param::ConvBias::Format::NCHW44) {
        return ssprintf("%s:%s:%s", NCHW44ParamTrait<T>::category.c_str(),
                        base.c_str(), p.to_string().c_str());
    }
    megdnn_throw("Invalid format");
    return "";
}

#define FOREACH_CONV_BIAS_PARAM(cb) \
    cb(WinogradParam) cb(DirectParam) cb(MatmulParam) cb(DefaultParam)

#define cb(pt)                              \
    template <>                             \
    struct NCHWParamTrait<ConvBias::pt> {   \
        static const std::string category;  \
    };                                      \
    template <>                             \
    struct NCHW44ParamTrait<ConvBias::pt> { \
        static const std::string category;  \
    };
FOREACH_CONV_BIAS_PARAM(cb)
#undef cb

#define cb(pt, ct)                                                 \
    const std::string NCHWParamTrait<ConvBias::pt>::category = ct; \
    const std::string NCHW44ParamTrait<ConvBias::pt>::category = ct
cb(DirectParam, "DIRECT");
cb(MatmulParam, "MATMUL");
cb(DefaultParam, "DEFAULT");
#undef cb

const std::string NCHWParamTrait<ConvBias::WinogradParam>::category =
        "WINOGRAD";
const std::string NCHW44ParamTrait<ConvBias::WinogradParam>::category =
        "WINOGRAD_NCHW44";

#define cb(t)                                              \
    template std::string ConvBias::algo_name<ConvBias::t>( \
            const std::string& base, const ConvBias::t& p, \
            param::ConvBias::Format format);
FOREACH_CONV_BIAS_PARAM(cb)
#undef cb

ConvBias::WinogradParam ConvBias::parse_winograd_name(
        const std::string& algo_name) {
    ConvBias::WinogradParam ret = INVALID_WINOGRAD_PARAM;
    char base[128];
    char name[128];

    auto parse = [&](const std::string& algo_name,
                     const std::string& pre) -> auto {
        memset(name, 0, 128);
        sscanf(algo_name.c_str(), "%[^:]:%[^:]:%u:%u:%u", name, base,
               &(ret.channel_block_size), &(ret.output_block_size),
               &(ret.tile_size));
        if (strcmp(name, pre.c_str())) {
            ret = INVALID_WINOGRAD_PARAM;
            return false;
        }
        if (ret.tile_size == 0 || ret.output_block_size == 0 ||
            ret.channel_block_size == 0) {
            ret = INVALID_WINOGRAD_PARAM;
            return false;
        }
        return true;
    };

    if (parse(algo_name, "WINOGRAD_NCHW44")) {
        return ret;
    } else {
        parse(algo_name, "WINOGRAD");
        return ret;
    }
}

constexpr ConvBias::WinogradParam ConvBias::INVALID_WINOGRAD_PARAM;

void handle_bias_and_nonlinear(Handle* handle, param::ConvBias args,
                               const TensorND* conv_dst_tensor,
                               const TensorND* dst_tensor,
                               const TensorND* bias_tensor) {
    using NonlineMode = param::ConvBias::NonlineMode;
    switch (args.nonlineMode) {
#define cb(_mode)                                                          \
    case NonlineMode::_mode: {                                             \
        if (conv_dst_tensor->layout.dtype.category() !=                    \
            DTypeCategory::QUANTIZED) {                                    \
            auto nonlinear = handle->create_operator<ElemwiseForward>();   \
            if (bias_tensor->layout.ndim > 0) {                            \
                nonlinear->param().mode =                                  \
                        Elemwise::Param::Mode::FUSE_ADD_##_mode;           \
                nonlinear->exec({*conv_dst_tensor, *bias_tensor},          \
                                *dst_tensor);                              \
            } else {                                                       \
                nonlinear->param().mode = Elemwise::Param::Mode::_mode;    \
                nonlinear->exec({*conv_dst_tensor}, *dst_tensor);          \
            }                                                              \
        } else {                                                           \
            auto nonlinear = handle->create_operator<ElemwiseMultiType>(); \
            if (bias_tensor->layout.ndim > 0) {                            \
                nonlinear->param().mode =                                  \
                        ElemwiseMultiType::Param::Mode::QFUSE_ADD_##_mode; \
                nonlinear->exec({*conv_dst_tensor, *bias_tensor},          \
                                *dst_tensor);                              \
            } else {                                                       \
                nonlinear->param().mode =                                  \
                        ElemwiseMultiType::Param::Mode::Q##_mode;          \
                nonlinear->exec({*conv_dst_tensor}, *dst_tensor);          \
            }                                                              \
        }                                                                  \
        break;                                                             \
    }
        cb(RELU);
        cb(H_SWISH);
#undef cb
        case NonlineMode::SIGMOID: {
            megdnn_assert(conv_dst_tensor->layout.dtype.category() !=
                          DTypeCategory::QUANTIZED);
            auto nonlinear = handle->create_operator<ElemwiseForward>();
            if (bias_tensor->layout.ndim > 0) {
                nonlinear->param().mode =
                        Elemwise::Param::Mode::FUSE_ADD_SIGMOID;
                nonlinear->exec({*conv_dst_tensor, *bias_tensor},
                                *conv_dst_tensor);
            } else {
                nonlinear->param().mode = Elemwise::Param::Mode::SIGMOID;
                nonlinear->exec({*conv_dst_tensor}, *conv_dst_tensor);
            }
            break;
        }
        case NonlineMode::IDENTITY: {
            if (bias_tensor->layout.ndim > 0) {
                if (dst_tensor->layout.dtype.category() ==
                    DTypeCategory::QUANTIZED) {
                    auto nonlinear =
                            handle->create_operator<ElemwiseMultiType>();
                    nonlinear->param().mode =
                            ElemwiseMultiType::Param::Mode::QADD;
                    nonlinear->exec({*conv_dst_tensor, *bias_tensor},
                                    *dst_tensor);
                } else {
                    auto nonlinear = handle->create_operator<Elemwise>();
                    nonlinear->param().mode = Elemwise::Param::Mode::ADD;
                    nonlinear->exec({*conv_dst_tensor, *bias_tensor},
                                    *dst_tensor);
                }
            } else {
                if (conv_dst_tensor->layout.dtype != dst_tensor->layout.dtype) {
                    handle->create_operator<TypeCvt>()->exec({*conv_dst_tensor},
                                                             *dst_tensor);
                }
            }
            break;
        }
        default:
            megdnn_assert(false);
    }
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
