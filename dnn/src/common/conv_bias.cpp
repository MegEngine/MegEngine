/**
 * \file dnn/src/common/conv_bias.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/conv_bias.h"
#include "megdnn/oprs/nn.h"
#include "src/common/utils.h"

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
        const TensorLayout& dst, size_t workspace_in_bytes) {
    if ((param().format == param::ConvBias::Format::NCHW_WINOGRAD ||
         param().format == param::ConvBias::Format::NCHW88_WINOGRAD) &&
        src.dtype.category() == DTypeCategory::QUANTIZED) {
        megdnn_assert(filter.dtype.enumv() == DTypeEnum::QuantizedS16);
        megdnn_assert(src.dtype.enumv() == DTypeEnum::QuantizedS8 ||
                      src.dtype.enumv() == DTypeEnum::Quantized8Asymm);
    } else {
        megdnn_assert(src.dtype.enumv() == filter.dtype.enumv());
    }
    if (src.dtype.enumv() == DTypeEnum::QuantizedS8) {
        float scale_src = src.dtype.param<dtype::QuantizedS8>().scale;
        float scale_filter = 0.f;
        if (param().format == param::ConvBias::Format::NCHW_WINOGRAD ||
            param().format == param::ConvBias::Format::NCHW88_WINOGRAD) {
            scale_filter = filter.dtype.param<dtype::QuantizedS16>().scale;
        } else {
            scale_filter = filter.dtype.param<dtype::QuantizedS8>().scale;
        }
        float scale_bias = bias.dtype.param<dtype::QuantizedS32>().scale;
        megdnn_assert(std::abs(scale_src * scale_filter - scale_bias) < 1e-6,
                      "scale_src: %f scale_filter: %f scale_bias: %f",
                      scale_src, scale_filter, scale_bias);
    } else if (src.dtype.enumv() == DTypeEnum::Quantized8Asymm) {
        float scale_src = src.dtype.param<dtype::Quantized8Asymm>().scale;
        float scale_filter = 0.f;
        if (param().format == param::ConvBias::Format::NCHW_WINOGRAD ||
            param().format == param::ConvBias::Format::NCHW88_WINOGRAD) {
            scale_filter = filter.dtype.param<dtype::QuantizedS16>().scale;
        } else {
            scale_filter = filter.dtype.param<dtype::Quantized8Asymm>().scale;
        }
        float scale_bias = bias.dtype.param<dtype::QuantizedS32>().scale;
        megdnn_assert(std::abs(scale_src * scale_filter - scale_bias) < 1e-6,
                      "scale_src: %f scale_filter: %f scale_bias: %f",
                      scale_src, scale_filter, scale_bias);
    }

    auto ret = check_layout_fwd(src, filter, dst);
    megdnn_assert_contiguous(bias);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(src, filter, bias, z, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
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
            param().format == param::ConvBias::Format::NCHW_WINOGRAD) {
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
        } else if (param().format == param::ConvBias::Format::NCHW4) {
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
        } else if (param().format == param::ConvBias::Format::NCHW32) {
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
        megdnn_assert(param().format != param::ConvBias::Format::NCHW88_WINOGRAD);
        megdnn_assert(z.dtype.enumv() == dst.dtype.enumv());
        megdnn_assert(z.eq_shape(dst));
    }
    return ret;
}

template <typename T>
struct ParamTrait;

std::string ConvBias::WinogradParam::to_string() const {
    return ssprintf("%u:%u:%u", channel_block_size, output_block_size,
                    tile_size);
}

template <typename T>
std::string ConvBias::algo_name(const std::string& base, const T& p) {
    return ssprintf("%s:%s:%s", ParamTrait<T>::category.c_str(), base.c_str(),
                    p.to_string().c_str());
}

#define FOREACH_CONV_BIAS_PARAM(cb) \
    cb(WinogradParam)               \
    cb(DirectParam)                 \
    cb(MatmulParam)                 \
    cb(DefaultParam)

#define cb(pt)                             \
    template <>                            \
    struct ParamTrait<ConvBias::pt> {      \
        static const std::string category; \
    };
FOREACH_CONV_BIAS_PARAM(cb)
#undef cb

#define cb(pt, ct) const std::string ParamTrait<ConvBias::pt>::category = ct
cb(WinogradParam, "WINOGRAD");
cb(DirectParam, "DIRECT");
cb(MatmulParam, "MATMUL");
cb(DefaultParam, "DEFAULT");
#undef cb

#define cb(t)                                              \
    template std::string ConvBias::algo_name<ConvBias::t>( \
            const std::string& base, const ConvBias::t& p);
FOREACH_CONV_BIAS_PARAM(cb)
#undef cb

ConvBias::WinogradParam ConvBias::parse_winograd_name(
        const std::string& algo_name) {
    ConvBias::WinogradParam ret = INVALID_WINOGRAD_PARAM;
    char base[128];
    sscanf(algo_name.c_str(), "WINOGRAD:%[^:]:%u:%u:%u", base,
           &(ret.channel_block_size), &(ret.output_block_size),
           &(ret.tile_size));
    if (ret.tile_size == 0 || ret.output_block_size == 0 ||
        ret.channel_block_size == 0) {
        megdnn_log_warn("the algo name %s is not suitable for winograd",
                        algo_name.c_str());
        return INVALID_WINOGRAD_PARAM;
    }
    return ret;
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

//! Only used for naive implementation. DO NOT use the following function in
//! other backends.
void handle_z_inp_and_activation(Handle* handle,
                                 param::ConvBias::NonlineMode nonline_mode,
                                 const TensorND& conv_bias_tensor,
                                 const TensorND& z_tensor,
                                 const TensorND& dst_tensor,
                                 dt_byte* workspace_ptr) {
    auto res = dst_tensor, z_float = z_tensor;
    if (z_tensor.layout.ndim > 0 &&
        z_tensor.layout.dtype.category() != DTypeCategory::FLOAT) {
        dt_byte *res_float_workspace_ptr = nullptr,
                *z_float_workspace_ptr = nullptr;
        megdnn_assert(z_tensor.layout.eq_shape(dst_tensor.layout));
        res_float_workspace_ptr = workspace_ptr;
        z_float_workspace_ptr = res_float_workspace_ptr +
                                TensorLayout{z_tensor.layout, dtype::Float32()}
                                        .span()
                                        .dist_byte();
        res = TensorND{res_float_workspace_ptr,
                       TensorLayout{dst_tensor.layout, dtype::Float32()}};
        z_float = TensorND{z_float_workspace_ptr,
                           TensorLayout{z_tensor.layout, dtype::Float32()}};
    }
    // ====================sfb + z_tensor=====================
    if (z_tensor.layout.ndim > 0) {
        if (z_tensor.layout.dtype.category() != DTypeCategory::FLOAT) {
            auto&& type_cvt = handle->create_operator<TypeCvt>();
            type_cvt->exec(conv_bias_tensor, res);
            type_cvt->exec(z_tensor, z_float);
        }
        auto add_opr = handle->create_operator<ElemwiseForward>();
        add_opr->param().mode = Elemwise::Param::Mode::ADD;
        add_opr->exec({res, z_float}, res);
    } else {
        res = conv_bias_tensor;
    }

    using NonlineMode = param::ConvBias::NonlineMode;

    switch (nonline_mode) {
#define cb(_mode)                                                          \
    case NonlineMode::_mode: {                                             \
        if (res.layout.dtype.category() != DTypeCategory::QUANTIZED) {     \
            auto nonlinear = handle->create_operator<ElemwiseForward>();   \
            nonlinear->param().mode = Elemwise::Param::Mode::_mode;        \
            if (res.layout.dtype == dst_tensor.layout.dtype) {             \
                nonlinear->exec({res}, dst_tensor);                        \
            } else {                                                       \
                nonlinear->exec({res}, res);                               \
                handle->create_operator<TypeCvt>()->exec(res, dst_tensor); \
            }                                                              \
        } else {                                                           \
            auto nonlinear = handle->create_operator<ElemwiseMultiType>(); \
            nonlinear->param().mode =                                      \
                    ElemwiseMultiType::Param::Mode::Q##_mode;              \
            nonlinear->exec({res}, dst_tensor);                            \
        }                                                                  \
        break;                                                             \
    }
        cb(RELU);
        cb(H_SWISH);
#undef cb
        case NonlineMode::SIGMOID: {
            megdnn_assert(res.layout.dtype.category() !=
                          DTypeCategory::QUANTIZED);
            auto nonlinear = handle->create_operator<ElemwiseForward>();
            nonlinear->param().mode = Elemwise::Param::Mode::SIGMOID;
            nonlinear->exec({res}, res);
            if (res.raw_ptr != dst_tensor.raw_ptr) {
                handle->create_operator<TypeCvt>()->exec(res, dst_tensor);
            }
            break;
        }
        case NonlineMode::IDENTITY: {
            if (res.raw_ptr != dst_tensor.raw_ptr) {
                handle->create_operator<TypeCvt>()->exec(res, dst_tensor);
            }
            break;
        }
        default:
            megdnn_assert(false);
    }
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
