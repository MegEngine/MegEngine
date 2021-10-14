/**
 * \file dnn/src/common/conv_bias.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/common/conv_bias.h"
#include "src/common/opr_delegate.h"
#include "src/common/utils.h"

namespace megdnn {
namespace {

void do_check_exec_common(
        ConvBiasForward* opr, const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& bias, const TensorLayout& z, const TensorLayout& dst,
        size_t workspace_in_bytes,
        const ConvBiasForward::PreprocessedFilter* preprocessed_filter) {
    megdnn_assert(
            (src.dtype.enumv() == filter.dtype.enumv()) ||
            (src.dtype.enumv() == DTypeEnum::Quantized4Asymm &&
             filter.dtype.enumv() == DTypeEnum::QuantizedS4));
    // check compatibility of bias's scale
    if (src.dtype.category() == DTypeCategory::QUANTIZED) {
        if (bias.dtype.enumv() == DTypeEnum::QuantizedS32) {
            float scale_expected = mul_scale(src.dtype, filter.dtype);
            float scale_bias = bias.dtype.param<dtype::QuantizedS32>().scale;
            megdnn_assert(
                    std::abs(scale_expected - scale_bias) < 1e-6,
                    "scale_src: %f scale_filter: %f scale_bias: %f",
                    get_scale(src.dtype), get_scale(filter.dtype), scale_bias);
        } else {
            megdnn_assert(bias.dtype.enumv() == DTypeEnum::Float32);
        }
    }

    megdnn_assert_contiguous(bias);
    auto required_workspace_in_bytes =
            opr->get_workspace_in_bytes(src, filter, bias, z, dst, preprocessed_filter);
    megdnn_assert(
            workspace_in_bytes >= required_workspace_in_bytes,
            "worksapce have size of %zu, but need %zu", workspace_in_bytes,
            required_workspace_in_bytes);
    if (bias.ndim != 0) {
        //! bias.layout == dst.layout failed, no assert information
        auto check_eq = [](const TensorLayout& bias, const TensorLayout& dst) {
            if (dst.dtype.category() == DTypeCategory::QUANTIZED) {
                return bias.eq_shape(dst);
            } else {
                return bias.eq_layout(dst);
            }
        };
        if (check_eq(bias, dst)) {
            return;
        }
        if (opr->param().format == param::ConvBias::Format::NCHW ||
            opr->param().format == param::ConvBias::Format::NCHW4_NCHW) {
            megdnn_assert(bias.shape[0] == 1);
            megdnn_assert(
                    bias.shape[1] == dst.shape[1], "bias:%s, dst:%s",
                    bias.to_string().c_str(), dst.to_string().c_str());
            megdnn_assert(bias.shape[2] == 1);
            megdnn_assert(bias.shape[3] == 1);
        } else if (
                opr->param().format == param::ConvBias::Format::NHWC ||
                opr->param().format == param::ConvBias::Format::NCHW4_NHWC) {
            megdnn_assert(bias.shape[0] == 1);
            megdnn_assert(bias.shape[1] == 1);
            megdnn_assert(bias.shape[2] == 1);
            megdnn_assert(
                    bias.shape[3] == dst.shape[3], "bias:%s, dst:%s",
                    bias.to_string().c_str(), dst.to_string().c_str());
        } else if (
                opr->param().format == param::ConvBias::Format::NCHW4 ||
                opr->param().format == param::ConvBias::Format::NCHW44 ||
                opr->param().format == param::ConvBias::Format::NCHW44_DOT ||
                opr->param().format == param::ConvBias::Format::NCHW32_NCHW4) {
            megdnn_assert(bias.shape[0] == 1);
            megdnn_assert(
                    bias.shape[1] == dst.shape[1], "bias:%s, dst:%s",
                    bias.to_string().c_str(), dst.to_string().c_str());
            megdnn_assert(bias.shape[2] == 1);
            megdnn_assert(bias.shape[3] == 1);
            megdnn_assert(bias.shape[4] == 4);
        } else if (
                opr->param().format == param::ConvBias::Format::NCHW8 ||
                opr->param().format == param::ConvBias::Format::NCHW88) {
            megdnn_assert(bias.shape[0] == 1);
            megdnn_assert(
                    bias.shape[1] == dst.shape[1], "bias:%s, dst:%s",
                    bias.to_string().c_str(), dst.to_string().c_str());
            megdnn_assert(bias.shape[2] == 1);
            megdnn_assert(bias.shape[3] == 1);
            megdnn_assert(bias.shape[4] == 8);
        } else if (
                opr->param().format == param::ConvBias::Format::NCHW32 ||
                opr->param().format == param::ConvBias::Format::NCHW4_NCHW32) {
            megdnn_assert(bias.shape[0] == 1);
            megdnn_assert(
                    bias.shape[1] == dst.shape[1], "bias:%s, dst:%s",
                    bias.to_string().c_str(), dst.to_string().c_str());
            megdnn_assert(bias.shape[2] == 1);
            megdnn_assert(bias.shape[3] == 1);
            megdnn_assert(bias.shape[4] == 32);
        } else if (opr->param().format == param::ConvBias::Format::CHWN4) {
            megdnn_assert(
                    bias.shape[0] == dst.shape[0], "bias:%s, dst:%s",
                    bias.to_string().c_str(), dst.to_string().c_str());
            megdnn_assert(bias.shape[1] == 1);
            megdnn_assert(bias.shape[2] == 1);
            megdnn_assert(bias.shape[3] == 1);
            megdnn_assert(bias.shape[4] == 4);
        } else if (opr->param().format == param::ConvBias::Format::NCHW64) {
            megdnn_assert(bias.shape[0] == 1);
            megdnn_assert(
                    bias.shape[1] == dst.shape[1], "bias:%s, dst:%s",
                    bias.to_string().c_str(), dst.to_string().c_str());
            megdnn_assert(bias.shape[2] == 1);
            megdnn_assert(bias.shape[3] == 1);
            megdnn_assert(bias.shape[4] == 64);
        } else {
            megdnn_assert(opr->param().format == param::ConvBias::Format::NHWCD4);
            megdnn_assert(bias.shape[0] == 1);
            megdnn_assert(bias.shape[1] == 1);
            megdnn_assert(
                    bias.shape[2] == dst.shape[2], "bias:%s, dst:%s",
                    bias.to_string().c_str(), dst.to_string().c_str());
            megdnn_assert(bias.shape[3] == 1);
            megdnn_assert(bias.shape[4] == 4);
        }
    }

    if (z.ndim != 0) {
        megdnn_assert(opr->param().format != param::ConvBias::Format::NCHW4_NCHW32);
        megdnn_assert(opr->param().format != param::ConvBias::Format::NCHW32_NCHW4);
        megdnn_assert(z.dtype.enumv() == dst.dtype.enumv());
        megdnn_assert(z.eq_shape(dst));
    }
}

}  // namespace

void ConvBiasForward::deduce_dtype(
        DType src, DType filter, DType /* bias */, DType /* z */, DType& dst) {
    check_or_deduce_dtype_fwd(src, filter, dst);
}

void ConvBiasForward::deduce_layout(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& /* bias */, const TensorLayout& /* z */,
        TensorLayout& dst) {
    deduce_layout_fwd(src, filter, dst);
}

ConvBiasForward::CanonizedFilterMeta ConvBiasForward::check_exec(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& bias,
        const TensorLayout& z, const TensorLayout& dst, size_t workspace_in_bytes,
        const PreprocessedFilter* preprocessed_filter) {
    do_check_exec_common(
            this, src, filter, bias, z, dst, workspace_in_bytes, preprocessed_filter);
    auto ret = check_layout_fwd(src, filter, dst);
    return ret;
}

ConvBiasForward::CanonizedFilterMeta ConvBiasForward::check_exec_allow_noncontiguous(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& bias,
        const TensorLayout& z, const TensorLayout& dst, size_t workspace_in_bytes,
        const PreprocessedFilter* preprocessed_filter) {
    do_check_exec_common(
            this, src, filter, bias, z, dst, workspace_in_bytes, preprocessed_filter);
    TensorLayout dst_expected;
    dst_expected.dtype = dst.dtype;
    auto ret = deduce_layout_fwd(src, filter, dst_expected);
    megdnn_assert_eq_shape(dst_expected, dst);
    return ret;
}

template <typename T>
struct NCHWParamTrait;

template <typename T>
struct NCHW44ParamTrait;

std::string ConvBias::WinogradParam::to_string() const {
    return ssprintf("%u:%u:%u", channel_block_size, output_block_size, tile_size);
}

template <typename T>
std::string ConvBias::algo_name(
        const std::string& base, const T& p, param::ConvBias::Format format) {
    if (format == param::ConvBias::Format::NCHW) {
        return ssprintf(
                "%s:%s:%s", NCHWParamTrait<T>::category.c_str(), base.c_str(),
                p.to_string().c_str());
    } else if (format == param::ConvBias::Format::NCHW44) {
        return ssprintf(
                "%s:%s:%s", NCHW44ParamTrait<T>::category.c_str(), base.c_str(),
                p.to_string().c_str());
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

const std::string NCHWParamTrait<ConvBias::WinogradParam>::category = "WINOGRAD";
const std::string NCHW44ParamTrait<ConvBias::WinogradParam>::category =
        "WINOGRAD_NCHW44";

#define cb(t)                                              \
    template std::string ConvBias::algo_name<ConvBias::t>( \
            const std::string& base, const ConvBias::t& p, \
            param::ConvBias::Format format);
FOREACH_CONV_BIAS_PARAM(cb)
#undef cb

ConvBias::WinogradParam ConvBias::parse_winograd_name(const std::string& algo_name) {
    ConvBias::WinogradParam ret = INVALID_WINOGRAD_PARAM;
    char base[128];
    char name[128];

    auto parse = [&](const std::string& algo_name, const std::string& pre) -> auto {
        memset(name, 0, 128);
        sscanf(algo_name.c_str(), "%[^:]:%[^:]:%u:%u:%u", name, base,
               &(ret.channel_block_size), &(ret.output_block_size), &(ret.tile_size));
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

void handle_bias_and_nonlinear(
        Handle* handle, param::ConvBias args, const TensorND* conv_dst_tensor,
        const TensorND* dst_tensor, const TensorND* bias_tensor) {
    using NonlineMode = param::ConvBias::NonlineMode;
    switch (args.nonlineMode) {
#define cb(_mode)                                                                   \
    case NonlineMode::_mode: {                                                      \
        if (conv_dst_tensor->layout.dtype.category() != DTypeCategory::QUANTIZED) { \
            auto nonlinear = handle->create_operator<ElemwiseForward>();            \
            if (bias_tensor->layout.ndim > 0) {                                     \
                nonlinear->param().mode = Elemwise::Param::Mode::FUSE_ADD_##_mode;  \
                nonlinear->exec({*conv_dst_tensor, *bias_tensor}, *dst_tensor);     \
            } else {                                                                \
                nonlinear->param().mode = Elemwise::Param::Mode::_mode;             \
                nonlinear->exec({*conv_dst_tensor}, *dst_tensor);                   \
            }                                                                       \
        } else {                                                                    \
            auto nonlinear = handle->create_operator<ElemwiseMultiType>();          \
            if (bias_tensor->layout.ndim > 0) {                                     \
                nonlinear->param().mode =                                           \
                        ElemwiseMultiType::Param::Mode::QFUSE_ADD_##_mode;          \
                nonlinear->exec({*conv_dst_tensor, *bias_tensor}, *dst_tensor);     \
            } else {                                                                \
                nonlinear->param().mode = ElemwiseMultiType::Param::Mode::Q##_mode; \
                nonlinear->exec({*conv_dst_tensor}, *dst_tensor);                   \
            }                                                                       \
        }                                                                           \
        break;                                                                      \
    }
        cb(RELU);
        cb(H_SWISH);
#undef cb
        case NonlineMode::SIGMOID: {
            megdnn_assert(
                    conv_dst_tensor->layout.dtype.category() !=
                    DTypeCategory::QUANTIZED);
            auto nonlinear = handle->create_operator<ElemwiseForward>();
            if (bias_tensor->layout.ndim > 0) {
                nonlinear->param().mode = Elemwise::Param::Mode::FUSE_ADD_SIGMOID;
                nonlinear->exec({*conv_dst_tensor, *bias_tensor}, *conv_dst_tensor);
            } else {
                nonlinear->param().mode = Elemwise::Param::Mode::SIGMOID;
                nonlinear->exec({*conv_dst_tensor}, *conv_dst_tensor);
            }
            break;
        }
        case NonlineMode::IDENTITY: {
            if (bias_tensor->layout.ndim > 0) {
                if (dst_tensor->layout.dtype.category() == DTypeCategory::QUANTIZED) {
                    auto nonlinear = handle->create_operator<ElemwiseMultiType>();
                    nonlinear->param().mode = ElemwiseMultiType::Param::Mode::QADD;
                    nonlinear->exec({*conv_dst_tensor, *bias_tensor}, *dst_tensor);
                } else {
                    auto nonlinear = handle->create_operator<Elemwise>();
                    nonlinear->param().mode = Elemwise::Param::Mode::ADD;
                    nonlinear->exec({*conv_dst_tensor, *bias_tensor}, *dst_tensor);
                }
            } else {
                if (conv_dst_tensor->layout.dtype != dst_tensor->layout.dtype) {
                    handle->create_operator<TypeCvt>()->exec(
                            {*conv_dst_tensor}, *dst_tensor);
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
