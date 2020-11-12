/**
 * \file dnn/src/naive/conv_bias/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/conv_bias/opr_impl.h"
#include "src/naive/convolution/helper.h"

#include <cstring>
#include "megdnn/dtype.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "src/naive/lowbit_utils.h"
#include "src/common/conv_bias.h"
#include "src/common/opr_delegate.h"

#include "midout.h"
MIDOUT_DECL(megdnn_naive_conv_bias_fwd)

namespace megdnn {
namespace naive {

//! Only used for naive implementation. DO NOT use the following function in
//! other backends.
void handle_z_inp_and_activation_naive(
        param::ConvBias::NonlineMode nonline_mode,
        const TensorND& conv_bias_tensor, const TensorND& z_tensor,
        const TensorND& dst_tensor, dt_byte* workspace_ptr) {
    auto res = dst_tensor, z_float = z_tensor;
    //!create naive inplace handle
    auto handle = inplace_cpu_handle(2);
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

namespace convolution {

template <>
void forward_bias<dt_quint4, dt_quint4, dt_qint32, dt_qint32>(
        _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in bias,
        _megdnn_tensor_out dst, dt_byte* workspace_ptr,
        const ConvBiasForward::CanonizedFilterMeta& filter_meta) {
    auto convert_layout = [](const TensorLayout& layout) {
        auto ret = layout;
        auto param = layout.dtype.param<dtype::Quantized4Asymm>();
        ret.dtype = dtype::Quantized8Asymm(param.scale, param.zero_point);
        return ret;
    };
    TensorND new_src = {workspace_ptr, convert_layout(src.layout)};
    TensorND new_flt = {workspace_ptr + new_src.layout.span().dist_byte(),
                        convert_layout(filter.layout)};

    uint4_to_uint8(src, new_src);
    uint4_to_uint8(filter, new_flt);
    auto new_filter_meta = filter_meta;
    new_filter_meta.dtype = new_flt.layout.dtype;
    forward_bias<dt_quint8, dt_quint8, dt_qint32, dt_qint32>(
            new_src, new_flt, bias, dst, nullptr, new_filter_meta);
}
}  // namespace convolution

size_t ConvBiasForwardImpl::get_workspace_in_bytes(const TensorLayout& src,
                                                   const TensorLayout& flt,
                                                   const TensorLayout& bias,
                                                   const TensorLayout& z,
                                                   const TensorLayout& dst,
                                                   const PreprocessedFilter*) {
    size_t float_workspace_size = 0;

    if (z.ndim > 0 && z.dtype.category() != DTypeCategory::FLOAT) {
        megdnn_assert(z.eq_shape(dst));
        // (w * f + b).astype(float) + (z).astype(float)
        float_workspace_size =
                2 * TensorLayout{z, dtype::Float32()}.span().dist_byte();
    }

    if (bias.dtype.enumv() != dst.dtype.enumv()) {
        return float_workspace_size +
               TensorLayout{dst, bias.dtype}.span().dist_byte();
    } else if (src.dtype.enumv() == DTypeEnum::Quantized4Asymm &&
               dst.dtype.enumv() == DTypeEnum::QuantizedS32) {
        return float_workspace_size +
               (src.span().dist_elem() + flt.span().dist_elem()) *
                       sizeof(uint8_t);
    }
    return float_workspace_size;
}

void ConvBiasForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                               _megdnn_tensor_in bias, _megdnn_tensor_in z,
                               _megdnn_tensor_out dst,
                               const PreprocessedFilter* preprocessed_filter,
                               _megdnn_workspace workspace) {
    MIDOUT_BEGIN(megdnn_naive_conv_bias_fwd) {
        dt_byte *workspace_ptr = workspace.raw_ptr;
        // ============================w * f + b================================

        auto filter_meta =
                check_exec(src.layout, filter.layout, bias.layout, z.layout,
                           dst.layout, workspace.size, preprocessed_filter);
        auto sfb = dst;
        if (bias.layout.dtype.enumv() != dst.layout.dtype.enumv()) {
            // intermediate result
            sfb = TensorND{workspace_ptr,
                           TensorLayout{dst.layout, bias.layout.dtype}};
            workspace_ptr += sfb.layout.span().dist_byte();
        }
#define DISPATCH_RAW(in_dt, bias_dt, out_dt, cmode, func)                      \
    else if (src.layout.dtype.enumv() == DTypeTrait<dtype::in_dt>::enumv &&    \
             filter.layout.dtype.enumv() == DTypeTrait<dtype::in_dt>::enumv && \
             bias.layout.dtype.enumv() == DTypeTrait<dtype::bias_dt>::enumv && \
             sfb.layout.dtype.enumv() == DTypeTrait<dtype::out_dt>::enumv &&   \
             param().compute_mode == Param::ComputeMode::cmode) {              \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                          \
                func(src, filter, bias, sfb, workspace_ptr, filter_meta));     \
    }
#define DISPATCH(in_dt, out_dt)                                          \
    DISPATCH_RAW(                                                        \
            in_dt, out_dt, out_dt, DEFAULT,                              \
            (convolution::forward_bias<DTypeTrait<dtype::in_dt>::ctype,  \
                                       DTypeTrait<dtype::in_dt>::ctype,  \
                                       DTypeTrait<dtype::out_dt>::ctype, \
                                       DTypeTrait<dtype::out_dt>::ctype>))
        if (0) {}
        DISPATCH(Float32, Float32)
        DISPATCH(Int8, Int16)
        DISPATCH(Int8, Int32)
        DISPATCH(QuantizedS8, QuantizedS32)
        DISPATCH(QuantizedS8, Float32)
        DISPATCH(Quantized8Asymm, QuantizedS32)
        DISPATCH(Quantized4Asymm, QuantizedS32)
        DISPATCH_RAW(QuantizedS8, QuantizedS32, QuantizedS32, FLOAT32,
                     (convolution::forward_bias<dt_int8, dt_int8, dt_int32,
                                                dt_int32>))
#if !MEGDNN_DISABLE_FLOAT16
        DISPATCH(Float16, Float16)
        DISPATCH_RAW(Float16, Float16, Float16, FLOAT32,
                     (convolution::forward_bias<dt_float16, dt_float16,
                                                dt_float16, dt_float32>))
#endif
        else {
            megdnn_throw(ssprintf(
                    "unsupported naive ConvBias(%s, %s, %s, %s) -> %s",
                    src.layout.dtype.name(), filter.layout.dtype.name(),
                    bias.layout.dtype.name(), z.layout.dtype.name(),
                    dst.layout.dtype.name()));
        }
#undef DISPATCH
#undef DISPATCH_RAW
        MEGDNN_DISPATCH_CPU_KERN_OPR(handle_z_inp_and_activation_naive(
                param().nonlineMode, sfb, z, dst, workspace_ptr));
    }
    MIDOUT_END();
}

std::vector<ConvBiasForward::Algorithm*>
ConvBiasForwardImpl::get_all_algorithms(const TensorLayout&,
                                        const TensorLayout&,
                                        const TensorLayout&,
                                        const TensorLayout&,
                                        const TensorLayout&) {
    return {static_cast<HandleImpl*>(handle())->default_conv_bias_fwd_algo()};
}

ConvBiasForward::Algorithm* ConvBiasForwardImpl::get_algorithm_heuristic(
        const TensorLayout& /* src */, const TensorLayout& /* filter */,
        const TensorLayout& /* bias */, const TensorLayout& /* z */,
        const TensorLayout& /* dst */, size_t /* workspace_limit_in_bytes */,
        bool reproducible) {
    auto algo =
            static_cast<HandleImpl*>(handle())->default_conv_bias_fwd_algo();
    if (reproducible) {
        megdnn_assert(algo->is_reproducible(),
                      "require reproducible algorithm, but heuristic "
                      "algorithm(%s) is not "
                      "reproducible",
                      algo->name());
    }
    return algo;
}

const char* ConvBiasForwardImpl::get_algorithm_set_name() const {
    return "DEFAULT";
}

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
