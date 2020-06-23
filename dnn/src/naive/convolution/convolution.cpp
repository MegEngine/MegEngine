/**
 * \file dnn/src/naive/convolution/convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./opr_impl.h"
#include "./helper.h"

#include "src/naive/handle.h"
#include "src/naive/handle.h"
#include "src/common/utils.h"
#include "megdnn/dtype.h"
#include "megdnn/tensor_iter.h"

#include <cstring>

#include "midout.h"
MIDOUT_DECL(megdnn_naive_conv_fwd)

using namespace megdnn;
using namespace naive;

void ConvolutionForwardImpl::exec(_megdnn_tensor_in src,
                                  _megdnn_tensor_in filter,
                                  _megdnn_tensor_out dst,
                                  const PreprocessedFilter* preprocessed_filter,
                                  _megdnn_workspace workspace) {
    MIDOUT_BEGIN(megdnn_naive_conv_fwd) {
        auto filter_meta = check_exec(src.layout, filter.layout, dst.layout,
                                      workspace.size, preprocessed_filter);
        using ComputeMode = Param::ComputeMode;
#define DISPATCH_CMODE(in_dt, out_dt, in_ct, out_ct, comp_ct, cmode)      \
    do {                                                                  \
        using namespace dtype;                                            \
        if (src.layout.dtype.enumv() == DTypeTrait<in_dt>::enumv &&       \
            dst.layout.dtype.enumv() == DTypeTrait<out_dt>::enumv &&      \
            param().compute_mode == cmode) {                              \
            MEGDNN_DISPATCH_CPU_KERN_OPR(                                 \
                    (convolution::forward<in_ct, in_ct, out_ct, comp_ct>( \
                            src, filter, dst, filter_meta)););            \
            return;                                                       \
        }                                                                 \
    } while (0);
#define DISPATCH(in_dt, out_dt, in_ct, out_ct, comp_ct) \
    DISPATCH_CMODE(in_dt, out_dt, in_ct, out_ct, comp_ct, ComputeMode::DEFAULT)
#define cb(dt)                                                     \
    DISPATCH(dt, dt, DTypeTrait<dt>::ctype, DTypeTrait<dt>::ctype, \
             DTypeTrait<dt>::ctype)
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb);
#undef cb
        DISPATCH(Int8, Int16, dt_int8, dt_int16, dt_int16);
        DISPATCH(Int8, Int32, dt_int8, dt_int32, dt_int32);
        DISPATCH(QuantizedS8, QuantizedS32, dt_int8, dt_int32, dt_int32);
        MEGDNN_INC_FLOAT16(DISPATCH_CMODE(Float16, Float16, dt_float16,
                                          dt_float16, dt_float32,
                                          ComputeMode::FLOAT32));
        MEGDNN_INC_FLOAT16(DISPATCH_CMODE(BFloat16, BFloat16, dt_bfloat16,
                                          dt_bfloat16, dt_float32,
                                          ComputeMode::FLOAT32));
        DISPATCH(Quantized8Asymm, QuantizedS32, dt_quint8, dt_qint32,
                 dt_qint32);
        DISPATCH(QuantizedS8, QuantizedS8, dt_int8, dt_int8, dt_int32);
#undef DISPATCH
        megdnn_throw(ssprintf("unsupported Conv(%s, %s) -> %s with cmode = %d",
                              src.layout.dtype.name(),
                              filter.layout.dtype.name(),
                              dst.layout.dtype.name(),
                              static_cast<int>(param().compute_mode)));
    }
    MIDOUT_END();
}

size_t ConvolutionBackwardDataImpl::get_workspace_in_bytes(const TensorLayout& filter,
                                                   const TensorLayout& diff,
                                                   const TensorLayout& grad) {
    size_t workspace_size = 0;
    auto flt_dt = filter.dtype.enumv();
    auto grad_dt = grad.dtype.enumv();
    auto diff_dt = diff.dtype.enumv();
#if !MEGDNN_DISABLE_FLOAT16
    if (flt_dt == DTypeEnum::Float16 || flt_dt == DTypeEnum::BFloat16) {
        megdnn_assert(flt_dt == grad_dt && flt_dt == diff_dt);
        workspace_size = grad.span().dist_elem() * dtype::Float32().size();
    }
#endif
    if ((flt_dt == DTypeEnum::Int8 || flt_dt == DTypeEnum::QuantizedS8) &&
        (diff_dt == DTypeEnum::Int8 || diff_dt == DTypeEnum::QuantizedS8) &&
        (grad_dt == DTypeEnum::Int8 || grad_dt == DTypeEnum::QuantizedS8)) {
        workspace_size =
                TensorLayout{grad, dtype::QuantizedS32()}.span().dist_byte();
    }

    return workspace_size;
}

void ConvolutionBackwardDataImpl::exec(_megdnn_tensor_in filter,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    auto filter_meta = check_exec(
            filter.layout, diff.layout, grad.layout, workspace.size);
    using ComputeMode = Param::ComputeMode;
    auto cmode = param().compute_mode;
#define cb(dt)                                                              \
    do {                                                                    \
        if (filter.layout.dtype == dt() && cmode == ComputeMode::DEFAULT) { \
            using ctype = DTypeTrait<dt>::ctype;                            \
            MEGDNN_DISPATCH_CPU_KERN_OPR(                                   \
                    (convolution::backward_data<ctype, ctype, ctype>(       \
                            filter, diff, grad, filter_meta)););            \
            return;                                                         \
        }                                                                   \
    } while (0);
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb);
#undef cb
#if !MEGDNN_DISABLE_FLOAT16
    if (filter.layout.dtype == dtype::Float16() &&
        cmode == ComputeMode::FLOAT32) {
        TensorND grad_fp32;
        grad_fp32.layout = grad.layout;
        grad_fp32.layout.dtype = dtype::Float32();
        grad_fp32.raw_ptr = workspace.raw_ptr;
        auto&& type_cvt = handle()->create_operator<TypeCvt>();
        type_cvt->exec(grad, grad_fp32);
        MEGDNN_DISPATCH_CPU_KERN_OPR(
                (convolution::backward_data<dt_float16, dt_float16, dt_float32>(
                        filter, diff, grad_fp32, filter_meta)););
        type_cvt->exec(grad_fp32, grad);
        return;
    }
    if (filter.layout.dtype == dtype::BFloat16() &&
        cmode == ComputeMode::FLOAT32) {
        TensorND grad_fp32;
        grad_fp32.layout = grad.layout;
        grad_fp32.layout.dtype = dtype::Float32();
        grad_fp32.raw_ptr = workspace.raw_ptr;
        auto&& type_cvt = handle()->create_operator<TypeCvt>();
        type_cvt->exec(grad, grad_fp32);
        MEGDNN_DISPATCH_CPU_KERN_OPR(
                (convolution::backward_data<dt_bfloat16, dt_bfloat16, dt_float32>(
                        filter, diff, grad_fp32, filter_meta)););
        type_cvt->exec(grad_fp32, grad);
        return;
    }
#endif
    auto flt_dt = filter.layout.dtype.enumv();
    auto grad_dt = grad.layout.dtype.enumv();
    if ((flt_dt == DTypeEnum::Int8 || flt_dt == DTypeEnum::QuantizedS8) &&
        (grad_dt == DTypeEnum::Int8 || grad_dt == DTypeEnum::QuantizedS8)) {
        auto res = grad;

        auto resf_s = filter.layout.dtype.param<dtype::QuantizedS8>().scale *
          diff.layout.dtype.param<dtype::QuantizedS8>().scale;
        res = TensorND{workspace.raw_ptr,
          TensorLayout{grad.layout, dtype::QuantizedS32(resf_s)}};
        MEGDNN_DISPATCH_CPU_KERN_OPR(
                (convolution::backward_data<dt_qint8, dt_qint8, dt_qint32>(
                        filter, diff, res, filter_meta)););
        handle()->create_operator<TypeCvt>()->exec(res, grad);

        return;
    }
    if ((flt_dt == DTypeEnum::Int8 || flt_dt == DTypeEnum::QuantizedS8) &&
        (grad_dt == DTypeEnum::Int32 || grad_dt == DTypeEnum::QuantizedS32)) {
        MEGDNN_DISPATCH_CPU_KERN_OPR(
                (convolution::backward_data<dt_int8, dt_int8, dt_int32>(
                        filter, diff, grad, filter_meta)););
        return;
    }
    if (flt_dt == DTypeEnum::Quantized8Asymm &&
        grad_dt == DTypeEnum::QuantizedS32) {
        MEGDNN_DISPATCH_CPU_KERN_OPR(
                (convolution::backward_data<dt_quint8, dt_quint8, dt_qint32>(
                        filter, diff, grad, filter_meta)););
        return;
    }
    megdnn_throw(ssprintf(
            "unsupported ConvolutionBackwardData(%s, %s) -> %s with cmode = %d",
            filter.layout.dtype.name(), diff.layout.dtype.name(),
            grad.layout.dtype.name(), static_cast<int>(cmode)));
}

size_t ConvolutionBackwardFilterImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& diff,
        const TensorLayout& grad) {
    size_t workspace_size = 0;
#if !MEGDNN_DISABLE_FLOAT16
    auto src_dt = src.dtype.enumv();
    auto grad_dt = grad.dtype.enumv();
    auto diff_dt = diff.dtype.enumv();
    if (src_dt == DTypeEnum::Float16 || src_dt == DTypeEnum::BFloat16) {
        megdnn_assert(src_dt == grad_dt && src_dt == diff_dt);
        workspace_size = grad.span().dist_elem() * dtype::Float32().size();
    }
#endif

    return workspace_size;
}

void ConvolutionBackwardFilterImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    auto filter_meta = check_exec(
            src.layout, diff.layout, grad.layout, workspace.size);
    using ComputeMode = Param::ComputeMode;
    auto cmode = param().compute_mode;
#define cb(dt)                                                            \
    do {                                                                  \
        if (src.layout.dtype == dt() && cmode == ComputeMode::DEFAULT) {  \
            using ctype = DTypeTrait<dt>::ctype;                          \
            MEGDNN_DISPATCH_CPU_KERN(                                     \
                    static_cast<HandleImpl*>(handle()),                   \
                    convolution::backward_filter<                         \
                            ctype MEGDNN_COMMA ctype MEGDNN_COMMA ctype>( \
                            src, diff, grad, filter_meta););              \
            return;                                                       \
        }                                                                 \
    } while (0);
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb);
#undef cb
#if !MEGDNN_DISABLE_FLOAT16
    if (src.layout.dtype == dtype::Float16() && cmode == ComputeMode::FLOAT32) {
        TensorND grad_fp32;
        grad_fp32.layout = grad.layout;
        grad_fp32.layout.dtype = dtype::Float32();
        grad_fp32.raw_ptr = workspace.raw_ptr;
        auto&& type_cvt = handle()->create_operator<TypeCvt>();
        type_cvt->exec(grad, grad_fp32);
        MEGDNN_DISPATCH_CPU_KERN_OPR(
                (convolution::backward_filter<dt_float16, dt_float16,
                                              dt_float32>(src, diff, grad_fp32,
                                                          filter_meta)););
        type_cvt->exec(grad_fp32, grad);
        return;
    }
    if (src.layout.dtype == dtype::BFloat16() &&
        cmode == ComputeMode::FLOAT32) {
        TensorND grad_fp32;
        grad_fp32.layout = grad.layout;
        grad_fp32.layout.dtype = dtype::Float32();
        grad_fp32.raw_ptr = workspace.raw_ptr;
        auto&& type_cvt = handle()->create_operator<TypeCvt>();
        type_cvt->exec(grad, grad_fp32);
        MEGDNN_DISPATCH_CPU_KERN_OPR(
                (convolution::backward_filter<dt_bfloat16, dt_bfloat16,
                                              dt_float32>(src, diff, grad_fp32,
                                                          filter_meta)););
        type_cvt->exec(grad_fp32, grad);
        return;
    }

#endif

    megdnn_assert_internal(0);
}

std::vector<ConvolutionForward::Algorithm *>
ConvolutionForwardImpl:: get_all_algorithms(const TensorLayout &,
        const TensorLayout &, const TensorLayout &)
{
    return {static_cast<HandleImpl *>(handle())->default_conv_fwd_algo()};
}

ConvolutionForward::Algorithm* ConvolutionForwardImpl::get_algorithm_heuristic(
        const TensorLayout& /* src */, const TensorLayout& /* diff */,
        const TensorLayout& /* grad */, size_t /* workspace_limit_in_bytes */,
        bool reproducible) {
    auto algo =
            static_cast<HandleImpl*>(handle())->default_conv_fwd_algo();
    if (reproducible) {
        megdnn_assert(algo->is_reproducible(),
                      "require reproducible algorithm, but heuristic "
                      "algorithm(%s) is not "
                      "reproducible",
                      algo->name());
    }
    return algo;
}

std::vector<ConvolutionBackwardData::Algorithm *>
ConvolutionBackwardDataImpl:: get_all_algorithms(const TensorLayout &,
        const TensorLayout &, const TensorLayout &)
{
    return {static_cast<HandleImpl *>(handle())->default_conv_bwd_data_algo()};
}

ConvolutionBackwardData::Algorithm*
ConvolutionBackwardDataImpl::get_algorithm_heuristic(
        const TensorLayout& /* filter */, const TensorLayout& /* diff */,
        const TensorLayout& /* grad */, size_t /* workspace_limit_in_bytes */,
        bool reproducible) {
    auto algo =
            static_cast<HandleImpl*>(handle())->default_conv_bwd_data_algo();
    if (reproducible) {
        megdnn_assert(algo->is_reproducible(),
                      "require reproducible algorithm, but heuristic "
                      "algorithm(%s) is not "
                      "reproducible",
                      algo->name());
    }
    return algo;
}

std::vector<ConvolutionBackwardFilter::Algorithm *>
ConvolutionBackwardFilterImpl:: get_all_algorithms(const TensorLayout &,
        const TensorLayout &, const TensorLayout &)
{
    return {static_cast<HandleImpl*>(handle())->default_conv_bwd_filter_algo()};
}

ConvolutionBackwardFilter::Algorithm*
ConvolutionBackwardFilterImpl::get_algorithm_heuristic(
        const TensorLayout& /* src */, const TensorLayout& /* diff */,
        const TensorLayout& /* grad */, size_t /* workspace_limit_in_bytes */,
        bool reproducible) {
    auto algo =
            static_cast<HandleImpl*>(handle())->default_conv_bwd_filter_algo();
    if (reproducible) {
        megdnn_assert(algo->is_reproducible(),
                      "require reproducible algorithm, but heuristic "
                      "algorithm(%s) is not "
                      "reproducible",
                      algo->name());
    }
    return algo;
}

const char* ConvolutionForwardImpl::get_algorithm_set_name() const {
    return "DEFAULT";
}

const char* ConvolutionBackwardDataImpl::get_algorithm_set_name() const {
    return "DEFAULT";
}

const char* ConvolutionBackwardFilterImpl::get_algorithm_set_name() const {
    return "DEFAULT";
}

// vim: syntax=cpp.doxygen
