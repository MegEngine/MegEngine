/**
 * \file dnn/src/fallback/conv_bias/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/conv_bias/algos.h"
#include "src/fallback/conv_bias/conv1x1/algos.h"
#include "src/fallback/conv_bias/conv1x1/algos_conv1x1_gemv.h"
#include "src/fallback/conv_bias/im2col/algos.h"
#include "megdnn/opr_param_defs.h"
#include "src/common/opr_delegate.h"
#include "src/fallback/conv_bias/winograd/strategy.h"
#include "src/naive/convolution/helper.h"
#include "src/common/algo_base.h"

#include "midout.h"

using namespace megdnn;
using namespace fallback;

namespace {

param::Convolution get_param_convolution(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    param::Convolution::Mode mode;
    param::Convolution::Sparse sparse;
    if (param.filter_meta.should_flip) {
        mode = param::Convolution::Mode::CONVOLUTION;
    } else {
        mode = param::Convolution::Mode::CROSS_CORRELATION;
    }
    return param::Convolution{mode,
                              param.filter_meta.padding[0],
                              param.filter_meta.padding[1],
                              param.filter_meta.stride[0],
                              param.filter_meta.stride[1],
                              param.filter_meta.dilation[1],
                              param.filter_meta.dilation[0],
                              sparse = param::Convolution::Sparse::DENSE,
                              param.filter_meta.format};
}

TensorLayoutArray get_layouts(const ConvBiasImpl::NCBKernSizeParam& p) {
    megdnn_assert(p.filter_meta.format == param::ConvBias::Format::NCHW);
    UNPACK_CONV_NCB_KERN_SIZES(p);
    MEGDNN_MARK_USED_VAR(SH);
    MEGDNN_MARK_USED_VAR(SW);
    MEGDNN_MARK_USED_VAR(PH);
    MEGDNN_MARK_USED_VAR(PW);
    MEGDNN_MARK_USED_VAR(OW);
    MEGDNN_MARK_USED_VAR(OH);
    MEGDNN_MARK_USED_VAR(N);
    TensorLayout src_layout({1, IC, IH, IW}, p.src_type);
    TensorLayout filter_layout({OC, IC, FH, FW}, p.filter_type);
    TensorLayout bias_layout{{}, p.bias_type};
    if (p.bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        bias_layout = TensorLayout({1, OC, 1, 1}, p.bias_type);
    } else if (p.bias_mode == BiasMode::BIAS) {
        bias_layout = TensorLayout({1, OC, OH, OW}, p.bias_type);
    }
    TensorLayout dst_layout = TensorLayout({1, OC, OH, OW}, p.dst_type);
    return {src_layout, filter_layout, bias_layout, dst_layout};
}

void kern_default(const ConvBiasImpl::NCBKernParam& p) {
    dt_byte* workspace_ptr = static_cast<dt_byte*>(p.workspace_ptr);

    auto filter_meta_ptr =
            reinterpret_cast<const ConvBiasForward::CanonizedFilterMeta*>(
                    &p.filter_meta);
    auto filter_meta = *filter_meta_ptr;
    auto layouts = get_layouts(p);

    TensorND src{reinterpret_cast<dt_byte*>(const_cast<void*>(p.src_ptr)),
                 layouts[0]};
    TensorND filter{const_cast<void*>(p.filter_ptr), layouts[1]};
    auto bias_ptr = reinterpret_cast<dt_byte*>(const_cast<void*>(p.bias_ptr));
    TensorND bias{bias_ptr, layouts[2]};
    TensorND dst{reinterpret_cast<dt_byte*>(const_cast<void*>(p.dst_ptr)),
                 layouts[3]};

    auto sfb = dst;
    if (bias.layout.dtype.enumv() != dst.layout.dtype.enumv()) {
        // intermediate result
        sfb = TensorND{workspace_ptr,
                       TensorLayout{dst.layout, bias.layout.dtype}};
    }
#define DISPATCH_RAW(in_dt, bias_dt, out_dt, cmode, func)                      \
    else if (src.layout.dtype.enumv() == DTypeTrait<dtype::in_dt>::enumv &&    \
             filter.layout.dtype.enumv() == DTypeTrait<dtype::in_dt>::enumv && \
             (!bias.layout.dtype.valid() ||                                    \
              bias.layout.dtype.enumv() ==                                     \
                      DTypeTrait<dtype::bias_dt>::enumv) &&                    \
             sfb.layout.dtype.enumv() == DTypeTrait<dtype::out_dt>::enumv &&   \
             p.compute_mode == param::ConvBias::ComputeMode::cmode) {          \
        func(src, filter, bias, sfb, workspace_ptr, filter_meta);              \
    }
#define DISPATCH(in_dt, out_dt)                             \
    DISPATCH_RAW(in_dt, out_dt, out_dt, DEFAULT,            \
                 (megdnn::naive::convolution::forward_bias< \
                         DTypeTrait<dtype::in_dt>::ctype,   \
                         DTypeTrait<dtype::in_dt>::ctype,   \
                         DTypeTrait<dtype::out_dt>::ctype,  \
                         DTypeTrait<dtype::out_dt>::ctype>))
    if (0) {
    }
    DISPATCH(Float32, Float32)
    DISPATCH(Int8, Int16)
    DISPATCH(Int8, Int32)
    DISPATCH(QuantizedS8, QuantizedS32)
    DISPATCH(Quantized8Asymm, QuantizedS32)
#if !MEGDNN_DISABLE_FLOAT16
    DISPATCH(Float16, Float16)
    DISPATCH_RAW(
            Float16, Float16, Float16, FLOAT32,
            (megdnn::naive::convolution::forward_bias<dt_float16, dt_float16,
                                                      dt_float16, dt_float32>))
#endif
    else {
        megdnn_throw(
                ssprintf("unsupported naive ConvBias(%s, %s, %s) -> %s",
                         src.layout.dtype.name(), filter.layout.dtype.name(),
                         bias.layout.dtype.name(), dst.layout.dtype.name()));
    }
#undef DISPATCH
#undef DISPATCH_RAW

    auto res = sfb;
    using NonlineMode = param::ConvBias::NonlineMode;
    switch (p.nonlineMode) {
#define cb(_mode)                                                             \
    case NonlineMode::_mode: {                                                \
        if (res.layout.dtype.category() != DTypeCategory::QUANTIZED) {        \
            auto nonlinear =                                                  \
                    inplace_cpu_handle()->create_operator<ElemwiseForward>(); \
            nonlinear->param().mode = Elemwise::Param::Mode::_mode;           \
            nonlinear->exec({res}, dst);                                      \
        } else {                                                              \
            auto nonlinear = inplace_cpu_handle()                             \
                                     ->create_operator<ElemwiseMultiType>();  \
            nonlinear->param().mode =                                         \
                    ElemwiseMultiType::Param::Mode::Q##_mode;                 \
            nonlinear->exec({res}, dst);                                      \
        }                                                                     \
        break;                                                                \
    }
        cb(RELU);
        cb(H_SWISH);
#undef cb
        case NonlineMode::SIGMOID: {
            megdnn_assert(res.layout.dtype.category() !=
                          DTypeCategory::QUANTIZED);
            auto nonlinear =
                    inplace_cpu_handle()->create_operator<ElemwiseForward>();
            nonlinear->param().mode = Elemwise::Param::Mode::SIGMOID;
            nonlinear->exec({res}, res);
            if (res.raw_ptr != dst.raw_ptr) {
                inplace_cpu_handle()->create_operator<TypeCvt>()->exec(res,
                                                                       dst);
            }
            break;
        }
        case NonlineMode::IDENTITY: {
            if (res.raw_ptr != dst.raw_ptr) {
                inplace_cpu_handle()->create_operator<TypeCvt>()->exec(res,
                                                                       dst);
            }
            break;
        }
        default:
            megdnn_assert(false);
    }
}
}  // namespace

MIDOUT_DECL(megdnn_fallback_naive)

/* ======================= AlgoNaive ======================== */

bool ConvBiasImpl::AlgoNaive::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MIDOUT_BEGIN(megdnn_fallback_naive, 0) {
        return param.filter_meta.format == param::ConvBias::Format::NCHW;
    }
    MIDOUT_END();
    return false;
}

size_t ConvBiasImpl::AlgoNaive::get_workspace(const NCBKernSizeParam& p) const {
    MIDOUT_BEGIN(megdnn_fallback_naive, 1) {
        auto layouts = get_layouts(p);
        //! When group>1 or n>1, this algo will parallel by group and n
        size_t nr_threads = p.nr_threads;
        auto conv_opr =
                inplace_cpu_handle()->create_operator<ConvolutionForward>();
        conv_opr->param() = get_param_convolution(p);
        if (p.dst_type.enumv() == DTypeEnum::QuantizedS8 ||
            p.dst_type.enumv() == DTypeEnum::Quantized8Asymm) {
            TensorLayout conv_dst_layout;
            conv_opr->deduce_layout(layouts[0], layouts[1], conv_dst_layout);
            WorkspaceBundle bundle(nullptr,
                                   {conv_dst_layout.span().dist_byte()});
            return bundle.total_size_in_bytes() * nr_threads;
        }
        return 0;
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoNaive::dispatch_kerns(
        const NCBKernSizeParam& p) const {
    size_t workspace_size = get_workspace(p);
    //! When group>1 or n>1, this algo will parallel by group and n
    size_t nr_threads = p.nr_threads;
    size_t GROUP = p.filter_meta.group;
    size_t N = p.n;
    size_t workspace_per_thread = workspace_size / nr_threads;
    auto kern = [workspace_per_thread](
                        const NCBKernParam& param,
                        const NCBKernIndex& ncb_index) {
        MIDOUT_BEGIN(megdnn_fallback_naive, 2) {
            size_t group_id = ncb_index.ndrange_id[0];
            size_t batch_id = ncb_index.ndrange_id[1];
            size_t thread_id = ncb_index.thread_id;
            auto thread_param = param;
            thread_param.workspace_ptr = reinterpret_cast<void*>(
                    reinterpret_cast<ptrdiff_t>(param.workspace_ptr) +
                    thread_id * workspace_per_thread);
            thread_param.filter_ptr = param.filter<void>(group_id);
            thread_param.dst_ptr = param.dst<void>(batch_id, group_id);
            thread_param.src_ptr = param.src<void>(batch_id, group_id);
            thread_param.bias_ptr = param.bias<void>(batch_id, group_id);
            kern_default(thread_param);
        }
        MIDOUT_END();
    };
    return {{kern, {GROUP, N, 1_z}}};
}

MIDOUT_DECL(megdnn_fallback_winograd)
/* ======================= AlgoWinogradF32 ======================== */

bool ConvBiasImpl::AlgoWinogradF32::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_fallback_winograd, 1, 0) {
        using Strategy = fallback::winograd::winograd_2x3_1x1_f;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param = megdnn::winograd::ConvBias<Strategy>(
                                      strategy, UNIT_TILE_SIZE, param)
                                      .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               (param.filter_meta.format == param::ConvBias::Format::NCHW ||
                (param.filter_meta.format ==
                         param::ConvBias::Format::NCHW_WINOGRAD &&
                 param.output_block_size == 2 &&
                 param.winograd_matmul_format ==
                         param::MatrixMul::Format::DEFAULT)) &&
               param.filter_meta.should_flip &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 3) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float32;
    }
    MIDOUT_END();
    return false;
}

size_t ConvBiasImpl::AlgoWinogradF32::get_workspace(
        const NCBKernSizeParam& p) const {
    MEGDNN_MARK_USED_VAR(p);
    MIDOUT_BEGIN(megdnn_fallback_winograd, 1, 1) {
        fallback::winograd::winograd_2x3_1x1_f strategy(
                p.src_type, p.filter_type, p.dst_type);
        return megdnn::winograd::ConvBias<
                       fallback::winograd::winograd_2x3_1x1_f>(
                       strategy, UNIT_TILE_SIZE, p)
                .get_workspace_size(p, m_matmul_algo);
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoWinogradF32::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_fallback_winograd, 1, 2) {
        fallback::winograd::winograd_2x3_1x1_f strategy(
                param.src_type, param.filter_type, param.dst_type);

        auto winograd_impl = megdnn::winograd::ConvBias<
                fallback::winograd::winograd_2x3_1x1_f>(strategy,
                                                        UNIT_TILE_SIZE, param);
        return winograd_impl.get_kerns(param, m_matmul_algo);
    }
    MIDOUT_END();
    return {};
}

/* ======================= AlgoWinogradF32 4x4 ======================== */

bool ConvBiasImpl::AlgoWinogradF32_4x4::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_fallback_winograd, 2, 0) {
        if (param.filter_meta.icpg % 4 != 0 || param.filter_meta.ocpg % 4 != 0)
            return false;
        using Strategy = fallback::winograd::winograd_2x3_4x4_f;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy,
                                           param::MatrixMul::Format::MK4>(
                        strategy, UNIT_TILE_SIZE, param)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               (param.filter_meta.format == param::ConvBias::Format::NCHW ||
                (param.filter_meta.format ==
                         param::ConvBias::Format::NCHW_WINOGRAD &&
                 param.output_block_size == 2 &&
                 param.winograd_matmul_format ==
                         param::MatrixMul::Format::MK4)) &&
               param.filter_meta.should_flip &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 3) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float32;
    }
    MIDOUT_END();
    return false;
}

size_t ConvBiasImpl::AlgoWinogradF32_4x4::get_workspace(
        const NCBKernSizeParam& p) const {
    MEGDNN_MARK_USED_VAR(p);
    MIDOUT_BEGIN(megdnn_fallback_winograd, 2, 1) {
        fallback::winograd::winograd_2x3_4x4_f strategy(
                p.src_type, p.filter_type, p.dst_type);
        return megdnn::winograd::ConvBias<
                       fallback::winograd::winograd_2x3_4x4_f,
                       param::MatrixMul::Format::MK4>(strategy, UNIT_TILE_SIZE,
                                                      p)
                .get_workspace_size(p, m_matmul_algo);
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoWinogradF32_4x4::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_fallback_winograd, 2, 2) {
        fallback::winograd::winograd_2x3_4x4_f strategy(
                param.src_type, param.filter_type, param.dst_type);

        auto winograd_impl = megdnn::winograd::ConvBias<
                fallback::winograd::winograd_2x3_4x4_f,
                param::MatrixMul::Format::MK4>(strategy, UNIT_TILE_SIZE, param);
        return winograd_impl.get_kerns(param, m_matmul_algo);
    }
    MIDOUT_END();
    return {};
}

/* ======================= AlgoWinogradQS8 ======================== */

bool ConvBiasImpl::AlgoWinogradQS8::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_fallback_winograd, 3, 0) {
        using Strategy = fallback::winograd::winograd_2x3_1x1_qs8;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param = megdnn::winograd::ConvBias<Strategy>(
                                      strategy, UNIT_TILE_SIZE, param)
                                      .get_matmul_kern_param(param);

        return m_matmul_algo->usable(matmul_param) &&
               (param.filter_meta.format == param::ConvBias::Format::NCHW ||
                (param.filter_meta.format ==
                         param::ConvBias::Format::NCHW_WINOGRAD &&
                 param.output_block_size == 2 &&
                 param.winograd_matmul_format ==
                         param::MatrixMul::Format::DEFAULT)) &&
               param.filter_meta.should_flip &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 3) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::QuantizedS8;
    }
    MIDOUT_END();
    return false;
}

size_t ConvBiasImpl::AlgoWinogradQS8::get_workspace(
        const NCBKernSizeParam& p) const {
    MEGDNN_MARK_USED_VAR(p);
    MIDOUT_BEGIN(megdnn_fallback_winograd, 3, 1) {
        fallback::winograd::winograd_2x3_1x1_qs8 strategy(
                p.src_type, p.filter_type, p.dst_type);
        return megdnn::winograd::ConvBias<
                       fallback::winograd::winograd_2x3_1x1_qs8>(
                       strategy, UNIT_TILE_SIZE, p)
                .get_workspace_size(p, m_matmul_algo);
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoWinogradQS8::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_fallback_winograd, 3, 2) {
        fallback::winograd::winograd_2x3_1x1_qs8 strategy(
                param.src_type, param.filter_type, param.dst_type);

        auto winograd_impl = megdnn::winograd::ConvBias<
                fallback::winograd::winograd_2x3_1x1_qs8>(
                strategy, UNIT_TILE_SIZE, param);
        return winograd_impl.get_kerns(param, m_matmul_algo);
    }
    MIDOUT_END();
    return {};
}

/* ======================= AlgoWinogradQS8 8x8 ======================== */

bool ConvBiasImpl::AlgoWinogradQS8_8x8::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_fallback_winograd, 4, 0) {
        if (param.filter_meta.icpg % 8 != 0 || param.filter_meta.ocpg % 8 != 0)
            return false;
        using Strategy = fallback::winograd::winograd_2x3_8x8_qs8;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy,
                                           param::MatrixMul::Format::MK8>(
                        strategy, UNIT_TILE_SIZE, param)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               (param.filter_meta.format == param::ConvBias::Format::NCHW ||
                (param.filter_meta.format ==
                         param::ConvBias::Format::NCHW_WINOGRAD &&
                 param.output_block_size == 2 &&
                 param.winograd_matmul_format ==
                         param::MatrixMul::Format::MK8)) &&
               param.filter_meta.should_flip &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 3) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::QuantizedS8;
    }
    MIDOUT_END();
    return false;
}

size_t ConvBiasImpl::AlgoWinogradQS8_8x8::get_workspace(
        const NCBKernSizeParam& p) const {
    MEGDNN_MARK_USED_VAR(p);
    MIDOUT_BEGIN(megdnn_fallback_winograd, 4, 1) {
        fallback::winograd::winograd_2x3_8x8_qs8 strategy(
                p.src_type, p.filter_type, p.dst_type);
        return megdnn::winograd::ConvBias<
                       fallback::winograd::winograd_2x3_8x8_qs8,
                       param::MatrixMul::Format::MK8>(strategy, UNIT_TILE_SIZE,
                                                      p)
                .get_workspace_size(p, m_matmul_algo);
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoWinogradQS8_8x8::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_fallback_winograd, 4, 2) {
        fallback::winograd::winograd_2x3_8x8_qs8 strategy(
                param.src_type, param.filter_type, param.dst_type);

        auto winograd_impl = megdnn::winograd::ConvBias<
                fallback::winograd::winograd_2x3_8x8_qs8,
                param::MatrixMul::Format::MK8>(strategy, UNIT_TILE_SIZE, param);
        return winograd_impl.get_kerns(param, m_matmul_algo);
    }
    MIDOUT_END();
    return {};
}

// vim: syntax=cpp.doxygen
