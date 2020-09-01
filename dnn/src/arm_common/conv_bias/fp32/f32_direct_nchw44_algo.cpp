/**
 * \file dnn/src/arm_common/conv_bias/fp32/f32_direct_nchw44_algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied.
 */

#include "megdnn/oprs.h"
#include "src/arm_common/conv_bias/block_helper.h"
#include "src/arm_common/conv_bias/fp32/algos.h"
#include "src/arm_common/conv_bias/fp32/f32_direct_nchw44_kern.h"

#include "src/arm_common/elemwise_op.h"

#include "midout.h"

using namespace megdnn;
using namespace arm_common;
using conv_fun = std::function<void(
        const WorkspaceBundle& bundle,
        const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index,
        const CpuNDRange& workspace_ids, const CpuNDRange& ncb_range)>;
MIDOUT_DECL(megdnn_arm_common_conv_bias_fp32_nchw44_stride1)
namespace {

static inline size_t get_perthread_cache_bytes(const int ic, const int ih2,
                                               const int iw2) {
    // border_size is used to avoid read illegal memory
    int border_size = 64 * 2;
    return ic * ih2 * iw2 * sizeof(float) + border_size;
}
static void get_rectified_size(
        const megdnn::fallback::ConvBiasImpl::NCBKernSizeParam& param, int& ih2,
        int& iw2, int& oh2, int& ow2) {
    constexpr int nr_elements_in_cacheline = 64 / sizeof(float);
    int ic = param.filter_meta.icpg;
    int iw = param.isz[1];
    int oh = param.osz[0];
    int ow = param.osz[1];
    auto&& fm = param.filter_meta;
    const int stride_h = static_cast<int>(fm.stride[0]);
    const int filter_h = static_cast<int>(fm.spatial[0]);

    oh2 = oh;
    ow2 = ow;

    int block_oh = l2_block_helper(param.nr_threads, oh,
                                   ic * iw * sizeof(float) * stride_h);
    ih2 = block_oh * stride_h + filter_h - stride_h;
    iw2 = round_up(iw + 2 * static_cast<int>(fm.padding[1]),
                   nr_elements_in_cacheline);
}

static WorkspaceBundle get_bundle(const ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    int ic = fm.icpg;
    int ih2, iw2, oh2, ow2;
    get_rectified_size(param, ih2, iw2, oh2, ow2);

    size_t src_size = get_perthread_cache_bytes(ic, ih2, iw2);
    return {nullptr, {src_size * param.nr_threads}};
};

template <size_t filter, BiasMode bias_mode, typename Op, int stride>
static void do_conv_kern(const WorkspaceBundle& bundle,
                         const ConvBiasImpl::NCBKernParam& kern_param,
                         const ConvBiasImpl::NCBKernIndex& ncb_index,
                         const CpuNDRange&, const CpuNDRange&) {
    const int oh = kern_param.osz[0];
    const int ow = kern_param.osz[1];
    const int fh = kern_param.filter_meta.spatial[0];
    const int fw = kern_param.filter_meta.spatial[1];
    const int ic = kern_param.filter_meta.icpg;
    const int oc = kern_param.filter_meta.ocpg;
    const int ih = kern_param.isz[0];
    const int iw = kern_param.isz[1];
    const int stride_h = kern_param.filter_meta.stride[0];
    const int ph = kern_param.filter_meta.padding[0];
    const int pw = kern_param.filter_meta.padding[1];
    int ih2 = 0;
    int iw2 = 0;
    int oh2 = 0;
    int ow2 = 0;
    get_rectified_size(kern_param, ih2, iw2, oh2, ow2);

    constexpr int pack_c = 4;
    const int batch_id = ncb_index.ndrange_id[0];
    const int group_id = ncb_index.ndrange_id[1];
    constexpr int oc_idx = 0;
    int oc_block = oc;
    int oh_block = l2_block_helper(kern_param.nr_threads, oh2,
                                   ic * iw * sizeof(float) * stride_h);
    const int oh_idx = ncb_index.ndrange_id[2];
    const int oh_block_real = std::min(oh - oh_idx * oh_block, oh_block);
    const int ih_real = oh_block_real * stride_h + fh - stride_h;
    const int src_top_pad = std::max(ph - oh_idx * oh_block * stride_h, 0);
    const int src_bottom_pad = std::max(
            (oh_idx * oh_block + oh_block_real - 1) * stride_h + fh - ih - ph,
            0);
    const int remain_right_pad = std::max(iw2 - iw - pw, 0);
    const int src_offset =
            std::max(oh_idx * oh_block * stride_h - ph, 0) * iw * pack_c;
    const float* origin_sptr = static_cast<const float*>(kern_param.src<float>(
                                       batch_id, group_id, 0, 1, 1)) +
                               src_offset;
    const size_t src_size = get_perthread_cache_bytes(ic, ih2, iw2);
    float* sptr = reinterpret_cast<float*>((int8_t*)bundle.get(0) +
                                           ncb_index.thread_id * src_size);

    conv_bias::pack_src_fp32_nchw44<stride>(
            sptr, origin_sptr, ph, pw, remain_right_pad,
            ih_real - src_top_pad - src_bottom_pad, iw, iw2, src_top_pad,
            src_bottom_pad, ic, ih * iw);

    const float* fptr =
            kern_param.filter<dt_float32>(group_id) + oc_idx * fh * fw * ic;
    float_t* dst = kern_param.dst<float_t>(batch_id, group_id) +
                   oh_idx * oh_block * ow * pack_c;
    const int bias_offset = bias_mode == BiasMode::BIAS
                                    ? oh_idx * oh_block * ow * pack_c
                                    : oc_idx;
    const float* bptr =
            kern_param.bias<dt_float32>(batch_id, group_id, oc_idx, 1, pack_c) +
            bias_offset;

    Op op;
    conv_bias::conv_direct_fp32_nchw44<bias_mode, Op, filter, stride>(
            sptr, fptr, bptr, nullptr, dst, oc_block, ic, ih_real, iw2, oh,
            oh_block_real, ow, op, ph, pw);
}

}  // namespace

/* ===================== stride1 algo ===================== */
bool ConvBiasImpl::AlgoF32DirectNCHW44::usable(const NCBKernSizeParam& param,
                                               AlgoSelectionStrategy) const {
    auto&& fm = param.filter_meta;
    auto fh = fm.spatial[0];
    int oc = fm.ocpg;
    int ic = fm.icpg;
    bool ok_type = ((param.src_type.enumv() == DTypeEnum::Float32 &&
                     param.filter_type.enumv() == DTypeEnum::Float32 &&
                     (param.dst_type.enumv() == DTypeEnum::Float32))) &&
                   (fm.format == param::Convolution::Format::NCHW44);
    bool ok_src_dst = (oc % 4 == 0 && oc >= 4 && ic % 4 == 0 && ic >= 4);
    bool ok_filter = fm.spatial_ndim == 2 && fh == fm.spatial[1] &&
                     (fh == 2 || fh == 3 || fh == 5 || fh == 7);
    bool ok_slide = fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
                    ((fm.stride[0] == 1 && fm.stride[1] == 1) ||
                     (fm.stride[0] == 2 && fm.stride[1] == 2));
    bool ok_conv = !fm.should_flip;
    bool avaible = ok_type && ok_src_dst && ok_filter && ok_slide && ok_conv;
    return avaible;
}

size_t ConvBiasImpl::AlgoF32DirectNCHW44::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_fp32_nchw44_stride1,
                 midout_iv("AlgoF32DirectNCHW44::get_workspace"_hash)) {
        return get_bundle(param).total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoF32DirectNCHW44::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    auto fm = param.filter_meta;
    const int batch = param.n;
    const int group = fm.group;
    WorkspaceBundle wbundle = get_bundle(param);
    conv_fun do_conv_fun = nullptr;
    // NOTE: remain_w is not used to gen hash of midout for compatible with
// shape runtime
#define DO_CONV_KERN_FUN(filter, bias_mode, op, stride)              \
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_fp32_nchw44_stride1,    \
                 midout_iv(#filter #bias_mode #stride #op##_hash)) { \
        do_conv_fun = do_conv_kern<filter, bias_mode, op, stride>;   \
    }                                                                \
    MIDOUT_END();

#define GET_STRIDE_PARAM(filter, bias_mode, op)         \
    switch (fm.stride[0]) {                             \
        case 1:                                         \
            DO_CONV_KERN_FUN(filter, bias_mode, op, 1); \
            break;                                      \
        case 2:                                         \
            DO_CONV_KERN_FUN(filter, bias_mode, op, 2); \
            break;                                      \
                                                        \
        default:                                        \
            megdnn_assert(0);                           \
    }

#define GET_OP_PARAM(filter, bias_mode)                                \
    switch (param.nonlineMode) {                                       \
        case param::ConvBias::NonlineMode::IDENTITY:                   \
            GET_STRIDE_PARAM(filter, bias_mode, NoneOp<dt_float32>)    \
            break;                                                     \
        case param::ConvBias::NonlineMode::RELU:                       \
            GET_STRIDE_PARAM(filter, bias_mode, ReluOp<dt_float32>)    \
            break;                                                     \
        case param::ConvBias::NonlineMode::H_SWISH:                    \
            GET_STRIDE_PARAM(filter, bias_mode, HSwishOp<dt_float32>)  \
            break;                                                     \
        case param::ConvBias::NonlineMode::SIGMOID:                    \
            GET_STRIDE_PARAM(filter, bias_mode, SigmoidOp<dt_float32>) \
            break;                                                     \
        default:                                                       \
            megdnn_assert(0);                                          \
            break;                                                     \
    }

#define GET_BIAS_MODE_PARAM(filter)                                \
    switch (param.bias_mode) {                                     \
        case BiasMode::NO_BIAS:                                    \
            GET_OP_PARAM(filter, BiasMode::NO_BIAS)                \
            break;                                                 \
        case BiasMode::BROADCAST_CHANNEL_BIAS:                     \
            GET_OP_PARAM(filter, BiasMode::BROADCAST_CHANNEL_BIAS) \
            break;                                                 \
        case BiasMode::BIAS:                                       \
            GET_OP_PARAM(filter, BiasMode::BIAS)                   \
            break;                                                 \
        default:                                                   \
            megdnn_assert(0);                                      \
            break;                                                 \
    }

#define DISPATCH_CONV_KERN()                \
    switch (param.filter_meta.spatial[0]) { \
        case 2:                             \
            GET_BIAS_MODE_PARAM(2)          \
            break;                          \
        case 3:                             \
            GET_BIAS_MODE_PARAM(3)          \
            break;                          \
        case 5:                             \
            GET_BIAS_MODE_PARAM(5)          \
            break;                          \
        case 7:                             \
            GET_BIAS_MODE_PARAM(7)          \
            break;                          \
        default:                            \
            megdnn_assert(0);               \
            break;                          \
    }

    DISPATCH_CONV_KERN();

#undef DO_CONV_KERN_FUN
#undef GET_REMAIN_W_PARAM
#undef GET_OP_PARAM
#undef GET_BIAS_MODE_PARAM
#undef DISPATCH_CONV_KERN

    megdnn_assert(do_conv_fun);

    SmallVector<ConvBiasImpl::NCBKern> ret_kerns;
    int oh = param.osz[0];
    int ic = param.filter_meta.icpg;
    int iw = param.isz[1];
    int stride_h = param.filter_meta.stride[0];
    int oh_block = l2_block_helper(param.nr_threads, oh,
                                   ic * iw * sizeof(float) * stride_h);
    CpuNDRange ncb_range = {static_cast<size_t>(batch),
                            static_cast<size_t>(group),
                            static_cast<size_t>(div_ceil(oh, oh_block))};
    auto do_conv = [wbundle, do_conv_fun, ncb_range](
                           const NCBKernParam& kern_param,
                           const NCBKernIndex& ncb_index) mutable {
        wbundle.set(kern_param.workspace_ptr);
        do_conv_fun(wbundle, kern_param, ncb_index, ncb_index.ndrange_id,
                    ncb_range);
    };
    ret_kerns.push_back({do_conv, ncb_range});
    return ret_kerns;
}

// vim: syntax=cpp.doxygen
