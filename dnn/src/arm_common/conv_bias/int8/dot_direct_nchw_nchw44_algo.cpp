/**
 * \file
 * dnn/src/arm_common/conv_bias/int8/dot_direct_nchw_nchw44_algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied.
 */
#if __ARM_FEATURE_DOTPROD
#include "megdnn/oprs.h"
#include "src/arm_common/conv_bias/block_helper.h"
#include "src/arm_common/conv_bias/int8/algos.h"
#include "src/arm_common/conv_bias/int8/dot_direct_nchw_nchw44_kern.h"
#include "src/arm_common/elemwise_op.h"
#include "src/common/nchw_nchwxx_valid.h"

#include "midout.h"

using namespace megdnn;
using namespace arm_common;
using conv_fun = std::function<void(
        const WorkspaceBundle& bundle,
        const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index,
        const CpuNDRange& workspace_ids, const CpuNDRange& ncb_range)>;
MIDOUT_DECL(megdnn_arm_common_conv_bias_int8_nchw44_dot)
namespace {
static inline size_t get_perthread_cache_bytes(const int ic, const int ih2,
                                               const int iw2,
                                               const int stride) {
    //! border_size is used to avoid read illegal memory
    constexpr int cacheline_size = 64;
    constexpr int border_size = 2 * cacheline_size;
    const int pack_iw_len = stride == 1 ? 4 : 1;
    return round_up(
            ic * ih2 * iw2 * pack_iw_len * (int)sizeof(int8_t) + border_size,
            cacheline_size);
}
static inline size_t get_temp_bytes(const int iw, const int pw) {
    //! border_size is used to avoid read illegal memory
    constexpr int cacheline_size = 64;
    constexpr int border_size = 1 * cacheline_size;

    return round_up(iw + pw * 2, cacheline_size) + border_size;
}
static void get_rectified_size(
        const megdnn::fallback::ConvBiasImpl::NCBKernSizeParam& param, int& ih2,
        int& iw2) {
    auto&& fm = param.filter_meta;
    const int stride_h = static_cast<int>(fm.stride[0]);
    const int filter_h = static_cast<int>(fm.spatial[0]);
    int ic = param.filter_meta.icpg;
    int iw = param.isz[1];
    int oh = param.osz[0];
    int block_oh = l2_block_helper(param.nr_threads, oh,
                                   ic * iw * sizeof(int8_t) * stride_h);
    ih2 = block_oh * stride_h + filter_h - stride_h;
    iw2 = iw + 2 * static_cast<int>(fm.padding[1]);
}

static WorkspaceBundle get_bundle(const ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    int ic = fm.icpg;
    int fh = fm.spatial[0];
    int fw = fm.spatial[1];
    int iw = param.isz[1];
    int pw = param.filter_meta.padding[1];
    int stride_w = param.filter_meta.stride[1];
    int ih2, iw2;
    get_rectified_size(param, ih2, iw2);

    size_t src_size = get_perthread_cache_bytes(ic, ih2, iw2, stride_w);
    size_t weight_size = fm.group * fm.icpg * fm.ocpg * fh * round_up(fw, 4);
    size_t temp_size = 0;
    if (fm.stride[0] == 1) {
        temp_size = get_temp_bytes(iw, pw);
    }
    return {nullptr,
            {src_size * param.nr_threads, weight_size,
             temp_size * param.nr_threads}};
};

void do_weight_trans(const WorkspaceBundle& bundle,
                     const ConvBiasImpl::NCBKernParam& kern_param,
                     const ConvBiasImpl::NCBKernIndex&, const CpuNDRange&) {
    const int ic = kern_param.filter_meta.icpg;
    const int oc = kern_param.filter_meta.ocpg;
    const int fh = kern_param.filter_meta.spatial[0];
    const int fw = kern_param.filter_meta.spatial[1];
    const int fw2 = round_up(fw, 4);
    auto packed_weight = reinterpret_cast<int8_t*>(bundle.get(1));
    auto origin_weight = kern_param.filter<dt_int8>();
    dot_direct_nchw_nchw44::pack_weight_int8_nchw_nchw44_dot(
            packed_weight, origin_weight, oc, ic, fh, fw, fw2);
}

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
    const int stride_w = kern_param.filter_meta.stride[1];
    const int ph = kern_param.filter_meta.padding[0];
    const int pw = kern_param.filter_meta.padding[1];
    int ih2 = 0;
    int iw2 = 0;
    get_rectified_size(kern_param, ih2, iw2);

    constexpr int pack_c = 4;
    const int batch_id = ncb_index.ndrange_id[0];
    const int group_id = ncb_index.ndrange_id[1];
    constexpr int oc_idx = 0;
    int oc_block = oc;
    int oh_block = l2_block_helper(kern_param.nr_threads, oh,
                                   ic * iw * sizeof(int8_t) * stride_h);
    const int oh_idx = ncb_index.ndrange_id[2];
    const int oh_block_real = std::min(oh - oh_idx * oh_block, oh_block);
    const int ih_real = oh_block_real * stride_h + fh - stride_h;
    const int src_top_pad = std::max(ph - oh_idx * oh_block * stride_h, 0);
    const int src_bottom_pad = std::max(
            (oh_idx * oh_block + oh_block_real - 1) * stride_h + fh - ih - ph,
            0);
    const int remain_right_pad = std::max(iw2 - iw - pw, 0);
    const int src_offset = std::max(oh_idx * oh_block * stride_h - ph, 0) * iw;
    const int8_t* origin_sptr =
            static_cast<const int8_t*>(
                    kern_param.src<int8_t>(batch_id, group_id, 0, 1, 1)) +
            src_offset;
    const size_t src_size = get_perthread_cache_bytes(ic, ih2, iw2, stride_w);
    int8_t* sptr = reinterpret_cast<int8_t*>(bundle.get(0)) +
                   ncb_index.thread_id * src_size;
    int8_t* tmp_ptr = nullptr;
    if (stride == 1) {
        const size_t tmp_size = get_temp_bytes(iw, pw);
        tmp_ptr = reinterpret_cast<int8_t*>(bundle.get(2)) +
                  ncb_index.thread_id * tmp_size;
    }
    dot_direct_nchw_nchw44::pack_src_int8_nchw_nchw44_dot<stride>(
            sptr, origin_sptr, ph, pw, remain_right_pad,
            ih_real - src_top_pad - src_bottom_pad, iw, iw2, src_top_pad,
            src_bottom_pad, ic, ih * iw, tmp_ptr);

    const int8_t* fptr =
            reinterpret_cast<int8_t*>(bundle.get(1)) + oc_idx * fh * fw * ic;
    int8_t* dst = kern_param.dst<int8_t>(batch_id, group_id) +
                  oh_idx * oh_block * ow * pack_c;

    const int bias_offset = oc_idx;
    const int32_t* bptr =
            kern_param.bias<dt_int32>(batch_id, group_id) + bias_offset;

    float scale_bias = kern_param.bias_type.param<dtype::QuantizedS32>().scale;
    float scale_dst = kern_param.dst_type.param<dtype::QuantizedS8>().scale;
    Op op(scale_bias, scale_dst);
    dot_direct_nchw_nchw44::conv_direct_int8_nchw_nchw44_dot<bias_mode, Op,
                                                             filter, stride>(
            sptr, fptr, bptr, nullptr, dst, oc_block, ic, ih_real, iw2, oh,
            oh_block_real, ow, op);
}

}  // namespace

bool ConvBiasImpl::AlgoDotS8DirectNCHWNCHW44::usable(
        const NCBKernSizeParam& param, AlgoSelectionStrategy) const {
    return nchw_nchwxx_valid<NchwNchwxxType::NCHW44_INT8_DOT>(
            param.src_type.enumv(), param.filter_type.enumv(),
            param.dst_type.enumv(), param.filter_meta, param.bias_mode,
            param.nonlineMode);
}

size_t ConvBiasImpl::AlgoDotS8DirectNCHWNCHW44::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8_nchw44_dot,
                 midout_iv("AlgoDotS8DirectNCHWNCHW44::get_workspace"_hash)) {
        return get_bundle(param).total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoDotS8DirectNCHWNCHW44::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    auto fm = param.filter_meta;
    const int batch = param.n;
    const int group = fm.group;
    WorkspaceBundle bundle = get_bundle(param);
    conv_fun do_conv_fun = nullptr;
    // NOTE: remain_w is not used to gen hash of midout for compatible with
// shape runtime
#define DO_CONV_KERN_FUN(stride, filter, bias_mode, op)              \
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8_nchw44_dot,        \
                 midout_iv(#stride #filter #bias_mode #op##_hash)) { \
        do_conv_fun = do_conv_kern<filter, bias_mode, op, stride>;   \
    }                                                                \
    MIDOUT_END();

#define GET_OP_PARAM(stride, filter, bias_mode)                          \
    switch (param.nonlineMode) {                                         \
        case param::ConvBias::NonlineMode::IDENTITY:                     \
            DO_CONV_KERN_FUN(stride, filter, bias_mode,                  \
                             TypeCvtOp<dt_qint32 MEGDNN_COMMA dt_qint8>) \
            break;                                                       \
        case param::ConvBias::NonlineMode::RELU:                         \
            DO_CONV_KERN_FUN(stride, filter, bias_mode,                  \
                             ReluOp<dt_qint32 MEGDNN_COMMA dt_qint8>)    \
            break;                                                       \
        case param::ConvBias::NonlineMode::H_SWISH:                      \
            DO_CONV_KERN_FUN(stride, filter, bias_mode,                  \
                             HSwishOp<dt_qint32 MEGDNN_COMMA dt_qint8>)  \
            break;                                                       \
        default:                                                         \
            megdnn_assert(0);                                            \
            break;                                                       \
    }

#define GET_BIAS_MODE_PARAM(stride, filter)                                \
    switch (param.bias_mode) {                                             \
        case BiasMode::NO_BIAS:                                            \
            GET_OP_PARAM(stride, filter, BiasMode::NO_BIAS)                \
            break;                                                         \
        case BiasMode::BROADCAST_CHANNEL_BIAS:                             \
            GET_OP_PARAM(stride, filter, BiasMode::BROADCAST_CHANNEL_BIAS) \
            break;                                                         \
        default:                                                           \
            megdnn_assert(0);                                              \
            break;                                                         \
    }

#define DISPATCH_CONV_KERN(stride)          \
    switch (param.filter_meta.spatial[0]) { \
        case 2:                             \
            GET_BIAS_MODE_PARAM(stride, 2)  \
            break;                          \
        case 3:                             \
            GET_BIAS_MODE_PARAM(stride, 3)  \
            break;                          \
        case 5:                             \
            GET_BIAS_MODE_PARAM(stride, 5)  \
            break;                          \
        case 7:                             \
            GET_BIAS_MODE_PARAM(stride, 7)  \
            break;                          \
        default:                            \
            megdnn_assert(0);               \
            break;                          \
    }

    switch (param.filter_meta.stride[0]) {
        case 1:
            DISPATCH_CONV_KERN(1);
            break;
        case 2:
            DISPATCH_CONV_KERN(2);
            break;
        default:
            megdnn_assert(0);
            break;
    }

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
                                   ic * iw * sizeof(int8_t) * stride_h);

    CpuNDRange ncb_range = {static_cast<size_t>(batch),
                            static_cast<size_t>(group),
                            static_cast<size_t>(div_ceil(oh, oh_block))};

    auto do_trans_weight = [bundle](const NCBKernParam& kern_param,
                                    const NCBKernIndex& ncb_index) mutable {
        bundle.set(kern_param.workspace_ptr);
        do_weight_trans(bundle, kern_param, ncb_index, ncb_index.ndrange_id);
    };
    ret_kerns.push_back({do_trans_weight, {1}});

    auto do_conv = [bundle, do_conv_fun, ncb_range](
                           const NCBKernParam& kern_param,
                           const NCBKernIndex& ncb_index) mutable {
        bundle.set(kern_param.workspace_ptr);
        do_conv_fun(bundle, kern_param, ncb_index, ncb_index.ndrange_id,
                    ncb_range);
    };
    ret_kerns.push_back({do_conv, ncb_range});
    return ret_kerns;
}
#endif

// vim: syntax=cpp.doxygen
