/**
 * \file dnn/src/arm_common/conv_bias/int8/direct_dotpord_nchw44_algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#if __ARM_FEATURE_DOTPROD

#include "src/arm_common/conv_bias/block_helper.h"
#include "src/arm_common/conv_bias/int8/algos.h"
#include "src/arm_common/conv_bias/int8/direct_dotprod_nchw44.h"
#include "src/arm_common/elemwise_op.h"

#include "midout.h"

using namespace megdnn;
using namespace arm_common;

MIDOUT_DECL(megdnn_arm_common_conv_bias_int8)

using direct_fun =
        std::function<void(const WorkspaceBundle& bundle,
                           const ConvBiasImpl::NCBKernParam& ncb_param,
                           const ConvBiasImpl::NCBKernIndex& ncb_index)>;

namespace {

static void get_rectified_size(
        const megdnn::fallback::ConvBiasImpl::NCBKernSizeParam& param, int& ih,
        int& iw, int& oh, int& ow) {
    int IC = param.filter_meta.icpg;
    int IW = param.isz[1];
    int OH = param.osz[0];
    int OW = param.osz[1];

    oh = OH;
    ow = OW;

    constexpr int cacheline = 64 / sizeof(int8_t);
    int oh_tile_size =
            l2_block_helper(param.nr_threads, OH, IC * IW * sizeof(int8_t) * 2);
    auto&& fm = param.filter_meta;
    const int SH = static_cast<int>(fm.stride[0]);
    const int FH = static_cast<int>(fm.spatial[0]);
    const int PW = static_cast<int>(fm.padding[1]);
    ih = oh_tile_size * SH + FH - SH;
    iw = round_up(IW + 2 * PW, cacheline);
}

static inline int get_perthread_cache_bytes(const int ic, const int ih,
                                            const int iw) {
    // border_size is used to avoid read illegal memory
    int border_size = 64 * 2;
    return ic * ih * iw * sizeof(int8_t) + border_size;
}

static WorkspaceBundle get_bundle(const ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    int IC = fm.icpg;
    int ih2, iw2, oh2, ow2;
    get_rectified_size(param, ih2, iw2, oh2, ow2);

    int bytes_of_copy_per_thread = get_perthread_cache_bytes(IC, ih2, iw2);
    return {nullptr, {bytes_of_copy_per_thread * param.nr_threads}};
}

template <typename dst_type, size_t filter_size, BiasMode bias_mode,
          typename Op, int stride>
static void conv_kern(const WorkspaceBundle& bundle,
                      const ConvBiasImpl::NCBKernParam& ncb_param,
                      const ConvBiasImpl::NCBKernIndex& ncb_index) {
    const int OH = ncb_param.osz[0];
    const int OW = ncb_param.osz[1];
    const int FH = ncb_param.filter_meta.spatial[0];
    const int IC = ncb_param.filter_meta.icpg;
    const int OC = ncb_param.filter_meta.ocpg;
    const int IH = ncb_param.isz[0];
    const int IW = ncb_param.isz[1];
    const int SH = ncb_param.filter_meta.stride[0];
    const int PH = ncb_param.filter_meta.padding[0];
    const int PW = ncb_param.filter_meta.padding[1];

    int ih2 = 0;
    int iw2 = 0;
    int oh2 = 0;
    int ow2 = 0;
    get_rectified_size(ncb_param, ih2, iw2, oh2, ow2);

    constexpr int IC_PACK_SIZE = 4;
    constexpr int OC_PACK_SIZE = 4;

    const int batch_id = ncb_index.ndrange_id[0];
    const int group_id = ncb_index.ndrange_id[1];
    const int oh_tile_id = ncb_index.ndrange_id[2];
    const int thread_id = ncb_index.thread_id;

    const int oh_tile_size = l2_block_helper(ncb_param.nr_threads, OH,
                                             IC * IW * sizeof(int8_t) * 2);
    const int oh_start_row = oh_tile_id * oh_tile_size;
    const int ih_start_row = std::max(oh_start_row * SH - PH, 0);

    const int oh_real_size = std::min(OH - oh_start_row, oh_tile_size);
    const int ih_real_size = oh_real_size * SH + FH - SH;

    const int rows_padding_at_top = std::max(PH - oh_start_row * SH, 0);
    const int rows_padding_at_bottom =
            std::max((oh_start_row + oh_real_size - 1) * SH + FH - IH - PH, 0);
    const int cols_padding_at_left = PW;
    const int cols_padding_at_right = std::max(iw2 - IW - PW, 0);

    //! src layout{IC/4, IH, IW, 4}
    const int bytes_of_src_offset =
            ih_start_row * IW * IC_PACK_SIZE * sizeof(int8_t);
    const int8_t* copy_src = static_cast<const int8_t*>(
            ncb_param.src<int8_t>(batch_id, group_id) + bytes_of_src_offset);

    const int bytes_of_copy_per_thread =
            get_perthread_cache_bytes(IC, ih2, iw2);
    int8_t* copy_dst = reinterpret_cast<int8_t*>(bundle.get(0)) +
                       thread_id * bytes_of_copy_per_thread;

    const int rows_copy_from_src =
            ih_real_size - rows_padding_at_top - rows_padding_at_bottom;

    direct_dotprod_nchw44::copy_packed_src_int8_nchw44<stride>(
            copy_dst, iw2, copy_src, IW, IC, IH * IW, rows_copy_from_src,
            cols_padding_at_left, cols_padding_at_right, rows_padding_at_top,
            rows_padding_at_bottom);

    const int8_t* weights = ncb_param.filter<int8_t>(group_id);

    dst_type* dst = ncb_param.dst<dst_type>(batch_id, group_id) +
                    oh_start_row * OW * OC_PACK_SIZE;

    //! only broadcast or no_bias
    const int32_t* bias = ncb_param.bias<int32_t>(batch_id, group_id);

    Op op = Op(1.0f, 4.0f);
    if (ncb_param.dst_type.enumv() == DTypeEnum::QuantizedS8) {
        float scale_bias =
                ncb_param.bias_type.param<dtype::QuantizedS32>().scale;
        float scale_dst = ncb_param.dst_type.param<dtype::QuantizedS8>().scale;
        op = Op(scale_bias, scale_dst);
    }
    direct_dotprod_nchw44::conv_direct_sdot_int8_nchw44<
            dst_type, stride, bias_mode, Op, filter_size>(
            dst, OH, OW, copy_dst, ih_real_size, iw2, weights, bias,
            oh_real_size, OC, IC, op);
}

}  // namespace

bool ConvBiasImpl::AlgoDotS8Direct_NCHW44::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy algo_selection_strategy) const {
    MEGDNN_MARK_USED_VAR(algo_selection_strategy);
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    auto FW = fm.spatial[1];
    auto SH = fm.stride[0];
    auto SW = fm.stride[1];
    auto OC = fm.ocpg;
    auto IC = fm.icpg;

    //! src and filter are qint8, dst is qint8.
    bool data_type_ok = param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                        param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
                        (param.dst_type.enumv() == DTypeEnum::QuantizedS8 ||
                         param.dst_type.enumv() == DTypeEnum::QuantizedS32);

    if (param.bias_type.valid()) {
        data_type_ok &= param.bias_type.enumv() == DTypeEnum::QuantizedS32;
    }

    data_type_ok |= param.src_type.enumv() == DTypeEnum::Int8 &&
                    param.filter_type.enumv() == DTypeEnum::Int8 &&
                    param.dst_type.enumv() == DTypeEnum::Int32;

    bool layout_ok = fm.format == param::Convolution::Format::NCHW44_DOT &&
                     IC % 4 == 0 && OC % 4 == 0;

    bool param_ok = !fm.should_flip && fm.spatial_ndim == 2 &&
                    fm.dilation[0] == 1 && fm.dilation[1] == 1 && FH == FW &&
                    (FH >= 2 && FH <= 7);

    bool stride_ok = SH == SW && (SH == 1 || SH == 2);

    return data_type_ok && layout_ok && param_ok && stride_ok;
}

bool ConvBiasImpl::AlgoDotS8Direct_NCHW44::is_preferred(
        const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    return true;
}

size_t ConvBiasImpl::AlgoDotS8Direct_NCHW44::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("ALGODOTS8DIRECT_NCHW44::get_workspace"_hash)) {
        return get_bundle(param).total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoDotS8Direct_NCHW44::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("ALGODOTS8DIRECT_NCHW44::dispatch_kerns"_hash)) {
        auto fm = param.filter_meta;
        size_t BATCH = param.n;
        size_t GROUP = fm.group;
        WorkspaceBundle wbundle = get_bundle(param);
        direct_fun kernel = nullptr;
        bool quantized = param.dst_type.enumv() == DTypeEnum::QuantizedS8;

#define DO_CONV_KERN_FUN(dst_type, filter, bias_mode, op, stride)      \
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,                     \
                 midout_iv(#dst_type #filter #bias_mode #op##_hash)) { \
        kernel = conv_kern<dst_type, filter, bias_mode, op, stride>;   \
    }                                                                  \
    MIDOUT_END();

#define GET_OP_PARAM(i, bias_mode, stride)                                   \
    switch (param.nonlineMode) {                                             \
        case param::ConvBias::NonlineMode::IDENTITY:                         \
            if (quantized) {                                                 \
                DO_CONV_KERN_FUN(dt_int8, i, bias_mode,                      \
                                 TypeCvtOp<dt_qint32 MEGDNN_COMMA dt_qint8>, \
                                 stride)                                     \
            } else {                                                         \
                DO_CONV_KERN_FUN(dt_int32, i, bias_mode,                     \
                                 NoneOp<dt_qint32 MEGDNN_COMMA dt_qint8>,    \
                                 stride)                                     \
            }                                                                \
            break;                                                           \
        case param::ConvBias::NonlineMode::RELU:                             \
            if (quantized) {                                                 \
                DO_CONV_KERN_FUN(dt_int8, i, bias_mode,                      \
                                 ReluOp<dt_qint32 MEGDNN_COMMA dt_qint8>,    \
                                 stride)                                     \
            } else {                                                         \
                megdnn_assert("No support NoQuantized RELU");                \
            }                                                                \
            break;                                                           \
        case param::ConvBias::NonlineMode::H_SWISH:                          \
            if (quantized) {                                                 \
                DO_CONV_KERN_FUN(dt_int8, i, bias_mode,                      \
                                 HSwishOp<dt_qint32 MEGDNN_COMMA dt_qint8>,  \
                                 stride)                                     \
            } else {                                                         \
                megdnn_assert("No support NoQuantized H_SWISH");             \
            }                                                                \
            break;                                                           \
        default:                                                             \
            megdnn_assert(0);                                                \
            break;                                                           \
    }

#define GET_STRIDE_PARAM(filter, bias_mode)     \
    switch (fm.stride[0]) {                     \
        case 1:                                 \
            GET_OP_PARAM(filter, bias_mode, 1); \
            break;                              \
        case 2:                                 \
            GET_OP_PARAM(filter, bias_mode, 2); \
            break;                              \
        default:                                \
            megdnn_assert(0);                   \
    }

#define GET_BIAS_MODE_PARAM(filter)                                    \
    switch (param.bias_mode) {                                         \
        case BiasMode::NO_BIAS:                                        \
            GET_STRIDE_PARAM(filter, BiasMode::NO_BIAS)                \
            break;                                                     \
        case BiasMode::BROADCAST_CHANNEL_BIAS:                         \
            GET_STRIDE_PARAM(filter, BiasMode::BROADCAST_CHANNEL_BIAS) \
            break;                                                     \
        default:                                                       \
            megdnn_assert(0);                                          \
            break;                                                     \
    }

#define SELECT_CONV_KERN()                  \
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

        SELECT_CONV_KERN()

#undef DO_CONV_KERN_FUN
#undef GET_OP_PARAM
#undef GET_STRIDE_PARAM
#undef GET_BIAS_MODE_PARAM
#undef SELECT_CONV_KERN

        megdnn_assert(kernel);

        SmallVector<ConvBiasImpl::NCBKern> ret_kerns;
        int OH = param.osz[0];
        int IC = param.filter_meta.icpg;
        int IW = param.isz[1];
        int oh_tile_size = l2_block_helper(param.nr_threads, OH,
                                           IC * IW * sizeof(int8_t) * 2);
        size_t oh_tiles = static_cast<size_t>(div_ceil(OH, oh_tile_size));

        auto do_conv = [wbundle, kernel](
                               const NCBKernParam& ncb_param,
                               const NCBKernIndex& ncb_index) mutable {
            wbundle.set(ncb_param.workspace_ptr);
            kernel(wbundle, ncb_param, std::move(ncb_index));
        };

        ret_kerns.push_back({do_conv, {BATCH, GROUP, oh_tiles}});
        return ret_kerns;
    }
    MIDOUT_END();
    return {};
}

#endif

// vim: syntax=cpp.doxygen
