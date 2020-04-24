/**
 * \file dnn/src/arm_common/conv_bias/int8/direct_stride2_nchw_nchw44_algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megdnn/oprs.h"
#include "src/arm_common/conv_bias/int8/algos.h"
#include "src/arm_common/conv_bias/int8/direct_stride2_nchw_nchw44_kern.h"
#include "src/arm_common/conv_bias/int8/strategy.h"
#include "src/arm_common/elemwise_op.h"
#include "src/common/opr_delegate.h"

#include "midout.h"

using namespace megdnn;
using namespace arm_common;
using conv_fun = std::function<void(
        WorkspaceBundle bundle, const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index,
        const CpuNDRange& workspace_ids, const CpuNDRange& ncb_range)>;
MIDOUT_DECL(megdnn_arm_common_conv_bias_int8_nchw_nchw44_stride2)

static void get_rectified_size(
        const megdnn::fallback::ConvBiasImpl::NCBKernSizeParam& param,
        size_t& IH2, size_t& IW2, size_t& OH2, size_t& OW2) {
    auto&& fm = param.filter_meta;
    size_t IH = param.isz[0];
    size_t IW = param.isz[1];
    size_t OH = param.osz[0];
    size_t OW = param.osz[1];

    OH2 = OH;
    OW2 = OW;
    IH2 = round_up(IH + 2 * fm.padding[0], static_cast<size_t>(2));
    IW2 = IW + 2 * fm.padding[1];
}
static WorkspaceBundle get_bundle(const ConvBiasImpl::NCBKernSizeParam& param) {
    constexpr size_t src_expand = 4;
    auto&& fm = param.filter_meta;
    size_t group = fm.group;
    size_t batch = param.n;
    size_t IC = fm.icpg;
    size_t OC = fm.ocpg;
    size_t FH = fm.spatial[0];
    size_t FW = fm.spatial[1];
    size_t IH2, IW2, OH2, OW2;
    get_rectified_size(param, IH2, IW2, OH2, OW2);
    megdnn_assert(group == 1, "only support group == 1 now");
    size_t src_size =
            batch * group * IC * IH2 * IW2 * sizeof(int8_t) * src_expand;
    size_t weight_size = group * OC * IC * FH * FW * sizeof(int8_t);
    return {nullptr, {src_size, weight_size}};
};

static void copy_padding_kern(WorkspaceBundle bundle,
                              const ConvBiasImpl::NCBKernParam& kern_param,
                              const ConvBiasImpl::NCBKernIndex& ncb_index,
                              const CpuNDRange& workspace_ids) {
    size_t IH = kern_param.isz[0];
    size_t IW = kern_param.isz[1];
    size_t IC = kern_param.filter_meta.icpg;
    size_t PH = kern_param.filter_meta.padding[0];
    size_t PW = kern_param.filter_meta.padding[1];
    size_t GROUP = kern_param.filter_meta.group;

    size_t IH2, IW2, OH2, OW2;
    get_rectified_size(kern_param, IH2, IW2, OH2, OW2);
    size_t padding_group_size = IH2 * IW2 * IC;
    bundle.set(kern_param.workspace_ptr);
    //! Used for get the workspace offset
    constexpr int expend_element = 4;
    // TODO: block dim is better to get from arg
    size_t workspace_ic_block = 1;
    size_t workspace_batch_id = workspace_ids[0];
    size_t workspace_group_id = workspace_ids[1];
    size_t workspace_ic_id = workspace_ids[2];
    size_t workspace_ic = workspace_ic_id * workspace_ic_block;
    size_t batch_id = ncb_index.ndrange_id[0];
    size_t group_id = ncb_index.ndrange_id[1];

    const int8_t* sptr = static_cast<const int8_t*>(
            kern_param.src<int8_t>(batch_id, group_id, workspace_ic_id, 1, 1));
    //! copy to sptr_base to eliminate padding effect
    int8_t* sptr_base = static_cast<int8_t*>(bundle.get(0)) +
                        (workspace_batch_id * GROUP * padding_group_size +
                         workspace_group_id * padding_group_size +
                         workspace_ic * IH2 * IW2) *
                                expend_element;
    conv_bias::pack_nchw_src_for_nchw44_conv(sptr, sptr_base, 1, PH, PH, PW, PW,
                                             IH, IW);
}

template <size_t filter, BiasMode bias_mode, typename Op>
static void do_conv_kern(WorkspaceBundle bundle,
                         const ConvBiasImpl::NCBKernParam& kern_param,
                         const ConvBiasImpl::NCBKernIndex& ncb_index,
                         const CpuNDRange& workspace_ids,
                         const CpuNDRange& ncb_range) {
    size_t OH = kern_param.osz[0];
    size_t OW = kern_param.osz[1];
    size_t FH = kern_param.filter_meta.spatial[0];
    size_t FW = kern_param.filter_meta.spatial[1];
    size_t IC = kern_param.filter_meta.icpg;
    size_t OC = kern_param.filter_meta.ocpg;
    size_t GROUP = kern_param.filter_meta.group;
    size_t IH2, IW2, OH2, OW2;
    get_rectified_size(kern_param, IH2, IW2, OH2, OW2);
    bool need_post_process =
            kern_param.dst_type.enumv() == DTypeEnum::QuantizedS8;
    //! if dst_type is qint32, the op is not used, just fill with (1.0f,4.0f)
    Op op = Op(1.0f, 4.0f);
    if (need_post_process) {
        float scale_bias =
                kern_param.bias_type.param<dtype::QuantizedS32>().scale;
        float scale_dst = kern_param.dst_type.param<dtype::QuantizedS8>().scale;
        op = Op(scale_bias, scale_dst);
    }
    size_t padding_group_size = IH2 * IW2 * IC;
    bundle.set(kern_param.workspace_ptr);

    constexpr size_t pack_c = 4;
    constexpr size_t src_expand_size = 4;
    const size_t workspace_batch_id = workspace_ids[0];
    const size_t workspace_group_id = workspace_ids[1];
    const size_t batch_id = ncb_index.ndrange_id[0];
    const size_t group_id = ncb_index.ndrange_id[1];
    const size_t oc_id = ncb_index.ndrange_id[2];
    const size_t oc_block_num = ncb_range[2];
    size_t nr_pack_per_step = div_ceil(div_ceil(OC, pack_c), oc_block_num);
    size_t oc_block = nr_pack_per_step * pack_c;
    const size_t oc_idx = oc_id * oc_block;
    if (oc_id == (oc_block_num - 1)) {
        oc_block = OC - oc_id * nr_pack_per_step * pack_c;
    }
    megdnn_assert(oc_block % pack_c == 0,
                  "oc must be devisible by 4, but oc = %zu", oc_block);
    const int8_t* sptr =
            static_cast<int8_t*>(bundle.get(0)) +
            workspace_batch_id * GROUP * padding_group_size * src_expand_size +
            workspace_group_id * padding_group_size * src_expand_size;

    const int8_t* fptr =
            kern_param.filter<dt_int8>(group_id) + oc_idx * FH * FW * IC;
    void* dst = reinterpret_cast<void*>(
            reinterpret_cast<ptrdiff_t>(
                    kern_param.dst<void>(batch_id, group_id)) +
            oc_idx * OH * OW);
    const int32_t* bptr =
            kern_param.bias<dt_int32>(batch_id, group_id) + oc_idx;
    auto packed_weight = reinterpret_cast<int8_t*>(bundle.get(1)) +
                         group_id * OC * IC * FH * FW + oc_idx * IC * FH * FW;

    conv_bias::pack_nchw44_weight_for_nchw_conv(fptr, packed_weight, IC, FH, FW,
                                                oc_block);
#define KERN1_NCHW44_CONV(filter)                                             \
    conv_bias::conv_direct_stride2_##filter##x##filter##_int8_nchw_nchw44<    \
            bias_mode, Op>(sptr, packed_weight, bptr, nullptr,                \
                           static_cast<int8_t*>(dst), oc_block, IC, IH2, IW2, \
                           OH, OW, op)
    DISPATCH_FILTER(filter, KERN1_NCHW44_CONV);
#undef KERN1_NCHW44_CONV
}

/* ===================== stride2 algo ===================== */
bool ConvBiasImpl::AlgoS8DirectStride2NCHWNCHW44::usable(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param,
        AlgoSelectionStrategy algo_selection_strategy) const {
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    auto OC = fm.ocpg;
    bool avaible =          //! src and filter are qint8, dst is qint8
            fm.icpg < 4 &&  // must be nchw input
            ((param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
              param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
              (param.dst_type.enumv() == DTypeEnum::QuantizedS8))) &&
            (fm.format == param::Convolution::Format::NCHW44) &&
            (OC % 4 == 0 && OC >= 4) && !fm.should_flip && fm.group == 1 &&
            fm.spatial_ndim == 2 && fm.dilation[0] == 1 &&
            fm.dilation[1] == 1 && fm.stride[0] == 2 && fm.stride[1] == 2 &&
            FH == fm.spatial[1] && (FH == 3 || FH == 5 || FH == 7) &&
            fm.group == 1 && param.bias_mode != BiasMode::BIAS;
    return avaible;
}

bool ConvBiasImpl::AlgoS8DirectStride2NCHWNCHW44::is_preferred(
        megdnn::fallback::ConvBiasImpl* conv_bias_impl_ptr,
        const NCBKernSizeParam& param) const {
    // TODO: benchmark and fix
    return false;
}

size_t ConvBiasImpl::AlgoS8DirectStride2NCHWNCHW44::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    return get_bundle(param).total_size_in_bytes();
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoS8DirectStride2NCHWNCHW44::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    auto fm = param.filter_meta;
    size_t N = param.n;
    size_t OC = fm.ocpg;
    size_t group = fm.group;
    WorkspaceBundle wbundle = get_bundle(param);
    conv_fun do_conv_fun = nullptr;
// NOTE: remain_w is not used to gen hash of midout for compatible with changing
// shape runtime
#define DO_CONV_KERN_FUN(filter, bias_mode, op)                        \
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8_nchw_nchw44_stride2, \
                 midout_iv(#filter #bias_mode #op##_hash)) {           \
        do_conv_fun = do_conv_kern<filter, bias_mode, op>;             \
    }                                                                  \
    MIDOUT_END();

#define GET_OP_PARAM(filter, bias_mode)                                  \
    switch (param.nonlineMode) {                                         \
        case param::ConvBias::NonlineMode::IDENTITY:                     \
            DO_CONV_KERN_FUN(filter, bias_mode,                          \
                             TypeCvtOp<dt_qint32 MEGDNN_COMMA dt_qint8>) \
            break;                                                       \
        case param::ConvBias::NonlineMode::RELU:                         \
            DO_CONV_KERN_FUN(filter, bias_mode,                          \
                             ReluOp<dt_qint32 MEGDNN_COMMA dt_qint8>)    \
            break;                                                       \
        case param::ConvBias::NonlineMode::H_SWISH:                      \
            DO_CONV_KERN_FUN(filter, bias_mode,                          \
                             HSwishOp<dt_qint32 MEGDNN_COMMA dt_qint8>)  \
            break;                                                       \
        default:                                                         \
            megdnn_assert(0);                                            \
            break;                                                       \
    }
#define GET_BIAS_MODE_PARAM(filter)                                \
    switch (param.bias_mode) {                                     \
        case BiasMode::NO_BIAS:                                    \
            GET_OP_PARAM(filter, BiasMode::NO_BIAS)                \
            break;                                                 \
        case BiasMode::BROADCAST_CHANNEL_BIAS:                     \
            GET_OP_PARAM(filter, BiasMode::BROADCAST_CHANNEL_BIAS) \
            break;                                                 \
        default:                                                   \
            megdnn_assert(0);                                      \
            break;                                                 \
    }

#define DISPATCH_CONV_KERN()                \
    switch (param.filter_meta.spatial[0]) { \
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
    WorkspaceBundle bundle = wbundle;

    constexpr size_t pack_oc = 8;
    size_t oc_step = pack_oc;
    auto copy_padding = [bundle](const NCBKernParam& kern_param,
                                 const NCBKernIndex& ncb_index) {
        copy_padding_kern(bundle, kern_param, ncb_index, ncb_index.ndrange_id);
    };
    ret_kerns.push_back({copy_padding, {N, group, fm.icpg}});

    CpuNDRange ncb_range = {N, group, div_ceil(OC, oc_step)};
    auto do_conv = [bundle, do_conv_fun, ncb_range](
                           const NCBKernParam& kern_param,
                           const NCBKernIndex& ncb_index) {
        do_conv_fun(bundle, kern_param, ncb_index, ncb_index.ndrange_id,
                    ncb_range);
    };
    ret_kerns.push_back({do_conv, ncb_range});

    return ret_kerns;
}

// vim: syntax=cpp.doxygen
