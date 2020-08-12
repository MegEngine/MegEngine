/**
 * \file dnn/src/arm_common/conv_bias/int8/direct_nchw44_algo.cpp
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
#include "src/arm_common/conv_bias/int8/direct.h"
#include "src/arm_common/conv_bias/int8/direct_nchw44_kern.h"
#include "src/arm_common/elemwise_op.h"
#include "src/common/opr_delegate.h"

#include "midout.h"

using namespace megdnn;
using namespace arm_common;
using conv_fun = std::function<void(
        const WorkspaceBundle& bundle,
        const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index,
        const CpuNDRange& workspace_ids, const CpuNDRange& ncb_range)>;
MIDOUT_DECL(megdnn_arm_common_conv_bias_int8_nchw44)

static void get_rectified_size(
        const megdnn::fallback::ConvBiasImpl::NCBKernSizeParam& param, int& ih2,
        int& iw2) {
    auto&& fm = param.filter_meta;
    int ih = param.isz[0];
    int iw = param.isz[1];
    int ph = fm.padding[0];
    int pw = fm.padding[1];

    ih2 = ih + ph * 2;
    iw2 = iw + pw * 2;
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
    int IH2, IW2;
    get_rectified_size(param, IH2, IW2);
    if (group == 1) {
        size_t src_size =
                batch * group * IC * IH2 * IW2 * sizeof(int8_t) * src_expand;
        size_t weight_size = group * OC * IC * FH * FW * sizeof(int8_t);
        return {nullptr, {src_size, weight_size}};
    } else {
        size_t src_size =
                param.nr_threads * IC * IH2 * IW2 * sizeof(int8_t) * src_expand;
        size_t weight_size = group * OC * IC * FH * FW * sizeof(int8_t);
        return {nullptr, {src_size, weight_size}};
    }
};

static void copy_padding_kern(const WorkspaceBundle& bundle,
                              const ConvBiasImpl::NCBKernParam& kern_param,
                              const ConvBiasImpl::NCBKernIndex& ncb_index,
                              const CpuNDRange& workspace_ids) {
    int IH = kern_param.isz[0];
    int IW = kern_param.isz[1];
    int IC = kern_param.filter_meta.icpg;
    int PH = kern_param.filter_meta.padding[0];
    int PW = kern_param.filter_meta.padding[1];
    int GROUP = kern_param.filter_meta.group;

    int IH2, IW2;
    get_rectified_size(kern_param, IH2, IW2);
    int padding_group_size = IH2 * IW2 * IC;
    //! Used for get the workspace offset
    constexpr int pack_ic = 4;
    constexpr int expend_element = 4;
    // TODO: block dim is better to get from arg
    size_t workspace_ic_block = 4;
    size_t workspace_batch_id = workspace_ids[0];
    size_t workspace_group_id = workspace_ids[1];
    size_t workspace_ic_id = workspace_ids[2];
    size_t workspace_ic = workspace_ic_id * workspace_ic_block;
    size_t batch_id = ncb_index.ndrange_id[0];
    size_t group_id = ncb_index.ndrange_id[1];
    size_t group_pack_size = 1;

    int nr_pad_w = PW * pack_ic * expend_element;
    int nr_pad_h = PH * IW2 * pack_ic * expend_element;
    int row_last_pad = (IW2 - IW - PW) * pack_ic * expend_element;
    int col_last_pad = (IH2 - IH - PH) * IW2 * pack_ic * expend_element;
    const int8_t* sptr = static_cast<const int8_t*>(kern_param.src<int8_t>(
            batch_id, group_id, workspace_ic_id, group_pack_size, pack_ic));

    //! copy to sptr_base to eliminate padding effect
    int8_t* sptr_base = static_cast<int8_t*>(bundle.get(0)) +
                        (workspace_batch_id * GROUP * padding_group_size +
                         workspace_group_id * padding_group_size +
                         workspace_ic * IH2 * IW2) *
                                expend_element;
    size_t nr_ic = workspace_ic_block;
    if (GROUP > 1) {
        nr_ic = IC;
    }
    rep_step(ic_idx, nr_ic, pack_ic) {
        std::memset(sptr_base, 0, nr_pad_h * sizeof(int8_t));
        sptr_base += nr_pad_h;
        rep(ih_idx, IH) {
            std::memset(sptr_base, 0, nr_pad_w * sizeof(int8_t));
            sptr_base += nr_pad_w;
            int8_direct_nchw44::nchw44_pack_src(sptr, sptr_base, IW);
            sptr_base += IW * pack_ic * expend_element;
            sptr += IW * pack_ic;
            std::memset(sptr_base, 0, row_last_pad * sizeof(int8_t));
            sptr_base += row_last_pad;
        }
        std::memset(sptr_base, 0, col_last_pad * sizeof(int8_t));
        sptr_base += col_last_pad;
    }
}

template <size_t filter, BiasMode bias_mode, typename Op, typename DstType,
          int stride>
static void do_conv_kern(const WorkspaceBundle& bundle,
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
    int IH2, IW2;
    get_rectified_size(kern_param, IH2, IW2);
    bool need_post_process =
            kern_param.dst_type.enumv() == DTypeEnum::QuantizedS8;
    //! if dst_type is qint32, the op is not used, just fill with (1.0f,4.0f)
    Op op(1.f, 4.f);
    if (need_post_process) {
        float scale_bias =
                kern_param.bias_type.param<dtype::QuantizedS32>().scale;
        float scale_dst = kern_param.dst_type.param<dtype::QuantizedS8>().scale;
        op = Op(scale_bias, scale_dst);
    }
    size_t padding_group_size = IH2 * IW2 * IC;

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
    DstType* dst = reinterpret_cast<DstType*>(
            kern_param.dst<void>(batch_id, group_id, oc_idx));
    const int32_t* bptr =
            kern_param.bias<dt_int32>(batch_id, group_id) + oc_idx;
    auto packed_weight = reinterpret_cast<int8_t*>(bundle.get(1)) +
                         group_id * OC * IC * FH * FW + oc_idx * IC * FH * FW;
    int8_direct_nchw44::nchw44_pack_filter(fptr, packed_weight,
                                           oc_block / 4 * IC / 4 * FH * FW);
    int8_direct_nchw44::conv_direct_int8_nchw44<bias_mode, Op, filter, DstType,
                                                stride>(
            sptr, packed_weight, bptr, nullptr, static_cast<DstType*>(dst),
            oc_block, IC, IH2, IW2, OH, OW, op);
}

bool ConvBiasImpl::AlgoS8DirectNCHW44::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy algo_selection_strategy) const {
    MEGDNN_MARK_USED_VAR(algo_selection_strategy);
    auto&& fm = param.filter_meta;
    const int fh = fm.spatial[0];
    const int fw = fm.spatial[1];
    const int oc = fm.ocpg;
    const int ic = fm.icpg;
    const bool avaible =  //! src and filter are qint8, dst is qint8 or qint32
            ((param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
              param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
              (param.dst_type.enumv() == DTypeEnum::QuantizedS8 ||
               param.dst_type.enumv() == DTypeEnum::QuantizedS32))) &&
            (fm.format == param::Convolution::Format::NCHW44) &&
            (oc % 4 == 0 && ic % 4 == 0 && oc >= 4) && !fm.should_flip &&
            fm.spatial_ndim == 2 && fm.dilation[0] == 1 &&
            fm.dilation[1] == 1 && fm.stride[0] == fm.stride[1] &&
            (fm.stride[0] == 2 || fm.stride[0] == 1) && fh == fw &&
            (fh == 2 || fh == 3 || fh == 5 || fh == 7) &&
            param.bias_mode != BiasMode::BIAS;
    return avaible;
}

bool ConvBiasImpl::AlgoS8DirectNCHW44::is_preferred(
                const NCBKernSizeParam& param) const {
    // TODO: benchmark and fix
    MEGDNN_MARK_USED_VAR(param);
    return false;
}

size_t ConvBiasImpl::AlgoS8DirectNCHW44::get_workspace(
         const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8_nchw44,
                 midout_iv("AlgoS8DirectNCHW44::get_workspace"_hash)) {
        return get_bundle(param).total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoS8DirectNCHW44::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    auto fm = param.filter_meta;
    size_t N = param.n;
    size_t IC = fm.icpg;
    size_t OC = fm.ocpg;
    size_t group = fm.group;
    size_t fh = fm.spatial[0];
    size_t fw = fm.spatial[1];
    WorkspaceBundle wbundle = get_bundle(param);
    conv_fun do_conv_fun = nullptr;

    bool need_post_process = param.dst_type.enumv() == DTypeEnum::QuantizedS8;
// NOTE: remain_w is not used to gen hash of midout for compatible with changing
// shape runtime
#define DO_CONV_KERN_FUN(stride, dst_type, filter, bias_mode, op)              \
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8_nchw44,                      \
                 midout_iv(#stride #dst_type #filter #bias_mode #op##_hash)) { \
        do_conv_fun = do_conv_kern<filter, bias_mode, op, dst_type, stride>;   \
    }                                                                          \
    MIDOUT_END();

#define GET_OP_PARAM(stride, filter, bias_mode)                              \
    if (need_post_process) {                                                 \
        switch (param.nonlineMode) {                                         \
            case param::ConvBias::NonlineMode::IDENTITY:                     \
                DO_CONV_KERN_FUN(stride, dt_qint8, filter, bias_mode,        \
                                                                             \
                                 TypeCvtOp<dt_qint32 MEGDNN_COMMA dt_qint8>) \
                break;                                                       \
            case param::ConvBias::NonlineMode::RELU:                         \
                DO_CONV_KERN_FUN(stride, dt_qint8, filter, bias_mode,        \
                                                                             \
                                 ReluOp<dt_qint32 MEGDNN_COMMA dt_qint8>)    \
                break;                                                       \
            case param::ConvBias::NonlineMode::H_SWISH:                      \
                DO_CONV_KERN_FUN(stride, dt_qint8, filter, bias_mode,        \
                                                                             \
                                 HSwishOp<dt_qint32 MEGDNN_COMMA dt_qint8>)  \
                break;                                                       \
            default:                                                         \
                megdnn_assert(0, "no supported noline mode");                \
                break;                                                       \
        }                                                                    \
    } else {                                                                 \
        switch (param.nonlineMode) {                                         \
            case param::ConvBias::NonlineMode::IDENTITY:                     \
                DO_CONV_KERN_FUN(stride, dt_int32, filter, bias_mode,        \
                                 NoneOp<dt_int32>)                           \
                break;                                                       \
            default:                                                         \
                megdnn_assert(                                               \
                        0,                                                   \
                        "only support IDENTITY mode when dst is not qint8"); \
                break;                                                       \
        }                                                                    \
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
            megdnn_throw(ssprintf("Unsupport stride size %u for the first conv",
                                  param.filter_meta.stride[0])
                                 .c_str());
            break;
    }

#undef DO_CONV_KERN_FUN
#undef GET_REMAIN_W_PARAM
#undef GET_OP_PARAM
#undef GET_BIAS_MODE_PARAM
#undef DISPATCH_CONV_KERN

    megdnn_assert(do_conv_fun);

    SmallVector<ConvBiasImpl::NCBKern> ret_kerns;

    constexpr size_t pack_oc = 4;
    size_t oc_step = pack_oc;
    if (fh == 2 && fw == 2 && OC >= 8) {
        oc_step = 8;
    }
    if (group == 1) {
        CpuNDRange ncb_range = {N, group, div_ceil(OC, oc_step)};
        auto copy_padding = [wbundle](const NCBKernParam& kern_param,
                                      const NCBKernIndex& ncb_index) mutable {
            wbundle.set(kern_param.workspace_ptr);
            copy_padding_kern(wbundle, kern_param, ncb_index,
                              ncb_index.ndrange_id);
        };
        constexpr size_t pack_ic = 4;
        ret_kerns.push_back({copy_padding, {N, group, div_ceil(IC, pack_ic)}});
        auto do_conv = [wbundle, do_conv_fun, ncb_range](
                               const NCBKernParam& kern_param,
                               const NCBKernIndex& ncb_index) mutable {
            wbundle.set(kern_param.workspace_ptr);
            do_conv_fun(wbundle, kern_param, ncb_index, ncb_index.ndrange_id,
                        ncb_range);
        };
        ret_kerns.push_back({do_conv, ncb_range});
    } else {
        CpuNDRange ncb_range = {N, group, 1};
        auto do_conv = [wbundle, do_conv_fun, ncb_range](
                               const NCBKernParam& kern_param,
                               const NCBKernIndex& ncb_index) mutable {
            wbundle.set(kern_param.workspace_ptr);
            copy_padding_kern(wbundle, kern_param, ncb_index,
                              {0, ncb_index.thread_id, 0});
            do_conv_fun(wbundle, kern_param, ncb_index,
                        {0, ncb_index.thread_id, 0}, ncb_range);
        };
        ret_kerns.push_back({do_conv, ncb_range});
    }

    return ret_kerns;
}

// vim: syntax=cpp.doxygen
