/**
 * \file dnn/src/arm_common/conv_bias/int8/direct_nchw_nchw44_algo.cpp
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
#include "src/arm_common/conv_bias/int8/direct_nchw_nchw44_kern.h"
#include "src/arm_common/conv_bias/int8/strategy.h"
#include "src/arm_common/elemwise_op.h"
#include "src/common/nchw_nchwxx_valid.h"
#include "src/common/opr_delegate.h"

#include "midout.h"

using namespace megdnn;
using namespace arm_common;
using conv_fun = std::function<void(
        const WorkspaceBundle& bundle,
        const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index,
        const CpuNDRange& workspace_ids, const CpuNDRange& ncb_range)>;
MIDOUT_DECL(megdnn_arm_common_conv_bias_int8_nchw_nchw44)

static void get_rectified_size(
        const megdnn::fallback::ConvBiasImpl::NCBKernSizeParam& param, int& ih2,
        int& iw2, int& oh2, int& ow2) {
    auto&& fm = param.filter_meta;
    int ih = param.isz[0];
    int iw = param.isz[1];
    int oh = param.osz[0];
    int ow = param.osz[1];
    int ph = fm.padding[0];
    int pw = fm.padding[1];
    int stride_h = fm.stride[0];

    oh2 = oh;
    ow2 = ow;
    ih2 = stride_h == 2 ? round_up(ih + 2 * ph, 2) : ih + 2 * ph;
    iw2 = iw + 2 * pw;
}
static inline size_t get_temp_bytes(const int iw, const int pw) {
    //! border_size is used to avoid read illegal memory
    constexpr int cacheline_size = 64;
    constexpr int border_size = 1 * cacheline_size;

    return round_up(iw + pw * 2, cacheline_size) + border_size;
}
static WorkspaceBundle get_bundle(const ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    int group = fm.group;
    int batch = param.n;
    int ic = fm.icpg;
    int oc = fm.ocpg;
    int fh = fm.spatial[0];
    int fw = fm.spatial[1];
    int stride_h = fm.stride[0];
    int iw = param.isz[1];
    int pw = fm.padding[1];
    int ih2, iw2, oh2, ow2;
    const size_t src_expand = stride_h == 2 ? 4 : 16;
    get_rectified_size(param, ih2, iw2, oh2, ow2);
    megdnn_assert(group == 1, "only support group == 1 now");
    size_t src_size =
            batch * group * ic * ih2 * iw2 * sizeof(int8_t) * src_expand;
    size_t weight_size = group * oc * ic * fh * fw * sizeof(int8_t);
    size_t tmp_size = 0;
    if (stride_h == 1) {
        weight_size = group * oc * ic * fh * round_up(fw, 4) * sizeof(int8_t);
        tmp_size = get_temp_bytes(iw, pw);
    }
    return {nullptr, {src_size, weight_size, tmp_size * param.nr_threads}};
};

static void copy_padding_kern(const WorkspaceBundle& bundle,
                              const ConvBiasImpl::NCBKernParam& kern_param,
                              const ConvBiasImpl::NCBKernIndex& ncb_index,
                              const CpuNDRange& workspace_ids) {
    int ih = kern_param.isz[0];
    int iw = kern_param.isz[1];
    int ic = kern_param.filter_meta.icpg;
    int ph = kern_param.filter_meta.padding[0];
    int pw = kern_param.filter_meta.padding[1];
    int group = kern_param.filter_meta.group;
    int stride_h = kern_param.filter_meta.stride[0];

    int ih2, iw2, oh2, ow2;
    get_rectified_size(kern_param, ih2, iw2, oh2, ow2);
    int padding_group_size = ih2 * iw2 * ic;
    //! Used for get the workspace offset
    const int src_expand = stride_h == 2 ? 4 : 16;

    //! TODO: block dim is better to get from arg
    int workspace_ic_block = 1;
    int workspace_batch_id = workspace_ids[0];
    int workspace_group_id = workspace_ids[1];
    int workspace_ic_id = workspace_ids[2];
    int workspace_ic = workspace_ic_id * workspace_ic_block;
    int batch_id = ncb_index.ndrange_id[0];
    int group_id = ncb_index.ndrange_id[1];

    const int8_t* sptr = static_cast<const int8_t*>(
            kern_param.src<int8_t>(batch_id, group_id, workspace_ic_id, 1, 1));
    //! copy to sptr_base to eliminate padding effect
    int8_t* sptr_base = static_cast<int8_t*>(bundle.get(0)) +
                        (workspace_batch_id * group * padding_group_size +
                         workspace_group_id * padding_group_size +
                         workspace_ic * ih2 * iw2) *
                                src_expand;
    if (stride_h == 1) {
        const size_t tmp_size = get_temp_bytes(iw, pw);
        int8_t* tmp_ptr = reinterpret_cast<int8_t*>(bundle.get(2)) +
                          ncb_index.thread_id * tmp_size;
        int8_direct_nchw_nchw44::pack_nchw_src_for_nchw44_conv<1>(
                sptr, sptr_base, 1, ph, ph, pw, pw, ih, iw, iw2, pw, tmp_ptr);
    } else {
        int8_direct_nchw_nchw44::pack_nchw_src_for_nchw44_conv<2>(
                sptr, sptr_base, 1, ph, ph, pw, pw, ih, iw, iw2, pw, nullptr);
    }
}
static void pack_weight(const WorkspaceBundle& bundle,
                        const ConvBiasImpl::NCBKernParam& kern_param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) {
    const int group_id = ncb_index.ndrange_id[0];
    int fh = kern_param.filter_meta.spatial[0];
    int fw = kern_param.filter_meta.spatial[1];
    int oc = kern_param.filter_meta.ocpg;
    int ic = kern_param.filter_meta.icpg;
    int stride_h = kern_param.filter_meta.stride[0];
    int fw2 = stride_h == 2 ? fw : round_up(fw, 4);
    int oc_block = oc;
    int oc_idx = 0;
    const int8_t* fptr =
            kern_param.filter<dt_int8>(group_id) + oc_idx * fh * fw * ic;
    auto packed_weight = reinterpret_cast<int8_t*>(bundle.get(1)) +
                         group_id * oc * ic * fh * fw2 + oc_idx * ic * fh * fw2;

    if (stride_h == 1) {
        int8_direct_nchw_nchw44::pack_nchw44_weight_for_nchw_conv<1>(
                fptr, packed_weight, ic, fh, fw, oc_block);
    } else {
        int8_direct_nchw_nchw44::pack_nchw44_weight_for_nchw_conv<2>(
                fptr, packed_weight, ic, fh, fw, oc_block);
    }
}
template <size_t filter, BiasMode bias_mode, typename Op, int stride>
static void do_conv_kern(const WorkspaceBundle& bundle,
                         const ConvBiasImpl::NCBKernParam& kern_param,
                         const ConvBiasImpl::NCBKernIndex& ncb_index,
                         const CpuNDRange& workspace_ids,
                         const CpuNDRange& ncb_range) {
    int oh = kern_param.osz[0];
    int ow = kern_param.osz[1];
    int fh = kern_param.filter_meta.spatial[0];
    int fw = kern_param.filter_meta.spatial[1];
    int fw2 = stride == 2 ? fw : round_up(fw, 4);
    int ic = kern_param.filter_meta.icpg;
    int oc = kern_param.filter_meta.ocpg;
    int group = kern_param.filter_meta.group;
    int ih2, iw2, oh2, ow2;
    get_rectified_size(kern_param, ih2, iw2, oh2, ow2);
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
    int padding_group_size = ih2 * iw2 * ic;

    constexpr int pack_c = 4;
    constexpr int src_expand_size = stride == 2 ? 4 : 16;
    const int workspace_batch_id = workspace_ids[0];
    const int workspace_group_id = workspace_ids[1];
    const int batch_id = ncb_index.ndrange_id[0];
    const int group_id = ncb_index.ndrange_id[1];
    const int oc_id = ncb_index.ndrange_id[2];
    const int oc_block_num = ncb_range[2];
    int nr_pack_per_step = div_ceil(div_ceil(oc, pack_c), oc_block_num);
    int oc_block = nr_pack_per_step * pack_c;
    const int oc_idx = oc_id * oc_block;
    if (oc_id == (oc_block_num - 1)) {
        oc_block = oc - oc_id * nr_pack_per_step * pack_c;
    }
    megdnn_assert(oc_block % pack_c == 0,
                  "oc must be devisible by 4, but oc = %d", oc_block);
    const int8_t* sptr =
            static_cast<int8_t*>(bundle.get(0)) +
            workspace_batch_id * group * padding_group_size * src_expand_size +
            workspace_group_id * padding_group_size * src_expand_size;

    int8_t* dst = reinterpret_cast<int8_t*>(
            reinterpret_cast<ptrdiff_t>(
                    kern_param.dst<void>(batch_id, group_id)) +
            oc_idx * oh * ow);

    const int32_t* bptr =
            kern_param.bias<dt_int32>(batch_id, group_id) + oc_idx;
    int8_t* packed_weight = reinterpret_cast<int8_t*>(bundle.get(1)) +
                            group_id * oc * ic * fh * fw2 +
                            oc_idx * ic * fh * fw2;
    int8_direct_nchw_nchw44::conv_direct_int8_nchw_nchw44<bias_mode, Op, filter,
                                                          stride>(
            sptr, packed_weight, bptr, nullptr, dst, oc_block, ic, ih2, iw2, oh,
            ow, op);
}

bool ConvBiasImpl::AlgoS8DirectNCHWNCHW44::usable(const NCBKernSizeParam& param,
                                                  AlgoSelectionStrategy) const {
    return nchw_nchwxx_valid<NchwNchwxxType::NCHW44_INT8>(
            param.src_type.enumv(), param.filter_type.enumv(),
            param.dst_type.enumv(), param.filter_meta, param.bias_mode,
            param.nonlineMode);
}

bool ConvBiasImpl::AlgoS8DirectNCHWNCHW44::is_preferred(
        const NCBKernSizeParam& param) const {
    // TODO: benchmark and fix
    MEGDNN_MARK_USED_VAR(param);
    return false;
}

size_t ConvBiasImpl::AlgoS8DirectNCHWNCHW44::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8_nchw_nchw44,
                 midout_iv("AlgoS8DirectNCHWNCHW44::get_workspace"_hash)) {
        return get_bundle(param).total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoS8DirectNCHWNCHW44::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    auto fm = param.filter_meta;
    size_t N = param.n;
    size_t OC = fm.ocpg;
    size_t group = fm.group;
    WorkspaceBundle bundle = get_bundle(param);
    conv_fun do_conv_fun = nullptr;
// NOTE: remain_w is not used to gen hash of midout for compatible with changing
// shape runtime
#define DO_CONV_KERN_FUN(stride, filter, bias_mode, op)              \
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8_nchw_nchw44,       \
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

    constexpr size_t pack_oc = 8;
    size_t oc_step = pack_oc;
    auto copy_padding = [bundle](const NCBKernParam& kern_param,
                                 const NCBKernIndex& ncb_index) mutable {
        bundle.set(kern_param.workspace_ptr);
        copy_padding_kern(bundle, kern_param, ncb_index, ncb_index.ndrange_id);
    };
    ret_kerns.push_back({copy_padding, {N, group, fm.icpg}});

    auto do_pack_weight = [bundle](const NCBKernParam& kern_param,
                                   const NCBKernIndex& ncb_index) mutable {
        bundle.set(kern_param.workspace_ptr);
        pack_weight(bundle, kern_param, ncb_index);
    };
    ret_kerns.push_back({do_pack_weight, {static_cast<size_t>(group)}});

    CpuNDRange ncb_range = {N, group, div_ceil(OC, oc_step)};
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

// vim: syntax=cpp.doxygen
