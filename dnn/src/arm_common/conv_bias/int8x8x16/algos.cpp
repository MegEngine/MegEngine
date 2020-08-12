/**
 * \file dnn/src/arm_common/conv_bias/int8x8x16/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/int8x8x16/algos.h"
#include "src/arm_common/conv_bias/int8x8x16/channel_wise_nchw44.h"
#include "src/arm_common/conv_bias/int8x8x16/channel_wise_nchw44_8x8x16.h"
#include "src/arm_common/conv_bias/int8x8x16/conv_direct.h"
#include "src/arm_common/conv_bias/int8x8x16/conv_stride2.h"

#include "midout.h"

MIDOUT_DECL(megdnn_arm_common_conv_bias_int8816_kimpl)


using namespace megdnn;
using namespace arm_common;

namespace {
bool need_dst_copy_str1(
        const megdnn::fallback::ConvolutionImpl::NCBKernSizeParam& param) {
    if (param.osz[0] % 1 != 0 || param.osz[1] % 8 != 0)
        return true;
    return false;
}
bool need_src_copy_str1(
        const megdnn::fallback::ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;

    if (fm.padding[0] != 0 || fm.padding[1] != 0)
        return true;

    return need_dst_copy_str1(param);
}
void get_rectified_size_str1(size_t IH, size_t IW, size_t OH, size_t OW,
                             size_t PH, size_t PW, size_t& IH2, size_t& IW2,
                             size_t& OH2, size_t& OW2) {
    OH2 = OH;
    OW2 = (OW + 7) & ~7;
    IH2 = OH2 + (IH - OH) + 2 * PH;
    IW2 = OW2 + (IW - OW) + 2 * PW;
}
bool need_dst_copy_str2(
        const megdnn::fallback::ConvBiasImpl::NCBKernSizeParam& param) {
    // If the size of output is not multiples of 8, we need to copy it.
    if (param.osz[0] % 8 != 0 || param.osz[1] % 8 != 0)
        return true;
    return false;
}
bool need_src_copy_str2(
        const megdnn::fallback::ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    // If padding is not zero, we need to copy to eliminate padding effect.
    if (fm.padding[0] != 0 || fm.padding[1] != 0)
        return true;

    return need_dst_copy_str2(param);
}
void get_rectified_size_str2(size_t IH, size_t IW, size_t OH, size_t OW,
                             size_t FH, size_t FW, size_t PH, size_t PW,
                             size_t& IH2, size_t& IW2, size_t& OH2,
                             size_t& OW2) {
    MEGDNN_MARK_USED_VAR(PH);
    MEGDNN_MARK_USED_VAR(PW);
    OH2 = (OH + 7) & ~7;
    OW2 = (OW + 7) & ~7;
    IH2 = 2 * OH2 + FH - 2;
    IW2 = 2 * OW2 + FW - 2;
    // Because stride is 2, sometimes IH/W == IH/W2 + 1
    // Do a max update to handle this case.
    IH2 = std::max(IH2, IH);
    IW2 = std::max(IW2, IW);
}
}  // namespace

/* ===================== direct algo ===================== */
bool ConvBiasImpl::AlgoI8x8x16Direct::usable(const NCBKernSizeParam& param,
                                             AlgoSelectionStrategy) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8816_kimpl,
                 midout_iv("AlgoI8x8x16Direct::usable"_hash)) {
        auto&& fm = param.filter_meta;
        auto FH = fm.spatial[0];
        return param.bias_mode == BiasMode::NO_BIAS &&
               param.nonlineMode == NonlineMode::IDENTITY &&
               fm.format == param::ConvBias::Format::NCHW && !fm.should_flip &&
               param.src_type.enumv() == DTypeEnum::Int8 &&
               param.filter_type.enumv() == DTypeEnum::Int8 &&
               param.dst_type.enumv() == DTypeEnum::Int16 &&
               fm.spatial_ndim == 2 && fm.dilation[0] == 1 &&
               fm.dilation[1] == 1 && fm.stride[0] == 1 && fm.stride[1] == 1 &&
               FH == fm.spatial[1] && (FH == 2 || FH == 3 || FH == 5);
    }
    MIDOUT_END();
    return false;
}
WorkspaceBundle ConvBiasImpl::AlgoI8x8x16Direct::get_bundle(
        const NCBKernSizeParam& param) const {
    auto&& fm = param.filter_meta;
    size_t nr_threads = param.nr_threads;
    size_t group = fm.group, batch = param.n;
    auto IC = fm.icpg, IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto PH = fm.padding[0], PW = fm.padding[1];
    size_t OH2, OW2, IH2, IW2;
    bool large_group = group >= param.nr_threads;
    get_rectified_size_str1(IH, IW, OH, OW, PH, PW, IH2, IW2, OH2, OW2);
    size_t part0 = 0u, part1 = 0u;
    if (need_src_copy_str1(param)) {
        part0 = large_group ? IC * IH2 * IW2 * sizeof(int8_t) * nr_threads
                            : IC * IH2 * IW2 * sizeof(int8_t) * group * batch;
    }
    if (need_dst_copy_str1(param)) {
        part1 = OH2 * OW2 * sizeof(int16_t) * nr_threads + 16;
    }
    return {nullptr, {part0, part1}};
}
size_t ConvBiasImpl::AlgoI8x8x16Direct::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8816_kimpl,
                 midout_iv("AlgoI8x8x16Direct::get_workspace"_hash)) {
        auto bundle = get_bundle(param);
        return bundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}
//! Process one input channel copy padding
void ConvBiasImpl::AlgoI8x8x16Direct::copy_padding_kern(
        const WorkspaceBundle& bundle,
        const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index,
        const CpuNDRange& workspace_ids) {
    size_t IH = kern_param.isz[0];
    size_t IW = kern_param.isz[1];
    size_t IC = kern_param.filter_meta.icpg;
    size_t OH = kern_param.osz[0];
    size_t OW = kern_param.osz[1];
    size_t PH = kern_param.filter_meta.padding[0];
    size_t PW = kern_param.filter_meta.padding[1];
    size_t GROUP = kern_param.filter_meta.group;
    size_t OH2, OW2, IH2, IW2;
    get_rectified_size_str1(IH, IW, OH, OW, PH, PW, IH2, IW2, OH2, OW2);
    bool need_src_copy_var = need_src_copy_str1(kern_param);
    size_t padding_group_size = IH2 * IW2 * IC;

    //! Used for get the workspace offset
    size_t workspace_group_id = workspace_ids[0],
           workspace_batch_id = workspace_ids[1],
           channel_id = workspace_ids[2];
    size_t group_id = ncb_index.ndrange_id[0],
           batch_id = ncb_index.ndrange_id[1];
    const int8_t* sptr = kern_param.src<int8_t>(batch_id, group_id, channel_id);
    if (need_src_copy_var) {
        //! copy to sptr_base to eliminate padding effect
        int8_t* sptr_base = static_cast<int8_t*>(bundle.get(0)) +
                            workspace_group_id * padding_group_size +
                            workspace_batch_id * GROUP * padding_group_size +
                            channel_id * IH2 * IW2;
        std::memset(sptr_base, 0, sizeof(int8_t) * IH2 * IW2);
        rep(ih, IH) {
            std::memcpy(sptr_base + (ih + PH) * IW2 + PW, sptr + ih * IW,
                        sizeof(int8_t) * IW);
        }
    }
};
//! compute one output channel
void ConvBiasImpl::AlgoI8x8x16Direct::do_conv_kern(
        const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
        const NCBKernIndex& ncb_index, const CpuNDRange& workspace_ids) {
    size_t OH = kern_param.osz[0];
    size_t OW = kern_param.osz[1];
    size_t IH = kern_param.isz[0];
    size_t IW = kern_param.isz[1];
    size_t FH = kern_param.filter_meta.spatial[0];
    size_t FW = kern_param.filter_meta.spatial[1];
    size_t IC = kern_param.filter_meta.icpg;
    size_t PH = kern_param.filter_meta.padding[0];
    size_t PW = kern_param.filter_meta.padding[1];
    size_t GROUP = kern_param.filter_meta.group;
    size_t OH2, OW2, IH2, IW2;
    get_rectified_size_str1(IH, IW, OH, OW, PH, PW, IH2, IW2, OH2, OW2);
    bool need_src_copy_var = need_src_copy_str1(kern_param);
    bool need_dst_copy_var = need_dst_copy_str1(kern_param);
    size_t padding_group_size = IH2 * IW2 * IC;
    //! Choose the compute kernel
    using Func =
            std::function<void(const int8_t*, const int8_t*, int16_t*, size_t,
                               size_t, size_t, size_t, size_t, size_t)>;
    Func fun_not_add_to_dst = nullptr, fun_add_to_dst = nullptr;
    if (FH == 2) {
        fun_not_add_to_dst =
                conv_bias::conv_direct_2x2_sc_int8_int8_int16<false>;
        fun_add_to_dst = conv_bias::conv_direct_2x2_sc_int8_int8_int16<true>;
    } else if (FH == 3) {
        fun_not_add_to_dst =
                conv_bias::conv_direct_3x3_sc_int8_int8_int16<false>;
        fun_add_to_dst = conv_bias::conv_direct_3x3_sc_int8_int8_int16<true>;
    } else if (FH == 5) {
        fun_not_add_to_dst =
                conv_bias::conv_direct_5x5_sc_int8_int8_int16<false>;
        fun_add_to_dst = conv_bias::conv_direct_5x5_sc_int8_int8_int16<true>;
    }

    //! Used for get the workspace offset
    size_t workspace_group_id = workspace_ids[0],
           workspace_batch_id = workspace_ids[1], oc = workspace_ids[2];

    size_t group_id = ncb_index.ndrange_id[0],
           batch_id = ncb_index.ndrange_id[1];

    const int8_t* sptr = kern_param.src<dt_int8>(batch_id, group_id);
    const int8_t* filter =
            kern_param.filter<dt_int8>(group_id) + oc * FH * FW * IC;
    int16_t* dst = kern_param.dst<dt_int16>(batch_id, group_id, oc);
    if (need_src_copy_var) {
        sptr = static_cast<int8_t*>(bundle.get(0)) +
               workspace_group_id * padding_group_size +
               workspace_batch_id * GROUP * padding_group_size;
    }
    int16_t* dptr = nullptr;
    if (need_dst_copy_var) {
        dptr = static_cast<int16_t*>(bundle.get(1)) +
               ncb_index.thread_id * OH2 * OW2;
    } else {
        dptr = dst;
    }
    fun_not_add_to_dst(sptr, filter, dptr, IH2, IW2, OH2, OW2, 0, 0);
    for (size_t ic = 1; ic < IC; ++ic) {
        fun_add_to_dst(sptr + ic * IH2 * IW2, filter + ic * FH * FW, dptr, IH2,
                       IW2, OH2, OW2, 0, 0);
    }
    if (need_dst_copy_var) {
        rep(oh, OH) {
            std::memcpy(dst + oh * OW, dptr + oh * OW2, sizeof(int16_t) * OW);
        }
    }
}
SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoI8x8x16Direct::get_kimpls(
        const NCBKernSizeParam& param) const {
    auto fm = param.filter_meta;
    size_t N = param.n;
    size_t IC = param.filter_meta.icpg;
    size_t OC = param.filter_meta.ocpg;
    size_t group = fm.group;
    bool large_group = group >= param.nr_threads;
    WorkspaceBundle bundle = get_bundle(param);
    SmallVector<NCBKern> ret_kerns;
    if (large_group) {
        auto exec_one_group = [bundle](const NCBKernParam& kern_param,
                                        const NCBKernIndex& ncb_index) mutable {
            auto fm = kern_param.filter_meta;
            size_t IC = fm.icpg;
            size_t OC = fm.ocpg;
            bundle.set(kern_param.workspace_ptr);
            for (size_t ic = 0; ic < IC; ic++) {
                copy_padding_kern(bundle, kern_param, ncb_index,
                                  {ncb_index.thread_id, 0, ic});
            }
            for (size_t oc = 0; oc < OC; oc++) {
                do_conv_kern(bundle, kern_param, ncb_index,
                             {ncb_index.thread_id, 0, oc});
            }
        };
        ret_kerns.push_back({exec_one_group, {group, N, 1_z}});
    } else {
        auto copy_padding = [bundle](const NCBKernParam& kern_param,
                                     const NCBKernIndex& ncb_index) mutable {
            bundle.set(kern_param.workspace_ptr);
            copy_padding_kern(bundle, kern_param, ncb_index,
                              ncb_index.ndrange_id);
        };
        ret_kerns.push_back({copy_padding, {group, N, IC}});
        auto do_conv = [bundle](const NCBKernParam& kern_param,
                                const NCBKernIndex& ncb_index) mutable {
            bundle.set(kern_param.workspace_ptr);
            do_conv_kern(bundle, kern_param, ncb_index, ncb_index.ndrange_id);
        };
        ret_kerns.push_back({do_conv, {group, N, OC}});
    }
    return ret_kerns;
}
SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoI8x8x16Direct::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8816_kimpl,
                 midout_iv("AlgoI8x8x16Direct::dispatch_kerns"_hash)) {
        return get_kimpls(param);
    }
    MIDOUT_END();
    return {};
}

/* ===================== stride-2 algo ===================== */
bool ConvBiasImpl::AlgoI8x8x16Stride2::usable(const NCBKernSizeParam& param,
                                              AlgoSelectionStrategy) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8816_kimpl,
                 midout_iv("AlgoI8x8x16Stride2::usable"_hash)) {
        auto&& fm = param.filter_meta;
        auto FH = fm.spatial[0];
        return param.bias_mode == BiasMode::NO_BIAS &&
               param.nonlineMode == NonlineMode::IDENTITY &&
               fm.format == param::ConvBias::Format::NCHW && !fm.should_flip &&
               param.src_type.enumv() == DTypeEnum::Int8 &&
               param.filter_type.enumv() == DTypeEnum::Int8 &&
               param.dst_type.enumv() == DTypeEnum::Int16 &&
               fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
               fm.stride[0] == 2 && fm.stride[1] == 2 && FH == fm.spatial[1] &&
               (FH == 2 || FH == 3 || FH == 5);
    }
    MIDOUT_END();
    return false;
}
WorkspaceBundle ConvBiasImpl::AlgoI8x8x16Stride2::get_bundle(
        const NCBKernSizeParam& param) const {
    auto&& fm = param.filter_meta;
    size_t nr_threads = param.nr_threads;
    size_t group = fm.group, batch = param.n;
    auto IC = fm.icpg, IH = param.isz[0], IW = param.isz[1];
    auto OH = param.osz[0], OW = param.osz[1];
    auto PH = fm.padding[0], PW = fm.padding[1];
    auto FH = fm.spatial[0], FW = fm.spatial[1];
    size_t OH2, OW2, IH2, IW2;
    get_rectified_size_str2(IH, IW, OH, OW, FH, FW, PH, PW, IH2, IW2, OH2, OW2);
    size_t part0 = 0u, part1 = 0u;
    bool large_group = group >= param.nr_threads;
    if (need_src_copy_str2(param)) {
        part0 = large_group ? IC * IH2 * IW2 * sizeof(int8_t) * nr_threads
                            : IC * IH2 * IW2 * sizeof(int8_t) * group * batch;
    }
    if (need_dst_copy_str2(param)) {
        part1 = OH2 * OW2 * sizeof(int16_t) * nr_threads + 16;
    }
    return {nullptr, {part0, part1}};
}
size_t ConvBiasImpl::AlgoI8x8x16Stride2::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8816_kimpl,
                 midout_iv("AlgoI8x8x16Stride2::get_workspace"_hash)) {
        auto bundle = get_bundle(param);
        return bundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}
//! Process one input channel copy padding
void ConvBiasImpl::AlgoI8x8x16Stride2::copy_padding_kern(
        const WorkspaceBundle& bundle,
        const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index,
        const CpuNDRange& workspace_ids) {
    size_t IH = kern_param.isz[0];
    size_t IW = kern_param.isz[1];
    size_t IC = kern_param.filter_meta.icpg;
    size_t OH = kern_param.osz[0];
    size_t OW = kern_param.osz[1];
    size_t PH = kern_param.filter_meta.padding[0];
    size_t PW = kern_param.filter_meta.padding[1];
    auto FH = kern_param.filter_meta.spatial[0],
         FW = kern_param.filter_meta.spatial[1];
    size_t GROUP = kern_param.filter_meta.group;
    size_t IH2, IW2, OH2, OW2;
    get_rectified_size_str2(IH, IW, OH, OW, FH, FW, PH, PW, IH2, IW2, OH2, OW2);
    bool need_src_copy_var = need_src_copy_str2(kern_param);
    size_t padding_group_size = IH2 * IW2 * IC;

    size_t workspace_group_id = workspace_ids[0],
           workspace_batch_id = workspace_ids[1],
           channel_id = workspace_ids[2];
    size_t group_id = ncb_index.ndrange_id[0],
           batch_id = ncb_index.ndrange_id[1];
    const int8_t* sptr = kern_param.src<int8_t>(batch_id, group_id, channel_id);
    if (need_src_copy_var) {
        //! copy to sptr_base to eliminate padding effect
        int8_t* sptr_base = static_cast<int8_t*>(bundle.get(0)) +
                            workspace_group_id * padding_group_size +
                            workspace_batch_id * GROUP * padding_group_size +
                            channel_id * IH2 * IW2;
        std::memset(sptr_base, 0, sizeof(int8_t) * IH2 * IW2);
        rep(ih, IH) {
            std::memcpy(sptr_base + (ih + PH) * IW2 + PW, sptr + ih * IW,
                        sizeof(int8_t) * IW);
        }
    }
};
//! compute one output channel
void ConvBiasImpl::AlgoI8x8x16Stride2::do_conv_kern(
        const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
        const NCBKernIndex& ncb_index, const CpuNDRange& workspace_ids) {
    size_t OH = kern_param.osz[0];
    size_t OW = kern_param.osz[1];
    size_t IH = kern_param.isz[0];
    size_t IW = kern_param.isz[1];
    size_t FH = kern_param.filter_meta.spatial[0];
    size_t FW = kern_param.filter_meta.spatial[1];
    size_t IC = kern_param.filter_meta.icpg;
    size_t PH = kern_param.filter_meta.padding[0];
    size_t PW = kern_param.filter_meta.padding[1];
    size_t GROUP = kern_param.filter_meta.group;
    size_t IH2, IW2, OH2, OW2;
    get_rectified_size_str2(IH, IW, OH, OW, FH, FW, PH, PW, IH2, IW2, OH2, OW2);
    bool need_src_copy_var = need_src_copy_str2(kern_param);
    bool need_dst_copy_var = need_dst_copy_str2(kern_param);
    size_t padding_group_size = IH2 * IW2 * IC;
    //! Choose the compute kernel
    using Func =
            std::function<void(const int8_t*, const int8_t*, int16_t*, size_t,
                               size_t, size_t, size_t, size_t, size_t)>;
    Func fun_not_add_to_dst = nullptr, fun_add_to_dst = nullptr;
    if (FH == 2) {
        fun_not_add_to_dst =
                conv_bias::conv_stride2_2x2_sc_int8_int8_int16<false>;
        fun_add_to_dst = conv_bias::conv_stride2_2x2_sc_int8_int8_int16<true>;
    } else if (FH == 3) {
        fun_not_add_to_dst =
                conv_bias::conv_stride2_3x3_sc_int8_int8_int16<false>;
        fun_add_to_dst = conv_bias::conv_stride2_3x3_sc_int8_int8_int16<true>;
    } else if (FH == 5) {
        fun_not_add_to_dst =
                conv_bias::conv_stride2_5x5_sc_int8_int8_int16<false>;
        fun_add_to_dst = conv_bias::conv_stride2_5x5_sc_int8_int8_int16<true>;
    }

    //! Used for get the workspace offset
    size_t workspace_group_id = workspace_ids[0],
           workspace_batch_id = workspace_ids[1], oc = workspace_ids[2];
    size_t group_id = ncb_index.ndrange_id[0],
           batch_id = ncb_index.ndrange_id[1];
    const int8_t* sptr = kern_param.src<dt_int8>(batch_id, group_id);
    const int8_t* filter =
            kern_param.filter<dt_int8>(group_id) + oc * FH * FW * IC;
    int16_t* dst = kern_param.dst<dt_int16>(batch_id, group_id, oc);
    if (need_src_copy_var) {
        sptr = static_cast<int8_t*>(bundle.get(0)) +
               workspace_group_id * padding_group_size +
               workspace_batch_id * GROUP * padding_group_size;
    }
    int16_t* dptr = nullptr;
    if (need_dst_copy_var) {
        dptr = static_cast<int16_t*>(bundle.get(1)) +
               ncb_index.thread_id * OH2 * OW2;
    } else {
        dptr = dst;
    }
    fun_not_add_to_dst(sptr, filter, dptr, IH2, IW2, OH2, OW2, 0, 0);
    for (size_t ic = 1; ic < IC; ++ic) {
        fun_add_to_dst(sptr + ic * IH2 * IW2, filter + ic * FH * FW, dptr, IH2,
                       IW2, OH2, OW2, 0, 0);
    }
    if (need_dst_copy_var) {
        rep(oh, OH) {
            std::memcpy(dst + oh * OW, dptr + oh * OW2, sizeof(int16_t) * OW);
        }
    }
}
SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoI8x8x16Stride2::get_kimpls(
        const NCBKernSizeParam& param) const {
    auto fm = param.filter_meta;
    size_t N = param.n;
    size_t IC = param.filter_meta.icpg;
    size_t OC = param.filter_meta.ocpg;
    size_t group = fm.group;
    bool large_group = group >= param.nr_threads;
    WorkspaceBundle bundle = get_bundle(param);
    SmallVector<NCBKern> ret_kerns;
    if (large_group) {
        auto exec_one_group = [bundle](const NCBKernParam& kern_param,
                                        const NCBKernIndex& ncb_index) mutable {
            auto fm = kern_param.filter_meta;
            size_t IC = fm.icpg;
            size_t OC = fm.ocpg;
            bundle.set(kern_param.workspace_ptr);
            for (size_t ic = 0; ic < IC; ic++) {
                copy_padding_kern(bundle, kern_param, ncb_index,
                                  {ncb_index.thread_id, 0, ic});
            }
            for (size_t oc = 0; oc < OC; oc++) {
                do_conv_kern(bundle, kern_param, ncb_index,
                             {ncb_index.thread_id, 0, oc});
            }
        };
        ret_kerns.push_back({exec_one_group, {group, N, 1_z}});
    } else {
        auto copy_padding = [bundle](const NCBKernParam& kern_param,
                                     const NCBKernIndex& ncb_index) mutable {
            bundle.set(kern_param.workspace_ptr);
            copy_padding_kern(bundle, kern_param, ncb_index,
                              ncb_index.ndrange_id);
        };
        ret_kerns.push_back({copy_padding, {group, N, IC}});
        auto do_conv = [bundle](const NCBKernParam& kern_param,
                                const NCBKernIndex& ncb_index) mutable {
            bundle.set(kern_param.workspace_ptr);
            do_conv_kern(bundle, kern_param, ncb_index, ncb_index.ndrange_id);
        };
        ret_kerns.push_back({do_conv, {group, N, OC}});
    }
    return ret_kerns;
}
SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoI8x8x16Stride2::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8816_kimpl,
                 midout_iv("AlgoI8x8x16Stride2::dispatch_kerns"_hash)) {
        return get_kimpls(param);
    }
    MIDOUT_END();
    return {};
}
bool ConvBiasImpl::AlgoI8x8x16Stride2Filter2::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8816_kimpl,
                 midout_iv("AlgoI8x8x16Stride2Filter2::usable"_hash)) {
        return param.bias_mode == BiasMode::NO_BIAS &&
               param.nonlineMode == NonlineMode::IDENTITY &&
               param.nr_threads == 1_z &&
               conv_bias::can_conv_int8x8x16_stride2_flt2(param);
    }
    MIDOUT_END();
    return false;
}

size_t ConvBiasImpl::AlgoI8x8x16Stride2Filter2::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8816_kimpl,
                 midout_iv("AlgoI8x8x16Stride2Filter2::get_workspace"_hash)) {
        return conv_bias::get_workspace_in_bytes_conv_int8x8x16_stride2_flt2(
                param);
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoI8x8x16Stride2Filter2::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    // return {conv_bias::conv_int8x8x16_stride2_flt2,true};
    auto kern = [](const NCBKernParam& param, const NCBKernIndex& ncb_index) {
        MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8816_kimpl,
                 midout_iv("AlgoI8x8x16Stride2Filter2::dispatch_kerns"_hash)) {
            auto ncb_param = param;
            ncb_param.src_ptr = param.src<void>(0, ncb_index.ndrange_id[0]);
            ncb_param.dst_ptr = param.dst<void>(0, ncb_index.ndrange_id[0]);
            ncb_param.filter_ptr = param.filter<void>(ncb_index.ndrange_id[0]);
            ncb_param.bias_ptr = param.bias<void>(0, ncb_index.ndrange_id[0]);
            conv_bias::conv_int8x8x16_stride2_flt2(ncb_param);
        }
        MIDOUT_END();
    };
    size_t group = param.filter_meta.group;
    return {{kern, {group, 1_z, 1_z}}};
}

/* =====================8int8x8x16 channel_wise_nchw44  stride1 stride2 algo ===================== */
bool ConvBiasImpl::AlgoS8x8x16ChanWiseStride1Stride2NCHW44::usable(
        const NCBKernSizeParam& param, AlgoSelectionStrategy) const {
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    bool avaible =
            //! src and filter are int8, dst is int16
            (param.src_type.enumv() == DTypeEnum::Int8 &&
             param.filter_type.enumv() == DTypeEnum::Int8 &&
             param.dst_type.enumv() == DTypeEnum::Int16) &&
            fm.format == param::Convolution::Format::NCHW44 &&
            param.bias_mode != megdnn::BiasMode::BIAS &&
            param.nonlineMode == megdnn::NonlineMode::IDENTITY &&
            !fm.should_flip && fm.spatial_ndim == 2 && fm.dilation[0] == 1 &&
            fm.dilation[1] == 1 &&
            (fm.stride[0] == fm.stride[1] &&
             (fm.stride[0] == 1 || fm.stride[0] == 2)) &&
            FH == fm.spatial[1] && (FH == 2 || FH == 3 || FH == 5) &&
            fm.icpg == 1 && fm.ocpg == 1 && fm.group % 4 == 0;
    return avaible;
}

size_t ConvBiasImpl::AlgoS8x8x16ChanWiseStride1Stride2NCHW44::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(
            megdnn_arm_common_conv_bias_int8816_kimpl,
            midout_iv(
                    "AlgoS8x8x16ChanWiseStride1Stride2NCHW44::get_workspace"_hash)) {
        size_t stride_h = param.filter_meta.stride[0];
        size_t stride_w = param.filter_meta.stride[1];
        megdnn_assert(stride_h == stride_w);
        if (stride_h == 1) {
            return channel_wise_nchw44_8x8x16::stride1::get_bundle(param)
                    .total_size_in_bytes();
        } else if (stride_h == 2) {
            return channel_wise_nchw44_8x8x16::stride2::get_bundle(param)
                    .total_size_in_bytes();
        } else {
            return 0;
        }
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoS8x8x16ChanWiseStride1Stride2NCHW44::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    size_t stride_h = param.filter_meta.stride[0];
    size_t stride_w = param.filter_meta.stride[1];
    if (stride_h == stride_w && stride_h == 1) {
        MIDOUT_BEGIN(
                megdnn_arm_common_conv_bias_int8816_kimpl,
                midout_iv(
                        "AlgoS8x8x16ChanWiseStride1Stride2NCHW44_dispatch_kerns"_hash)) {
            return channel_wise_nchw44_8x8x16::stride1::get_kimpls(param);
        }
        MIDOUT_END();
        return {};
    } else if (stride_h == stride_w && stride_h == 2) {
        MIDOUT_BEGIN(
                megdnn_arm_common_conv_bias_int8816_kimpl,
                midout_iv(
                        "AlgoS8x8x16ChanWiseStride2NCHW44_dispatch_kerns"_hash)) {
            return channel_wise_nchw44_8x8x16::stride2::get_kimpls(param);
        }
        MIDOUT_END();
        return {};
    } else {
        return {};
    }
}

// vim: syntax=cpp.doxygen
