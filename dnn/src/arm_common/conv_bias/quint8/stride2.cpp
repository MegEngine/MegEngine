/**
 * \file dnn/src/arm_common/conv_bias/quint8/stride2.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/quint8/stride2.h"
#include "megdnn/oprs.h"
#include "src/arm_common/conv_bias/quint8/direct.h"
#include "src/arm_common/elemwise_op.h"
#include "src/common/opr_delegate.h"

using namespace megdnn;
using namespace arm_common;
using namespace direct_quint8_stride2;

namespace {
bool need_dst_copy(
        const megdnn::fallback::ConvBiasImpl::NCBKernSizeParam& param) {
    return param.osz[1] % 8;
}
bool need_src_copy(
        const megdnn::fallback::ConvBiasImpl::NCBKernSizeParam& param) {
    if (param.filter_meta.padding[0] || param.filter_meta.padding[1]) {
        return true;
    }
    return need_dst_copy(param);
}
void get_rectified_size(
        const megdnn::fallback::ConvBiasImpl::NCBKernSizeParam& param,
        size_t& IH2, size_t& IW2, size_t& OH2, size_t& OW2) {
    auto&& fm = param.filter_meta;
    size_t SW = fm.stride[1];
    size_t IH = param.isz[0];
    size_t IW = param.isz[1];
    size_t OH = param.osz[0];
    size_t OW = param.osz[1];
    size_t FH = fm.spatial[0];
    size_t FW = fm.spatial[1];

    OH2 = OH;
    OW2 = (OW + 7) & ~7;
    IH2 = SW * OH + FH - SW;
    IW2 = SW * OW2 + FW - SW;
    // Because stride is 2, sometimes IW == IW2+1. Do a max update to
    // handle this case.
    IH2 = std::max(IH2, IH);
    IW2 = std::max(IW2, IW);
}
}  // namespace

bool direct_quint8_stride2::can_conv_direct_stride2_quint8(
        const NCBKernSizeParam& param) {
    // Semantically it means avaiable,
    // but we use it as preferred actually.
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    auto OC = fm.ocpg;
    auto IC = fm.icpg;
    bool avaible =
            param.src_type.enumv() == DTypeEnum::Quantized8Asymm &&
            param.filter_type.enumv() == DTypeEnum::Quantized8Asymm &&
            (param.dst_type.enumv() == DTypeEnum::QuantizedS32 ||
             param.dst_type.enumv() == DTypeEnum::Quantized8Asymm) &&
            fm.format == param::Convolution::Format::NCHW && !fm.should_flip &&
            fm.spatial_ndim == 2 && fm.dilation[0] == 1 &&
            fm.dilation[1] == 1 && fm.stride[0] == 2 && fm.stride[1] == 2 &&
            FH == fm.spatial[1] && (FH == 2 || FH == 3 || FH == 5 || FH == 7);
    if (param.bias_type.valid()) {
        avaible &= param.bias_type.enumv() == DTypeEnum::QuantizedS32;
    }
    bool preferred = (((FH == 2 || FH == 3) &&
                       (IC == 1 || (IC <= 8 && OC <= 12) || OC <= 8)) ||
                      (FH == 5 && ((IC == 1 && OC <= 16) || OC <= 12)) ||
                      (FH == 7 && OC <= 16)) &&
                     (param.bias_mode != BiasMode::BIAS);
    return avaible && preferred;
}

WorkspaceBundle direct_quint8_stride2::get_bundle(
        const ConvBiasImpl::NCBKernSizeParam& param, bool m_large_group) {
    auto&& fm = param.filter_meta;
    size_t nr_threads = param.nr_threads;
    size_t group = fm.group, batch = param.n;
    size_t IC = fm.icpg;
    size_t IH2, IW2, OH2, OW2;
    get_rectified_size(param, IH2, IW2, OH2, OW2);
    size_t src_size = 0, dst_size = 0;
    if (need_src_copy(param)) {
        src_size = m_large_group
                           ? IC * IH2 * IW2 * sizeof(uint8_t) * nr_threads
                           : IC * IH2 * IW2 * sizeof(uint8_t) * group * batch;
    };
    if (need_dst_copy(param)) {
        dst_size = OH2 * OW2 * param.dst_type.size() * nr_threads;
    }
    if (IC > 1) {
        size_t temp_size = OH2 * OW2 * sizeof(int32_t) * nr_threads;
        return {nullptr, {src_size, dst_size, temp_size}};
    } else {
        return {nullptr, {src_size, dst_size}};
    };
}
//! Process one input channel copy padding
void direct_quint8_stride2::copy_padding_kern(
        WorkspaceBundle bundle, const ConvBiasImpl::NCBKernParam& kern_param,
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
    bool need_src_copy_var = need_src_copy(kern_param);
    size_t padding_group_size = IH2 * IW2 * IC;
    bundle.set(kern_param.workspace_ptr);

    //! Used for get the workspace offset
    size_t workspace_group_id = workspace_ids[0],
           workspace_batch_id = workspace_ids[1], channel_id = workspace_ids[2],
           group_id = ncb_index.ndrange_id[0],
           batch_id = ncb_index.ndrange_id[1];

    const uint8_t* sptr =
            kern_param.src<uint8_t>(batch_id, group_id, channel_id);
    if (need_src_copy_var) {
        //! copy to sptr_base to eliminate padding effect
        uint8_t* sptr_base = static_cast<uint8_t*>(bundle.get(0)) +
                             workspace_group_id * padding_group_size +
                             workspace_batch_id * GROUP * padding_group_size +
                             channel_id * IH2 * IW2;
        uint8_t _src_zp =
                kern_param.src_type.param<dtype::Quantized8Asymm>().zero_point;
        std::memset(sptr_base, _src_zp, sizeof(uint8_t) * IH2 * IW2);
        rep(ih, IH) {
            std::memcpy(sptr_base + (ih + PH) * IW2 + PW, sptr + ih * IW,
                        sizeof(uint8_t) * IW);
        }
    }
};
//! compute one output channel
template <size_t filter, BiasMode bias_mode, typename Op>
void direct_quint8_stride2::do_conv_kern(WorkspaceBundle bundle,
                                         const NCBKernParam& kern_param,
                                         const NCBKernIndex& ncb_index,
                                         const CpuNDRange& workspace_ids) {
    size_t OH = kern_param.osz[0];
    size_t OW = kern_param.osz[1];
    size_t FH = kern_param.filter_meta.spatial[0];
    size_t FW = kern_param.filter_meta.spatial[1];
    size_t IC = kern_param.filter_meta.icpg;
    size_t GROUP = kern_param.filter_meta.group;
    size_t IH2, IW2, OH2, OW2;
    get_rectified_size(kern_param, IH2, IW2, OH2, OW2);
    bool need_src_copy_var = need_src_copy(kern_param);
    bool need_dst_copy_var = need_dst_copy(kern_param);
    bool need_post_process =
            (kern_param.dst_type.enumv() == DTypeEnum::Quantized8Asymm);

#define SUB128(n) static_cast<int8_t>(static_cast<int32_t>(n) - 128)
    uint8_t _src_zp =
            kern_param.src_type.param<dtype::Quantized8Asymm>().zero_point;
    int8_t src_zp = SUB128(_src_zp);
    int8_t filter_zp = SUB128(
            kern_param.filter_type.param<dtype::Quantized8Asymm>().zero_point);
    int32_t src_filter_zp = static_cast<int32_t>(filter_zp) *
                            static_cast<int32_t>(src_zp) * IC * FH * FW;
#undef SUB128
    Op op = Op(1.0f, 1.0f, 0);
    if (need_post_process) {
        float scale_bias =
                kern_param.bias_type.param<dtype::QuantizedS32>().scale;
        float scale_dst =
                kern_param.dst_type.param<dtype::Quantized8Asymm>().scale;
        uint8_t dst_zp =
                kern_param.dst_type.param<dtype::Quantized8Asymm>().zero_point;
        op = Op(scale_bias, scale_dst, dst_zp);
    }
    size_t padding_group_size = IH2 * IW2 * IC;

    bundle.set(kern_param.workspace_ptr);
    //! Used for get the workspace offset
    size_t workspace_group_id = workspace_ids[0],
           workspace_batch_id = workspace_ids[1], oc = workspace_ids[2],
           group_id = ncb_index.ndrange_id[0],
           batch_id = ncb_index.ndrange_id[1];

    const uint8_t* sptr = kern_param.src<uint8_t>(batch_id, group_id);
    const uint8_t* fptr = kern_param.filter<uint8_t>(group_id) + oc * FH * FW * IC;
    void* dst = kern_param.dst<void>(batch_id, group_id, oc);
    const int32_t* bptr = kern_param.bias<int32_t>(batch_id, group_id, oc);
    if (need_src_copy_var) {
        sptr = static_cast<uint8_t*>(bundle.get(0)) +
               workspace_group_id * padding_group_size +
               workspace_batch_id * GROUP * padding_group_size;
    }
    void* dptr = nullptr;
    int32_t* tptr = nullptr;
    if (need_dst_copy_var) {
        dptr = reinterpret_cast<void*>(
                reinterpret_cast<ptrdiff_t>(bundle.get(1)) +
                ncb_index.thread_id * OH2 * OW2 * kern_param.dst_type.size());
    } else {
        dptr = dst;
    }

#define KERN0_NEED_POST_PROCESS(filter, first_ic, last_ic)           \
    conv_bias::conv_direct_stride2_##filter##x##filter##_quint8<     \
            first_ic, last_ic, bias_mode, Op>(                       \
            sptr + ic * IH2 * IW2, fptr + ic * FH * FW, bptr, tptr,  \
            static_cast<uint8_t*>(dptr), IH2, IW2, OH2, OW2, src_zp, \
            filter_zp, src_filter_zp, op)

#define KERN0_NO_POST_PROCESS(filter, first_ic, last_ic)                      \
    conv_bias::conv_direct_stride2_##filter##x##filter##_quint8<              \
            first_ic, last_ic, bias_mode, Op>(                                \
            sptr + ic * IH2 * IW2, fptr + ic * FH * FW, bptr,                 \
            static_cast<int32_t*>(dptr), nullptr, IH2, IW2, OH2, OW2, src_zp, \
            filter_zp, src_filter_zp, op)

#define KERN1_NEED_POST_PROCESS(filter)                \
    KERN0_NEED_POST_PROCESS(filter, true, false);      \
    for (ic = 1; ic < IC - 1; ++ic) {                  \
        KERN0_NEED_POST_PROCESS(filter, false, false); \
    }                                                  \
    KERN0_NEED_POST_PROCESS(filter, false, true);

#define KERN1_NO_POST_PROCESS(filter)                \
    KERN0_NO_POST_PROCESS(filter, true, false);      \
    for (ic = 1; ic < IC; ++ic) {                    \
        KERN0_NO_POST_PROCESS(filter, false, false); \
    }
    if (need_post_process) {
        size_t ic = 0;
        if (IC == 1) {
            DISPATCH_FILTER(filter, KERN0_NEED_POST_PROCESS, true, true)
        } else {
            tptr = static_cast<int32_t*>(bundle.get(2)) +
                   ncb_index.thread_id * OH2 * OW2 * kern_param.dst_type.size();
            DISPATCH_FILTER(filter, KERN1_NEED_POST_PROCESS)
        }
    } else {
        size_t ic = 0;
        if (IC == 1) {
            DISPATCH_FILTER(filter, KERN0_NO_POST_PROCESS, true, false)
        } else {
            DISPATCH_FILTER(filter, KERN1_NO_POST_PROCESS)
        }
    }
#undef KERN0
#undef KERN1_NEED_POST_PROCESS
#undef KERN1_NO_POST_PROCESS
    if (need_dst_copy_var) {
        rep(oh, OH) {
            std::memcpy(reinterpret_cast<void*>(
                                reinterpret_cast<ptrdiff_t>(dst) +
                                oh * OW * kern_param.dst_type.size()),
                        reinterpret_cast<void*>(
                                reinterpret_cast<ptrdiff_t>(dptr) +
                                oh * OW2 * kern_param.dst_type.size()),
                        kern_param.dst_type.size() * OW);
        }
    }
}

SmallVector<ConvBiasImpl::NCBKern> direct_quint8_stride2::get_kimpls(
        const NCBKernSizeParam& param, bool m_large_group) {
    auto fm = param.filter_meta;
    size_t N = param.n;
    size_t IC = param.filter_meta.icpg;
    size_t OC = param.filter_meta.ocpg;
    size_t group = fm.group;
    WorkspaceBundle wbundle = get_bundle(param, m_large_group);
    conv_fun do_conv_fun = nullptr;

#define DO_CONV_KERN_FUN(filter, bias_mode, op) \
    do_conv_fun = do_conv_kern<filter, bias_mode, op>;

#define GET_OP_PARAM(i, bias_mode)                                        \
    switch (param.nonlineMode) {                                          \
        case param::ConvBias::NonlineMode::IDENTITY:                      \
            DO_CONV_KERN_FUN(i, bias_mode,                                \
                             TypeCvtOp<dt_qint32 MEGDNN_COMMA dt_quint8>) \
            break;                                                        \
        case param::ConvBias::NonlineMode::RELU:                          \
            DO_CONV_KERN_FUN(i, bias_mode,                                \
                             ReluOp<dt_qint32 MEGDNN_COMMA dt_quint8>)    \
            break;                                                        \
        case param::ConvBias::NonlineMode::H_SWISH:                       \
            DO_CONV_KERN_FUN(i, bias_mode,                                \
                             HSwishOp<dt_qint32 MEGDNN_COMMA dt_quint8>)  \
            break;                                                        \
        default:                                                          \
            megdnn_assert(0);                                             \
            break;                                                        \
    }

#define GET_BIAS_MODE_PARAM(i)                                \
    switch (param.bias_mode) {                                \
        case BiasMode::NO_BIAS:                               \
            GET_OP_PARAM(i, BiasMode::NO_BIAS)                \
            break;                                            \
        case BiasMode::BROADCAST_CHANNEL_BIAS:                \
            GET_OP_PARAM(i, BiasMode::BROADCAST_CHANNEL_BIAS) \
            break;                                            \
        default:                                              \
            megdnn_assert(0);                                 \
            break;                                            \
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
    megdnn_assert(do_conv_fun);

    SmallVector<ConvBiasImpl::NCBKern> ret_kerns;
    if (m_large_group) {
        auto exec_one_group = [wbundle, do_conv_fun](
                                      const NCBKernParam& kern_param,
                                      const NCBKernIndex& ncb_index) {
            auto fm = kern_param.filter_meta;
            size_t IC = fm.icpg;
            size_t OC = fm.ocpg;
            WorkspaceBundle bundle = wbundle;
            for (size_t ic = 0; ic < IC; ic++) {
                copy_padding_kern(bundle, kern_param, ncb_index,
                                  {ncb_index.thread_id, 0, ic});
            }
            for (size_t oc = 0; oc < OC; oc++) {
                do_conv_fun(bundle, kern_param, ncb_index,
                            {ncb_index.thread_id, 0, oc});
            }
        };
        ret_kerns.push_back({exec_one_group, {group, N, 1_z}});
    }else {
        WorkspaceBundle bundle = wbundle;
        auto copy_padding = [bundle](const NCBKernParam& kern_param,
                                     const NCBKernIndex& ncb_index) {
            copy_padding_kern(bundle, kern_param, ncb_index,
                              ncb_index.ndrange_id);
        };
        ret_kerns.push_back({copy_padding, {group, N, IC}});
        auto do_conv = [bundle, do_conv_fun](const NCBKernParam& kern_param,
                                             const NCBKernIndex& ncb_index) {
            do_conv_fun(bundle, kern_param, ncb_index, ncb_index.ndrange_id);
        };
        ret_kerns.push_back({do_conv, {group, N, OC}});
    }
    return ret_kerns;
}
// vim: syntax=cpp.doxygen
