/**
 * \file dnn/src/arm_common/conv_bias/f16/channel_wise_nchw88_algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/conv_bias/f16/algos.h"
#include "src/arm_common/conv_bias/f16/channel_wise_nchw88_kern.h"
#include "src/arm_common/elemwise_op.h"

#include "midout.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

using namespace megdnn;
using namespace arm_common;
using namespace fp16;

using conv_fun = std::function<void(
        const __fp16* src, const __fp16* filter, const __fp16* bias, __fp16* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW,
        const size_t PH, size_t PW)>;

MIDOUT_DECL(conv_bias_fp16_channel_wise_nchw88)

bool ConvBiasImpl::AlgoF16ChannelWiseNCHW88::usable(
        const NCBKernSizeParam& param, AlgoSelectionStrategy) const {
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    size_t OC = fm.ocpg;
    size_t IC = fm.icpg;
    size_t GROUP = fm.group;
    bool ok_type =
            (param.src_type.enumv() == DTypeEnum::Float16 &&
             param.filter_type.enumv() == DTypeEnum::Float16 &&
             param.bias_type.enumv() == DTypeEnum::Float16 &&
             param.dst_type.enumv() == DTypeEnum::Float16);
    bool ok_format = OC == 1 && IC == 1 && GROUP % 8 == 0 &&
                     fm.format == param::Convolution::Format::NCHW88;
    bool ok_filter = fm.spatial_ndim == 2 && FH == fm.spatial[1] &&
                     (FH == 2 || FH == 3 || FH == 5);
    bool ok_slide = fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
                    fm.stride[0] == fm.stride[1] &&
                    (fm.stride[0] == 1 || fm.stride[0] == 2);
    bool ok_conv = !fm.should_flip;
    bool ok_comp = param.compute_mode == Param::ComputeMode::DEFAULT;
    return ok_type && ok_format && ok_filter && ok_slide && ok_conv && ok_comp;
}

size_t ConvBiasImpl::AlgoF16ChannelWiseNCHW88::get_workspace(
        const NCBKernSizeParam&) const {
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoF16ChannelWiseNCHW88::
        dispatch_kerns(const NCBKernSizeParam& param) const {
    const constexpr size_t pack_group_size = 8_z;
    auto fm = param.filter_meta;
    const int batch = param.n;
    const int group = fm.group;
    const int stride = fm.stride[0];

    conv_fun do_conv_fun = nullptr;
    // NOTE: remain_w is not used to gen hash of midout for compatible with
// shape runtime
#define DO_CONV_KERN_FUN(_stride, filter, bias_mode, op)                           \
    MIDOUT_BEGIN(                                                                  \
            conv_bias_fp16_channel_wise_nchw88,                                    \
            midout_iv(#_stride #filter #bias_mode #op##_hash)) {                   \
        do_conv_fun =                                                              \
                channel_wise_nchw88::do_conv_kern_##_stride##_##filter##x##filter< \
                        bias_mode, op>;                                            \
    }                                                                              \
    MIDOUT_END();

#define GET_OP_PARAM(_stride, filter, bias_mode)                            \
    switch (param.nonlineMode) {                                            \
        case param::ConvBias::NonlineMode::IDENTITY:                        \
            DO_CONV_KERN_FUN(_stride, filter, bias_mode, NoneOp<__fp16>)    \
            break;                                                          \
        case param::ConvBias::NonlineMode::RELU:                            \
            DO_CONV_KERN_FUN(_stride, filter, bias_mode, ReluOp<__fp16>)    \
            break;                                                          \
        case param::ConvBias::NonlineMode::SIGMOID:                         \
            DO_CONV_KERN_FUN(_stride, filter, bias_mode, SigmoidOp<__fp16>) \
            break;                                                          \
        case param::ConvBias::NonlineMode::H_SWISH:                         \
            DO_CONV_KERN_FUN(_stride, filter, bias_mode, HSwishOp<__fp16>)  \
            break;                                                          \
        default:                                                            \
            megdnn_assert(0, "not supported nonline mode");                 \
            break;                                                          \
    }

#define GET_BIAS_MODE_PARAM(_stride, filter)                                \
    switch (param.bias_mode) {                                              \
        case BiasMode::NO_BIAS:                                             \
            GET_OP_PARAM(_stride, filter, BiasMode::NO_BIAS)                \
            break;                                                          \
        case BiasMode::BROADCAST_CHANNEL_BIAS:                              \
            GET_OP_PARAM(_stride, filter, BiasMode::BROADCAST_CHANNEL_BIAS) \
            break;                                                          \
        case BiasMode::BIAS:                                                \
            GET_OP_PARAM(_stride, filter, BiasMode::BIAS)                   \
            break;                                                          \
        default:                                                            \
            megdnn_assert(0, "not supported bias mode");                    \
            break;                                                          \
    }

#define DISPATCH_CONV_KERN(_stride)                   \
    switch (param.filter_meta.spatial[0]) {           \
        case 2:                                       \
            GET_BIAS_MODE_PARAM(_stride, 2)           \
            break;                                    \
        case 3:                                       \
            GET_BIAS_MODE_PARAM(_stride, 3)           \
            break;                                    \
        case 5:                                       \
            GET_BIAS_MODE_PARAM(_stride, 5)           \
            break;                                    \
        default:                                      \
            megdnn_assert(0, "not supported stride"); \
            break;                                    \
    }

#define DISPATCH_STRIDE()            \
    if (1 == stride) {               \
        DISPATCH_CONV_KERN(stride1); \
    } else {                         \
        DISPATCH_CONV_KERN(stride2); \
    }

    DISPATCH_STRIDE();

#undef DO_CONV_KERN_FUN
#undef GET_REMAIN_W_PARAM
#undef GET_OP_PARAM
#undef GET_BIAS_MODE_PARAM
#undef DISPATCH_CONV_KERN
#undef DISPATCH_STRIDE

    megdnn_assert(do_conv_fun, "conv filter not supported");

    SmallVector<ConvBiasImpl::NCBKern> ret_kerns;

    CpuNDRange ncb_range = {
            static_cast<size_t>(batch), static_cast<size_t>(group / pack_group_size)};
    auto do_conv = [do_conv_fun](
                           const NCBKernParam& kern_param,
                           const NCBKernIndex& ncb_index) {
        size_t PH = kern_param.filter_meta.padding[0];
        size_t PW = kern_param.filter_meta.padding[1];
        size_t OH = kern_param.osz[0];
        size_t OW = kern_param.osz[1];
        size_t IH = kern_param.isz[0];
        size_t IW = kern_param.isz[1];

        size_t batch_id = ncb_index.ndrange_id[0];
        size_t group_id = ncb_index.ndrange_id[1];
        const __fp16* sptr = reinterpret_cast<const __fp16*>(
                kern_param.src<dt_float16>(batch_id, group_id, 0, pack_group_size));
        const __fp16* fptr = reinterpret_cast<const __fp16*>(
                kern_param.filter<dt_float16>(group_id, pack_group_size));
        __fp16* dst = reinterpret_cast<__fp16*>(
                kern_param.dst<dt_float16>(batch_id, group_id, 0, pack_group_size));
        const __fp16* bptr = reinterpret_cast<const __fp16*>(
                kern_param.bias<dt_float16>(batch_id, group_id, 0, pack_group_size));

        do_conv_fun(sptr, fptr, bptr, dst, IH, IW, OH, OW, PH, PW);
    };
    ret_kerns.push_back({do_conv, ncb_range});
    return ret_kerns;
}

#endif

// vim: syntax=cpp.doxygen
