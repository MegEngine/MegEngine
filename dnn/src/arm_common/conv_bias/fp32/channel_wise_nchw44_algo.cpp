/**
 * \file dnn/src/arm_common/conv_bias/fp32/channel_wise_nchw44_algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/conv_bias/fp32/algos.h"
#include "src/arm_common/conv_bias/fp32/channel_wise_nchw44_kern.h"
#include "src/arm_common/elemwise_op.h"

#include "midout.h"

using namespace megdnn;
using namespace arm_common;
using conv_fun = std::function<void(
        const float* src, const float* filter, const float* bias, float* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW,
        const size_t PH, size_t PW)>;

MIDOUT_DECL(conv_bias_fp32_channel_wise_nchw44)

bool ConvBiasImpl::AlgoF32ChannelWiseNCHW44::usable(
        const NCBKernSizeParam& param, AlgoSelectionStrategy) const {
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    size_t OC = fm.ocpg;
    size_t IC = fm.icpg;
    size_t GROUP = fm.group;
    bool ok_type = (param.src_type.enumv() == DTypeEnum::Float32 &&
                    param.filter_type.enumv() == DTypeEnum::Float32 &&
                    (param.dst_type.enumv() == DTypeEnum::Float32));
    bool ok_format = OC == 1 && IC == 1 && GROUP % 4 == 0 &&
                     fm.format == param::Convolution::Format::NCHW44;
    bool ok_filter = fm.spatial_ndim == 2 && FH == fm.spatial[1] &&
                     (FH == 2 || FH == 3 || FH == 5);
    bool ok_slide = fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
                    fm.stride[0] == fm.stride[1] &&
                    (fm.stride[0] == 1 || fm.stride[0] == 2);
    bool ok_conv = !fm.should_flip;
    bool avaible = ok_type && ok_format && ok_filter && ok_slide && ok_conv;
    return avaible;
}

size_t ConvBiasImpl::AlgoF32ChannelWiseNCHW44::get_workspace(
        const NCBKernSizeParam&) const {
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoF32ChannelWiseNCHW44::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    const constexpr size_t pack_group_size = 4_z;
    auto fm = param.filter_meta;
    const int batch = param.n;
    const int group = fm.group;
    const int stride = fm.stride[0];

    conv_fun do_conv_fun = nullptr;
    // NOTE: remain_w is not used to gen hash of midout for compatible with
// shape runtime
#define DO_CONV_KERN_FUN(_stride, filter, bias_mode, op)                     \
    MIDOUT_BEGIN(conv_bias_fp32_channel_wise_nchw44,                         \
                 midout_iv(#_stride #filter #bias_mode #op##_hash)) {        \
        do_conv_fun = channel_wise_nchw44_float::                            \
                do_conv_kern_##_stride##_##filter##x##filter<bias_mode, op>; \
    }                                                                        \
    MIDOUT_END();

#define GET_OP_PARAM(_stride, filter, bias_mode)                               \
    switch (param.nonlineMode) {                                               \
        case param::ConvBias::NonlineMode::IDENTITY:                           \
            DO_CONV_KERN_FUN(_stride, filter, bias_mode, NoneOp<dt_float32>)   \
            break;                                                             \
        case param::ConvBias::NonlineMode::RELU:                               \
            DO_CONV_KERN_FUN(_stride, filter, bias_mode, ReluOp<dt_float32>)   \
            break;                                                             \
        case param::ConvBias::NonlineMode::SIGMOID:                            \
            DO_CONV_KERN_FUN(_stride, filter, bias_mode,                       \
                             SigmoidOp<dt_float32>)                            \
            break;                                                             \
        case param::ConvBias::NonlineMode::H_SWISH:                            \
            DO_CONV_KERN_FUN(_stride, filter, bias_mode, HSwishOp<dt_float32>) \
            break;                                                             \
        default:                                                               \
            megdnn_assert(0);                                                  \
            break;                                                             \
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
            megdnn_assert(0);                                               \
            break;                                                          \
    }

#define DISPATCH_CONV_KERN(_stride)         \
    switch (param.filter_meta.spatial[0]) { \
        case 2:                             \
            GET_BIAS_MODE_PARAM(_stride, 2) \
            break;                          \
        case 3:                             \
            GET_BIAS_MODE_PARAM(_stride, 3) \
            break;                          \
        case 5:                             \
            GET_BIAS_MODE_PARAM(_stride, 5) \
            break;                          \
        default:                            \
            megdnn_assert(0);               \
            break;                          \
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

    megdnn_assert(do_conv_fun);

    SmallVector<ConvBiasImpl::NCBKern> ret_kerns;

    CpuNDRange ncb_range = {static_cast<size_t>(batch),
                            static_cast<size_t>(group / pack_group_size)};
    auto do_conv = [do_conv_fun](const NCBKernParam& kern_param,
                                 const NCBKernIndex& ncb_index) {
        size_t PH = kern_param.filter_meta.padding[0];
        size_t PW = kern_param.filter_meta.padding[1];
        size_t OH = kern_param.osz[0];
        size_t OW = kern_param.osz[1];
        size_t IH = kern_param.isz[0];
        size_t IW = kern_param.isz[1];

        size_t batch_id = ncb_index.ndrange_id[0];
        size_t group_id = ncb_index.ndrange_id[1];
        const float* sptr =
                kern_param.src<float>(batch_id, group_id, 0, pack_group_size);
        const float* fptr = kern_param.filter<float>(group_id, pack_group_size);
        float* dst =
                kern_param.dst<float>(batch_id, group_id, 0, pack_group_size);
        const float* bptr =
                kern_param.bias<float>(batch_id, group_id, 0, pack_group_size);
        //! copy in case of illegal read src when padding is zero
        do_conv_fun(sptr, fptr, bptr, dst, IH, IW, OH, OW, PH, PW);
    };
    ret_kerns.push_back({do_conv, ncb_range});
    return ret_kerns;
}

//vim: syntax=cpp.doxygen
