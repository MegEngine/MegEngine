/**
 * \file dnn/src/arm_common/conv_bias/int8/channel_wise_nchw44.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/int8/channel_wise_nchw44.h"
#include "megdnn/oprs.h"
#include "src/arm_common/conv_bias/int8/channel_wise_kernel.h"
#include "src/arm_common/elemwise_op.h"
#include "src/common/opr_delegate.h"

#include "midout.h"

using namespace megdnn;
using namespace arm_common;
using namespace channel_wise_nchw44;

namespace {
void get_rectified_size(
        const megdnn::fallback::ConvBiasImpl::NCBKernSizeParam& param,
        size_t& IH2, size_t& IW2) {
    auto&& fm = param.filter_meta;
    auto SW = fm.stride[1];
    auto OH = param.osz[0];
    auto OW = param.osz[1];
    auto FH = fm.spatial[0];
    auto FW = fm.spatial[1];

    size_t OW2 = (OW + 3) & ~3;
    IH2 = SW * OH + FH - SW;
    IW2 = SW * OW2 + FW - SW;
}
}  // namespace

MIDOUT_DECL(megdnn_arm_common_conv_bias_int8_nchw44_stride1)
MIDOUT_DECL(megdnn_arm_common_conv_bias_int8_nchw44_stride2)

bool stride1::is_available(const NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    bool avaible =
            //! src and filter are qint8, dst is qint8 or qint32
            ((param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
              param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
              (param.dst_type.enumv() == DTypeEnum::QuantizedS8 ||
               param.dst_type.enumv() == DTypeEnum::QuantizedS32)) ||
             //! src and filter are int8, dst is int32
             (param.src_type.enumv() == DTypeEnum::Int8 &&
              param.filter_type.enumv() == DTypeEnum::Int8 &&
              param.dst_type.enumv() == DTypeEnum::Int32)) &&
            fm.format == param::Convolution::Format::NCHW44 &&
            !fm.should_flip && fm.spatial_ndim == 2 && fm.dilation[0] == 1 &&
            fm.dilation[1] == 1 && fm.stride[0] == 1 && fm.stride[1] == 1 &&
            FH == fm.spatial[1] && (FH == 2 || FH == 3 || FH == 5) &&
            fm.icpg == 1 && fm.ocpg == 1 && fm.group % 4 == 0;
    return avaible;
}

WorkspaceBundle stride1::get_bundle(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    size_t nr_threads = param.nr_threads;
    size_t IH2, IW2;
    get_rectified_size(param, IH2, IW2);
    constexpr size_t pack_ic_size = 4_z;
    //! The extra 16B is used to void ivalid read in kernel compute
    size_t src_size = IH2 * IW2 * pack_ic_size * sizeof(int8_t) + 16;
    SmallVector<size_t> sizes(nr_threads, src_size);
    return {nullptr, sizes};
}

//! compute one output channel
template <bool quantized, size_t filter, BiasMode bias_mode, typename Op>
void stride1::do_conv_kern(const WorkspaceBundle& bundle,
                           const NCBKernParam& kern_param,
                           const NCBKernIndex& ncb_index) {
    size_t PH = kern_param.filter_meta.padding[0];
    size_t PW = kern_param.filter_meta.padding[1];
    size_t OH = kern_param.osz[0];
    size_t OW = kern_param.osz[1];
    size_t IH = kern_param.isz[0];
    size_t IW = kern_param.isz[1];
    size_t IH2, IW2;
    get_rectified_size(kern_param, IH2, IW2);
    Op op = Op(1.0f, 1.0f);
    if (quantized) {
        float scale_bias =
                kern_param.bias_type.param<dtype::QuantizedS32>().scale;
        float scale_dst = kern_param.dst_type.param<dtype::QuantizedS8>().scale;
        op = Op(scale_bias, scale_dst);
    }

    constexpr size_t pack_group_size = 4_z;
    constexpr size_t pack_ic_size = 4_z;

    size_t thread_id = ncb_index.thread_id, batch_id = ncb_index.ndrange_id[0];
    size_t group_id = ncb_index.ndrange_id[1];
    int8_t* padding_src = static_cast<int8_t*>(bundle.get(thread_id));
    const int8_t* sptr =
            kern_param.src<dt_int8>(batch_id, group_id, 0, pack_group_size);
    const int8_t* fptr = kern_param.filter<dt_int8>(group_id, pack_group_size);
    void* dst = kern_param.dst<void>(batch_id, group_id, 0, pack_group_size);
    const int32_t* bptr =
            kern_param.bias<dt_int32>(batch_id, group_id, 0, pack_group_size);
    //! copy in case of illegal read src when padding is zero
    std::memset(padding_src, 0, sizeof(int8_t) * IH2 * IW2 * pack_ic_size);
    rep(ih, IH) {
        std::memcpy(padding_src + ((ih + PH) * IW2 + PW) * pack_ic_size,
                    sptr + ih * IW * pack_ic_size,
                    sizeof(int8_t) * IW * pack_ic_size);
    }
    sptr = padding_src;

#define KERN(_size)                                                    \
    direct_stride1_##_size##x##_size##_int8<quantized, bias_mode, Op>( \
            sptr, fptr, bptr, dst, IH2, IW2, OH, OW, op);
    DISPATCH_FILTER_CHANNEL_WISE(filter, KERN);
#undef KERN
}

SmallVector<ConvBiasImpl::NCBKern> stride1::get_kimpls(
        const NCBKernSizeParam& param) {
    auto fm = param.filter_meta;
    size_t N = param.n;
    size_t group = fm.group / 4;
    megdnn_assert(fm.group % 4 == 0,
                  "nchw44 channel wise conv with group is not times of 4");
    WorkspaceBundle wbundle = get_bundle(param);
    bool quantized = param.dst_type.enumv() == DTypeEnum::QuantizedS8;
    conv_fun do_conv_fun = nullptr;

#define DO_CONV_KERN_FUN(quantized, filter, bias_mode, op)              \
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8_nchw44_stride1,       \
                 midout_iv(#quantized #filter #bias_mode #op##_hash)) { \
        do_conv_fun = do_conv_kern<quantized, filter, bias_mode, op>;   \
    }                                                                   \
    MIDOUT_END();

#define GET_OP_PARAM(i, bias_mode)                                           \
    switch (param.nonlineMode) {                                             \
        case param::ConvBias::NonlineMode::IDENTITY:                         \
            if (quantized) {                                                 \
                DO_CONV_KERN_FUN(true, i, bias_mode,                         \
                                 TypeCvtOp<dt_qint32 MEGDNN_COMMA dt_qint8>) \
            } else {                                                         \
                DO_CONV_KERN_FUN(false, i, bias_mode,                        \
                                 NoneOp<dt_qint32 MEGDNN_COMMA dt_qint8>)    \
            }                                                                \
            break;                                                           \
        case param::ConvBias::NonlineMode::RELU:                             \
            if (quantized) {                                                 \
                DO_CONV_KERN_FUN(true, i, bias_mode,                         \
                                 ReluOp<dt_qint32 MEGDNN_COMMA dt_qint8>)    \
            } else {                                                         \
                DO_CONV_KERN_FUN(false, i, bias_mode,                        \
                                 NoneOp<dt_qint32 MEGDNN_COMMA dt_qint8>)    \
            }                                                                \
            break;                                                           \
        case param::ConvBias::NonlineMode::H_SWISH:                          \
            if (quantized) {                                                 \
                DO_CONV_KERN_FUN(true, i, bias_mode,                         \
                                 HSwishOp<dt_qint32 MEGDNN_COMMA dt_qint8>)  \
            } else {                                                         \
                DO_CONV_KERN_FUN(false, i, bias_mode,                        \
                                 NoneOp<dt_qint32 MEGDNN_COMMA dt_qint8>)    \
            }                                                                \
            break;                                                           \
        default:                                                             \
            megdnn_assert(0);                                                \
            break;                                                           \
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
        default:                            \
            megdnn_assert(0);               \
            break;                          \
    }

    DISPATCH_CONV_KERN();
    megdnn_assert(do_conv_fun);

    SmallVector<ConvBiasImpl::NCBKern> ret_kerns;
    auto exec_one_group = [wbundle, do_conv_fun](
                                  const NCBKernParam& kern_param,
                                  const NCBKernIndex& ncb_index) mutable {
        wbundle.set(kern_param.workspace_ptr);
        do_conv_fun(wbundle, kern_param, ncb_index);
    };
    ret_kerns.push_back({exec_one_group, {N, group}});
    return ret_kerns;
#undef DO_CONV_KERN_FUN
}

bool stride2::is_available(const NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    bool avaible =
            //! src and filter are qint8, dst is qint8 or qint32
            ((param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
              param.filter_type.enumv() == DTypeEnum::QuantizedS8 &&
              (param.dst_type.enumv() == DTypeEnum::QuantizedS8 ||
               param.dst_type.enumv() == DTypeEnum::QuantizedS32)) ||
             //! src and filter are int8, dst is int32
             (param.src_type.enumv() == DTypeEnum::Int8 &&
              param.filter_type.enumv() == DTypeEnum::Int8 &&
              param.dst_type.enumv() == DTypeEnum::Int32)) &&
            fm.format == param::Convolution::Format::NCHW44 &&
            !fm.should_flip && fm.spatial_ndim == 2 && fm.dilation[0] == 1 &&
            fm.dilation[1] == 1 && fm.stride[0] == 2 && fm.stride[1] == 2 &&
            FH == fm.spatial[1] && (FH == 2 || FH == 3 || FH == 5) &&
            fm.icpg == 1 && fm.ocpg == 1 && fm.group % 4 == 0;
    return avaible;
}

WorkspaceBundle stride2::get_bundle(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    size_t nr_threads = param.nr_threads;
    size_t IH2, IW2;
    get_rectified_size(param, IH2, IW2);
    constexpr size_t pack_ic_size = 4_z;
    //! The extra 16B is used to void ivalid read in kernel compute
    size_t src_size = IH2 * IW2 * pack_ic_size * sizeof(int8_t) + 16;
    SmallVector<size_t> sizes(nr_threads, src_size);
    return {nullptr, sizes};
}

//! compute one output channel
template <bool quantized, size_t filter, BiasMode bias_mode, typename Op>
void stride2::do_conv_kern(const WorkspaceBundle& bundle,
                           const NCBKernParam& kern_param,
                           const NCBKernIndex& ncb_index) {
    size_t PH = kern_param.filter_meta.padding[0];
    size_t PW = kern_param.filter_meta.padding[1];
    size_t OH = kern_param.osz[0];
    size_t OW = kern_param.osz[1];
    size_t IH = kern_param.isz[0];
    size_t IW = kern_param.isz[1];
    size_t IH2, IW2;
    get_rectified_size(kern_param, IH2, IW2);
    Op op = Op(1.0f, 1.0f);
    if (quantized) {
        float scale_bias =
                kern_param.bias_type.param<dtype::QuantizedS32>().scale;
        float scale_dst = kern_param.dst_type.param<dtype::QuantizedS8>().scale;
        op = Op(scale_bias, scale_dst);
    }

    constexpr size_t pack_group_size = 4_z;
    constexpr size_t pack_ic_size = 4_z;

    size_t thread_id = ncb_index.thread_id, batch_id = ncb_index.ndrange_id[0];
    size_t group_id = ncb_index.ndrange_id[1];
    int8_t* padding_src = static_cast<int8_t*>(bundle.get(thread_id));
    const int8_t* sptr =
            kern_param.src<dt_int8>(batch_id, group_id, 0, pack_group_size);
    const int8_t* fptr = kern_param.filter<dt_int8>(group_id, pack_group_size);
    void* dst = kern_param.dst<void>(batch_id, group_id, 0, pack_group_size);
    const int32_t* bptr =
            kern_param.bias<dt_int32>(batch_id, group_id, 0, pack_group_size);
    //! copy in case of illegal read src when padding is zero
    std::memset(padding_src, 0, sizeof(int8_t) * IH2 * IW2 * pack_ic_size);
    rep(ih, IH) {
        std::memcpy(padding_src + ((ih + PH) * IW2 + PW) * pack_ic_size,
                    sptr + ih * IW * pack_ic_size,
                    sizeof(int8_t) * IW * pack_ic_size);
    }
    sptr = padding_src;

#define KERN(_size)                                                    \
    direct_stride2_##_size##x##_size##_int8<quantized, bias_mode, Op>( \
            sptr, fptr, bptr, dst, IH2, IW2, OH, OW, op);
    DISPATCH_FILTER_CHANNEL_WISE(filter, KERN);
#undef KERN
}

SmallVector<ConvBiasImpl::NCBKern> stride2::get_kimpls(
        const NCBKernSizeParam& param) {
    auto fm = param.filter_meta;
    size_t N = param.n;
    size_t group = fm.group / 4;
    megdnn_assert(fm.group % 4 == 0,
                  "nchw44 channel wise conv with group is not times of 4");
    WorkspaceBundle wbundle = get_bundle(param);
    bool quantized = param.dst_type.enumv() == DTypeEnum::QuantizedS8;
    conv_fun do_conv_fun = nullptr;

#define DO_CONV_KERN_FUN(quantized, filter, bias_mode, op)              \
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8_nchw44_stride2,       \
                 midout_iv(#quantized #filter #bias_mode #op##_hash)) { \
        do_conv_fun = do_conv_kern<quantized, filter, bias_mode, op>;   \
    }                                                                   \
    MIDOUT_END();

    DISPATCH_CONV_KERN();
    megdnn_assert(do_conv_fun);

    SmallVector<ConvBiasImpl::NCBKern> ret_kerns;
    auto exec_one_group = [wbundle, do_conv_fun](
                                  const NCBKernParam& kern_param,
                                  const NCBKernIndex& ncb_index) mutable {
        wbundle.set(kern_param.workspace_ptr);
        do_conv_fun(wbundle, kern_param, ncb_index);
    };
    ret_kerns.push_back({exec_one_group, {N, group}});
    return ret_kerns;
#undef DISPATCH_CONV_KERN
#undef GET_BIAS_MODE_PARAM
#undef GET_OP_PARAM
}

// vim: syntax=cpp.doxygen
