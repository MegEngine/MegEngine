/**
 * \file src/x86/conv_bias/int8/avx2_chanwsie_stride2.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/x86/conv_bias/int8/avx2_chanwise_stride2.h"
#include "src/x86/conv_bias/int8/avx2_chanwise_kern.h"
#include "src/x86/elemwise_op.h"

namespace megdnn {
namespace x86 {
namespace avx2_chanwise_stride2 {

template <size_t filter, BiasMode bias_mode, bool is_quantized, typename Op>
void conv_kimpl(const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
                const NCBKernIndex& ncb_index) {
    size_t OH = kern_param.osz[0];
    size_t OW = kern_param.osz[1];
    size_t IH2, IW2, OH2, OW2;
    get_rectified_size(kern_param, IH2, IW2, OH2, OW2);
    bool need_src_copy_var = need_src_copy(kern_param);
    bool need_dst_copy_var = need_dst_copy(kern_param);
    bool need_post_process =
            kern_param.dst_type.enumv() == DTypeEnum::QuantizedS8;

    Op op = Op(1.0f, 4.0f);
    if (need_post_process) {
        float scale_bias =
                kern_param.bias_type.param<dtype::QuantizedS32>().scale;
        float scale_dst = kern_param.dst_type.param<dtype::QuantizedS8>().scale;
        op = Op(scale_bias, scale_dst);
    }
    size_t padding_group_size = IH2 * IW2;
    size_t workspace_group_id = ncb_index.thread_id;
    size_t group_id = ncb_index.ndrange_id[0],
           batch_id = ncb_index.ndrange_id[1];

    const int8_t* sptr = kern_param.src<dt_int8>(batch_id, group_id);
    const int8_t* fptr = kern_param.filter<dt_int8>(group_id);
    void* dst = kern_param.dst<void>(batch_id, group_id);
    const int32_t* bptr = kern_param.bias<dt_int32>(batch_id, group_id);
    if (need_src_copy_var) {
        sptr = static_cast<int8_t*>(bundle.get(0)) +
               workspace_group_id * padding_group_size;
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

#define KERN_NEED_POST_PROCESS(filter)                                         \
    avx2_chanwise_direct_stride2_##filter##x##filter##_int8<bias_mode, true,   \
                                                            Op>(               \
            sptr, fptr, bptr, tptr, static_cast<int8_t*>(dptr), IH2, IW2, OH2, \
            OW2, op)

#define KERN_NO_POST_PROCESS(filter)                                          \
    avx2_chanwise_direct_stride2_##filter##x##filter##_int8<bias_mode, false, \
                                                            Op>(              \
            sptr, fptr, bptr, static_cast<int32_t*>(dptr), nullptr, IH2, IW2, \
            OH2, OW2, op)

    if (need_post_process) {
        tptr = static_cast<int32_t*>(bundle.get(2)) +
               ncb_index.thread_id * OH2 * OW2 * kern_param.dst_type.size();
        DISPATCH_FILTER(filter, KERN_NEED_POST_PROCESS)
    } else {
        DISPATCH_FILTER(filter, KERN_NO_POST_PROCESS)
    }

#undef KERN_NEED_POST_PROCESS
#undef KERN_NO_POST_PROCESS
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
};
SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& kern_param,
                                const WorkspaceBundle& bundle) {
    MEGDNN_MARK_USED_VAR(kern_param);
    auto fm = kern_param.filter_meta;
    size_t group = fm.group;
    size_t n = kern_param.n;

    SmallVector<NCBKern> ncb_kerns;
    conv_fun do_conv_fun = nullptr;

#define DO_CONV_KERN_FUN(filter, bias_mode, is_quantized, op) \
    do_conv_fun = conv_kimpl<filter, bias_mode, is_quantized, op>;

#define GET_OP_PARAM(i, bias_mode, is_quantized)                             \
    switch (kern_param.nonlineMode) {                                        \
        case param::ConvBias::NonlineMode::IDENTITY:                         \
            DO_CONV_KERN_FUN(i, bias_mode, is_quantized,                     \
                             TypeCvtOp<SIMDType::AVX2 MEGDNN_COMMA dt_qint32 \
                                               MEGDNN_COMMA dt_qint8>)       \
            break;                                                           \
        case param::ConvBias::NonlineMode::RELU:                             \
            DO_CONV_KERN_FUN(i, bias_mode, is_quantized,                     \
                             ReluOp<SIMDType::AVX2 MEGDNN_COMMA dt_qint32    \
                                            MEGDNN_COMMA dt_qint8>)          \
            break;                                                           \
        case param::ConvBias::NonlineMode::H_SWISH:                          \
            DO_CONV_KERN_FUN(i, bias_mode, is_quantized,                     \
                             HSwishOp<SIMDType::AVX2 MEGDNN_COMMA dt_qint32  \
                                              MEGDNN_COMMA dt_qint8>)        \
            break;                                                           \
        default:                                                             \
            megdnn_assert(0, "do not support nonlineMode: %d",               \
                          static_cast<int>(kern_param.nonlineMode));         \
            break;                                                           \
    }

#define GET_BIAS_MODE_PARAM(i, is_quantized)                                \
    switch (kern_param.bias_mode) {                                         \
        case BiasMode::NO_BIAS:                                             \
            GET_OP_PARAM(i, BiasMode::NO_BIAS, is_quantized)                \
            break;                                                          \
        case BiasMode::BROADCAST_CHANNEL_BIAS:                              \
            GET_OP_PARAM(i, BiasMode::BROADCAST_CHANNEL_BIAS, is_quantized) \
            break;                                                          \
        default:                                                            \
            megdnn_assert(0, "do not support bias mode: %d",                \
                          static_cast<int>(kern_param.bias_mode));          \
            break;                                                          \
    }

#define GET_QUANTIZED(i)                                                  \
    switch (kern_param.dst_type.enumv()) {                                \
        case DTypeEnum::QuantizedS8:                                      \
            GET_BIAS_MODE_PARAM(i, true)                                  \
            break;                                                        \
        case DTypeEnum::QuantizedS32:                                     \
            GET_BIAS_MODE_PARAM(i, false)                                 \
            break;                                                        \
        case DTypeEnum::Int32:                                            \
            GET_BIAS_MODE_PARAM(i, false)                                 \
            break;                                                        \
        default:                                                          \
            megdnn_assert(0, "do not support dtype: %d",                  \
                          static_cast<int>(kern_param.dst_type.enumv())); \
            break;                                                        \
    }

#define DISPATCH_CONV_KERN()                                              \
    switch (kern_param.filter_meta.spatial[0]) {                          \
        case 2:                                                           \
            GET_QUANTIZED(2)                                              \
            break;                                                        \
        case 3:                                                           \
            GET_QUANTIZED(3)                                              \
            break;                                                        \
        case 5:                                                           \
            GET_QUANTIZED(5)                                              \
            break;                                                        \
        case 7:                                                           \
            GET_QUANTIZED(7)                                              \
            break;                                                        \
        default:                                                          \
            megdnn_assert(                                                \
                    0, "do not support kernel: %d",                       \
                    static_cast<int>(kern_param.filter_meta.spatial[0])); \
            break;                                                        \
    }

    DISPATCH_CONV_KERN();

    auto exec_one_group = [bundle = bundle, do_conv_fun](
                                  const NCBKernParam& kern_param,
                                  const NCBKernIndex& ncb_index) mutable {
        bundle.set(kern_param.workspace_ptr);
        copy_padding_kern(bundle, kern_param, ncb_index);
        do_conv_fun(bundle, kern_param, ncb_index);
    };
    ncb_kerns.push_back({exec_one_group, {group, n, 1_z}});

    return ncb_kerns;
}

}  // namespace avx2_chanwise_stride2
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
