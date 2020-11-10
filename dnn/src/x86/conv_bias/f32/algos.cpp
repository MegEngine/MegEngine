/**
 * \file dnn/src/x86/conv_bias/f32/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/x86/conv_bias/f32/algos.h"
#include <unordered_map>
#include "src/common/opr_delegate.h"
#include "src/common/utils.h"
#include "src/fallback/convolution/img2col_helper.h"
#include "src/x86/conv_bias/f32/do_conv_stride2.h"
#include "src/x86/conv_bias/opr_impl.h"
#include "src/x86/conv_bias/postprocess_helper.h"
#include "src/x86/convolution/convolution_direct_special_cases.h"
#include "src/x86/handle.h"

#include "midout.h"

using namespace megdnn;
using namespace x86;
namespace {
bool need_dst_copy(const fallback::ConvBiasImpl::NCBKernSizeParam& param) {
    if (param.osz[0] % 8 != 0 || param.osz[1] % 8 != 0) {
        // If the size of output is not multiples of 8, we need to copy it.
        return true;
    }
    return false;
}

bool need_src_copy(const fallback::ConvBiasImpl::NCBKernSizeParam& param) {
    if (param.filter_meta.padding[0] || param.filter_meta.padding[1]) {
        // If padding is not zero, we need to copy to eliminate padding effect.
        return true;
    }
    return need_dst_copy(param);
}

void get_rectified_size(size_t IH, size_t IW, size_t OH, size_t OW, size_t FH,
                        size_t FW, size_t PH, size_t PW, size_t& IH2,
                        size_t& IW2, size_t& OH2, size_t& OW2) {
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

#define GET_KERN                                                               \
    auto fm = param.filter_meta;                                               \
    size_t N = param.n;                                                        \
    size_t IC = param.filter_meta.icpg;                                        \
    size_t OC = param.filter_meta.ocpg;                                        \
    size_t group = fm.group;                                                   \
    bool large_group = group >= param.nr_threads;                              \
    WorkspaceBundle bundle = get_bundle(param);                                \
    SmallVector<NCBKern> ret_kerns;                                            \
    if (large_group) {                                                         \
        auto exec_one_group = [bundle](                                        \
                                      const NCBKernParam& kern_param,          \
                                      const NCBKernIndex& ncb_index) mutable { \
            bundle.set(kern_param.workspace_ptr);                              \
            auto fm = kern_param.filter_meta;                                  \
            size_t IC = fm.icpg;                                               \
            size_t OC = fm.ocpg;                                               \
            for (size_t ic = 0; ic < IC; ic++) {                               \
                copy_padding_kern(bundle, kern_param, ncb_index,               \
                                  {ncb_index.thread_id, 0, ic});               \
            }                                                                  \
            for (size_t oc = 0; oc < OC; oc++) {                               \
                do_conv_kern(bundle, kern_param, ncb_index,                    \
                             {ncb_index.thread_id, 0, oc});                    \
            }                                                                  \
        };                                                                     \
        ret_kerns.push_back({exec_one_group, {group, N, 1_z}});                \
    } else {                                                                   \
        auto copy_padding = [bundle](const NCBKernParam& kern_param,           \
                                     const NCBKernIndex& ncb_index) mutable {  \
            bundle.set(kern_param.workspace_ptr);                              \
            copy_padding_kern(bundle, kern_param, ncb_index,                   \
                              ncb_index.ndrange_id);                           \
        };                                                                     \
        ret_kerns.push_back({copy_padding, {group, N, IC}});                   \
        auto do_conv = [bundle](const NCBKernParam& kern_param,                \
                                const NCBKernIndex& ncb_index) mutable {       \
            bundle.set(kern_param.workspace_ptr);                              \
            do_conv_kern(bundle, kern_param, ncb_index, ncb_index.ndrange_id); \
        };                                                                     \
        ret_kerns.push_back({do_conv, {group, N, OC}});                        \
    }                                                                          \
    return ret_kerns;

/* ===================== direct algo ===================== */

bool ConvBiasImpl::AlgoDirect::usable(const NCBKernSizeParam& param,
                                      AlgoSelectionStrategy) const {
    auto&& fm = param.filter_meta;
    return fm.format == Param::Format::NCHW && fm.spatial_ndim == 2 &&
           param.src_type.enumv() == DTypeEnum::Float32 &&
           param.filter_type.enumv() == DTypeEnum::Float32 &&
           param.dst_type.enumv() == DTypeEnum::Float32 &&
           fm.dilation[0] == 1 && fm.dilation[1] == 1 && fm.spatial[0] <= 7 &&
           fm.stride[0] == 1 && fm.stride[1] == 1;
}
WorkspaceBundle ConvBiasImpl::AlgoDirect::get_bundle(
        const NCBKernSizeParam& param) const {
    auto&& fm = param.filter_meta;
    size_t nr_threads = param.nr_threads;
    size_t group = fm.group, batch = param.n;
    auto IC = fm.icpg, IH = param.isz[0], IW = param.isz[1];
    auto FH = fm.spatial[0], FW = fm.spatial[1];
    auto OH = param.osz[0], OW = param.osz[1];
    size_t OH2, OW2, IH2, IW2;
    get_rectified_img_size(IH, IW, FH, FW, OH, OW, fm.padding[0], fm.padding[1],
                           IH2, IW2, OH2, OW2);
    size_t part0 = 0u, part1 = 0u;
    bool large_group = group >= param.nr_threads;
    if (IH != IH2 || IW != IW2) {
        part0 = large_group ? IC * IH2 * IW2 * sizeof(float) * nr_threads
                            : IC * IH2 * IW2 * sizeof(float) * group * batch;
    }
    if (OH != OH2 || OW != OW2) {
        part1 = OH2 * OW2 * sizeof(float) * nr_threads;
    }
    return {nullptr, {part0, part1}};
}
size_t ConvBiasImpl::AlgoDirect::get_workspace(
        const NCBKernSizeParam& param) const {
    return get_bundle(param).total_size_in_bytes();
}

//! Process one input channel copy padding
void ConvBiasImpl::AlgoDirect::copy_padding_kern(
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
    size_t FH = kern_param.filter_meta.spatial[0];
    size_t FW = kern_param.filter_meta.spatial[1];
    size_t GROUP = kern_param.filter_meta.group;
    size_t OH2, OW2, IH2, IW2;
    get_rectified_img_size(IH, IW, FH, FW, OH, OW, PH, PW, IH2, IW2, OH2, OW2);
    bool rectify_src = (IH != IH2 || IW != IW2);
    size_t padding_group_size = IH2 * IW2 * IC;
    size_t batch_id = ncb_index.ndrange_id[1];
    size_t group_id = ncb_index.ndrange_id[0];
    size_t channel_id = workspace_ids[2];
    const float* sptr = static_cast<const float*>(
                                kern_param.src<float>(batch_id, group_id)) +
                        channel_id * IH * IW;

    //! Used for get the workspace offset
    size_t workspace_group_id = workspace_ids[0],
           workspace_batch_id = workspace_ids[1],
           workspace_channel_id = workspace_ids[2];
    //! If large group, each thread has its own worspace, set group_id with
    //! thread_id
    if (rectify_src) {
        //! copy to sptr_base to eliminate padding effect
        float* sptr_base = static_cast<float*>(bundle.get(0)) +
                           workspace_group_id * padding_group_size +
                           workspace_batch_id * GROUP * padding_group_size +
                           workspace_channel_id * IH2 * IW2;
        std::memset(sptr_base, 0, sizeof(float) * IH2 * IW2);
        rep(ih, IH) {
            std::memcpy(sptr_base + (ih + PH) * IW2 + PW, sptr + ih * IW,
                        sizeof(float) * IW);
        }
    }
};
#define DISPATCH                                                \
    if (is_supported(SIMDType::FMA)) {                          \
        DISPATCH_SIMD(fma)                                      \
    } else if (is_supported(SIMDType::AVX)) {                   \
        DISPATCH_SIMD(avx)                                      \
    } else if (is_supported(SIMDType::SSE)) {                   \
        DISPATCH_SIMD(sse)                                      \
    } else {                                                    \
        megdnn_throw(megdnn_mangle("no fma/avx/sse detected")); \
    }

#define DISPATCH_SIMD(simd)             \
    if (is_xcorr) {                     \
        DISPATCH_SIMD_MODE(simd, xcorr) \
    } else {                            \
        DISPATCH_SIMD_MODE(simd, conv)  \
    }

#define DISPATCH_SIMD_MODE(simd, mode)                              \
    switch (FH) {                                                   \
        case 1:                                                     \
            DISPATCH_SIMD_MODE_FSIZE(simd, mode, 1);                \
            break;                                                  \
        case 2:                                                     \
            DISPATCH_SIMD_MODE_FSIZE(simd, mode, 2);                \
            break;                                                  \
        case 3:                                                     \
            DISPATCH_SIMD_MODE_FSIZE(simd, mode, 3);                \
            break;                                                  \
        case 4:                                                     \
            DISPATCH_SIMD_MODE_FSIZE(simd, mode, 4);                \
            break;                                                  \
        case 5:                                                     \
            DISPATCH_SIMD_MODE_FSIZE(simd, mode, 5);                \
            break;                                                  \
        case 6:                                                     \
            DISPATCH_SIMD_MODE_FSIZE(simd, mode, 6);                \
            break;                                                  \
        case 7:                                                     \
            DISPATCH_SIMD_MODE_FSIZE(simd, mode, 7);                \
            break;                                                  \
        default:                                                    \
            megdnn_throw(megdnn_mangle("unsupported filter size")); \
    }

#define DISPATCH_SIMD_MODE_FSIZE(simd, mode, fsize) \
    func = detail::convolution_##mode##_fh##fsize##_##simd;

//! compute one output channel
void ConvBiasImpl::AlgoDirect::do_conv_kern(const WorkspaceBundle& bundle,
                                            const NCBKernParam& kern_param,
                                            const NCBKernIndex& ncb_index,
                                            const CpuNDRange& workspace_ids) {
    size_t OH = kern_param.osz[0];
    size_t OW = kern_param.osz[1];
    size_t IH = kern_param.isz[0];
    size_t IW = kern_param.isz[1];
    size_t FH = kern_param.filter_meta.spatial[0];
    size_t FW = kern_param.filter_meta.spatial[1];
    size_t IC = kern_param.filter_meta.icpg;
    size_t PH = kern_param.filter_meta.padding[0];
    size_t PW = kern_param.filter_meta.padding[1];
    auto is_xcorr = !kern_param.filter_meta.should_flip;
    size_t GROUP = kern_param.filter_meta.group;
    size_t OH2, OW2, IH2, IW2;
    get_rectified_img_size(IH, IW, FH, FW, OH, OW, PH, PW, IH2, IW2, OH2, OW2);
    bool rectify_src = (IH != IH2 || IW != IW2);
    bool rectify_dst = (OH != OH2 || OW != OW2);
    size_t padding_group_size = IH2 * IW2 * IC;
    //! Choose the compute kernel
    std::function<void(const float*, const float*, float*, size_t, size_t,
                       size_t, size_t, size_t)>
            func = nullptr;
    DISPATCH;

    size_t bias_offset = 0;
    if (kern_param.bias_mode == megdnn::BiasMode::BIAS) {
        bias_offset = OH * OW;
    } else if (kern_param.bias_mode ==
               megdnn::BiasMode::BROADCAST_CHANNEL_BIAS) {
        bias_offset = 1_z;
    }
    size_t group_id = ncb_index.ndrange_id[0];
    size_t batch_id = ncb_index.ndrange_id[1];
    //! Used for get the workspace offset
    size_t workspace_group_id = workspace_ids[0],
           workspace_batch_id = workspace_ids[1], oc = workspace_ids[2];
    const float* sptr = kern_param.src<float>(batch_id, group_id);
    const float* filter =
            kern_param.filter<float>(group_id) + oc * FH * FW * IC;
    const float* bias_ptr =
            kern_param.bias<float>(batch_id, group_id) + oc * bias_offset;
    float* dst = kern_param.dst<float>(batch_id, group_id) + oc * OH * OW;
    if (rectify_src) {
        sptr = static_cast<float*>(bundle.get(0)) +
               workspace_group_id * padding_group_size +
               workspace_batch_id * GROUP * padding_group_size;
    }
    float* dptr = nullptr;
    if (rectify_dst) {
        dptr = static_cast<float*>(bundle.get(1)) +
               ncb_index.thread_id * OH2 * OW2;
    } else {
        dptr = dst;
    }
    std::memset(dptr, 0, sizeof(float) * OH2 * OW2);
    rep(ic, IC) {
        func(sptr + ic * IH2 * IW2, filter + ic * FH * FW, dptr, IH2, IW2, OH2,
             OW2, FW);
    }
    if (rectify_dst) {
        rep(oh, OH) {
            std::memcpy(dst + oh * OW, dptr + oh * OW2, sizeof(float) * OW);
        }
    }
    PostProcess<dt_float32>::run(dst, const_cast<float*>(bias_ptr), dst,
                                 kern_param.bias_mode, kern_param.nonlineMode,
                                 kern_param.bias_type, kern_param.dst_type, 1_z,
                                 1_z, OH, OW);
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoDirect::get_kimpls(
        const NCBKernSizeParam& param) const {
    GET_KERN;
}
/* ===================== direct-stride2 algo ===================== */
bool ConvBiasImpl::AlgoDirectStride2::usable(const NCBKernSizeParam& param,
                                             AlgoSelectionStrategy) const {
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    return param.filter_meta.format == param::ConvBias::Format::NCHW &&
           param.src_type.enumv() == DTypeEnum::Float32 &&
           param.filter_type.enumv() == DTypeEnum::Float32 &&
           param.dst_type.enumv() == DTypeEnum::Float32 && !fm.should_flip &&
           fm.spatial_ndim == 2 && fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
           fm.stride[0] == 2 && fm.stride[1] == 2 && FH == fm.spatial[1] &&
           (FH == 2 || FH == 3 || FH == 5 || FH == 7);
}

WorkspaceBundle ConvBiasImpl::AlgoDirectStride2::get_bundle(
        const NCBKernSizeParam& param) const {
    UNPACK_CONV_F32_NCB_KERN_SIZES(param);
    MEGDNN_MARK_USED_VAR(N);
    MEGDNN_MARK_USED_VAR(OC);
    MEGDNN_MARK_USED_VAR(SH);
    MEGDNN_MARK_USED_VAR(SW);
    size_t nr_threads = param.nr_threads;
    size_t group = param.filter_meta.group;
    size_t batch = param.n;
    size_t src_size = 0, dst_size = 0;
    size_t IH2, IW2, OH2, OW2;
    get_rectified_size(IH, IW, OH, OW, FH, FW, PH, PW, IH2, IW2, OH2, OW2);
    bool large_group = group >= param.nr_threads;                              \
    if (need_src_copy(param)) {
        src_size = large_group ? IC * IH2 * IW2 * sizeof(float) * nr_threads
                               : IC * IH2 * IW2 * sizeof(float) * group * batch;
    }
    if (need_dst_copy(param)) {
        // we only need one dst plane
        dst_size = OH2 * OW2 * sizeof(float) * nr_threads;
    }
    return WorkspaceBundle(nullptr, {src_size, dst_size});
}

size_t ConvBiasImpl::AlgoDirectStride2::get_workspace(
        const NCBKernSizeParam& param) const {
    return get_bundle(param).total_size_in_bytes();
}
//! Process one input channel copy padding
void ConvBiasImpl::AlgoDirectStride2::copy_padding_kern(
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
    size_t FH = kern_param.filter_meta.spatial[0];
    size_t FW = kern_param.filter_meta.spatial[1];
    size_t GROUP = kern_param.filter_meta.group;
    size_t OH2, OW2, IH2, IW2;
    get_rectified_size(IH, IW, OH, OW, FH, FW, PH, PW, IH2, IW2, OH2, OW2);
    bool rectify_src = need_src_copy(kern_param);
    size_t padding_group_size = IH2 * IW2 * IC;
    size_t group_id = ncb_index.ndrange_id[0];
    size_t batch_id = ncb_index.ndrange_id[1];
    size_t channel_id = workspace_ids[2];
    const float* sptr = static_cast<const float*>(
                                kern_param.src<float>(batch_id, group_id)) +
                        channel_id * IH * IW;
    //! Used for get the workspace offset
    size_t workspace_group_id = workspace_ids[0],
           workspace_batch_id = workspace_ids[1],
           workspace_channel_id = workspace_ids[2];
    if (rectify_src) {
        //! copy to sptr_base to eliminate padding effect
        float* sptr_base = static_cast<float*>(bundle.get(0)) +
                           workspace_group_id * padding_group_size +
                           workspace_batch_id * GROUP * padding_group_size +
                           workspace_channel_id * IH2 * IW2;
        std::memset(sptr_base, 0, sizeof(float) * IH2 * IW2);
        rep(ih, IH) {
            std::memcpy(sptr_base + (ih + PH) * IW2 + PW, sptr + ih * IW,
                        sizeof(float) * IW);
        }
    }
};

//! compute one output channel
void ConvBiasImpl::AlgoDirectStride2::do_conv_kern(
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
    get_rectified_size(IH, IW, OH, OW, FH, FW, PH, PW, IH2, IW2, OH2, OW2);
    bool rectify_src = need_src_copy(kern_param);
    bool rectify_dst = need_dst_copy(kern_param);
    size_t padding_group_size = IH2 * IW2 * IC;
    //! Choose the compute kernel
    using Func = std::function<void(const float*, const float*, float*, size_t,
                                    size_t, size_t, size_t, size_t, size_t)>;
    Func func_no_add_dst = nullptr, func_add_dst = nullptr;
    if (FH == 2) {
        func_no_add_dst = conv_general_simd::do_conv_2x2_stride2<false>;
        func_add_dst = conv_general_simd::do_conv_2x2_stride2<true>;
    } else if (FH == 3) {
        func_no_add_dst = conv_general_simd::do_conv_3x3_stride2<false>;
        func_add_dst = conv_general_simd::do_conv_3x3_stride2<true>;
    } else if (FH == 5) {
        func_no_add_dst = conv_general_simd::do_conv_5x5_stride2<false>;
        func_add_dst = conv_general_simd::do_conv_5x5_stride2<true>;
    } else if (FH == 7) {
        func_no_add_dst = conv_general_simd::do_conv_7x7_stride2<false>;
        func_add_dst = conv_general_simd::do_conv_7x7_stride2<true>;
    }

    size_t bias_offset = 0;
    if (kern_param.bias_mode == megdnn::BiasMode::BIAS) {
        bias_offset = OH * OW;
    } else if (kern_param.bias_mode ==
               megdnn::BiasMode::BROADCAST_CHANNEL_BIAS) {
        bias_offset = 1_z;
    }
    size_t group_id = ncb_index.ndrange_id[0];
    size_t batch_id = ncb_index.ndrange_id[1];
    //! Used for get the workspace offset
    size_t workspace_group_id = workspace_ids[0],
           workspace_batch_id = workspace_ids[1], oc = workspace_ids[2];
    const float* sptr = kern_param.src<float>(batch_id, group_id);
    const float* filter =
            kern_param.filter<float>(group_id) + oc * FH * FW * IC;
    const float* bias_ptr =
            kern_param.bias<float>(batch_id, group_id) + oc * bias_offset;
    float* dst = kern_param.dst<float>(batch_id, group_id) + oc * OH * OW;
    if (rectify_src) {
        sptr = static_cast<float*>(bundle.get(0)) +
               workspace_group_id * padding_group_size +
               workspace_batch_id * GROUP * padding_group_size;
    }
    float* dptr = nullptr;
    if (rectify_dst) {
        dptr = static_cast<float*>(bundle.get(1)) +
               ncb_index.thread_id * OH2 * OW2;
    } else {
        dptr = dst;
    }
    func_no_add_dst(sptr, filter, dptr, IH2, IW2, OH2, OW2, 0, 0);
    for (size_t ic = 1; ic < IC; ++ic) {
        func_add_dst(sptr + ic * IH2 * IW2, filter + ic * FH * FW, dptr, IH2,
                     IW2, OH2, OW2, 0, 0);
    }
    if (rectify_dst) {
        rep(oh, OH) {
            std::memcpy(dst + oh * OW, dptr + oh * OW2, sizeof(float) * OW);
        }
    }
    PostProcess<dt_float32>::run(dst, const_cast<float*>(bias_ptr), dst,
                                 kern_param.bias_mode, kern_param.nonlineMode,
                                 kern_param.bias_type, kern_param.dst_type, 1_z,
                                 1_z, OH, OW);
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoDirectStride2::get_kimpls(
        const NCBKernSizeParam& param) const {
    GET_KERN;
}

#if MEGDNN_X86_WITH_MKL_DNN
static inline void mkldnn_fp32_conv_instance(
        const ConvBiasImpl::NCBKernParam& param, const uint32_t ocpg,
        const uint32_t icpg, const uint32_t group, const uint32_t in,
        const uint32_t ic, const uint32_t oc, const uint32_t ih,
        const uint32_t iw, const uint32_t kh, const uint32_t kw,
        const uint32_t pad_h, const uint32_t pad_w, const uint32_t stride_h,
        const uint32_t stride_w, const uint32_t oh, const uint32_t ow,
        std::vector<dnnl::primitive>& net,
        std::vector<std::unordered_map<int, dnnl::memory>>& net_args,
        dnnl::engine& eng_mkldnn) {
    dnnl::memory::dims src_shape = {in, ic, ih, iw};
    dnnl::memory::dims weight_shape = {oc, ic, kh, kw};
    dnnl::memory::dims bias_shape = {oc};
    dnnl::memory::dims dst_shape = {in, oc, oh, ow};
    dnnl::memory::dims strides_shape = {stride_h, stride_w};
    dnnl::memory::dims padding_shape = {pad_h, pad_w};

    auto user_src_desc =
            dnnl::memory::desc({src_shape}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::nChw8c);
    if (group == 1 && ic < 8) {
        user_src_desc =
                dnnl::memory::desc({src_shape}, dnnl::memory::data_type::f32,
                                   dnnl::memory::format_tag::nchw);
    }
    auto user_src_mem = dnnl::memory(user_src_desc, eng_mkldnn,
                                     const_cast<void*>(param.src_ptr));

    auto weight_tag = dnnl::memory::format_tag::OIhw8i8o;
    if (group > 1) {
        weight_shape = {group, ocpg, icpg, kh, kw};
        if (oc == group && ic == group) {
            weight_tag = dnnl::memory::format_tag::Goihw8g;
        } else {
            weight_tag = dnnl::memory::format_tag::gOIhw8i8o;
        }
    } else if (group == 1 && ic < 8) {
        weight_tag = dnnl::memory::format_tag::Ohwi8o;
    }

    auto user_weights_desc = dnnl::memory::desc(
            {weight_shape}, dnnl::memory::data_type::f32, weight_tag);

    auto user_weights_mem = dnnl::memory(user_weights_desc, eng_mkldnn,
                                         const_cast<void*>(param.filter_ptr));
    auto user_bias_desc = dnnl::memory::desc();
    if (param.bias_mode == megdnn::BiasMode::BROADCAST_CHANNEL_BIAS) {
        user_bias_desc =
                dnnl::memory::desc({bias_shape}, dnnl::memory::data_type::f32,
                                   dnnl::memory::format_tag::x);
    }
    auto user_bias_mem = dnnl::memory(user_bias_desc, eng_mkldnn,
                                      const_cast<void*>(param.bias_ptr));
    auto user_dst_desc =
            dnnl::memory::desc({dst_shape}, dnnl::memory::data_type::f32,
                               dnnl::memory::format_tag::nChw8c);
    auto user_dst_mem = dnnl::memory(user_dst_desc, eng_mkldnn,
                                     const_cast<void*>(param.dst_ptr));
    auto conv_desc = dnnl::convolution_forward::desc(
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::convolution_auto, user_src_mem.get_desc(),
            user_weights_mem.get_desc(), user_bias_mem.get_desc(),
            user_dst_mem.get_desc(), strides_shape, padding_shape,
            padding_shape);

    dnnl::primitive_attr attr;
    if ((param.nonlineMode == NonlineMode::RELU ||
         param.nonlineMode == NonlineMode::SIGMOID) &&
        (param.bias_mode == megdnn::BiasMode::NO_BIAS ||
         param.bias_mode == megdnn::BiasMode::BROADCAST_CHANNEL_BIAS)) {
        auto post_tag = dnnl::algorithm::eltwise_linear;
        switch (param.nonlineMode) {
            case NonlineMode::RELU:
                post_tag = dnnl::algorithm::eltwise_relu;
                break;
            case NonlineMode::SIGMOID:
                post_tag = dnnl::algorithm::eltwise_logistic;
                break;
            default:
                megdnn_assert(0, "not supported nonline mode %d\n",
                              static_cast<int>(param.nonlineMode));
        }
        dnnl::post_ops ops;
        ops.append_eltwise(1.f, post_tag, 0.f, 0.f);
        attr.set_post_ops(ops);
    }

    auto conv_prim_desc = dnnl::convolution_forward::primitive_desc(
            conv_desc, attr, eng_mkldnn);

    net.push_back(dnnl::convolution_forward(conv_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, user_src_mem},
                        {DNNL_ARG_WEIGHTS, user_weights_mem},
                        {DNNL_ARG_BIAS, user_bias_mem},
                        {DNNL_ARG_DST, user_dst_mem}});
}

namespace {
struct NCBKernParamEqual {
    bool operator()(const fallback::ConvBiasImpl::NCBKernParam& x,
                    const fallback::ConvBiasImpl::NCBKernParam& y) const {
        bool flag = true;
        flag = flag && (x.src_ptr == y.src_ptr);
        flag = flag && (x.dst_ptr == y.dst_ptr);
        flag = flag && (x.filter_ptr == y.filter_ptr);
        flag = flag && (x.bias_ptr == y.bias_ptr);
        flag = flag && (x.isz == y.isz);
        flag = flag && (x.osz == y.osz);
        flag = flag && (x.src_type == y.src_type);
        flag = flag && (x.dst_type == y.dst_type);
        flag = flag && (x.filter_type == y.filter_type);
        flag = flag && (x.bias_type == y.bias_type);
        flag = flag && (x.filter_meta == y.filter_meta);
        flag = flag && (x.n == y.n);
        flag = flag && (x.bias_mode == y.bias_mode);
        flag = flag && (x.nonlineMode == y.nonlineMode);
        flag = flag && (x.bias_bs == y.bias_bs);
        return flag;
    };
};

struct NCBKernParamHash {
    std::size_t operator()(
            const fallback::ConvBiasImpl::NCBKernParam& param) const {
        std::size_t result = reinterpret_cast<std::size_t>(param.filter_ptr);
        result = result ^ (reinterpret_cast<std::size_t>(param.src_ptr) << 3);
        result = result ^ (reinterpret_cast<std::size_t>(param.dst_ptr) << 7);
        result = result ^ (static_cast<std::size_t>(param.n) << 11);
        return result;
    };
};

}  // namespace

void ConvBiasImpl::AlgoMkldnnConv::kern_mkldnn_fp32(const NCBKernParam& param,
                                                    const NCBKernIndex&) {
    const NCBKernParam& key = param;
    static std::unordered_map<NCBKernParam, std::vector<dnnl::primitive>,
                              NCBKernParamHash, NCBKernParamEqual>
            kern_net_map;
    static std::unordered_map<
            NCBKernParam, std::vector<std::unordered_map<int, dnnl::memory>>,
            NCBKernParamHash, NCBKernParamEqual>
            kern_net_arg_map;

    auto x86_handle = static_cast<HandleImpl*>(inplace_cpu_handle().get());
    megdnn_assert(x86_handle != nullptr, "x86 handle can not be null");
    auto eng_mkldnn = x86_handle->mkldnn_engine();
    auto stream_mkldnn = x86_handle->mkldnn_stream();
    auto&& fm = param.filter_meta;
    const uint32_t group = fm.group;
    const uint32_t in = param.n;
    const uint32_t ic = fm.icpg * group;
    const uint32_t oc = fm.ocpg * group;
    const uint32_t ih = param.isz[0];
    const uint32_t iw = param.isz[1];
    const uint32_t kh = fm.spatial[0];
    const uint32_t kw = fm.spatial[1];
    const uint32_t pad_h = fm.padding[0];
    const uint32_t pad_w = fm.padding[1];
    const uint32_t stride_h = fm.stride[0];
    const uint32_t stride_w = fm.stride[1];
    const uint32_t oh = param.osz[0];
    const uint32_t ow = param.osz[1];

    if (kern_net_map.find(key) == kern_net_map.end()) {
        std::vector<dnnl::primitive> net;
        std::vector<std::unordered_map<int, dnnl::memory>> net_args;
        mkldnn_fp32_conv_instance(param, fm.ocpg, fm.icpg, group, in, ic, oc,
                                  ih, iw, kh, kw, pad_h, pad_w, stride_h,
                                  stride_w, oh, ow, net, net_args, eng_mkldnn);
        kern_net_map[key] = net;
        kern_net_arg_map[key] = net_args;
    }

    const auto& net = kern_net_map[key];
    const auto& net_args = kern_net_arg_map[key];
    for (size_t i = 0; i < net.size(); ++i) {
        net.at(i).execute(stream_mkldnn, net_args.at(i));
    }
    stream_mkldnn.wait();

    if ((param.bias_mode == megdnn::BiasMode::NO_BIAS ||
         param.bias_mode == megdnn::BiasMode::BROADCAST_CHANNEL_BIAS) &&
        (param.nonlineMode != NonlineMode::IDENTITY &&
         param.nonlineMode != NonlineMode::RELU &&
         param.nonlineMode != NonlineMode::SIGMOID)) {
        /**
         *NO_BIAS and BROADCAST_CHANNEL_BIAS has be done in mkldnn conv, but
         *it is necessary to do activition function not supported by mkldnn.
         *do not need any bias op
         **/
        PostProcess<float>::run(
                param.dst_ptr, const_cast<void*>(param.bias_ptr), param.dst_ptr,
                megdnn::BiasMode::NO_BIAS, param.nonlineMode, param.bias_type,
                param.dst_type, in, oc, oh, ow);
    } else if (param.bias_mode == megdnn::BiasMode::BIAS) {
        PostProcess<float>::run(
                param.dst_ptr, const_cast<void*>(param.bias_ptr), param.dst_ptr,
                param.bias_mode, param.nonlineMode, param.bias_type,
                param.dst_type, in, oc, oh, ow);
    }
}
#endif

// vim: syntax=cpp.doxygen
