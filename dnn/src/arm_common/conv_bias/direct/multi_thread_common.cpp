/**
 * \file dnn/src/arm_common/conv_bias/direct/multi_thread_common.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/direct/multi_thread_common.h"
#include "src/arm_common/conv_bias/postprocess_helper.h"
#include "src/fallback/matrix_mul/opr_impl.h"

using namespace megdnn;
using namespace arm_common;

namespace {
bool need_dst_copy(
        const megdnn::fallback::ConvBiasImpl::NCBKernSizeParam& param) {
    auto align = param.src_type.enumv() == DTypeEnum::Float32 ? 4 : 8;
    return param.osz[1] % align;
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
        size_t IH, size_t IW, size_t OH, size_t OW, size_t FH, size_t FW,
        size_t PH, size_t PW, size_t& IH2, size_t& IW2, size_t& OW2) {
    MEGDNN_MARK_USED_VAR(PW);
    MEGDNN_MARK_USED_VAR(PH);
    auto&& fm = param.filter_meta;
    auto SW = fm.stride[1];

    auto Align = param.src_type.enumv() == DTypeEnum::Float32 ? 3 : 7;
    OW2 = (OW + Align) & ~Align;
    IH2 = SW * OH + FH - SW;
    IW2 = SW * OW2 + FW - SW;
    // Because stride is 2, sometimes IW == IW2+1. Do a max update to
    // handle this case.
    IH2 = std::max(IH2, IH);
    IW2 = std::max(IW2, IW);
}

}  // namespace

template <typename io_ctype, typename compute_ctype>
WorkspaceBundle MultithreadDirectConvCommon<io_ctype, compute_ctype>::get_bundle(
        const ConvBiasImpl::NCBKernSizeParam& param, bool m_large_group) {
    auto&& fm = param.filter_meta;
    size_t nr_threads = param.nr_threads;
    size_t group = fm.group, batch = param.n;
    size_t IH2 = param.isz[0] + 2 * fm.padding[0];
    size_t IW2 = param.isz[1] + 2 * fm.padding[1];
    // part0: copied src
    // part1: copied filter
    size_t part0, part1;
    if (fm.padding[0] == 0 && fm.padding[1] == 0) {
        //! only the last plane need to be copied, add 16 Byte extra space in
        //! case of invalid read and write
        part0 = (param.isz[0] * param.isz[1]) * sizeof(io_ctype) + 16;
    } else if (m_large_group) {
        //! Serial in group, each thread process one group, parallel by group
        part0 = (IH2 * IW2 * fm.icpg * nr_threads) * sizeof(io_ctype) + 16;
    } else {
        //! Parallel in group, Then should copy every inputs to workspace
        part0 = (IH2 * IW2 * fm.icpg * group * batch) * sizeof(io_ctype) + 16;
    }
    if (param.filter_meta.should_flip) {
        if (m_large_group) {
            //! Serial in group, each thread has own workspace and then reuse
            part1 = fm.spatial[0] * fm.spatial[1] * fm.ocpg * fm.icpg *
                    nr_threads * sizeof(io_ctype);
        } else {
            part1 = fm.spatial[0] * fm.spatial[1] * fm.ocpg * fm.icpg * group *
                    sizeof(io_ctype);
        }
    } else {
        part1 = 0;
    }
    return {nullptr, {part0, part1}};
}
template <typename io_ctype, typename compute_ctype>
WorkspaceBundle
MultithreadDirectConvCommon<io_ctype, compute_ctype>::get_bundle_stride(
        const ConvBiasImpl::NCBKernSizeParam& param, bool m_large_group) {
    UNPACK_CONV_F32_NCB_KERN_SIZES(param);
    MEGDNN_MARK_USED_VAR(N);
    MEGDNN_MARK_USED_VAR(OC);
    MEGDNN_MARK_USED_VAR(SH);
    MEGDNN_MARK_USED_VAR(SW);
    auto&& fm = param.filter_meta;
    size_t nr_threads = param.nr_threads;
    size_t group = fm.group, batch = param.n;
    size_t IH2, IW2, OW2;
    get_rectified_size(param, IH, IW, OH, OW, FH, FW, PH, PW, IH2, IW2, OW2);

    size_t src_size = 0, dst_size = 0;
    // src_size: copied src
    // dst_size: copied dst
    if (need_src_copy(param)) {
        src_size = m_large_group
                           ? IC * IH2 * IW2 * sizeof(io_ctype) * nr_threads
                           : IC * IH2 * IW2 * sizeof(io_ctype) * group * batch;
    };
    if (need_dst_copy(param)) {
        //! add 16 Byte extra space in case of invalid read and write
        dst_size = OH * OW2 * sizeof(io_ctype) * nr_threads + 16;
    }
    return {nullptr, {src_size, dst_size}};
}

//! Process one output channel weight flip
template <typename io_ctype, typename compute_ctype>
void MultithreadDirectConvCommon<io_ctype, compute_ctype>::weight_flip_kern(
        const WorkspaceBundle& bundle,
        const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index,
        const CpuNDRange& workspace_ids) {
    size_t FH = kern_param.filter_meta.spatial[0];
    size_t FW = kern_param.filter_meta.spatial[1];
    size_t IC = kern_param.filter_meta.icpg;
    size_t OC = kern_param.filter_meta.ocpg;
    //! Used for get the workspace offset
    size_t workspace_group_id = workspace_ids[0], channel_id = workspace_ids[2],
           group_id = ncb_index.ndrange_id[0];
    const io_ctype* filter =
            kern_param.filter<io_ctype>(group_id) + channel_id * FH * FW * IC;
    io_ctype* filter_flip =
            static_cast<io_ctype*>(bundle.get(1)) +
            (workspace_group_id * IC * OC + channel_id * IC) * FH * FW;
    rep(ic, IC) {
        const io_ctype* filter_plane = filter + ic * FH * FW;
        io_ctype* filter_flip_plane = filter_flip + ic * FH * FW;
        rep(fh, FH) rep(fw, FW) {
            filter_flip_plane[fh * FW + fw] =
                    filter_plane[(FH - fh - 1) * FW + (FW - fw - 1)];
        }
    }
}

//! Process one input channel copy padding
template <typename io_ctype, typename compute_ctype>
void MultithreadDirectConvCommon<io_ctype, compute_ctype>::copy_padding_kern(
        const WorkspaceBundle& bundle,
        const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index,
        const CpuNDRange& workspace_ids) {
    size_t IH = kern_param.isz[0];
    size_t IW = kern_param.isz[1];
    size_t IC = kern_param.filter_meta.icpg;
    size_t PH = kern_param.filter_meta.padding[0];
    size_t PW = kern_param.filter_meta.padding[1];
    size_t IH2 = IH + 2 * PH;
    size_t IW2 = IW + 2 * PW;
    size_t padding_group_size = IH2 * IW2 * IC;
    size_t N = kern_param.n;
    size_t GROUP = kern_param.filter_meta.group;

    //! Used for get the workspace offset
    size_t workspace_group_id = workspace_ids[0],
           workspace_batch_id = workspace_ids[1], channel_id = workspace_ids[2];
    size_t batch_id = ncb_index.ndrange_id[1],
           group_id = ncb_index.ndrange_id[0];
    const io_ctype* sptr = static_cast<const io_ctype*>(
            kern_param.src<io_ctype>(batch_id, group_id, channel_id));
    if (PH > 0 || PW > 0) {
        //! copy to sptr_base to eliminate padding effect
        io_ctype* sptr_base = static_cast<io_ctype*>(bundle.get(0)) +
                              workspace_group_id * padding_group_size +
                              workspace_batch_id * GROUP * padding_group_size +
                              channel_id * IH2 * IW2;
        std::memset(sptr_base, 0, sizeof(io_ctype) * IH2 * IW2);
        rep(ih, IH) {
            std::memcpy(sptr_base + (ih + PH) * IW2 + PW, sptr + ih * IW,
                        sizeof(io_ctype) * IW);
        }
    } else if (batch_id + 1 == N && channel_id + 1 == IC &&
               group_id + 1 == GROUP) {
        //! copy last plane
        io_ctype* sptr_last_c = static_cast<io_ctype*>(bundle.get(0));
        std::memcpy(sptr_last_c, sptr, sizeof(io_ctype) * IH2 * IW2);
    }
};
//! Process one input channel copy padding
template <typename io_ctype, typename compute_ctype>
void MultithreadDirectConvCommon<io_ctype, compute_ctype>::
        copy_padding_kern_stride(const WorkspaceBundle& bundle,
                                 const ConvBiasImpl::NCBKernParam& kern_param,
                                 const ConvBiasImpl::NCBKernIndex& ncb_index,
                                 const CpuNDRange& workspace_ids) {
    size_t IH = kern_param.isz[0];
    size_t IW = kern_param.isz[1];
    size_t IC = kern_param.filter_meta.icpg;
    size_t PH = kern_param.filter_meta.padding[0];
    size_t PW = kern_param.filter_meta.padding[1];
    size_t FH = kern_param.filter_meta.spatial[0];
    size_t FW = kern_param.filter_meta.spatial[1];
    size_t OW = kern_param.osz[1];
    size_t OH = kern_param.osz[0];
    size_t IH2, IW2, OW2;
    size_t GROUP = kern_param.filter_meta.group;
    get_rectified_size(kern_param, IH, IW, OH, OW, FH, FW, PH, PW, IH2, IW2, OW2);
    size_t padding_group_size = IH2 * IW2 * IC;

    //! Used for get the workspace offset
    size_t workspace_group_id = workspace_ids[0],
           workspace_batch_id = workspace_ids[1];
    size_t channel_id = workspace_ids[2], batch_id = ncb_index.ndrange_id[1],
           group_id = ncb_index.ndrange_id[0];

    const io_ctype* sptr = static_cast<const io_ctype*>(
            kern_param.src<io_ctype>(batch_id, group_id, channel_id));
    if (need_src_copy(kern_param)) {
        //! copy to sptr_base to eliminate padding effect
        io_ctype* sptr_base = static_cast<io_ctype*>(bundle.get(0)) +
                              workspace_group_id * padding_group_size +
                              workspace_batch_id * GROUP * padding_group_size +
                              channel_id * IH2 * IW2;
        std::memset(sptr_base, 0, sizeof(io_ctype) * IH2 * IW2);
        rep(ih, IH) {
            std::memcpy(sptr_base + (ih + PH) * IW2 + PW, sptr + ih * IW,
                        sizeof(io_ctype) * IW);
        }
    }
};

//! compute one output channel
template <typename io_ctype, typename compute_ctype>
void MultithreadDirectConvCommon<io_ctype, compute_ctype>::do_conv_kern(
        const WorkspaceBundle& bundle,
        const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index,
        const kern_direct_conv_f32& fun, const CpuNDRange& workspace_ids) {
    size_t OH = kern_param.osz[0];
    size_t OW = kern_param.osz[1];
    size_t FH = kern_param.filter_meta.spatial[0];
    size_t FW = kern_param.filter_meta.spatial[1];
    size_t IC = kern_param.filter_meta.icpg;
    size_t OC = kern_param.filter_meta.ocpg;
    size_t PH = kern_param.filter_meta.padding[0];
    size_t PW = kern_param.filter_meta.padding[1];
    size_t IH2 = kern_param.isz[0] + 2 * PH;
    size_t IW2 = kern_param.isz[1] + 2 * PW;
    size_t padding_group_size = IH2 * IW2 * IC;
    size_t N = kern_param.n;
    size_t GROUP = kern_param.filter_meta.group;

    size_t group_id = ncb_index.ndrange_id[0],
           batch_id = ncb_index.ndrange_id[1];
    size_t channel_id = workspace_ids[2];

    const io_ctype* sptr = kern_param.src<io_ctype>(batch_id, group_id);
    const io_ctype* filter = kern_param.filter<io_ctype>(group_id);
    const io_ctype* bias_ptr =
            kern_param.bias<io_ctype>(batch_id, group_id, channel_id);
    io_ctype* dptr = kern_param.dst<io_ctype>(batch_id, group_id, channel_id);

    //! Used for get the workspace offset
    size_t workspace_batch_id = workspace_ids[1];
    size_t workspace_group_id = workspace_ids[0];

    io_ctype* sptr_base;
    io_ctype* sptr_last_c;
    auto fptr =
            kern_param.filter_meta.should_flip
                    ? static_cast<io_ctype*>(bundle.get(1)) +
                              (workspace_group_id * OC * IC + channel_id * IC) *
                                      FH * FW
                    : filter + channel_id * FH * FW * IC;
    if (PH > 0 || PW > 0) {
        sptr_base = static_cast<io_ctype*>(bundle.get(0)) +
                    workspace_group_id * padding_group_size +
                    workspace_batch_id * GROUP * padding_group_size;
        sptr_last_c = sptr_base + (IC - 1) * IH2 * IW2;
        //! Last batch, last group
    } else if (batch_id + 1 == N && group_id + 1 == GROUP) {
        sptr_base = const_cast<io_ctype*>(sptr);
        sptr_last_c = static_cast<io_ctype*>(bundle.get(0));
    } else {
        sptr_base = const_cast<io_ctype*>(sptr);
        sptr_last_c = sptr_base + (IC - 1) * IH2 * IW2;
    }
    std::memset(dptr, 0, sizeof(io_ctype) * (OH * OW));
    rep(ic, IC) {
        io_ctype* sptr_cur =
                (ic + 1 == IC ? sptr_last_c : sptr_base + ic * IH2 * IW2);
        fun(reinterpret_cast<const compute_ctype*>(sptr_cur),
            reinterpret_cast<const compute_ctype*>(fptr + ic * FH * FW),
            reinterpret_cast<compute_ctype*>(dptr), IH2, IW2, OH, OW, FH, FW);
    }
    PostProcess<compute_ctype>::run(dptr, const_cast<io_ctype*>(bias_ptr), dptr,
                                kern_param.bias_mode, kern_param.nonlineMode,
                                kern_param.bias_type, kern_param.dst_type, 1_z,
                                1_z, OH, OW);
};

//! compute one output channel
template <typename io_ctype, typename compute_ctype>
void MultithreadDirectConvCommon<io_ctype, compute_ctype>::do_conv_kern_stride(
        const WorkspaceBundle& bundle,
        const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index,
        const kern_direct_conv_f32_stride& fun,
        const CpuNDRange& workspace_ids) {
    size_t IH = kern_param.isz[0];
    size_t IW = kern_param.isz[1];
    size_t OH = kern_param.osz[0];
    size_t OW = kern_param.osz[1];
    size_t FH = kern_param.filter_meta.spatial[0];
    size_t FW = kern_param.filter_meta.spatial[1];
    size_t IC = kern_param.filter_meta.icpg;
    size_t PH = kern_param.filter_meta.padding[0];
    size_t PW = kern_param.filter_meta.padding[1];
    size_t IH2, IW2, OW2;
    get_rectified_size(kern_param, IH, IW, OH, OW, FH, FW, PH, PW, IH2, IW2, OW2);

    size_t padding_group_size = IH2 * IW2 * IC;
    size_t GROUP = kern_param.filter_meta.group;

    //! Used for get the workspace offset
    size_t group_id = ncb_index.ndrange_id[0],
           batch_id = ncb_index.ndrange_id[1];
    size_t channel_id = workspace_ids[2];

    const io_ctype* sptr = kern_param.src<io_ctype>(batch_id, group_id);
    const io_ctype* fptr =
            kern_param.filter<io_ctype>(group_id) + channel_id * FH * FW * IC;
    const io_ctype* bias_ptr =
            kern_param.bias<io_ctype>(batch_id, group_id, channel_id);
    io_ctype* dptr = kern_param.dst<io_ctype>(batch_id, group_id, channel_id);

    size_t workspace_batch_id = workspace_ids[1];
    size_t workspace_group_id = workspace_ids[0];

    io_ctype* sptr_base;
    io_ctype* dptr_base;
    if (need_src_copy(kern_param)) {
        sptr_base = static_cast<io_ctype*>(bundle.get(0)) +
                    workspace_group_id * padding_group_size +
                    workspace_batch_id * GROUP * padding_group_size;
    } else {
        sptr_base = const_cast<io_ctype*>(sptr);
    }
    if (need_dst_copy(kern_param)) {
        dptr_base = static_cast<io_ctype*>(bundle.get(1)) +
                    ncb_index.thread_id * OH * OW2;
    } else {
        dptr_base = dptr;
    }
    if (need_dst_copy(kern_param)) {
        std::memset(dptr_base, 0, sizeof(io_ctype) * (OH * OW2));
        fun(reinterpret_cast<const compute_ctype*>(sptr_base),
            reinterpret_cast<const compute_ctype*>(fptr),
            reinterpret_cast<compute_ctype*>(dptr_base), IH2, IW2, OH, OW2, IC);
        copy_plane_in_bytes(dptr, dptr_base, OH, OW * sizeof(io_ctype),
                            OW * sizeof(io_ctype), OW2 * sizeof(io_ctype));
    } else {
        std::memset(dptr_base, 0, sizeof(io_ctype) * (OH * OW));
        fun(reinterpret_cast<const compute_ctype*>(sptr_base),
            reinterpret_cast<const compute_ctype*>(fptr),
            reinterpret_cast<compute_ctype*>(dptr_base), IH2, IW2, OH, OW, IC);
    }
    PostProcess<compute_ctype>::run(dptr, const_cast<io_ctype*>(bias_ptr), dptr,
                                kern_param.bias_mode, kern_param.nonlineMode,
                                kern_param.bias_type, kern_param.dst_type, 1_z,
                                1_z, OH, OW);
};
template class megdnn::arm_common::MultithreadDirectConvCommon<float, float>;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template class megdnn::arm_common::MultithreadDirectConvCommon<dt_float16, __fp16>;
#endif
// vim: syntax=cpp.doxygen
