/**
 * \file dnn/src/x86/conv_bias/int8/chainwise_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "megdnn/arch.h"
#include "src/x86/conv_bias/opr_impl.h"

namespace megdnn {
namespace x86 {
using NCBKern = fallback::ConvBiasImpl::NCBKern;
using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;
using NCBKernParam = fallback::ConvBiasImpl::NCBKernParam;
using NCBKernIndex = fallback::ConvBiasImpl::NCBKernIndex;

static inline bool need_dst_copy(const NCBKernSizeParam& param) {
    return param.osz[1] % 16;
}

static inline bool need_src_copy(const NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    return (fm.padding[0] != 0 || fm.padding[1] != 0) ? true
                                                      : need_dst_copy(param);
}

static inline void get_rectified_size(const NCBKernSizeParam& param,
                                      size_t& IH2, size_t& IW2, size_t& OH2,
                                      size_t& OW2) {
    auto&& fm = param.filter_meta;
    auto SW = fm.stride[1];
    auto OH = param.osz[0];
    auto OW = param.osz[1];
    auto FH = fm.spatial[0];
    auto FW = fm.spatial[1];

    OH2 = OH;
    OW2 = (OW + 15) & ~15;
    IH2 = SW * OH + FH - SW;
    IW2 = SW * OW2 + FW - SW;
}

static inline void copy_padding_kern(
        const WorkspaceBundle& bundle, const ConvBiasImpl::NCBKernParam& kern_param,
        const ConvBiasImpl::NCBKernIndex& ncb_index) {
    size_t IW = kern_param.isz[1];
    size_t IH = kern_param.isz[0];
    size_t PH = kern_param.filter_meta.padding[0];
    size_t PW = kern_param.filter_meta.padding[1];

    size_t IH2, IW2, OH2, OW2;
    get_rectified_size(kern_param, IH2, IW2, OH2, OW2);
    bool need_src_copy_var = need_src_copy(kern_param);
    size_t padding_group_size = IH2 * IW2;

    size_t group_id = ncb_index.ndrange_id[0],
           batch_id = ncb_index.ndrange_id[1],
           channel_id = ncb_index.ndrange_id[2];
    size_t workspace_group_id = ncb_index.thread_id;
    const int8_t* sptr = kern_param.src<int8_t>(batch_id, group_id, channel_id);
    if (need_src_copy_var) {
        int8_t* sptr_base = static_cast<int8_t*>(bundle.get(0)) +
                            workspace_group_id * padding_group_size;
        std::memset(sptr_base, 0, sizeof(int8_t) * IH2 * IW2);
        rep(ih, std::min(IH, IH2)) {
            std::memcpy(sptr_base + (ih + PH) * IW2 + PW, sptr + ih * IW,
                        sizeof(int8_t) * IW);
        }
    }
};

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
