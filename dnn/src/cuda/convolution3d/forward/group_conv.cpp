/**
 * \file dnn/src/cuda/convolution3d/forward/group_conv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algo.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution3d;

void Convolution3DForwardImpl::AlgoGroupConvGeneral::modify_size_args(
        Convolution3DForwardImpl::AlgoBase::SizeArgs &args,
        TensorLayout &src_pg, TensorLayout &dst_pg) {
    src_pg = *args.src_layout;
    dst_pg = *args.dst_layout;
    auto nr_grp = args.filter_meta.group;
    args.filter_meta.group = 1;
    size_t c_pos;
    if (args.filter_meta.format == Param::Format::NCDHW) {
        c_pos = 1;
    } else {
        megdnn_assert(args.filter_meta.format == Param::Format::NDHWC,
                "invalid conv format");
        c_pos = 4;
    }
    src_pg.shape[c_pos] /= nr_grp;
    dst_pg.shape[c_pos] /= nr_grp;
    args.src_layout = &src_pg;
    args.dst_layout = &dst_pg;
}

Convolution3DForwardImpl::AlgoGroupConvGeneral::AlgoGroupConvGeneral(
        AlgoBase *impl):
    m_impl{impl} {
    m_name = "group_conv3d:";
    m_name += impl->name();
}

bool Convolution3DForwardImpl::AlgoGroupConvGeneral::is_available(
        const SizeArgs &args) const {
    auto sub_args = args;
    TensorLayout src_pg, dst_pg;
    modify_size_args(sub_args, src_pg, dst_pg);
    return m_impl->is_available(sub_args);
}

size_t Convolution3DForwardImpl::AlgoGroupConvGeneral::get_workspace_in_bytes(
        const SizeArgs &args) const {
    auto sub_args = args;
    TensorLayout src_pg, dst_pg;
    modify_size_args(sub_args, src_pg, dst_pg);
    return m_impl->get_workspace_in_bytes(sub_args);
}

void Convolution3DForwardImpl::AlgoGroupConvGeneral::exec(
        const ExecArgs &args) const {
    auto sub_args = args;
    TensorND tsrc{*args.src_tensor}, tdst{*args.dst_tensor},
             tflt{*args.filter_tensor};
    modify_size_args(sub_args, tsrc.layout, tdst.layout);
    sub_args.src_tensor = &tsrc;
    sub_args.dst_tensor = &tdst;
    sub_args.filter_tensor = &tflt;

    size_t c_pos;
    if (args.filter_meta.format == Param::Format::NCDHW) {
        c_pos = 1;
    } else {
        megdnn_assert(args.filter_meta.format == Param::Format::NDHWC,
                "invalid conv format");
        c_pos = 4;
    }

    auto grp = args.filter_meta.group;

    auto &&fm = args.filter_meta;
    auto strd_src = tsrc.layout.stride[c_pos] * fm.icpg * tsrc.layout.dtype.size(),
         strd_dst = tdst.layout.stride[c_pos] * fm.ocpg * tdst.layout.dtype.size(),
         strd_flt = fm.icpg * fm.ocpg *
             fm.spatial[0] * fm.spatial[1] * fm.spatial[2] * 
             tflt.layout.dtype.size();
    for (uint32_t g = 0; g < grp; ++ g) {
        m_impl->exec(sub_args);
        incr_voidp(tsrc.raw_ptr, strd_src);
        incr_voidp(tdst.raw_ptr, strd_dst);
        incr_voidp(tflt.raw_ptr, strd_flt);
    }
}

// vim: syntax=cpp.doxygen

