/**
 * \file dnn/src/cuda/convolution/backward_filter/group_conv.cpp
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
using namespace convolution;

void ConvolutionBackwardFilterImpl::AlgoGroupConvGeneral::modify_size_args(
        ConvolutionBackwardFilterImpl::AlgoBase::SizeArgs &args,
        TensorLayout &src_pg, TensorLayout &diff_pg) {
    src_pg = *args.src_layout;
    diff_pg = *args.diff_layout;
    auto nr_grp = args.grad_filter_meta.group;
    args.grad_filter_meta.group = 1;
    src_pg.shape[1] /= nr_grp;
    diff_pg.shape[1] /= nr_grp;
    args.src_layout = &src_pg;
    args.diff_layout = &diff_pg;
}

ConvolutionBackwardFilterImpl::AlgoGroupConvGeneral::AlgoGroupConvGeneral(
        AlgoBase *impl):
    m_impl{impl}
{
    m_name = "group_conv:";
    m_name += impl->name();
}

bool ConvolutionBackwardFilterImpl::AlgoGroupConvGeneral::is_available(
        const SizeArgs &args) const {
    if (args.src_layout->dtype == args.src_layout->dtype &&
        args.diff_layout->dtype == dtype::BFloat16()) {
        return false;
    }
    auto sub_args = args;
    TensorLayout src_pg, diff_pg;
    modify_size_args(sub_args, src_pg, diff_pg);
    return m_impl->is_available(sub_args);
}

size_t ConvolutionBackwardFilterImpl::AlgoGroupConvGeneral::
get_workspace_in_bytes(const SizeArgs &args) const {
    auto sub_args = args;
    TensorLayout src_pg, diff_pg;
    modify_size_args(sub_args, src_pg, diff_pg);
    return m_impl->get_workspace_in_bytes(sub_args);
}

void ConvolutionBackwardFilterImpl::AlgoGroupConvGeneral::exec(
        const ExecArgs &args) const {
    auto sub_args = args;
    TensorND tsrc{*args.src_tensor}, tdiff{*args.diff_tensor},
             tgrad{*args.grad_tensor};
    modify_size_args(sub_args, tsrc.layout, tdiff.layout);
    sub_args.src_tensor = &tsrc;
    sub_args.diff_tensor = &tdiff;
    sub_args.grad_tensor = &tgrad;

    auto &&fm = args.grad_filter_meta;
    auto grp = fm.group;

    auto strd_src = (
                 tsrc.layout.stride[1] * fm.icpg * tsrc.layout.dtype.size()),
         strd_diff = (
                 tdiff.layout.stride[1] * fm.ocpg * tdiff.layout.dtype.size()),
         strd_grad = (fm.icpg * fm.ocpg *
                 fm.spatial[0] * fm.spatial[1] * tgrad.layout.dtype.size());
    for (uint32_t g = 0; g < grp; ++ g) {
        m_impl->exec(sub_args);
        incr_voidp(tsrc.raw_ptr, strd_src);
        incr_voidp(tdiff.raw_ptr, strd_diff);
        incr_voidp(tgrad.raw_ptr, strd_grad);
    }
}

// vim: syntax=cpp.doxygen

