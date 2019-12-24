/**
 * \file dnn/src/cuda/convolution/backward_data/group_conv.cpp
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

void ConvolutionBackwardDataImpl::AlgoGroupConvGeneral::modify_size_args(
        ConvolutionBackwardDataImpl::AlgoBase::SizeArgs &args,
        TensorLayout &diff_pg, TensorLayout &grad_pg) {
    diff_pg = *args.diff_layout;
    grad_pg = *args.grad_layout;
    auto nr_grp = args.filter_meta.group;
    args.filter_meta.group = 1;
    diff_pg.shape[1] /= nr_grp;
    grad_pg.shape[1] /= nr_grp;
    args.diff_layout = &diff_pg;
    args.grad_layout = &grad_pg;
}

ConvolutionBackwardDataImpl::AlgoGroupConvGeneral::AlgoGroupConvGeneral(
        AlgoBase *impl):
    m_impl{impl}
{
    m_name = "group_conv:";
    m_name += impl->name();
}

bool ConvolutionBackwardDataImpl::AlgoGroupConvGeneral::is_available(
        const SizeArgs &args) const {
    if (args.diff_layout->dtype == args.filter_layout->dtype &&
        args.diff_layout->dtype == dtype::BFloat16()) {
        return false;
    }
    auto sub_args = args;
    TensorLayout diff_pg, grad_pg;
    modify_size_args(sub_args, diff_pg, grad_pg);
    return m_impl->is_available(sub_args);
}

size_t ConvolutionBackwardDataImpl::AlgoGroupConvGeneral::
get_workspace_in_bytes(const SizeArgs &args) const {
    auto sub_args = args;
    TensorLayout diff_pg, grad_pg;
    modify_size_args(sub_args, diff_pg, grad_pg);
    return m_impl->get_workspace_in_bytes(sub_args);
}

void ConvolutionBackwardDataImpl::AlgoGroupConvGeneral::exec(
        const ExecArgs &args) const {
    auto sub_args = args;
    TensorND tflt{*args.filter_tensor}, tdiff{*args.diff_tensor},
             tgrad{*args.grad_tensor};
    modify_size_args(sub_args, tdiff.layout, tgrad.layout);
    sub_args.filter_tensor = &tflt;
    sub_args.diff_tensor = &tdiff;
    sub_args.grad_tensor = &tgrad;
    auto grp = args.filter_meta.group;

    auto &&fm = args.filter_meta;
    auto strd_flt = (fm.icpg * fm.ocpg *
            fm.spatial[0] * fm.spatial[1] * tflt.layout.dtype.size()),
         strd_diff = (
                 tdiff.layout.stride[1] * fm.ocpg * tdiff.layout.dtype.size()),
         strd_grad = (
                 tgrad.layout.stride[1] * fm.icpg * tgrad.layout.dtype.size());
    for (uint32_t g = 0; g < grp; ++ g) {
        m_impl->exec(sub_args);
        incr_voidp(tflt.raw_ptr, strd_flt);
        incr_voidp(tdiff.raw_ptr, strd_diff);
        incr_voidp(tgrad.raw_ptr, strd_grad);
    }
}

// vim: syntax=cpp.doxygen

