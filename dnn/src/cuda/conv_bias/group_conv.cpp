/**
 * \file dnn/src/cuda/conv_bias/group_conv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/conv_bias.h"
#include "src/cuda/conv_bias/algo.h"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

void ConvBiasForwardImpl::AlgoGroupConvGeneral::modify_size_args(
        ConvBiasForwardImpl::AlgoBase::SizeArgs& args, TensorLayout& src_pg,
        TensorLayout& dst_pg, TensorLayout& bias_pg) {
    src_pg = *args.src_layout;
    dst_pg = *args.dst_layout;
    bias_pg = *args.bias_layout;
    auto nr_grp = args.filter_meta.group;
    args.filter_meta.group = 1;
    size_t c_pos;
    if (args.filter_meta.format == Param::Format::NCHW ||
        args.filter_meta.format == Param::Format::NCHW4) {
        c_pos = 1;
    } else {
        megdnn_assert(args.filter_meta.format == Param::Format::NHWC,
                      "invalid conv format");
        c_pos = 3;
    }
    src_pg.shape[c_pos] /= nr_grp;
    dst_pg.shape[c_pos] /= nr_grp;
    bias_pg.ndim = 0;
    args.src_layout = &src_pg;
    args.dst_layout = &dst_pg;
    args.bias_layout = &bias_pg;
    args.nonlinear_mode = Param::NonlineMode::IDENTITY;
}

ConvBiasForwardImpl::AlgoGroupConvGeneral::AlgoGroupConvGeneral(AlgoBase* impl)
        : m_impl{impl} {
    m_name = ConvBiasForward::algo_name<DirectParam>(
            ssprintf("%s:%s", "CUDA:GROUP_CONV", impl->name()), {});
}

bool ConvBiasForwardImpl::AlgoGroupConvGeneral::is_available(
        const SizeArgs& args) const {
    if (args.src_layout->dtype == args.filter_layout->dtype &&
        args.src_layout->dtype == dtype::BFloat16()) {
        return false;
    }
    if (args.z_layout->ndim > 0 || args.filter_meta.group <= 1)
        return false;
    auto&& param = args.opr->param();
    if (param.format == param::ConvBias::Format::NCHW8 ||
        param.format == param::ConvBias::Format::CHWN4 ||
        param.format == param::ConvBias::Format::NCHW32)
        return false;

    auto sub_args = args;
    TensorLayout src_pg, dst_pg, bias_pg;
    modify_size_args(sub_args, src_pg, dst_pg, bias_pg);
    return m_impl->is_available(sub_args);
}

WorkspaceBundle ConvBiasForwardImpl::AlgoGroupConvGeneral::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    auto dst_layout = *args.dst_layout;
    SmallVector<size_t> sizes;
    if (dst_layout.dtype.enumv() != args.bias_layout->dtype.enumv()) {
        dst_layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(args.src_layout->dtype,
                                            args.filter_layout->dtype,
                                            dst_layout.dtype);
        sizes.push_back(dst_layout.span().dist_byte());
    }

    auto sub_args = args;
    sub_args.dst_layout = &dst_layout;
    TensorLayout src_pg, dst_pg, bias_pg;
    modify_size_args(sub_args, src_pg, dst_pg, bias_pg);
    sizes.insert(sizes.begin(),
            m_impl->get_workspace_in_bytes(sub_args));
    return {ptr, std::move(sizes)};
}

size_t ConvBiasForwardImpl::AlgoGroupConvGeneral::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvBiasForwardImpl::AlgoGroupConvGeneral::exec(
        const ExecArgs& args) const {
    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);
    auto conv_dst_tensor = *args.dst_tensor;
    if (args.dst_layout->dtype.enumv() != args.bias_layout->dtype.enumv()) {
        conv_dst_tensor.raw_ptr = bundle.get(bundle.nr_workspace() - 1);
        conv_dst_tensor.layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(args.src_layout->dtype,
                                            args.filter_layout->dtype,
                                            conv_dst_tensor.layout.dtype);
    }
    {
        auto sub_args = args;
        sub_args.dst_tensor = &conv_dst_tensor;
        sub_args.dst_layout = &conv_dst_tensor.layout;
        TensorND tsrc{*args.src_tensor}, tdst{conv_dst_tensor}, tbias{*args.bias_tensor};
        SmallVector<size_t> flt_shape(0);
        std::vector<ptrdiff_t> flt_stride(0);
        size_t idx = 0;
        // check if the first dim is group
        if (args.filter_tensor->layout.ndim > args.src_layout->ndim)
            ++idx;
        for (; idx < args.filter_tensor->layout.ndim; ++idx) {
            flt_shape.push_back(args.filter_tensor->layout[idx]);
            flt_stride.push_back(args.filter_tensor->layout.stride[idx]);
        }
        TensorND tflt{args.filter_tensor->raw_ptr,
                      TensorLayout{flt_shape, flt_stride,
                                   args.filter_tensor->layout.dtype,
                                   args.filter_tensor->layout.format}};

        modify_size_args(sub_args, tsrc.layout, tdst.layout, tbias.layout);
        sub_args.src_tensor = &tsrc;
        sub_args.dst_tensor = &tdst;
        sub_args.filter_tensor = &tflt;
        sub_args.bias_tensor = &tbias;

        size_t c_pos;
        if (args.filter_meta.format == Param::Format::NCHW ||
            args.filter_meta.format == Param::Format::NCHW4) {
            c_pos = 1;
        } else {
            megdnn_assert(args.filter_meta.format == Param::Format::NHWC,
                          "invalid conv format");
            c_pos = 3;
        }

        auto grp = args.filter_meta.group;

        auto&& fm = args.filter_meta;
        auto strd_src = tsrc.layout.stride[c_pos] * fm.icpg *
                        tsrc.layout.dtype.size(),
             strd_dst = tdst.layout.stride[c_pos] * fm.ocpg *
                        tdst.layout.dtype.size(),
             strd_flt = fm.icpg * fm.ocpg * fm.spatial[0] * fm.spatial[1] *
                        tflt.layout.dtype.size();
        if (args.filter_meta.format == Param::Format::NCHW4) {
            strd_src >>= 2;
            strd_dst >>= 2;
        }
        for (uint32_t g = 0; g < grp; ++g) {
            m_impl->exec(sub_args);
            incr_voidp(tsrc.raw_ptr, strd_src);
            incr_voidp(tdst.raw_ptr, strd_dst);
            incr_voidp(tflt.raw_ptr, strd_flt);
        }
    }
    handle_bias_and_nonlinear(args.handle, args.nonlinear_mode,
                              &conv_dst_tensor, args.dst_tensor,
                              args.bias_tensor);
}

// vim: syntax=cpp.doxygen
