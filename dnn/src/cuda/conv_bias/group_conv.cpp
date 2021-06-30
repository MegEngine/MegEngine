/**
 * \file dnn/src/cuda/conv_bias/group_conv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <utility>
#include "src/common/conv_bias.h"
#include "src/cuda/conv_bias/algo.h"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

namespace {
std::pair<TensorLayoutArray, ConvBiasForwardImpl::Param> sub_opr_config(
        const ConvBiasForwardImpl::AlgoBase::SizeArgs& args) {
    TensorLayout src_pg = *args.src_layout;

    SmallVector<size_t> flt_shape(0);
    std::vector<ptrdiff_t> flt_stride(0);
    size_t idx = 0;
    // check if the first dim is group
    if (args.filter_layout->ndim > args.src_layout->ndim)
        ++idx;
    for (; idx < args.filter_layout->ndim; ++idx) {
        flt_shape.push_back(args.filter_layout->shape[idx]);
        flt_stride.push_back(args.filter_layout->stride[idx]);
    }
    TensorLayout filter_pg(flt_shape, flt_stride,
                               args.filter_layout->dtype,
                               args.filter_layout->format);
    TensorLayout bias_pg = *args.bias_layout;
    TensorLayout z_pg = *args.z_layout;
    TensorLayout dst_pg = *args.dst_layout;

    auto nr_grp = args.filter_meta.group;
    size_t c_pos;
    if (args.filter_meta.format == megdnn::param::ConvBias::Format::NCHW ||
        args.filter_meta.format == megdnn::param::ConvBias::Format::NCHW4) {
        c_pos = 1;
    } else {
        megdnn_assert(args.filter_meta.format ==
                              megdnn::param::ConvBias::Format::NHWC,
                      "invalid conv format");
        c_pos = 3;
    }
    src_pg.shape[c_pos] /= nr_grp;
    bias_pg.ndim = 0;
    dst_pg.shape[c_pos] /= nr_grp;

    megdnn::param::ConvBias param = args.opr->param();
    param.sparse = megdnn::param::ConvBias::Sparse::DENSE;
    param.nonlineMode =
            megdnn::param::ConvBias::NonlineMode::IDENTITY;
    std::pair<TensorLayoutArray, ConvBiasForwardImpl::Param> ret;
    ret.first = {src_pg, filter_pg, bias_pg, z_pg, dst_pg};
    ret.second = param;

    return ret;
}

std::pair<TensorLayoutArray, std::unique_ptr<ConvBiasForward>> prepare_sub_opr(
        const ConvBiasForwardImpl::AlgoBase::SizeArgs& args) {
    auto convbias_opr = args.handle->create_operator<ConvBias>();
    set_execution_policy<ConvBiasForward, ConvBiasForward*>(
            args.opr, convbias_opr.get());
    auto&& config = sub_opr_config(args);
    convbias_opr->param() = config.second;

    return {config.first, std::move(convbias_opr)};
}
}  // namespace

std::vector<Algorithm::SearchItem>
ConvBiasForwardImpl::AlgoGroupConvGeneral::get_subopr_list(
        const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    AlgoBase::SizeArgs args{static_cast<const ConvBiasForwardImpl*>(opr),
                            layouts[0],
                            layouts[1],
                            layouts[2],
                            layouts[3],
                            layouts[4]};
    auto&& config = sub_opr_config(args);

    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {{Algorithm::OprType::CONVBIAS_FORWARD, param_str, config.first}};
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

    auto config = prepare_sub_opr(args);
    return get_algorithm(static_cast<ConvBiasForwardImpl*>(config.second.get()),
                         config.first[0], config.first[1], config.first[2],
                         config.first[3], config.first[4]);
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

    auto config = prepare_sub_opr(args);
    size_t mm_ws = config.second->get_workspace_in_bytes(
                    config.first[0], config.first[1], config.first[2],
                    config.first[3], config.first[4], nullptr);

    sizes.insert(sizes.begin(), mm_ws);
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

        auto config = prepare_sub_opr(sub_args);
        TensorND tsrc{args.src_tensor->raw_ptr, config.first[0]};
        TensorND tfilter{args.filter_tensor->raw_ptr, config.first[1]};
        TensorND tbias{args.bias_tensor->raw_ptr, config.first[2]};
        TensorND tz{args.z_tensor->raw_ptr, config.first[3]};
        TensorND tdst{conv_dst_tensor.raw_ptr, config.first[4]};

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
                        tfilter.layout.dtype.size();
        if (args.filter_meta.format == Param::Format::NCHW4) {
            strd_src >>= 2;
            strd_dst >>= 2;
        }
        for (uint32_t g = 0; g < grp; ++g) {
            config.second->exec(tsrc, tfilter, tbias,
                            tz, tdst, nullptr, bundle.get_workspace(0));
            incr_voidp(tsrc.raw_ptr, strd_src);
            incr_voidp(tdst.raw_ptr, strd_dst);
            incr_voidp(tfilter.raw_ptr, strd_flt);
        }
    }
    handle_bias_and_nonlinear(args.handle, args.nonlinear_mode,
                              &conv_dst_tensor, args.dst_tensor,
                              args.bias_tensor);
}

// vim: syntax=cpp.doxygen
