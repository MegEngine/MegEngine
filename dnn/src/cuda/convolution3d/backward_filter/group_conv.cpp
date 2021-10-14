/**
 * \file dnn/src/cuda/convolution3d/backward_filter/group_conv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algo.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution3d;

namespace {
std::pair<TensorLayoutArray, Convolution3DBackwardFilterImpl::Param> sub_opr_config(
        const Convolution3DBackwardFilterImpl::AlgoBase::SizeArgs& args) {
    TensorLayout grad_pg = *args.grad_layout;
    TensorLayout src_pg = *args.src_layout;
    TensorLayout diff_pg = *args.diff_layout;

    grad_pg.remove_axis_inplace(0);
    auto nr_grp = args.grad_filter_meta.group;
    size_t c_pos = 1;
    src_pg.shape[c_pos] /= nr_grp;
    diff_pg.shape[c_pos] /= nr_grp;

    megdnn::param::Convolution3D param = args.opr->param();
    param.sparse = megdnn::param::Convolution3D::Sparse::DENSE;
    std::pair<TensorLayoutArray, Convolution3DBackwardFilterImpl::Param> ret;
    ret.first = {src_pg, diff_pg, grad_pg};
    ret.second = param;

    return ret;
}

std::pair<TensorLayoutArray, std::unique_ptr<Convolution3DBackwardFilter>>
prepare_sub_opr(const Convolution3DBackwardFilterImpl::AlgoBase::SizeArgs& args) {
    auto conv3d_backfilter_opr =
            args.handle->create_operator<Convolution3DBackwardFilter>();
    set_execution_policy<Convolution3DBackwardFilter, Convolution3DBackwardFilter*>(
            args.opr, conv3d_backfilter_opr.get());
    auto&& config = sub_opr_config(args);
    conv3d_backfilter_opr->param() = config.second;

    return {config.first, std::move(conv3d_backfilter_opr)};
}
}  // namespace

std::vector<Algorithm::SearchItem> Convolution3DBackwardFilterImpl::
        AlgoGroupConvGeneral::get_subopr_list(
                const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    AlgoBase::SizeArgs args{
            static_cast<const Convolution3DBackwardFilterImpl*>(opr), layouts[0],
            layouts[1], layouts[2]};
    auto&& config = sub_opr_config(args);

    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {
            {Algorithm::OprType::CONVOLUTION3D_BACKWARD_FILTER, param_str,
             config.first}};
}

bool Convolution3DBackwardFilterImpl::AlgoGroupConvGeneral::is_available(
        const SizeArgs& args) const {
    if (args.grad_filter_meta.group <= 1)
        return false;
    if (args.grad_filter_meta.format != Param::Format::NCDHW) {
        return false;
    }

    auto config = prepare_sub_opr(args);

    return has_available_algo<Convolution3DBackwardFilterImpl>(
            static_cast<Convolution3DBackwardFilterImpl*>(config.second.get()),
            config.first[0], config.first[1], config.first[2]);
}

WorkspaceBundle Convolution3DBackwardFilterImpl::AlgoGroupConvGeneral::
        get_workspace_bundle(void* ptr, const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);
    size_t sizes = config.second->get_workspace_in_bytes(
            config.first[0], config.first[1], config.first[2]);
    return {ptr, {sizes}};
}

size_t Convolution3DBackwardFilterImpl::AlgoGroupConvGeneral::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void Convolution3DBackwardFilterImpl::AlgoGroupConvGeneral::exec(
        const ExecArgs& args) const {
    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);
    {
        auto config = prepare_sub_opr(args);
        TensorND tsrc{args.src_tensor->raw_ptr, config.first[0]};
        TensorND tdiff{args.diff_tensor->raw_ptr, config.first[1]};
        TensorND tgrad{args.grad_tensor->raw_ptr, config.first[2]};

        size_t c_pos = 1;
        auto grp = args.grad_filter_meta.group;

        auto&& fm = args.grad_filter_meta;
        auto strd_src =
                     (tsrc.layout.stride[c_pos] * fm.icpg * tsrc.layout.dtype.size()),
             strd_diff =
                     (tdiff.layout.stride[c_pos] * fm.ocpg * tdiff.layout.dtype.size()),
             strd_grad =
                     (fm.icpg * fm.ocpg * fm.spatial[0] * fm.spatial[1] *
                      fm.spatial[2] * tgrad.layout.dtype.size());

        for (uint32_t g = 0; g < grp; ++g) {
            config.second->exec(tsrc, tdiff, tgrad, bundle.get_workspace(0));
            incr_voidp(tsrc.raw_ptr, strd_src);
            incr_voidp(tdiff.raw_ptr, strd_diff);
            incr_voidp(tgrad.raw_ptr, strd_grad);
        }
    }
}

// vim: syntax=cpp.doxygen
