/**
 * \file dnn/src/cuda/convolution3d/backward_data/group_conv.cpp
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
std::pair<TensorLayoutArray, Convolution3DBackwardDataImpl::Param>
sub_opr_config(const Convolution3DBackwardDataImpl::AlgoBase::SizeArgs& args) {
    TensorLayout filter_pg = *args.filter_layout;
    TensorLayout diff_pg = *args.diff_layout;
    TensorLayout grad_pg = *args.grad_layout;

    filter_pg.remove_axis_inplace(0);
    auto nr_grp = args.filter_meta.group;
    size_t c_pos = 1;
    diff_pg.shape[c_pos] /= nr_grp;
    grad_pg.shape[c_pos] /= nr_grp;

    megdnn::param::Convolution3D param = args.opr->param();
    param.sparse = megdnn::param::Convolution3D::Sparse::DENSE;
    std::pair<TensorLayoutArray, Convolution3DBackwardDataImpl::Param> ret;
    ret.first = {filter_pg, diff_pg, grad_pg};
    ret.second = param;

    return ret;
}

std::pair<TensorLayoutArray, std::unique_ptr<Convolution3DBackwardData>>
prepare_sub_opr(const Convolution3DBackwardDataImpl::AlgoBase::SizeArgs& args) {
    auto conv3d_backdata_opr =
            args.handle->create_operator<Convolution3DBackwardData>();
    set_execution_policy<Convolution3DBackwardData, Convolution3DBackwardData*>(
            args.opr, conv3d_backdata_opr.get());
    auto&& config = sub_opr_config(args);
    conv3d_backdata_opr->param() = config.second;

    return {config.first, std::move(conv3d_backdata_opr)};
}
}  // namespace

std::vector<Algorithm::SearchItem>
Convolution3DBackwardDataImpl::AlgoGroupConvGeneral::get_subopr_list(
        const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    AlgoBase::SizeArgs args{
            static_cast<const Convolution3DBackwardDataImpl*>(opr), layouts[0],
            layouts[1], layouts[2]};
    auto&& config = sub_opr_config(args);

    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {{Algorithm::OprType::CONVOLUTION3D_BACKWARD_DATA, param_str,
             config.first}};
}

bool Convolution3DBackwardDataImpl::AlgoGroupConvGeneral::is_available(
        const SizeArgs &args) const {
    if (args.filter_meta.group <= 1)
        return false;
    if (args.filter_meta.format != Param::Format::NCDHW) {
        return false;
    }

    auto config = prepare_sub_opr(args);
    AlgoBase::SizeArgs sub_args{
            static_cast<Convolution3DBackwardDataImpl*>(config.second.get()),
            config.first[0], config.first[1], config.first[2]};

    return has_available_algo<Convolution3DBackwardDataImpl>(sub_args);
}

WorkspaceBundle
Convolution3DBackwardDataImpl::AlgoGroupConvGeneral::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);
    size_t sizes = config.second->get_workspace_in_bytes(
            config.first[0], config.first[1], config.first[2]);
    return {ptr, {sizes}};
}

size_t
Convolution3DBackwardDataImpl::AlgoGroupConvGeneral::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void Convolution3DBackwardDataImpl::AlgoGroupConvGeneral::exec(
        const ExecArgs& args) const {
    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);
    {
        auto config = prepare_sub_opr(args);
        TensorND tfilter{args.filter_tensor->raw_ptr, config.first[0]};
        TensorND tdiff{args.diff_tensor->raw_ptr, config.first[1]};
        TensorND tgrad{args.grad_tensor->raw_ptr, config.first[2]};

        size_t c_pos = 1;
        auto grp = args.filter_meta.group;

        auto&& fm = args.filter_meta;
        auto strd_flt = (fm.icpg * fm.ocpg * fm.spatial[0] * fm.spatial[1] *
                         fm.spatial[2] * tfilter.layout.dtype.size()),
             strd_diff = (tdiff.layout.stride[c_pos] * fm.ocpg *
                          tdiff.layout.dtype.size()),
             strd_grad = (tgrad.layout.stride[c_pos] * fm.icpg *
                          tgrad.layout.dtype.size());

        for (uint32_t g = 0; g < grp; ++g) {
            config.second->exec(tfilter, tdiff, tgrad, bundle.get_workspace(0));
            incr_voidp(tfilter.raw_ptr, strd_flt);
            incr_voidp(tdiff.raw_ptr, strd_diff);
            incr_voidp(tgrad.raw_ptr, strd_grad);
        }
    }
}

// vim: syntax=cpp.doxygen

