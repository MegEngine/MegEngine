/**
 * \file dnn/src/cuda/convolution3d/forward/group_conv.cpp
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
std::pair<TensorLayoutArray, Convolution3DForwardImpl::Param> sub_opr_config(
        const Convolution3DForwardImpl::AlgoBase::SizeArgs& args) {
    TensorLayout src_pg = *args.src_layout;
    TensorLayout filter_pg = *args.filter_layout;
    TensorLayout dst_pg = *args.dst_layout;

    auto nr_grp = args.filter_meta.group;
    size_t c_pos;
    if (args.filter_meta.format == param::Convolution3D::Format::NCDHW) {
        c_pos = 1;
    } else {
        megdnn_assert(
                args.filter_meta.format == param::Convolution3D::Format::NDHWC,
                "invalid conv format");
        c_pos = 4;
    }
    filter_pg.remove_axis_inplace(0);
    src_pg.shape[c_pos] /= nr_grp;
    dst_pg.shape[c_pos] /= nr_grp;

    megdnn::param::Convolution3D param = args.opr->param();
    param.sparse = megdnn::param::Convolution3D::Sparse::DENSE;
    std::pair<TensorLayoutArray, Convolution3DForwardImpl::Param> ret;
    ret.first = {src_pg, filter_pg, dst_pg};
    ret.second = param;

    return ret;
}

std::pair<TensorLayoutArray, std::unique_ptr<Convolution3DForward>> prepare_sub_opr(
        const Convolution3DForwardImpl::AlgoBase::SizeArgs& args) {
    auto conv3d_opr = args.handle->create_operator<Convolution3D>();
    set_execution_policy<Convolution3DForward, Convolution3DForward*>(
            args.opr, conv3d_opr.get());
    auto&& config = sub_opr_config(args);
    conv3d_opr->param() = config.second;

    return {config.first, std::move(conv3d_opr)};
}
}  // namespace

std::vector<Algorithm::SearchItem> Convolution3DForwardImpl::AlgoGroupConvGeneral::
        get_subopr_list(
                const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    AlgoBase::SizeArgs args{
            static_cast<const Convolution3DForwardImpl*>(opr), layouts[0], layouts[1],
            layouts[2]};
    auto&& config = sub_opr_config(args);

    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {{Algorithm::OprType::CONVOLUTION3D_FORWARD, param_str, config.first}};
}

bool Convolution3DForwardImpl::AlgoGroupConvGeneral::is_available(
        const SizeArgs& args) const {
    if (args.filter_meta.group <= 1)
        return false;
    if (args.filter_meta.format != Param::Format::NCDHW &&
        args.filter_meta.format != Param::Format::NDHWC) {
        return false;
    }

    auto config = prepare_sub_opr(args);

    return has_available_algo<Convolution3DForwardImpl>(
            static_cast<Convolution3DForwardImpl*>(config.second.get()),
            config.first[0], config.first[1], config.first[2]);
}

WorkspaceBundle Convolution3DForwardImpl::AlgoGroupConvGeneral::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);
    size_t sizes = config.second->get_workspace_in_bytes(
            config.first[0], config.first[1], config.first[2]);
    return {ptr, {sizes}};
}

size_t Convolution3DForwardImpl::AlgoGroupConvGeneral::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void Convolution3DForwardImpl::AlgoGroupConvGeneral::exec(const ExecArgs& args) const {
    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);
    {
        auto config = prepare_sub_opr(args);
        TensorND tsrc{args.src_tensor->raw_ptr(), config.first[0]};
        TensorND tfilter{args.filter_tensor->raw_ptr(), config.first[1]};
        TensorND tdst{args.dst_tensor->raw_ptr(), config.first[2]};

        size_t c_pos;
        if (args.filter_meta.format == Param::Format::NCDHW) {
            c_pos = 1;
        } else {
            megdnn_assert(
                    args.filter_meta.format == Param::Format::NDHWC,
                    "invalid conv format");
            c_pos = 4;
        }

        auto grp = args.filter_meta.group;

        auto&& fm = args.filter_meta;
        auto strd_src = tsrc.layout.stride[c_pos] * fm.icpg * tsrc.layout.dtype.size(),
             strd_dst = tdst.layout.stride[c_pos] * fm.ocpg * tdst.layout.dtype.size(),
             strd_flt = fm.icpg * fm.ocpg * fm.spatial[0] * fm.spatial[1] *
                        fm.spatial[2] * tfilter.layout.dtype.size();

        for (uint32_t g = 0; g < grp; ++g) {
            config.second->exec(tsrc, tfilter, tdst, bundle.get_workspace(0));
            incr_refp(tsrc.get_ref_ptr(), strd_src);
            incr_refp(tdst.get_ref_ptr(), strd_dst);
            incr_refp(tfilter.get_ref_ptr(), strd_flt);
        }
    }
}

// vim: syntax=cpp.doxygen
