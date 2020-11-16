/**
 * \file dnn/src/cuda/deformable_conv/bwd_data/algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/deformable_conv/bwd_data/algo.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

using OprImpl = DeformableConvBackwardDataImpl;

OprImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&algo_matmul);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}
MEGDNN_DEF_GET_ALGO_FROM_DESC(DeformableConvBackwardDataImpl)

OprImpl::AlgoPack OprImpl::sm_algo_pack;

OprImpl::AlgoBase::SizeArgs::SizeArgs(
        OprImpl* o, const TensorLayout& im, const TensorLayout& filter,
        const TensorLayout& offset, const TensorLayout& mask,
        const TensorLayout& out_grad, const TensorLayout& im_grad,
        const TensorLayout& offset_grad, const TensorLayout& mask_grad)
        : SizeArgs(o, im,
                   o->make_canonized_filter_meta(im.ndim, filter, offset),
                   offset, mask, out_grad, im_grad, offset_grad, mask_grad) {}

OprImpl::AlgoBase::SizeArgs::SizeArgs(
        OprImpl* o, const TensorLayout& im, const CanonizedFilterMeta& filter,
        const TensorLayout& offset, const TensorLayout& mask,
        const TensorLayout& out_grad, const TensorLayout& im_grad,
        const TensorLayout& offset_grad, const TensorLayout& mask_grad)
        : opr(o),
          handle(concrete_handle(o->handle())),
          im_layout(im),
          filter_meta(filter),
          offset_layout(offset),
          mask_layout(mask),
          out_grad_layout(out_grad),
          im_grad_layout(im_grad),
          offset_grad_layout(offset_grad),
          mask_grad_layout(mask_grad) {}

OprImpl::AlgoBase::ExecArgs::ExecArgs(
        OprImpl* opr, _megdnn_tensor_in im, _megdnn_tensor_in filter,
        _megdnn_tensor_in offset, _megdnn_tensor_in mask,
        _megdnn_tensor_in out_grad, _megdnn_tensor_out im_grad,
        _megdnn_tensor_out offset_grad, _megdnn_tensor_out mask_grad,
        _megdnn_workspace ws)
        : SizeArgs(opr, im.layout, filter.layout, offset.layout, mask.layout,
                   out_grad.layout, im_grad.layout, offset_grad.layout,
                   mask_grad.layout),
          im_tensor(im),
          filter_tensor(filter),
          offset_tensor(offset),
          mask_tensor(mask),
          out_grad_tensor(out_grad),
          im_grad_tensor(im_grad),
          offset_grad_tensor(offset_grad),
          mask_grad_tensor(mask_grad),
          workspace(ws) {}

std::string OprImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& fm = filter_meta;
    MEGDNN_MARK_USED_VAR(fm);
    return ssprintf(
            "im=%s, filter=%u{%u,%u,%u,%u}, offset=%s, mask=%s, "
            "dst_grad=%s, "
            "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, "
            "dtype=%s,%s",
            megdnn_layout_msg(im_layout).c_str(), fm.group, fm.ocpg, fm.icpg,
            fm.spatial[0], fm.spatial[1],
            megdnn_layout_msg(offset_layout).c_str(),
            megdnn_layout_msg(mask_layout).c_str(),
            megdnn_layout_msg(out_grad_layout).c_str(), fm.padding[0],
            fm.padding[1], fm.stride[0], fm.stride[1], fm.dilation[0],
            fm.dilation[1], !fm.should_flip, im_layout.dtype.name(),
            out_grad_layout.dtype.name());
}

// vim: syntax=cpp.doxygen
