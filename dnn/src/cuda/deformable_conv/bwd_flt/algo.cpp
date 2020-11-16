/**
 * \file dnn/src/cuda/deformable_conv/bwd_flt/algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/utils.h"

#include "src/cuda/deformable_conv/bwd_flt/algo.h"

using namespace megdnn;
using namespace cuda;

using OprImpl = DeformableConvBackwardFilterImpl;

OprImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&algo_matmul);
    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}
MEGDNN_DEF_GET_ALGO_FROM_DESC(DeformableConvBackwardFilterImpl)

OprImpl::AlgoPack OprImpl::sm_algo_pack;

OprImpl::AlgoBase::SizeArgs::SizeArgs(OprImpl* o, const TensorLayout& im,
                                      const TensorLayout& offset,
                                      const TensorLayout& mask,
                                      const TensorLayout& out_grad,
                                      const TensorLayout& filter_grad)
        : SizeArgs(
                  o, im, offset, mask, out_grad,
                  o->make_canonized_filter_meta(im.ndim, filter_grad, offset)) {
}

OprImpl::AlgoBase::SizeArgs::SizeArgs(
        OprImpl* o, const TensorLayout& im, const TensorLayout& offset,
        const TensorLayout& mask, const TensorLayout& out_grad,
        const CanonizedFilterMeta& filter_grad_meta)
        : opr(o),
          handle(concrete_handle(o->handle())),
          im_layout(im),
          offset_layout(offset),
          mask_layout(mask),
          out_grad_layout(out_grad),
          filter_grad_meta(filter_grad_meta) {}

OprImpl::AlgoBase::ExecArgs::ExecArgs(OprImpl* opr, _megdnn_tensor_in im,
                                      _megdnn_tensor_in offset,
                                      _megdnn_tensor_in mask,
                                      _megdnn_tensor_in out_grad,
                                      _megdnn_tensor_out filter_grad,
                                      _megdnn_workspace ws)
        : SizeArgs(opr, im.layout, offset.layout, mask.layout, out_grad.layout,
                   filter_grad.layout),
          im_tensor(im),
          offset_tensor(offset),
          mask_tensor(mask),
          out_grad_tensor(out_grad),
          filter_grad_tensor(filter_grad),
          workspace(ws) {}

std::string OprImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& fm = filter_grad_meta;
    MEGDNN_MARK_USED_VAR(fm);
    return ssprintf("im=%s, offset=%s, mask=%s, dst_grad=%s, "
                     "filter_grad=%u{%u,%u,%u,%u},"
                     "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, "
                     "dtype=%s,%s",
                     megdnn_layout_msg(im_layout).c_str(),
                     megdnn_layout_msg(offset_layout).c_str(),
                     megdnn_layout_msg(mask_layout).c_str(),
                     megdnn_layout_msg(out_grad_layout).c_str(), fm.group,
                     fm.ocpg, fm.icpg, fm.spatial[0], fm.spatial[1],
                     fm.padding[0], fm.padding[1], fm.stride[0], fm.stride[1],
                     fm.dilation[0], fm.dilation[1], !fm.should_flip,
                     im_layout.dtype.name(), out_grad_layout.dtype.name());
}

// vim: syntax=cpp.doxygen
