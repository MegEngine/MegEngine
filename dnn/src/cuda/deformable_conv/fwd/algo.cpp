/**
 * \file dnn/src/cuda/deformable_conv/fwd/algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/utils.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

#include "src/cuda/deformable_conv/fwd/algo.h"

using namespace megdnn;
using namespace cuda;

using OprImpl = DeformableConvForwardImpl;

OprImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&algo_matmul);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(DeformableConvForwardImpl)

OprImpl::AlgoPack OprImpl::sm_algo_pack;

OprImpl::AlgoBase::SizeArgs::SizeArgs(OprImpl* o, const TensorLayout& im,
                                      const TensorLayout& filter,
                                      const TensorLayout& offset,
                                      const TensorLayout& mask,
                                      const TensorLayout& dst)
        : SizeArgs(o, im,
                   o->make_canonized_filter_meta(im.ndim, filter, offset),
                   offset, mask, dst) {}

OprImpl::AlgoBase::SizeArgs::SizeArgs(OprImpl* o, const TensorLayout& im,
                                      const CanonizedFilterMeta& filter,
                                      const TensorLayout& offset,
                                      const TensorLayout& mask,
                                      const TensorLayout& dst)
        : opr(o),
          handle(concrete_handle(o->handle())),
          im_layout(im),
          filter_meta(filter),
          offset_layout(offset),
          mask_layout(mask),
          dst_layout(dst) {}

OprImpl::AlgoBase::ExecArgs::ExecArgs(OprImpl* opr, _megdnn_tensor_in im,
                                      _megdnn_tensor_in filter,
                                      _megdnn_tensor_in offset,
                                      _megdnn_tensor_in mask,
                                      _megdnn_tensor_out dst,
                                      _megdnn_workspace workspace)
        : SizeArgs(opr, im.layout, filter.layout, offset.layout, mask.layout,
                   dst.layout),
          im_tensor(im),
          filter_tensor(filter),
          offset_tensor(offset),
          mask_tensor(mask),
          dst_tensor(dst),
          workspace(workspace) {}

std::string OprImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& fm = filter_meta;
    MEGDNN_MARK_USED_VAR(fm);
    return ssprintf(
            "im=%s, filter=%u{%u,%u,%u,%u}, offset=%s, mask=%s, dst=%s, "
            "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, dtype=%s,%s",
            im_layout.to_string().c_str(), fm.group, fm.ocpg, fm.icpg,
            fm.spatial[0], fm.spatial[1], offset_layout.to_string().c_str(),
            mask_layout.to_string().c_str(), dst_layout.to_string().c_str(),
            fm.padding[0], fm.padding[1], fm.stride[0], fm.stride[1],
            fm.dilation[0], fm.dilation[1], !fm.should_flip,
            im_layout.dtype.name(), dst_layout.dtype.name());
}

// vim: syntax=cpp.doxygen
