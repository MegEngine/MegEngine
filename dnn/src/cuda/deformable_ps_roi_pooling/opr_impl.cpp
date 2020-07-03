/**
 * \file dnn/src/cuda/deformable_ps_roi_pooling/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/deformable_ps_roi_pooling/kimpl/kern.cuh"
#include "src/cuda/deformable_ps_roi_pooling/opr_impl.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using KernParam = deformable_ps_roi_pooling::Param;

namespace {

void create_param(const DeformablePSROIPoolingBase* opr,
                  const TensorLayout& data, const TensorLayout& rois,
                  const TensorLayout& trans, KernParam& p) {
    auto&& param = opr->param();
    auto&& handle = concrete_handle(opr->handle());

    p.stream = handle->stream();
    p.no_trans = param.no_trans;
    p.pool_h = param.pooled_h;
    p.pool_w = param.pooled_w;
    p.part_sz = param.part_size;
    p.sample_per_part = param.sample_per_part;
    p.trans_std = param.trans_std;
    p.scale = param.spatial_scale;
    p.nr_cls = p.no_trans ? 1 : trans[1] / 2;
    p.nr_bbox = rois[0];
    p.IC = data[1];
    p.IH = data[2];
    p.IW = data[3];
}

}  // namespace

namespace megdnn {
namespace cuda {

void DeformablePSROIPoolingForwardImpl::exec(_megdnn_tensor_in data,
                                             _megdnn_tensor_in rois,
                                             _megdnn_tensor_in trans,
                                             _megdnn_tensor_out out_data,
                                             _megdnn_tensor_out out_count,
                                             _megdnn_workspace workspace) {
    KernParam p;

    check_exec(data.layout, rois.layout, trans.layout, out_data.layout,
               out_count.layout, workspace.size);

    create_param(this, data.layout, rois.layout, trans.layout, p);
    deformable_ps_roi_pooling::DeformablePSROIPoolForward(
            data, rois, trans, out_data, out_count, p);
}

void DeformablePSROIPoolingBackwardImpl::exec(
        _megdnn_tensor_in data, _megdnn_tensor_in rois, _megdnn_tensor_in trans,
        _megdnn_tensor_in out_diff, _megdnn_tensor_in out_count,
        _megdnn_tensor_out data_diff, _megdnn_tensor_out trans_diff,
        _megdnn_workspace workspace) {
    KernParam p;

    check_exec(data.layout, rois.layout, trans.layout, out_diff.layout,
               out_count.layout, data_diff.layout, trans_diff.layout,
               workspace.size);
    create_param(this, data.layout, rois.layout, trans.layout, p);
    deformable_ps_roi_pooling::DeformablePSROIPoolBackwardAcc(
            data, rois, trans, out_diff, out_count, data_diff, trans_diff, p);
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
