/**
 * \file dnn/src/cuda/roi_pooling/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/roi_pooling/opr_impl.h"

#include "src/cuda/roi_pooling/roi_pooling.cuh"
#include "src/cuda/utils.h"
#include "src/common/roi_pooling_helper.h"

namespace megdnn {
namespace cuda {

void ROIPoolingForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in rois,
        _megdnn_tensor_out dst,
        _megdnn_tensor_out index,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, rois.layout, dst.layout, index.layout,
            workspace.size);
    auto stream = cuda_stream(handle());
    auto nthreads = dst.layout.total_nr_elems();
    auto spatial_scale = m_param.scale;
    auto channels = src.layout.shape[1];
    auto height = src.layout.shape[2];
    auto width = src.layout.shape[3];
    auto pooled_height = dst.layout.shape[2];
    auto pooled_width = dst.layout.shape[3];
    using namespace ::megdnn::roi_pooling;
    using namespace ::megdnn::cuda::roi_pooling;
#define cb(DType) \
    if (src.layout.dtype == DType()) { \
        using T = typename DTypeTrait<DType>::ctype; \
        switch (param().mode) { \
            case param::ROIPooling::Mode::MAX: \
                forward_proxy<T, MaxPooler<T>>(nthreads, \
                        src.ptr<T>(), spatial_scale, channels, height, width, \
                        pooled_height, pooled_width, \
                        rois.ptr<T>(), dst.ptr<T>(), \
                        index.ptr<dt_int32>(), \
                        stream); \
                break; \
            case param::ROIPooling::Mode::AVERAGE: \
                forward_proxy<T, AveragePooler<T>>(nthreads, \
                        src.ptr<T>(), spatial_scale, channels, height, width, \
                        pooled_height, pooled_width, \
                        rois.ptr<T>(), dst.ptr<T>(), \
                        index.ptr<dt_int32>(), \
                        stream); \
                break; \
            default: \
                megdnn_assert_internal(false); \
        } \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
}

void ROIPoolingBackwardImpl::exec(_megdnn_tensor_in diff,
        _megdnn_tensor_in src,
        _megdnn_tensor_in rois,
        _megdnn_tensor_in index,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(diff.layout, src.layout, rois.layout, index.layout, grad.layout,
            workspace.size);
    auto stream = cuda_stream(handle());
    auto nthreads = grad.layout.total_nr_elems();
    auto num_rois = rois.layout.shape[0];
    auto spatial_scale = m_param.scale;
    auto channels = src.layout.shape[1];
    auto height = src.layout.shape[2];
    auto width = src.layout.shape[3];
    auto pooled_height = diff.layout.shape[2];
    auto pooled_width = diff.layout.shape[3];
    using namespace ::megdnn::roi_pooling;
    using namespace ::megdnn::cuda::roi_pooling;
#define cb(DType) \
    if (src.layout.dtype == DType()) { \
        using T = typename DTypeTrait<DType>::ctype; \
        switch (param().mode) { \
            case param::ROIPooling::Mode::MAX: \
                roi_pooling::backward_proxy<T, BwdMaxPooler<T>>(nthreads, \
                        diff.ptr<T>(), index.ptr<dt_int32>(), \
                        num_rois, spatial_scale, \
                        channels, height, width, \
                        pooled_height, pooled_width, \
                        grad.ptr<T>(), rois.ptr<T>(), \
                        stream); \
                break; \
            case param::ROIPooling::Mode::AVERAGE: \
                roi_pooling::backward_proxy<T, BwdAveragePooler<T>>(nthreads, \
                        diff.ptr<T>(), index.ptr<dt_int32>(), \
                        num_rois, spatial_scale, \
                        channels, height, width, \
                        pooled_height, pooled_width, \
                        grad.ptr<T>(), rois.ptr<T>(), \
                        stream); \
                break; \
            default: \
                megdnn_assert_internal(false); \
        } \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
}

} // namespace cuda
} // namespace megdnn
// vim: syntax=cpp.doxygen

