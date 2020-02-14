/**
 * \file dnn/src/cuda/roi_align/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/roi_align/opr_impl.h"

#include "src/common/roi_align_helper.h"
#include "src/cuda/roi_align/roi_align.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void ROIAlignForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in rois,
                               _megdnn_tensor_out dst, _megdnn_tensor_out index,
                               _megdnn_workspace workspace) {
    check_exec(src.layout, rois.layout, dst.layout, index.layout,
               workspace.size);
    auto stream = cuda_stream(handle());
    int nthreads = dst.layout.total_nr_elems();
    float spatial_scale = param().spatial_scale;
    float offset = param().offset;
    int sample_height = param().sample_height;
    int sample_width = param().sample_width;
    int channels = src.layout.shape[1];
    int height = src.layout.shape[2];
    int width = src.layout.shape[3];
    int pooled_height = dst.layout.shape[2];
    int pooled_width = dst.layout.shape[3];
    using namespace ::megdnn::roi_align;
    using namespace ::megdnn::cuda::roi_align;
#define cb(DType)                                                             \
    if (src.layout.dtype == DType()) {                                        \
        using T = typename DTypeTrait<DType>::ctype;                          \
        switch (param().mode) {                                               \
            case param::ROIAlign::Mode::MAX:                                  \
                forward_proxy<T, MaxPooler<T>>(                               \
                        nthreads, src.ptr<T>(), spatial_scale, offset,        \
                        channels, height, width, pooled_height, pooled_width, \
                        sample_height, sample_width, rois.ptr<T>(),           \
                        dst.ptr<T>(), index.ptr<dt_int32>(), stream);         \
                break;                                                        \
            case param::ROIAlign::Mode::AVERAGE:                              \
                forward_proxy<T, AveragePooler<T>>(                           \
                        nthreads, src.ptr<T>(), spatial_scale, offset,        \
                        channels, height, width, pooled_height, pooled_width, \
                        sample_height, sample_width, rois.ptr<T>(),           \
                        dst.ptr<T>(), index.ptr<dt_int32>(), stream);         \
                break;                                                        \
            default:                                                          \
                megdnn_assert_internal(false);                                \
        }                                                                     \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
}

void ROIAlignBackwardImpl::exec(_megdnn_tensor_in diff, _megdnn_tensor_in rois,
                                _megdnn_tensor_in index,
                                _megdnn_tensor_out grad,
                                _megdnn_workspace workspace) {
    check_exec(diff.layout, rois.layout, index.layout, grad.layout,
               workspace.size);
    auto stream = cuda_stream(handle());
    int nthreads = diff.layout.total_nr_elems();
    float spatial_scale = param().spatial_scale;
    float offset = param().offset;
    int sample_height = param().sample_height;
    int sample_width = param().sample_width;
    int channels = grad.layout.shape[1];
    int height = grad.layout.shape[2];
    int width = grad.layout.shape[3];
    int pooled_height = diff.layout.shape[2];
    int pooled_width = diff.layout.shape[3];
    using namespace ::megdnn::roi_align;
    using namespace ::megdnn::cuda::roi_align;
    cuda_check(cudaMemsetAsync(
            grad.raw_ptr, 0,
            grad.layout.total_nr_elems() * grad.layout.dtype.size(), stream));
#define cb(DType)                                                            \
    if (diff.layout.dtype == DType()) {                                      \
        using T = typename DTypeTrait<DType>::ctype;                         \
        switch (param().mode) {                                              \
            case param::ROIAlign::Mode::MAX:                                 \
                roi_align::backward_proxy<T, BwdMaxPooler<T>>(               \
                        nthreads, diff.ptr<T>(), index.ptr<dt_int32>(),      \
                        spatial_scale, offset, channels, height, width,      \
                        pooled_height, pooled_width, sample_height,          \
                        sample_width, rois.ptr<T>(), grad.ptr<T>(), stream); \
                break;                                                       \
            case param::ROIAlign::Mode::AVERAGE:                             \
                roi_align::backward_proxy<T, BwdAveragePooler<T>>(           \
                        nthreads, diff.ptr<T>(), index.ptr<dt_int32>(),      \
                        spatial_scale, offset, channels, height, width,      \
                        pooled_height, pooled_width, sample_height,          \
                        sample_width, rois.ptr<T>(), grad.ptr<T>(), stream); \
                break;                                                       \
            default:                                                         \
                megdnn_assert_internal(false);                               \
        }                                                                    \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
}

}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen

