/**
 * \file dnn/src/naive/roi_align/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/roi_align/opr_impl.h"

#include "src/common/roi_align_helper.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace roi_align;

namespace {

using Param = megdnn::ROIAlign::Param;

template <typename T, typename Pooler>
void forward_impl(_megdnn_tensor_in src, _megdnn_tensor_in rois,
                  _megdnn_tensor_in dst, _megdnn_tensor_out index,
                  float spatial_scale, float offset, const int sample_height,
                  const int sample_width) {
    size_t channels = src.layout[1], hi = src.layout[2], wi = src.layout[3];
    size_t pooled_height = dst.layout[2], pooled_width = dst.layout[3];

    size_t total_nr_elems = dst.layout.total_nr_elems();
    int height = hi, width = wi;
    for (size_t idx = 0; idx < total_nr_elems; ++idx) {
        int pw = idx % pooled_width;
        int ph = (idx / pooled_width) % pooled_height;
        int c = (idx / pooled_width / pooled_height) % channels;
        int n = idx / pooled_width / pooled_height / channels;

        auto rois_ptr = rois.ptr<T>() + n * 5;
        int roi_batch_ind = rois_ptr[0];
        float roi_start_w = rois_ptr[1] * spatial_scale - offset;
        float roi_start_h = rois_ptr[2] * spatial_scale - offset;
        float roi_end_w = rois_ptr[3] * spatial_scale - offset;
        float roi_end_h = rois_ptr[4] * spatial_scale - offset;

        float roi_width = std::max(roi_end_w - roi_start_w, ((float)(0.0)));
        float roi_height = std::max(roi_end_h - roi_start_h, ((float)(0.0)));
        float bin_size_h = static_cast<float>(roi_height) /
                           static_cast<float>(pooled_height);
        float bin_size_w = static_cast<float>(roi_width) /
                           static_cast<float>(pooled_width);

        auto feat_map_ptr =
                src.ptr<T>() + (roi_batch_ind * channels + c) * height * width;
        float sample_h_rate = 1.0f / float(sample_height);
        float sample_w_rate = 1.0f / float(sample_width);
        float hcenter;
        float wcenter;

        Pooler pooler;
        for (int h_iter = 0; h_iter < sample_height; ++h_iter) {
            for (int w_iter = 0; w_iter < sample_width; ++w_iter) {
                hcenter = roi_start_h +
                          bin_size_h * (ph + sample_h_rate * (h_iter + 0.5f));
                wcenter = roi_start_w +
                          bin_size_w * (pw + sample_w_rate * (w_iter + 0.5f));
                T val = bilinear_interp(feat_map_ptr, hcenter, wcenter, height,
                                        width);
                int idx = h_iter * sample_width + w_iter;
                pooler.feed(val, idx);
            }
        }
        pooler.writeback_val(dst.ptr<T>()[idx]);
        pooler.writeback_idx(index.ptr<dt_int32>()[idx]);
    }
}

template <typename T>
void forward(_megdnn_tensor_in src, _megdnn_tensor_in rois,
             _megdnn_tensor_out dst, _megdnn_tensor_out index,
             const Param& param) {
    using namespace ::megdnn::roi_align;
    switch (param.mode) {
        case param::ROIAlign::Mode::MAX:
            forward_impl<T, MaxPooler<T>>(
                    src, rois, dst, index, param.spatial_scale, param.offset,
                    param.sample_height, param.sample_width);
            break;
        case param::ROIAlign::Mode::AVERAGE:
            forward_impl<T, AveragePooler<T>>(
                    src, rois, dst, index, param.spatial_scale, param.offset,
                    param.sample_height, param.sample_width);
            break;
        default:
            megdnn_assert_internal(false);
    }
}

template <typename T, typename BwdPooler>
void backward_impl(_megdnn_tensor_in diff, _megdnn_tensor_in rois,
                   _megdnn_tensor_in index, _megdnn_tensor_out grad,
                   float spatial_scale, float offset, const int sample_height,
                   const int sample_width) {
    size_t channels = grad.layout[1], hi = grad.layout[2], wi = grad.layout[3];
    size_t pooled_height = diff.layout[2], pooled_width = diff.layout[3];

    size_t total_nr_elems = diff.layout.total_nr_elems();
    int height = hi, width = wi;

    for (size_t idx = 0; idx < total_nr_elems; ++idx) {
        int pw = idx % pooled_width;
        int ph = (idx / pooled_width) % pooled_height;
        int c = (idx / pooled_width / pooled_height) % channels;
        int n = idx / pooled_width / pooled_height / channels;

        auto rois_ptr = rois.ptr<T>() + n * 5;
        int roi_batch_ind = rois_ptr[0];
        float roi_start_w = rois_ptr[1] * spatial_scale - offset;
        float roi_start_h = rois_ptr[2] * spatial_scale - offset;
        float roi_end_w = rois_ptr[3] * spatial_scale - offset;
        float roi_end_h = rois_ptr[4] * spatial_scale - offset;

        float roi_width = std::max(roi_end_w - roi_start_w, ((float)(0.0)));
        float roi_height = std::max(roi_end_h - roi_start_h, ((float)(0.0)));
        float bin_size_h = static_cast<float>(roi_height) /
                           static_cast<float>(pooled_height);
        float bin_size_w = static_cast<float>(roi_width) /
                           static_cast<float>(pooled_width);

        // regularly sample from a sample_height * sample_width grid
        auto grad_ptr =
                grad.ptr<T>() + (roi_batch_ind * channels + c) * height * width;
        BwdPooler pooler{ph,         pw,        sample_height, sample_width,
                         height,     width,     roi_start_h,   roi_start_w,
                         bin_size_h, bin_size_w};
        pooler.update(static_cast<int>(idx), diff.ptr<T>(),
                      index.ptr<dt_int32>(), grad_ptr);
    }
}

template <typename T>
void backward(_megdnn_tensor_in diff, _megdnn_tensor_in rois,
              _megdnn_tensor_in index, _megdnn_tensor_out grad,
              const Param& param) {
    using namespace ::megdnn::roi_align;
    switch (param.mode) {
        case param::ROIAlign::Mode::MAX:
            backward_impl<T, BwdMaxPooler<T>>(
                    diff, rois, index, grad, param.spatial_scale, param.offset,
                    param.sample_height, param.sample_width);
            break;
        case param::ROIAlign::Mode::AVERAGE:
            backward_impl<T, BwdAveragePooler<T>>(
                    diff, rois, index, grad, param.spatial_scale, param.offset,
                    param.sample_height, param.sample_width);
            break;
        default:
            megdnn_assert_internal(false);
    }
}

}  // namespace

namespace megdnn {
namespace naive {

void ROIAlignForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in rois,
                               _megdnn_tensor_out dst, _megdnn_tensor_out index,
                               _megdnn_workspace workspace) {
    check_exec(src.layout, rois.layout, dst.layout, index.layout,
               workspace.size);
#define cb(DType)                                                            \
    if (src.layout.dtype == DType()) {                                       \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                        \
                forward<typename DTypeTrait<DType>::ctype>(src, rois, dst,   \
                                                           index, param())); \
        return;                                                              \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
}

void ROIAlignBackwardImpl::exec(_megdnn_tensor_in diff, _megdnn_tensor_in rois,
                                _megdnn_tensor_in index,
                                _megdnn_tensor_out grad,
                                _megdnn_workspace workspace) {
    check_exec(diff.layout, rois.layout, index.layout, grad.layout,
               workspace.size);
#define cb(DType)                                                              \
    if (diff.layout.dtype == DType()) {                                        \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                          \
                backward<typename DTypeTrait<DType>::ctype>(diff, rois, index, \
                                                            grad, param()));   \
        return;                                                                \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
}

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen

