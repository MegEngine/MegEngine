/**
 * \file dnn/src/naive/roi_pooling/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/roi_pooling/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "src/common/roi_pooling_helper.h"

namespace {

using namespace megdnn;
using Param = param::ROIPooling;

template <typename T, typename Pooler>
void forward_impl(_megdnn_tensor_in src,
        _megdnn_tensor_in rois,
        _megdnn_tensor_in dst,
        _megdnn_tensor_out index,
        float spatial_scale)
{
    auto C = src.layout.shape[1],
         IH = src.layout.shape[2],
         IW = src.layout.shape[3];
    auto OH = dst.layout.shape[2],
         OW = dst.layout.shape[3];

    auto total_nr_elem = dst.layout.total_nr_elems();
    auto pooled_height = OH, pooled_width = OW;
    auto height = IH, width = IW;
    auto channels = C;
    for (size_t i = 0; i < total_nr_elem; ++i) {
        int pw = i % pooled_width;
        int ph = (i / pooled_width) % pooled_height;
        int c = (i / pooled_width / pooled_height) % channels;
        int n = i / pooled_width / pooled_height / channels;
        auto rois_ptr = rois.ptr<T>() + n * 5;
        int roi_batch_ind = rois_ptr[0];
        int roi_start_w = round(rois_ptr[1] * spatial_scale);
        int roi_start_h = round(rois_ptr[2] * spatial_scale);
        int roi_end_w = round(rois_ptr[3] * spatial_scale);
        int roi_end_h = round(rois_ptr[4] * spatial_scale);
        // Force malformed ROIs to be 1x1
        int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
        float bin_size_h = static_cast<float>(roi_height)
            / static_cast<float>(pooled_height);
        float bin_size_w = static_cast<float>(roi_width)
            / static_cast<float>(pooled_width);

        int hstart = static_cast<int>(floor(static_cast<float>(ph)
                    * bin_size_h));
        int wstart = static_cast<int>(floor(static_cast<float>(pw)
                    * bin_size_w));
        int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                    * bin_size_h));
        int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                    * bin_size_w));
        // Add roi offsets and clip to input boundaries
        hstart = std::min<int>(std::max(hstart + roi_start_h, 0), height);
        hend = std::min<int>(std::max(hend + roi_start_h, 0), height);
        wstart = std::min<int>(std::max(wstart + roi_start_w, 0), width);
        wend = std::min<int>(std::max(wend + roi_start_w, 0), width);

        Pooler pooler;
        auto feat_map_ptr = src.ptr<T>() +
            (roi_batch_ind * channels + c) * height * width;
        for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
                int bottom_i = h * width + w;
                pooler.feed(feat_map_ptr[bottom_i], bottom_i);
            }
        }
        pooler.writeback_val(dst.ptr<T>()[i]);
        pooler.writeback_idx(index.ptr<dt_int32>()[i]);
    }
}

template <typename T>
void forward(_megdnn_tensor_in src,
        _megdnn_tensor_in rois,
        _megdnn_tensor_out dst,
        _megdnn_tensor_out index,
        const Param &param)
{
    using namespace ::megdnn::roi_pooling;
    switch (param.mode) {
        case param::ROIPooling::Mode::MAX:
            forward_impl<T, MaxPooler<T>>(src, rois, dst, index, param.scale);
            break;
        case param::ROIPooling::Mode::AVERAGE:
            forward_impl<T, AveragePooler<T>>(src, rois, dst, index, param.scale);
            break;
        default:
            megdnn_assert_internal(false);
    }
}

template <typename T, typename BwdPooler>
void backward_impl(_megdnn_tensor_in diff,
        _megdnn_tensor_in src,
        _megdnn_tensor_in rois,
        _megdnn_tensor_in index,
        _megdnn_tensor_out grad,
        float scale)
{
    auto batch_size = src.layout.shape[0],
         channels = src.layout.shape[1],
         height = src.layout.shape[2],
         width = src.layout.shape[3];
    auto total_nr_elem = batch_size * channels * height * width;
    auto num_rois = rois.layout.shape[0];
    auto spatial_scale = scale;

    auto pooled_height = diff.layout.shape[2],
         pooled_width = diff.layout.shape[3];
    for (size_t i = 0; i < total_nr_elem; ++i) {
        // (n, c, h, w) coords in bottom data
        int w = i % width;
        int h = (i / width) % height;
        int c = (i / width / height) % channels;
        int n = i / width / height / channels;

        T gradient = T(0);
        // Accumulate gradient over all ROIs that pooled this element
        for (size_t roi_n = 0; roi_n < num_rois; ++roi_n) {
            const T* offset_rois = rois.ptr<T>() + roi_n * 5;
            int roi_batch_ind = offset_rois[0];
            if (n != roi_batch_ind) {
                continue;
            }

            int roi_start_w = round(offset_rois[1] * spatial_scale);
            int roi_start_h = round(offset_rois[2] * spatial_scale);
            int roi_end_w = round(offset_rois[3] * spatial_scale);
            int roi_end_h = round(offset_rois[4] * spatial_scale);

            // Skip if ROI doesn't include (h, w)
            const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                                 h >= roi_start_h && h <= roi_end_h);
            if (!in_roi) {
                continue;
            }

            int offset = (roi_n * channels + c) * pooled_height * pooled_width;
            const T* offset_src_diff = diff.ptr<T>() + offset;
            const int* offset_fp_idx = index.ptr<dt_int32>() + offset;

            // Compute feasible set of pooled units that could have pooled
            // this bottom unit

            // Force malformed ROIs to be 1x1
            int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
            int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);

            float bin_size_h = static_cast<float>(roi_height)
                               / static_cast<float>(pooled_height);
            float bin_size_w = static_cast<float>(roi_width)
                               / static_cast<float>(pooled_width);

            int phstart = floor(static_cast<float>(h - roi_start_h) / bin_size_h);
            int phend = ceil(static_cast<float>(h - roi_start_h + 1) / bin_size_h);
            int pwstart = floor(static_cast<float>(w - roi_start_w) / bin_size_w);
            int pwend = ceil(static_cast<float>(w - roi_start_w + 1) / bin_size_w);

            phstart = std::min<int>(std::max(phstart, 0), pooled_height);
            phend = std::min<int>(std::max(phend, 0), pooled_height);
            pwstart = std::min<int>(std::max(pwstart, 0), pooled_width);
            pwend = std::min<int>(std::max(pwend, 0), pooled_width);

            for (int ph = phstart; ph < phend; ++ph) {
                for (int pw = pwstart; pw < pwend; ++pw) {
                    BwdPooler pooler;
                    pooler.update(ph, pw, h, w, bin_size_h, bin_size_w,
                            roi_start_h, roi_start_w,
                            pooled_height, pooled_width,
                            height, width,
                            offset_src_diff,
                            offset_fp_idx,
                            gradient);
                }
            }
        }
        grad.ptr<T>()[i] = gradient;
    }
}

template <typename T>
void backward(_megdnn_tensor_in diff,
        _megdnn_tensor_in src,
        _megdnn_tensor_in rois,
        _megdnn_tensor_in index,
        _megdnn_tensor_out grad,
        const Param &param)
{
    using namespace ::megdnn::roi_pooling;
    switch (param.mode) {
        case param::ROIPooling::Mode::MAX:
            backward_impl<T, BwdMaxPooler<T>>(diff, src, rois, index, grad,
                    param.scale);
            break;
        case param::ROIPooling::Mode::AVERAGE:
            backward_impl<T, BwdAveragePooler<T>>(diff, src, rois, index, grad,
                    param.scale);
            break;
        default:
            megdnn_assert_internal(false);
    }
}

} // anonymous namespace

namespace megdnn {
namespace naive {

void ROIPoolingForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in rois,
        _megdnn_tensor_out dst,
        _megdnn_tensor_out index,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, rois.layout, dst.layout, index.layout,
            workspace.size);
#define cb(DType) \
    if (src.layout.dtype == DType()) { \
        MEGDNN_DISPATCH_CPU_KERN_OPR( \
                forward<typename DTypeTrait<DType>::ctype>( \
                    src, rois, dst, index, param())); \
        return; \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
}

void ROIPoolingBackwardImpl::exec(_megdnn_tensor_in diff,
        _megdnn_tensor_in src,
        _megdnn_tensor_in rois,
        _megdnn_tensor_in index,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(diff.layout, src.layout, rois.layout,
            index.layout, grad.layout, workspace.size);
#define cb(DType) \
    if (diff.layout.dtype == DType()) { \
        MEGDNN_DISPATCH_CPU_KERN_OPR( \
                backward<typename DTypeTrait<DType>::ctype>( \
                    diff, src, rois, index, grad, param())); \
        return; \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
}

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen

