/**
 * \file dnn/src/naive/deformable_ps_roi_pooling/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/utils.h"
#include "src/naive/handle.h"

#include "src/naive/deformable_ps_roi_pooling/opr_impl.h"

using namespace megdnn;
using namespace naive;

using Fwd = DeformablePSROIPoolingForwardImpl;
using Bwd = DeformablePSROIPoolingBackwardImpl;

/* ============== Fwd Implementation ============== */

namespace {

float bilinear_interp(const float* data, const int /* IH */, const int IW,
                      const float h, const float w) {
    int h1 = floor(h), h2 = ceil(h);
    int w1 = floor(w), w2 = ceil(w);
    float dist_h = (float)(h - h1);
    float dist_w = (float)(w - w1);
    float value11 = data[h1 * IW + w1];
    float value12 = data[h2 * IW + w1];
    float value21 = data[h1 * IW + w2];
    float value22 = data[h2 * IW + w2];
    float value = (1 - dist_w) * (1 - dist_h) * value11 +
                  (1 - dist_w) * dist_h * value12 +
                  dist_w * (1 - dist_h) * value21 + dist_w * dist_h * value22;
    return value;
}

void deformable_ps_roi_pooling_forward(const float* data, const float* rois,
                                       const float* trans, float* out_data,
                                       float* out_count, int IC, int IH, int IW,
                                       bool no_trans, int nr_bbox, int nr_cls,
                                       int pool_h, int pool_w, int part_sz,
                                       int sample_per_part, float trans_std,
                                       float scale) {
    const int icpcls = IC / nr_cls;

    for (int n = 0; n < nr_bbox; ++n)
        for (int ic = 0; ic < IC; ++ic)
            for (int ph = 0; ph < pool_h; ++ph)
                for (int pw = 0; pw < pool_w; ++pw) {
                    int idx = ((n * IC + ic) * pool_h + ph) * pool_w + pw;
                    const float* rois_ptr = &rois[n * 5];

                    int roi_batch_idx = rois_ptr[0];

                    float roi_w_l =
                            static_cast<float>(round(rois_ptr[1])) * scale -
                            0.5;
                    float roi_h_l =
                            static_cast<float>(round(rois_ptr[2])) * scale -
                            0.5;
                    float roi_w_r =
                            static_cast<float>(round(rois_ptr[3]) + 1.) *
                                    scale -
                            0.5;
                    float roi_h_r =
                            static_cast<float>(round(rois_ptr[4]) + 1.) *
                                    scale -
                            0.5;

                    // Force too small ROIs to be 1x1
                    float roi_w = std::max(roi_w_r - roi_w_l, 0.1f);  // avoid 0
                    float roi_h = std::max(roi_h_r - roi_h_l, 0.1f);

                    // Compute w and h at bottom
                    float bin_sz_h = roi_h / static_cast<float>(pool_h);
                    float bin_sz_w = roi_w / static_cast<float>(pool_w);

                    float sub_bin_sz_h =
                            bin_sz_h / static_cast<float>(sample_per_part);
                    float sub_bin_sz_w =
                            bin_sz_w / static_cast<float>(sample_per_part);

                    int count = 0;
                    int cls_id = ic / icpcls;
                    float sum = 0, trans_x = 0, trans_y = 0;
                    float wstart = static_cast<float>(pw) * bin_sz_w + roi_w_l;
                    float hstart = static_cast<float>(ph) * bin_sz_h + roi_h_l;

                    if (!no_trans) {
                        int part_h = floor(static_cast<float>(ph) / pool_h *
                                           part_sz);
                        int part_w = floor(static_cast<float>(pw) / pool_w *
                                           part_sz);
                        int x_idx = (((n * nr_cls + cls_id) * 2) * part_sz +
                                     part_h) *
                                            part_sz +
                                    part_w;
                        int y_idx = (((n * nr_cls + cls_id) * 2 + 1) * part_sz +
                                     part_h) *
                                            part_sz +
                                    part_w;
                        trans_x = trans[x_idx] * static_cast<float>(trans_std);
                        trans_y = trans[y_idx] * static_cast<float>(trans_std);
                    }

                    wstart += trans_x * roi_w;
                    hstart += trans_y * roi_h;

                    const float* data_ptr =
                            data + (roi_batch_idx * IC + ic) * IH * IW;

                    for (int ih = 0; ih < sample_per_part; ih++) {
                        for (int iw = 0; iw < sample_per_part; iw++) {
                            float w = wstart + iw * sub_bin_sz_w;
                            float h = hstart + ih * sub_bin_sz_h;
                            // bilinear interpolation
                            if (w < -0.5 || w > IW - 0.5 || h < -0.5 ||
                                h > IH - 0.5)
                                continue;
                            w = std::min(std::max(w, 0.f), IW - 1.f);
                            h = std::min(std::max(h, 0.f), IH - 1.f);
                            float val = bilinear_interp(data_ptr, IH, IW, h, w);
                            sum += val, count++;
                        }
                    }
                    out_data[idx] = count == 0 ? (float)(0) : sum / count;
                    out_count[idx] = count;
                }
}

void deformable_ps_roi_pool_backward_acc_kernel(
        const float* data, const float* rois, const float* trans,
        const float* out_diff, const float* out_count, float* data_diff,
        float* trans_diff, int IC, int IH, int IW, bool no_trans, int nr_bbox,
        int nr_cls, int pool_h, int pool_w, int part_sz, int sample_per_part,
        float trans_std, float scale) {
    const int icpcls = IC / nr_cls;
    for (int n = 0; n < nr_bbox; ++n)
        for (int ic = 0; ic < IC; ++ic)
            for (int ph = 0; ph < pool_h; ++ph)
                for (int pw = 0; pw < pool_w; ++pw) {
                    const int idx = ((n * IC + ic) * pool_h + ph) * pool_w + pw;
                    const float* rois_ptr = &rois[n * 5];

                    int roi_batch_idx = rois_ptr[0];

                    float roi_w_l =
                            static_cast<float>(round(rois_ptr[1])) * scale -
                            0.5;
                    float roi_h_l =
                            static_cast<float>(round(rois_ptr[2])) * scale -
                            0.5;
                    float roi_w_r =
                            static_cast<float>(round(rois_ptr[3]) + 1.) *
                                    scale -
                            0.5;
                    float roi_h_r =
                            static_cast<float>(round(rois_ptr[4]) + 1.) *
                                    scale -
                            0.5;

                    // Force too small ROIs to be 1x1
                    float roi_w = std::max(roi_w_r - roi_w_l, 0.1f);  // avoid 0
                    float roi_h = std::max(roi_h_r - roi_h_l, 0.1f);

                    // Compute w and h at bottom
                    float bin_sz_h = roi_h / static_cast<float>(pool_h);
                    float bin_sz_w = roi_w / static_cast<float>(pool_w);

                    float sub_bin_sz_h =
                            bin_sz_h / static_cast<float>(sample_per_part);
                    float sub_bin_sz_w =
                            bin_sz_w / static_cast<float>(sample_per_part);

                    int part_h = 0, part_w = 0, cls_id = ic / icpcls;
                    float trans_x = 0, trans_y = 0;
                    float wstart = static_cast<float>(pw) * bin_sz_w + roi_w_l;
                    float hstart = static_cast<float>(ph) * bin_sz_h + roi_h_l;

                    if (!no_trans) {
                        part_h = floor(static_cast<float>(ph) / pool_h *
                                       part_sz);
                        part_w = floor(static_cast<float>(pw) / pool_w *
                                       part_sz);
                        int x_idx = (((n * nr_cls + cls_id) * 2) * part_sz +
                                     part_h) *
                                            part_sz +
                                    part_w;
                        int y_idx = (((n * nr_cls + cls_id) * 2 + 1) * part_sz +
                                     part_h) *
                                            part_sz +
                                    part_w;
                        trans_x = trans[x_idx] * static_cast<float>(trans_std);
                        trans_y = trans[y_idx] * static_cast<float>(trans_std);
                    }

                    wstart += trans_x * roi_w;
                    hstart += trans_y * roi_h;

                    if (out_count[idx] <= 0)
                        continue;

                    float diff_val = out_diff[idx] / out_count[idx];

                    const int data_idx = (roi_batch_idx * IC + ic) * IH * IW;

                    float* data_diff_ptr;
                    const float* data_ptr;

                    for (int ih = 0; ih < sample_per_part; ih++) {
                        for (int iw = 0; iw < sample_per_part; iw++) {
                            float w = wstart + iw * sub_bin_sz_w;
                            float h = hstart + ih * sub_bin_sz_h;
                            // bilinear interpolation
                            if (w < -0.5 || w > IW - 0.5 || h < -0.5 ||
                                h > IH - 0.5)
                                continue;
                            w = std::min(std::max(w, 0.f), IW - 1.f),
                            h = std::min(std::max(h, 0.f), IH - 1.f);
                            // backward on feature
                            int x0 = floor(w), x1 = ceil(w);
                            int y0 = floor(h), y1 = ceil(h);
                            float dist_x = w - x0, dist_y = h - y0;
                            float q00 = (1 - dist_x) * (1 - dist_y);
                            float q01 = (1 - dist_x) * dist_y;
                            float q10 = dist_x * (1 - dist_y);
                            float q11 = dist_x * dist_y;

                            data_diff_ptr = &data_diff[data_idx];

                            data_diff_ptr[y0 * IW + x0] += q00 * diff_val;
                            data_diff_ptr[y1 * IW + x0] += q01 * diff_val;
                            data_diff_ptr[y0 * IW + x1] += q10 * diff_val;
                            data_diff_ptr[y1 * IW + x1] += q11 * diff_val;

                            if (no_trans)
                                continue;

                            data_ptr = &data[data_idx];

                            float U00 = data_ptr[y0 * IW + x0];
                            float U01 = data_ptr[y1 * IW + x0];
                            float U10 = data_ptr[y0 * IW + x1];
                            float U11 = data_ptr[y1 * IW + x1];

                            float diff_x = (U11 * dist_y + U10 * (1 - dist_y) -
                                            U01 * dist_y - U00 * (1 - dist_y)) *
                                           trans_std * diff_val;
                            float diff_y = (U11 * dist_x + U01 * (1 - dist_x) -
                                            U10 * dist_x - U00 * (1 - dist_x)) *
                                           trans_std * diff_val;

                            diff_x *= roi_w, diff_y *= roi_h;

                            int diff_x_idx =
                                    (((n * nr_cls + cls_id) * 2) * part_sz +
                                     part_h) *
                                            part_sz +
                                    part_w;
                            int diff_y_idx =
                                    (((n * nr_cls + cls_id) * 2 + 1) * part_sz +
                                     part_h) *
                                            part_sz +
                                    part_w;

                            trans_diff[diff_x_idx] += diff_x;
                            trans_diff[diff_y_idx] += diff_y;
                        }
                    }
                }
}

}  // namespace

void Fwd::exec(_megdnn_tensor_in data, _megdnn_tensor_in rois,
               _megdnn_tensor_in trans, _megdnn_tensor_out out_data,
               _megdnn_tensor_out out_count, _megdnn_workspace workspace) {
    check_exec(data.layout, rois.layout, trans.layout, out_data.layout,
               out_count.layout, workspace.size);

    auto kern = [data, rois, trans, out_data,
                 out_count](const DeformablePSROIPoolingBase::Param& param) {
        bool no_trans = param.no_trans;
        size_t pool_h = param.pooled_h, pool_w = param.pooled_w;
        size_t part_sz = param.part_size,
               sample_per_part = param.sample_per_part;
        float trans_std = param.trans_std, scale = param.spatial_scale;

        size_t nr_bbox = rois.layout[0];
        size_t nr_cls = no_trans ? 1 : trans.layout[1] / 2;
        size_t IC = data.layout[1], IH = data.layout[2], IW = data.layout[3];

        const float* data_ptr = data.ptr<float>();
        const float* rois_ptr = rois.ptr<float>();
        const float* trans_ptr = trans.ptr<float>();

        float* out_data_ptr = out_data.ptr<float>();
        float* out_count_ptr = out_count.ptr<float>();

        auto&& out_data_elems = out_data.layout.total_nr_elems();
        auto&& out_count_elems = out_count.layout.total_nr_elems();
        size_t out_data_bytes = sizeof(float[out_data_elems]);
        size_t out_count_bytes = sizeof(float[out_count_elems]);

        memset(out_data_ptr, 0, out_data_bytes);
        memset(out_count_ptr, 0, out_count_bytes);

        deformable_ps_roi_pooling_forward(
                data_ptr, rois_ptr, trans_ptr, out_data_ptr, out_count_ptr, IC,
                IH, IW, no_trans, nr_bbox, nr_cls, pool_h, pool_w, part_sz,
                sample_per_part, trans_std, scale);
    };

    MEGDNN_DISPATCH_CPU_KERN_OPR(kern(param()));
    return;
}

/* ============== Bwd Implementation ============== */

void Bwd::exec(_megdnn_tensor_in data, _megdnn_tensor_in rois,
               _megdnn_tensor_in trans, _megdnn_tensor_in out_diff,
               _megdnn_tensor_in out_count, _megdnn_tensor_out data_diff,
               _megdnn_tensor_out trans_diff, _megdnn_workspace workspace) {
    check_exec(data.layout, rois.layout, trans.layout, out_diff.layout,
               out_count.layout, data_diff.layout, trans_diff.layout,
               workspace.size);
    auto kern = [data, rois, trans, out_diff, out_count, data_diff,
                 trans_diff](const DeformablePSROIPoolingBase::Param& param) {
        bool no_trans = param.no_trans;
        size_t pool_h = param.pooled_h, pool_w = param.pooled_w;
        size_t part_sz = param.part_size,
               sample_per_part = param.sample_per_part;
        float trans_std = param.trans_std, scale = param.spatial_scale;

        size_t nr_bbox = rois.layout[0];
        size_t nr_cls = no_trans ? 1 : trans.layout[1] / 2;
        size_t IC = data.layout[1], IH = data.layout[2], IW = data.layout[3];

        const float* data_ptr = data.ptr<float>();
        const float* rois_ptr = rois.ptr<float>();
        const float* trans_ptr = trans.ptr<float>();
        const float* out_diff_ptr = out_diff.ptr<float>();
        const float* out_count_ptr = out_count.ptr<float>();

        float* data_diff_ptr = data_diff.ptr<float>();
        float* trans_diff_ptr = trans_diff.ptr<float>();

        auto&& data_diff_elems = data_diff.layout.total_nr_elems();
        auto&& trans_diff_elems = trans_diff.layout.total_nr_elems();
        size_t data_diff_bytes = sizeof(float[data_diff_elems]);
        size_t trans_diff_bytes = sizeof(float[trans_diff_elems]);

        memset(data_diff_ptr, 0, data_diff_bytes);
        memset(trans_diff_ptr, 0, trans_diff_bytes);
        deformable_ps_roi_pool_backward_acc_kernel(
                data_ptr, rois_ptr, trans_ptr, out_diff_ptr, out_count_ptr,
                data_diff_ptr, trans_diff_ptr, IC, IH, IW, no_trans, nr_bbox,
                nr_cls, pool_h, pool_w, part_sz, sample_per_part, trans_std,
                scale);
    };

    MEGDNN_DISPATCH_CPU_KERN_OPR(kern(param()));
    return;
}

// vim: syntax=cpp.doxygen
