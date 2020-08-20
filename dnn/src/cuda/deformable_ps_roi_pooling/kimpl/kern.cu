/**
 * \file dnn/src/cuda/deformable_ps_roi_pooling/kimpl/kern.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/deformable_ps_roi_pooling/kimpl/kern.cuh"
#include "src/cuda/query_blocksize.cuh"

namespace {

using Param = megdnn::cuda::deformable_ps_roi_pooling::Param;

__device__ float bilinear_interp(const float* data, const int IH, const int IW,
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

__global__ void DeformablePSROIPoolForwardKern(Param p, const float* data,
                                               const float* rois,
                                               const float* trans,
                                               float* out_data,
                                               float* out_count) {
    const int loops = p.nr_bbox * p.IC * p.pool_h * p.pool_w;
    const int icpcls = p.IC / p.nr_cls;

    KERN_FOR(idx, loops) {
        const int pw = idx % p.pool_w;
        const int ph = (idx / p.pool_w) % p.pool_h;
        const int ic = (idx / p.pool_w / p.pool_h) % p.IC;
        const int n = (idx / p.pool_w / p.pool_h / p.IC);
        const float* rois_ptr = &rois[n * 5];

        int roi_batch_idx = rois_ptr[0];

        float roi_w_l = static_cast<float>(round(rois_ptr[1])) * p.scale - 0.5;
        float roi_h_l = static_cast<float>(round(rois_ptr[2])) * p.scale - 0.5;
        float roi_w_r =
                static_cast<float>(round(rois_ptr[3]) + 1.) * p.scale - 0.5;
        float roi_h_r =
                static_cast<float>(round(rois_ptr[4]) + 1.) * p.scale - 0.5;

        // Force too small ROIs to be 1x1
        float roi_w = max(roi_w_r - roi_w_l, 0.1);  // avoid 0
        float roi_h = max(roi_h_r - roi_h_l, 0.1);

        // Compute w and h at bottom
        float bin_sz_h = roi_h / static_cast<float>(p.pool_h);
        float bin_sz_w = roi_w / static_cast<float>(p.pool_w);

        float sub_bin_sz_h = bin_sz_h / static_cast<float>(p.sample_per_part);
        float sub_bin_sz_w = bin_sz_w / static_cast<float>(p.sample_per_part);

        int count = 0;
        int cls_id = ic / icpcls;
        float sum = 0, trans_x = 0, trans_y = 0;
        float hstart = static_cast<float>(ph) * bin_sz_h + roi_h_l;
        float wstart = static_cast<float>(pw) * bin_sz_w + roi_w_l;

        if (!p.no_trans) {
            int part_h = floor(static_cast<float>(ph) / p.pool_h * p.part_sz);
            int part_w = floor(static_cast<float>(pw) / p.pool_w * p.part_sz);
            int x_idx = (((n * p.nr_cls + cls_id) * 2) * p.part_sz + part_h) *
                                p.part_sz +
                        part_w;
            int y_idx =
                    (((n * p.nr_cls + cls_id) * 2 + 1) * p.part_sz + part_h) *
                            p.part_sz +
                    part_w;
            trans_x = trans[x_idx] * static_cast<float>(p.trans_std);
            trans_y = trans[y_idx] * static_cast<float>(p.trans_std);
        }

        wstart += trans_x * roi_w;
        hstart += trans_y * roi_h;

        const float* data_ptr =
                data + (roi_batch_idx * p.IC + ic) * p.IH * p.IW;

        for (int ih = 0; ih < p.sample_per_part; ih++) {
            for (int iw = 0; iw < p.sample_per_part; iw++) {
                float w = wstart + iw * sub_bin_sz_w;
                float h = hstart + ih * sub_bin_sz_h;
                // bilinear interpolation
                if (w < -0.5 || w > p.IW - 0.5 || h < -0.5 || h > p.IH - 0.5)
                    continue;
                w = min(max(w, 0.), p.IW - 1.);
                h = min(max(h, 0.), p.IH - 1.);
                float val = bilinear_interp(data_ptr, p.IH, p.IW, h, w);
                sum += val, count++;
            }
        }
        out_data[idx] = count == 0 ? (float)(0) : sum / count;
        out_count[idx] = count;
    }
}

__global__ void DeformablePSROIPoolBackwardAccKern(
        Param p, const float* data, const float* rois, const float* trans,
        const float* out_diff, const float* out_count, float* data_diff,
        float* trans_diff) {
    const int loops = p.nr_bbox * p.IC * p.pool_h * p.pool_w;
    const int icpcls = p.IC / p.nr_cls;

    KERN_FOR(idx, loops) {
        const int pw = idx % p.pool_w;
        const int ph = (idx / p.pool_w) % p.pool_h;
        const int ic = (idx / p.pool_w / p.pool_h) % p.IC;
        const int n = (idx / p.pool_w / p.pool_h / p.IC);

        const float* rois_ptr = &rois[n * 5];

        int roi_batch_idx = rois_ptr[0];

        float roi_w_l = static_cast<float>(round(rois_ptr[1])) * p.scale - 0.5;
        float roi_h_l = static_cast<float>(round(rois_ptr[2])) * p.scale - 0.5;
        float roi_w_r =
                static_cast<float>(round(rois_ptr[3]) + 1.) * p.scale - 0.5;
        float roi_h_r =
                static_cast<float>(round(rois_ptr[4]) + 1.) * p.scale - 0.5;

        // Force too small ROIs to be 1x1
        float roi_w = max(roi_w_r - roi_w_l, 0.1);  // avoid 0
        float roi_h = max(roi_h_r - roi_h_l, 0.1);

        // Compute w and h at bottom
        float bin_sz_h = roi_h / static_cast<float>(p.pool_h);
        float bin_sz_w = roi_w / static_cast<float>(p.pool_w);

        float sub_bin_sz_h = bin_sz_h / static_cast<float>(p.sample_per_part);
        float sub_bin_sz_w = bin_sz_w / static_cast<float>(p.sample_per_part);

        int part_h = 0, part_w = 0, cls_id = ic / icpcls;
        float trans_x = 0, trans_y = 0;
        float wstart = static_cast<float>(pw) * bin_sz_w + roi_w_l;
        float hstart = static_cast<float>(ph) * bin_sz_h + roi_h_l;

        if (!p.no_trans) {
            part_h = floor(static_cast<float>(ph) / p.pool_h * p.part_sz);
            part_w = floor(static_cast<float>(pw) / p.pool_w * p.part_sz);
            int x_idx = (((n * p.nr_cls + cls_id) * 2) * p.part_sz + part_h) *
                                p.part_sz +
                        part_w;
            int y_idx =
                    (((n * p.nr_cls + cls_id) * 2 + 1) * p.part_sz + part_h) *
                            p.part_sz +
                    part_w;
            trans_x = trans[x_idx] * static_cast<float>(p.trans_std);
            trans_y = trans[y_idx] * static_cast<float>(p.trans_std);
        }

        wstart += trans_x * roi_w;
        hstart += trans_y * roi_h;

        if (out_count[idx] <= 0)
            continue;

        float diff_val = out_diff[idx] / out_count[idx];

        const int data_idx = (roi_batch_idx * p.IC + ic) * p.IH * p.IW;

        float* data_diff_ptr;
        const float* data_ptr;

        for (int ih = 0; ih < p.sample_per_part; ih++) {
            for (int iw = 0; iw < p.sample_per_part; iw++) {
                float w = wstart + iw * sub_bin_sz_w;
                float h = hstart + ih * sub_bin_sz_h;
                // bilinear interpolation
                if (w < -0.5 || w > p.IW - 0.5 || h < -0.5 || h > p.IH - 0.5)
                    continue;
                w = min(max(w, 0.), p.IW - 1.), h = min(max(h, 0.), p.IH - 1.);
                // backward on feature
                int x0 = floor(w), x1 = ceil(w);
                int y0 = floor(h), y1 = ceil(h);
                float dist_x = w - x0, dist_y = h - y0;
                float q00 = (1 - dist_x) * (1 - dist_y);
                float q01 = (1 - dist_x) * dist_y;
                float q10 = dist_x * (1 - dist_y);
                float q11 = dist_x * dist_y;

                data_diff_ptr = &data_diff[data_idx];

                atomicAdd(&data_diff_ptr[y0 * p.IW + x0], q00 * diff_val);
                atomicAdd(&data_diff_ptr[y1 * p.IW + x0], q01 * diff_val);
                atomicAdd(&data_diff_ptr[y0 * p.IW + x1], q10 * diff_val);
                atomicAdd(&data_diff_ptr[y1 * p.IW + x1], q11 * diff_val);

                if (p.no_trans)
                    continue;

                data_ptr = &data[data_idx];

                float U00 = data_ptr[y0 * p.IW + x0];
                float U01 = data_ptr[y1 * p.IW + x0];
                float U10 = data_ptr[y0 * p.IW + x1];
                float U11 = data_ptr[y1 * p.IW + x1];

                float diff_x = (U11 * dist_y + U10 * (1 - dist_y) -
                                U01 * dist_y - U00 * (1 - dist_y)) *
                               p.trans_std * diff_val;
                float diff_y = (U11 * dist_x + U01 * (1 - dist_x) -
                                U10 * dist_x - U00 * (1 - dist_x)) *
                               p.trans_std * diff_val;

                diff_x *= roi_w, diff_y *= roi_h;

                int diff_x_idx =
                        (((n * p.nr_cls + cls_id) * 2) * p.part_sz + part_h) *
                                p.part_sz +
                        part_w;
                int diff_y_idx =
                        (((n * p.nr_cls + cls_id) * 2 + 1) * p.part_sz +
                         part_h) *
                                p.part_sz +
                        part_w;

                atomicAdd(&trans_diff[diff_x_idx], diff_x);
                atomicAdd(&trans_diff[diff_y_idx], diff_y);
            }
        }
    }
}
}  // namespace

namespace megdnn {
namespace cuda {
namespace deformable_ps_roi_pooling {

void DeformablePSROIPoolForward(const TensorND& data, const TensorND& rois,
                                const TensorND& trans, const TensorND& out_data,
                                const TensorND& out_count, Param& p) {
    const int loops = p.nr_bbox * p.IC * p.pool_h * p.pool_w;
    int nr_thds = query_blocksize_for_kernel(DeformablePSROIPoolForwardKern);
    const int blks = DIVUP(loops, nr_thds);

    const float* data_ptr = data.ptr<float>();
    const float* rois_ptr = rois.ptr<float>();
    const float* trans_ptr = p.no_trans ? NULL : trans.ptr<float>();

    float* out_data_ptr = out_data.ptr<float>();
    float* out_count_ptr = out_count.ptr<float>();

    auto&& out_data_elems = out_data.layout.total_nr_elems();
    auto&& out_count_elems = out_count.layout.total_nr_elems();
    size_t out_data_bytes = sizeof(float) * out_data_elems;
    size_t out_count_bytes = sizeof(float) * out_count_elems;

    cudaMemsetAsync(out_data_ptr, 0, out_data_bytes, p.stream);
    cudaMemsetAsync(out_count_ptr, 0, out_count_bytes, p.stream);

    DeformablePSROIPoolForwardKern<<<blks, nr_thds, 0, p.stream>>>(
            p, data_ptr, rois_ptr, trans_ptr, out_data_ptr, out_count_ptr);
    after_kernel_launch();
}

void DeformablePSROIPoolBackwardAcc(const TensorND& data, const TensorND& rois,
                                    const TensorND& trans,
                                    const TensorND& out_diff,
                                    const TensorND& out_count,
                                    const TensorND& data_diff,
                                    const TensorND& trans_diff, Param& p) {
    const int loops = p.nr_bbox * p.IC * p.pool_h * p.pool_w;
    int nr_thds =
            query_blocksize_for_kernel(DeformablePSROIPoolBackwardAccKern);
    const int blks = DIVUP(loops, nr_thds);

    const float* data_ptr = data.ptr<float>();
    const float* rois_ptr = rois.ptr<float>();
    const float* trans_ptr = p.no_trans ? NULL : trans.ptr<float>();
    const float* out_diff_ptr = out_diff.ptr<float>();
    const float* out_count_ptr = out_count.ptr<float>();

    float* data_diff_ptr = data_diff.ptr<float>();
    float* trans_diff_ptr = trans_diff.ptr<float>();

    auto&& data_diff_elems = data_diff.layout.total_nr_elems();
    auto&& trans_diff_elems = trans_diff.layout.total_nr_elems();
    size_t data_diff_bytes = sizeof(float) * data_diff_elems;
    size_t trans_diff_bytes = sizeof(float) * trans_diff_elems;

    cudaMemsetAsync(data_diff_ptr, 0, data_diff_bytes, p.stream);
    cudaMemsetAsync(trans_diff_ptr, 0, trans_diff_bytes, p.stream);

    DeformablePSROIPoolBackwardAccKern<<<blks, nr_thds, 0, p.stream>>>(
            p, data_ptr, rois_ptr, trans_ptr, out_diff_ptr, out_count_ptr,
            data_diff_ptr, trans_diff_ptr);
    after_kernel_launch();
}

}  // namespace deformable_ps_roi_pooling
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cuda syntax=cuda.doxygen
