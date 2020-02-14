/**
 * \file dnn/src/naive/deformable_conv/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/utils.h"
#include "src/naive/convolution/helper.h"
#include "src/naive/deformable_conv/opr_impl.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;

using Fwd = DeformableConvForwardImpl;
using BwdFlt = DeformableConvBackwardFilterImpl;
using BwdData = DeformableConvBackwardDataImpl;

using AlgoFwd = Fwd::Algorithm;
using AlgoBwdFlt = BwdFlt::Algorithm;
using AlgoBwdData = BwdData::Algorithm;

/* ============== Fwd Implementation ============== */

static float dmcn_bilinear(const float* bottom_data, const size_t stride,
                           const int H, const int W, float h, float w) {
    int h_low = floor(h), w_low = floor(w);
    int h_high = h_low + 1, w_high = w_low + 1;

    float lh = h - h_low, lw = w - w_low;
    float hh = 1 - lh, hw = 1 - lw;
    float v1 = 0, v2 = 0, v3 = 0, v4 = 0;

    if (h_low >= 0 && w_low >= 0)
        v1 = bottom_data[h_low * stride + w_low];
    if (h_low >= 0 && w_high <= W - 1)
        v2 = bottom_data[h_low * stride + w_high];
    if (h_high <= H - 1 && w_low >= 0)
        v3 = bottom_data[h_high * stride + w_low];
    if (h_high <= H - 1 && w_high <= W - 1)
        v4 = bottom_data[h_high * stride + w_high];

    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

static void deformable_conv_forward(
        const float* im, const float* filter, const float* offset,
        const float* mask, float* out, const size_t OC, const size_t IC,
        const size_t N, const size_t FH, const size_t FW, const size_t IH,
        const size_t IW, const size_t PH, const size_t PW, const size_t DH,
        const size_t DW, const size_t SH, const size_t SW, const size_t OH,
        const size_t OW, const size_t group, const size_t deformable_group) {
    const int icpg = IC / group;
    const int ocpg = OC / group;
    const int icpdg = IC / deformable_group;

    for (size_t n = 0; n < N; ++n) {
        for (size_t oc = 0; oc < OC; ++oc) {
            size_t g = oc / ocpg;
            size_t oc_in_group = oc % ocpg;
            size_t icpg_l = icpg * g, icpg_r = icpg * (g + 1);
            for (size_t oh = 0; oh < OH; ++oh)
                for (size_t ow = 0; ow < OW; ++ow) {
                    float sum = 0.f;
                    const int ih = oh * SH - PH;
                    const int iw = ow * SW - PW;
                    for (size_t ic = icpg_l; ic < icpg_r; ++ic) {
                        const size_t dg = ic / icpdg;
                        const size_t ic_in_group = ic % icpg;
                        const float* im_ptr = &im[(n * IC + ic) * IH * IW];
                        const float* filter_ptr =
                                &filter[((g * ocpg + oc_in_group) * icpg +
                                         ic_in_group) *
                                        FH * FW];
                        const float* offset_ptr =
                                &offset[(n * deformable_group + dg) * 2 * FH *
                                        FW * OH * OW];
                        const float* mask_ptr =
                                &mask[(n * deformable_group + dg) * FH * FW *
                                      OH * OW];

                        for (size_t fh = 0; fh < FH; ++fh)
                            for (size_t fw = 0; fw < FW; ++fw) {
                                size_t filter_idx = fh * FW + fw;
                                size_t offset_h_idx =
                                        ((2 * (fh * FW + fw)) * OH + oh) * OW +
                                        ow;
                                size_t offset_w_idx =
                                        ((2 * (fh * FW + fw) + 1) * OH + oh) *
                                                OW +
                                        ow;
                                size_t mask_idx =
                                        ((fh * FW + fw) * OH + oh) * OW + ow;
                                float flt = filter_ptr[filter_idx];
                                float offset_h = offset_ptr[offset_h_idx];
                                float offset_w = offset_ptr[offset_w_idx];
                                float m = mask_ptr[mask_idx];
                                float h = ((float)ih) + fh * DH + offset_h;
                                float w = ((float)iw) + fw * DW + offset_w;
                                float val = 0.f;

                                if (h > -1.f && w > -1.f && h < IH && w < IW)
                                    val = dmcn_bilinear(im_ptr, IW, IH, IW, h,
                                                        w);
                                sum += val * m * flt;
                            }
                    }
                    out[((n * OC + oc) * OH + oh) * OW + ow] = sum;
                }
        }
    }
}

void Fwd::exec(_megdnn_tensor_in im, _megdnn_tensor_in filter,
               _megdnn_tensor_in offset, _megdnn_tensor_in mask,
               _megdnn_tensor_out dst, _megdnn_workspace) {
    auto&& out = dst;
    auto filter_meta = make_canonized_filter_meta(im.layout.ndim, filter.layout,
                                                  offset.layout);
    size_t group = filter_meta.group,
           deformable_group = filter_meta.deformable_group, N = im.layout[0],
           IC = im.layout[1], IH = im.layout[2], IW = im.layout[3],
           SH = param().stride_h, SW = param().stride_w, PH = param().pad_h,
           PW = param().pad_w, DH = filter_meta.dilation[0],
           DW = filter_meta.dilation[1], FH = filter_meta.spatial[0],
           FW = filter_meta.spatial[1],
           OC = filter_meta.group * filter_meta.ocpg, OH = out.layout[2],
           OW = out.layout[3];

    const float* __restrict im_ptr = im.ptr<float>();
    const float* __restrict filter_ptr = filter.ptr<float>();
    const float* __restrict offset_ptr = offset.ptr<float>();
    const float* __restrict mask_ptr = mask.ptr<float>();
    float* __restrict dst_ptr = dst.ptr<float>();

    MEGDNN_DISPATCH_CPU_KERN_OPR(deformable_conv_forward(
            im_ptr, filter_ptr, offset_ptr, mask_ptr, dst_ptr, OC, IC, N, FH,
            FW, IH, IW, PH, PW, DH, DW, SH, SW, OH, OW, group,
            deformable_group));
    return;
}

/* ============== Bwd Implementation ============== */

static void deformable_conv_backward_weight(
        const float* im, const float* offset, const float* mask,
        const float* out_grad, float* weight_grad, const size_t OC,
        const size_t IC, const size_t N, const size_t FH, const size_t FW,
        const size_t IH, const size_t IW, const size_t PH, const size_t PW,
        const size_t DH, const size_t DW, const size_t SH, const size_t SW,
        const size_t OH, const size_t OW, const size_t group,
        const size_t deformable_group) {
    const int icpg = IC / group, ocpg = OC / group,
              icpdg = IC / deformable_group;

    memset(weight_grad, 0, sizeof(float[group * ocpg * icpg * FH * FW]));

    for (size_t n = 0; n < N; ++n) {
        for (size_t oc = 0; oc < OC; ++oc) {
            size_t g = oc / ocpg;
            size_t oc_in_group = oc % ocpg;
            size_t icpg_l = icpg * g, icpg_r = icpg * (g + 1);
            const float* out_grad_ptr = &out_grad[(n * OC + oc) * OH * OW];
            for (size_t oh = 0; oh < OH; ++oh)
                for (size_t ow = 0; ow < OW; ++ow) {
                    int ih = oh * SH - PH;
                    int iw = ow * SW - PW;
                    float o_grad = out_grad_ptr[oh * OW + ow];
                    for (size_t ic = icpg_l; ic < icpg_r; ic++) {
                        const size_t dg = ic / icpdg;
                        const size_t ic_in_group = ic % icpg;
                        const float* im_ptr = &im[(n * IC + ic) * IH * IW];
                        const float* offset_ptr =
                                &offset[(n * deformable_group + dg) * 2 * FH *
                                        FW * OH * OW];
                        const float* mask_ptr =
                                &mask[(n * deformable_group + dg) * FH * FW *
                                      OH * OW];
                        float* weight_grad_ptr =
                                &weight_grad[((g * ocpg + oc_in_group) * icpg +
                                              ic_in_group) *
                                             FH * FW];

                        for (size_t fh = 0; fh < FH; ++fh)
                            for (size_t fw = 0; fw < FW; ++fw) {
                                size_t offset_h_idx =
                                        ((2 * (fh * FW + fw)) * OH + oh) * OW +
                                        ow;
                                size_t offset_w_idx =
                                        ((2 * (fh * FW + fw) + 1) * OH + oh) *
                                                OW +
                                        ow;
                                size_t mask_idx =
                                        ((fh * FW + fw) * OH + oh) * OW + ow;
                                float offset_h = offset_ptr[offset_h_idx];
                                float offset_w = offset_ptr[offset_w_idx];
                                float m = mask_ptr[mask_idx];
                                float h = ((float)ih) + fh * DH + offset_h;
                                float w = ((float)iw) + fw * DW + offset_w;
                                float val = 0.f;

                                if (h > -1.f && w > -1.f && h < IH && w < IW)
                                    val = dmcn_bilinear(im_ptr, IW, IH, IW, h,
                                                        w);
                                weight_grad_ptr[fh * FW + fw] +=
                                        val * m * o_grad;
                            }
                    }
                }
        }
    }
}

static float dmcn_get_gradient_weight(const int H, const int W, const int h,
                                      const int w, const float argmax_h,
                                      const float argmax_w) {
    if (argmax_h <= -1.0f || argmax_h >= H || argmax_w <= -1.0f ||
        argmax_w >= W)
        return 0.f;

    const int argmax_h_low = floor(argmax_h);
    const int argmax_w_low = floor(argmax_w);
    const int argmax_h_high = argmax_h_low + 1;
    const int argmax_w_high = argmax_w_low + 1;

    float weight = 0.f;

    if (h == argmax_h_low && w == argmax_w_low)
        weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
    if (h == argmax_h_low && w == argmax_w_high)
        weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
    if (h == argmax_h_high && w == argmax_w_low)
        weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
    if (h == argmax_h_high && w == argmax_w_high)
        weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
    return weight;
}

static float dmcn_get_coordinate_weight(const float* im_data,
                                        const size_t im_stride, const int H,
                                        const int W, float argmax_h,
                                        float argmax_w, const int bp_dir) {
    if (argmax_h <= -1.f || argmax_h >= H || argmax_w <= -1.f || argmax_w >= W)
        return 0;

    float weight = 0.f;
    int argmax_h_low = floor(argmax_h), argmax_w_low = floor(argmax_w);
    int argmax_h_high = argmax_h_low + 1, argmax_w_high = argmax_w_low + 1;

    if (bp_dir == 0) {
        if (argmax_h_low >= 0 && argmax_w_low >= 0)
            weight += -1 * (argmax_w_low + 1 - argmax_w) *
                      im_data[argmax_h_low * im_stride + argmax_w_low];
        if (argmax_h_low >= 0 && argmax_w_high <= W - 1)
            weight += -1 * (argmax_w - argmax_w_low) *
                      im_data[argmax_h_low * im_stride + argmax_w_high];
        if (argmax_h_high <= H - 1 && argmax_w_low >= 0)
            weight += (argmax_w_low + 1 - argmax_w) *
                      im_data[argmax_h_high * im_stride + argmax_w_low];
        if (argmax_h_high <= H - 1 && argmax_w_high <= W - 1)
            weight += (argmax_w - argmax_w_low) *
                      im_data[argmax_h_high * im_stride + argmax_w_high];
    } else {
        if (argmax_h_low >= 0 && argmax_w_low >= 0)
            weight += -1 * (argmax_h_low + 1 - argmax_h) *
                      im_data[argmax_h_low * im_stride + argmax_w_low];
        if (argmax_h_low >= 0 && argmax_w_high <= W - 1)
            weight += (argmax_h_low + 1 - argmax_h) *
                      im_data[argmax_h_low * im_stride + argmax_w_high];
        if (argmax_h_high <= H - 1 && argmax_w_low >= 0)
            weight += -1 * (argmax_h - argmax_h_low) *
                      im_data[argmax_h_high * im_stride + argmax_w_low];
        if (argmax_h_high <= H - 1 && argmax_w_high <= W - 1)
            weight += (argmax_h - argmax_h_low) *
                      im_data[argmax_h_high * im_stride + argmax_w_high];
    }
    return weight;
}

static void deformable_conv_backward_data(
        const float* im, const float* flt, const float* offset,
        const float* mask, const float* out_grad, float* im_grad,
        float* offset_grad, float* mask_grad, const size_t OC, const size_t IC,
        const size_t N, const size_t FH, const size_t FW, const size_t IH,
        const size_t IW, const size_t PH, const size_t PW, const size_t SH,
        const size_t SW, const size_t DH, const size_t DW, const size_t OH,
        const size_t OW, const size_t group, const size_t deformable_group) {
    memset(im_grad, 0, sizeof(float[N * IC * IH * IW]));
    memset(offset_grad, 0,
           N * deformable_group * 2 * FH * FW * OH * OW * sizeof(float));
    memset(mask_grad, 0,
           N * deformable_group * FH * FW * OH * OW * sizeof(float));

    const int icpg = IC / group, ocpg = OC / group,
              icpdg = IC / deformable_group;

    size_t n, oc, ic, oh, ow, fh, fw, g, dg, oc_in_group, ic_in_group;
    const float *im_ptr, *flt_ptr, *offset_ptr, *mask_ptr;
    float *im_grad_ptr, *offset_grad_ptr, *mask_grad_ptr;

    int ih, iw;
    size_t m_idx, off_h_idx, off_w_idx;
    float h, w, col_grad, off_h, off_w, m;

    for (n = 0; n < N; ++n)
        for (g = 0; g < group; ++g) {
            const size_t ic_l = icpg * g, ic_r = icpg * (g + 1);
            const size_t oc_l = ocpg * g, oc_r = ocpg * (g + 1);
            for (oc = oc_l; oc < oc_r; ++oc) {
                oc_in_group = oc % ocpg;
                const float* out_grad_ptr = &out_grad[(n * OC + oc) * OH * OW];
                for (oh = 0; oh < OH; ++oh)
                    for (ow = 0; ow < OW; ++ow) {
                        ih = oh * SH - PH;
                        iw = ow * SW - PW;
                        float out_grad = out_grad_ptr[oh * OW + ow];
                        for (ic = ic_l; ic < ic_r; ic++) {
                            dg = ic / icpdg, ic_in_group = ic % icpg;
                            im_ptr = &im[(n * IC + ic) * IH * IW];
                            flt_ptr = &flt[((g * ocpg + oc_in_group) * icpg +
                                            ic_in_group) *
                                           FH * FW];
                            offset_ptr = &offset[(n * deformable_group + dg) *
                                                 2 * FH * FW * OH * OW];
                            mask_ptr = &mask[(n * deformable_group + dg) * FH *
                                             FW * OH * OW];

                            im_grad_ptr = &im_grad[(n * IC + ic) * IH * IW];
                            offset_grad_ptr =
                                    &offset_grad[(n * deformable_group + dg) *
                                                 2 * FH * FW * OH * OW];
                            mask_grad_ptr =
                                    &mask_grad[(n * deformable_group + dg) *
                                               FH * FW * OH * OW];
                            for (fh = 0; fh < FH; ++fh)
                                for (fw = 0; fw < FW; ++fw) {
                                    auto f = flt_ptr[fh * FW + fw];
                                    off_h_idx = ((2 * (fh * FW + fw)) * OH +
                                                 oh) * OW +
                                                ow;
                                    off_w_idx = ((2 * (fh * FW + fw) + 1) * OH +
                                                 oh) * OW +
                                                ow;
                                    m_idx = ((fh * FW + fw) * OH + oh) * OW +
                                            ow;
                                    off_h = offset_ptr[off_h_idx];
                                    off_w = offset_ptr[off_w_idx];
                                    m = mask_ptr[m_idx];

                                    h = ((float)ih) + fh * DH + off_h;
                                    w = ((float)iw) + fw * DW + off_w;
                                    col_grad = out_grad * f;

                                    if (h <= -1.f || w <= -1.f || h >= IH ||
                                        w >= IW) {
                                        h = w = -2.f;
                                    } else {
                                        mask_grad_ptr[m_idx] +=
                                                col_grad *
                                                dmcn_bilinear(im_ptr, IW, IH,
                                                              IW, h, w);
                                    }
                                    float weight_h = dmcn_get_coordinate_weight(
                                            im_ptr, IW, IH, IW, h, w, 0);
                                    float weight_w = dmcn_get_coordinate_weight(
                                            im_ptr, IW, IH, IW, h, w, 1);

                                    offset_grad_ptr[off_h_idx] +=
                                            col_grad * m * weight_h;
                                    offset_grad_ptr[off_w_idx] +=
                                            col_grad * m * weight_w;

                                    int ih_hat = (int)h, iw_hat = (int)w;
                                    for (int dy = ih_hat - 2; dy <= ih_hat + 2;
                                         dy++)
                                        for (int dx = iw_hat - 2;
                                             dx <= iw_hat + 2; dx++) {
                                            if (dy >= 0 && dy < (int)IH &&
                                                dx >= 0 && dx < (int)IW &&
                                                abs(h - dy) < 1.f &&
                                                abs(w - dx) < 1.f) {
                                                int im_idx = dy * IW + dx;
                                                float weight =
                                                        dmcn_get_gradient_weight(
                                                                IH, IW, dy, dx,
                                                                h, w);
                                                im_grad_ptr[im_idx] +=
                                                        weight * m * col_grad;
                                            }
                                        }
                                }
                        }
                    }
            }
        }
}

size_t BwdFlt::get_workspace_in_bytes(const TensorLayout& /* im */,
                                      const TensorLayout& /* offset */,
                                      const TensorLayout& /* mask */,
                                      const TensorLayout& /* out */,
                                      const TensorLayout& /* filter_grad */) {
    return 0ULL;
}

void BwdFlt::exec(_megdnn_tensor_in im, _megdnn_tensor_in offset,
                  _megdnn_tensor_in mask, _megdnn_tensor_in out_grad,
                  _megdnn_tensor_out filter_grad, _megdnn_workspace) {
    auto&& out = out_grad;
    auto fm = make_canonized_filter_meta(im.layout.ndim, filter_grad.layout,
                                         offset.layout);

    size_t group = fm.group, deformable_group = fm.deformable_group,
           N = im.layout[0], IC = im.layout[1], IH = im.layout[2],
           IW = im.layout[3], SH = param().stride_h, SW = param().stride_w,
           PH = param().pad_h, PW = param().pad_w, DH = fm.dilation[0],
           DW = fm.dilation[1], FH = fm.spatial[0], FW = fm.spatial[1],
           OC = fm.group * fm.ocpg, OH = out.layout[2], OW = out.layout[3];

    const float* __restrict im_ptr = im.ptr<float>();
    const float* __restrict offset_ptr = offset.ptr<float>();
    const float* __restrict mask_ptr = mask.ptr<float>();
    const float* __restrict out_grad_ptr = out_grad.ptr<float>();
    float* __restrict filter_grad_ptr = filter_grad.ptr<float>();
    // backward filter
    MEGDNN_DISPATCH_CPU_KERN_OPR(deformable_conv_backward_weight(
            im_ptr, offset_ptr, mask_ptr, out_grad_ptr, filter_grad_ptr, OC, IC,
            N, FH, FW, IH, IW, PH, PW, DH, DW, SH, SW, OH, OW, group,
            deformable_group));
}
size_t BwdData::get_workspace_in_bytes(const TensorLayout& /* im */,
                                       const TensorLayout& /* filter */,
                                       const TensorLayout& /* offset */,
                                       const TensorLayout& /* mask */,
                                       const TensorLayout& /* out_grad */,
                                       const TensorLayout& /* im_grad */,
                                       const TensorLayout& /* offset_grad */,
                                       const TensorLayout& /* mask_grad */) {
    return 0ULL;
}

void BwdData::exec(_megdnn_tensor_in im, _megdnn_tensor_in filter,
                   _megdnn_tensor_in offset, _megdnn_tensor_in mask,
                   _megdnn_tensor_in out_grad, _megdnn_tensor_out im_grad,
                   _megdnn_tensor_out offset_grad, _megdnn_tensor_out mask_grad,
                   _megdnn_workspace) {
    auto fm = make_canonized_filter_meta(im.layout.ndim, filter.layout,
                                         offset.layout);
    size_t group = fm.group, deformable_group = fm.deformable_group,
           N = im.layout[0], IC = im.layout[1], IH = im.layout[2],
           IW = im.layout[3], SH = param().stride_h, SW = param().stride_w,
           PH = param().pad_h, PW = param().pad_w, DH = fm.dilation[0],
           DW = fm.dilation[1], FH = fm.spatial[0], FW = fm.spatial[1],
           OC = fm.group * fm.ocpg, OH = out_grad.layout[2],
           OW = out_grad.layout[3];

    const float* __restrict im_ptr = im.ptr<float>();
    const float* __restrict filter_ptr = filter.ptr<float>();
    const float* __restrict offset_ptr = offset.ptr<float>();
    const float* __restrict mask_ptr = mask.ptr<float>();
    const float* __restrict out_grad_ptr = out_grad.ptr<float>();

    float* __restrict im_grad_ptr = im_grad.ptr<float>();
    float* __restrict offset_grad_ptr = offset_grad.ptr<float>();
    float* __restrict mask_grad_ptr = mask_grad.ptr<float>();

    // backward coordinate data
    MEGDNN_DISPATCH_CPU_KERN_OPR(deformable_conv_backward_data(
            im_ptr, filter_ptr, offset_ptr, mask_ptr, out_grad_ptr, im_grad_ptr,
            offset_grad_ptr, mask_grad_ptr, OC, IC, N, FH, FW, IH, IW, PH, PW,
            SH, SW, DH, DW, OH, OW, group, deformable_group));
}

// vim: syntax=cpp.doxygen
