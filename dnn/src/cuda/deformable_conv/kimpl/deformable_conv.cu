/**
 * \file dnn/src/cuda/deformable_conv/kimpl/deformable_conv.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/deformable_conv/kimpl/deformable_conv.cuh"

using namespace megdnn;
using namespace cuda;
using namespace deformable_conv;

namespace {

__device__ float dmcn_im2col_bilinear(const float* bottom_data,
                                      const int data_width, const int height,
                                      const int width, float h, float w) {
    int h_low = floor(h);
    int w_low = floor(w);
    int h_high = h_low + 1;
    int w_high = w_low + 1;

    float lh = h - h_low;
    float lw = w - w_low;
    float hh = 1 - lh, hw = 1 - lw;

    float v1 = 0;
    if (h_low >= 0 && w_low >= 0)
        v1 = bottom_data[h_low * data_width + w_low];
    float v2 = 0;
    if (h_low >= 0 && w_high <= width - 1)
        v2 = bottom_data[h_low * data_width + w_high];
    float v3 = 0;
    if (h_high <= height - 1 && w_low >= 0)
        v3 = bottom_data[h_high * data_width + w_low];
    float v4 = 0;
    if (h_high <= height - 1 && w_high <= width - 1)
        v4 = bottom_data[h_high * data_width + w_high];

    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

__device__ float dmcn_get_gradient_weight(float argmax_h, float argmax_w,
                                          const int h, const int w,
                                          const int height, const int width) {
    if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
        argmax_w >= width) {
        return 0;
    }

    int argmax_h_low = floor(argmax_h);
    int argmax_w_low = floor(argmax_w);
    int argmax_h_high = argmax_h_low + 1;
    int argmax_w_high = argmax_w_low + 1;

    float weight = 0;
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

__device__ float dmcn_get_coordinate_weight(float argmax_h, float argmax_w,
                                            const int height, const int width,
                                            const float* im_data,
                                            const int data_width,
                                            const int bp_dir) {
    if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
        argmax_w >= width) {
        return 0;
    }

    int argmax_h_low = floorf(argmax_h);
    int argmax_w_low = floorf(argmax_w);
    int argmax_h_high = argmax_h_low + 1;
    int argmax_w_high = argmax_w_low + 1;

    float weight = 0;

    if (bp_dir == 0) {
        if (argmax_h_low >= 0 && argmax_w_low >= 0)
            weight += -1 * (argmax_w_low + 1 - argmax_w) *
                      im_data[argmax_h_low * data_width + argmax_w_low];
        if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
            weight += -1 * (argmax_w - argmax_w_low) *
                      im_data[argmax_h_low * data_width + argmax_w_high];
        if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
            weight += (argmax_w_low + 1 - argmax_w) *
                      im_data[argmax_h_high * data_width + argmax_w_low];
        if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
            weight += (argmax_w - argmax_w_low) *
                      im_data[argmax_h_high * data_width + argmax_w_high];
    } else if (bp_dir == 1) {
        if (argmax_h_low >= 0 && argmax_w_low >= 0)
            weight += -1 * (argmax_h_low + 1 - argmax_h) *
                      im_data[argmax_h_low * data_width + argmax_w_low];
        if (argmax_h_low >= 0 && argmax_w_high <= width - 1)
            weight += (argmax_h_low + 1 - argmax_h) *
                      im_data[argmax_h_low * data_width + argmax_w_high];
        if (argmax_h_high <= height - 1 && argmax_w_low >= 0)
            weight += -1 * (argmax_h - argmax_h_low) *
                      im_data[argmax_h_high * data_width + argmax_w_low];
        if (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
            weight += (argmax_h - argmax_h_low) *
                      im_data[argmax_h_high * data_width + argmax_w_high];
    }

    return weight;
}

__global__ void deformable_im2col(Param p, const float* im, const float* offset,
                                  const float* mask, float* col) {
    size_t n = blockIdx.y;
    const size_t N = p.batch_sz;
    const size_t loops = p.IC * p.OH * p.OW;
    const size_t im_bs = p.IC * p.IH * p.IW;
    const size_t offset_bs = 2 * p.deformable_group * p.FH * p.FW * p.OH * p.OW;
    const size_t mask_bs = p.deformable_group * p.FH * p.FW * p.OH * p.OW;

    im = &im[n * im_bs];
    offset = &offset[n * offset_bs];
    mask = &mask[n * mask_bs];

    KERN_FOR(idx, loops) {
        const int ow = idx % p.OW;
        const int oh = (idx / p.OW) % p.OH;
        const int ic = (idx / p.OW / p.OH);
        const int dg = ic / p.icpdg;
        const int ih = oh * p.SH - p.PH;
        const int iw = ow * p.SW - p.PW;

        const float* im_ptr = &im[ic * p.IH * p.IW];
        const float* offset_ptr =
                &offset[(dg * 2 * p.FH * p.FW * p.OH + oh) * p.OW + ow];
        const float* mask_ptr =
                &mask[(dg * p.FH * p.FW * p.OH + oh) * p.OW + ow];
        float* col_ptr =
                &col[((((ic * p.FH * p.FW) * N + n) * p.OH + oh) * p.OW + ow)];

        for (int i = 0; i < p.FH; ++i)
            for (int j = 0; j < p.FW; ++j) {
                const float off_h =
                        offset_ptr[(2 * (i * p.FW + j)) * p.OH * p.OW];
                const float off_w =
                        offset_ptr[(2 * (i * p.FW + j) + 1) * p.OH * p.OW];
                const float m = mask_ptr[(i * p.FW + j) * p.OH * p.OW];

                float val = 0.f;
                const float h = ih + i * p.DH + off_h;
                const float w = iw + j * p.DW + off_w;
                if (h > -1 && h < p.IH && w > -1 && w < p.IW)
                    val = dmcn_im2col_bilinear(im_ptr, p.IW, p.IH, p.IW, h, w);
                col_ptr[(i * p.FW + j) * N * p.OH * p.OW] = val * m;
            }
    }
}

__global__ void deformable_col2im(Param p, const float* col,
                                  const float* offset, const float* mask,
                                  float* im) {
    size_t dg = blockIdx.y % p.deformable_group;
    size_t n = blockIdx.y / p.deformable_group;
    const size_t loops = p.FH * p.FW * p.OH * p.OW;
    const size_t N = p.batch_sz;
    const size_t im_bs = p.IC * p.IH * p.IW;
    const size_t offset_bs = 2 * p.deformable_group * p.FH * p.FW * p.OH * p.OW;
    const size_t mask_bs = p.deformable_group * p.FH * p.FW * p.OH * p.OW;

    offset = &offset[n * offset_bs];
    mask = &mask[n * mask_bs];
    im = &im[n * im_bs];

    KERN_FOR(idx, loops) {
        const int ow = (idx) % p.OW;
        const int oh = (idx / p.OW) % p.OH;
        const int fw = (idx / p.OW / p.OH) % p.FW;
        const int fh = (idx / p.OW / p.OH / p.FW) % p.FH;

        const float* offset_ptr = &offset[dg * 2 * p.FH * p.FW * p.OH * p.OW];
        const float* mask_ptr = &mask[dg * p.FH * p.FW * p.OH * p.OW];

        const int off_h_idx = ((2 * (fh * p.FW + fw)) * p.OH + oh) * p.OW + ow;
        const int off_w_idx =
                ((2 * (fh * p.FW + fw) + 1) * p.OH + oh) * p.OW + ow;
        const int mask_idx = ((fh * p.FW + fw) * p.OH + oh) * p.OW + ow;

        const float off_h = offset_ptr[off_h_idx];
        const float off_w = offset_ptr[off_w_idx];
        const float m = mask_ptr[mask_idx];

        const size_t ic_l = dg * p.icpdg, ic_r = (dg + 1) * p.icpdg;

        for (int ic = ic_l; ic < ic_r; ++ic) {
            const int ih = oh * p.SH - p.PH;
            const int iw = ow * p.SW - p.PW;

            const int col_idx =
                    (((((ic * p.FH) + fh) * p.FW + fw) * N + n) * p.OH + oh) *
                            p.OW +
                    ow;
            const float top_grad = col[col_idx] * m;

            const float h = ih + fh * p.DH + off_h;
            const float w = iw + fw * p.DW + off_w;

            const int h_hat = (int)h, w_hat = (int)w;
#pragma unroll
            for (int dy = -2; dy <= 2;
                 dy++) {  // use 0-1 is better, same for dx
#pragma unroll
                for (int dx = -2; dx <= 2; dx++) {
                    if (h_hat + dy >= 0 && h_hat + dy < p.IH &&
                        w_hat + dx >= 0 && w_hat + dx < p.IW &&
                        abs(h - (h_hat + dy)) < 1 &&
                        abs(w - (w_hat + dx)) < 1) {
                        int bottom_pos =
                                (ic * p.IH + h_hat + dy) * p.IW + w_hat + dx;
                        float weight = dmcn_get_gradient_weight(
                                h, w, h_hat + dy, w_hat + dx, p.IH, p.IW);
                        atomicAdd(&im[bottom_pos], weight * top_grad);
                    }
                }
            }
        }
    }
}

__global__ void deformable_col2coord(Param p, const float* im, const float* col,
                                     const float* offset, const float* mask,
                                     float* offset_grad, float* mask_grad) {
    size_t n = blockIdx.y;
    const size_t N = p.batch_sz;
    const size_t loops = p.deformable_group * p.FH * p.FW * 2 * p.OH * p.OW;
    const size_t im_bs = p.IC * p.IH * p.IW;
    const size_t offset_bs = p.deformable_group * p.FH * p.FW * 2 * p.OH * p.OW;
    const size_t mask_bs = p.deformable_group * p.FH * p.FW * p.OH * p.OW;

    im = &im[n * im_bs];
    offset = &offset[n * offset_bs];
    mask = &mask[n * mask_bs];

    offset_grad = &offset_grad[n * offset_bs];
    mask_grad = &mask_grad[n * mask_bs];

    KERN_FOR(idx, loops) {
        float val = 0, mval = 0;
        const int hw = idx % 2;
        const int ow = (idx / 2) % p.OW;
        const int oh = (idx / 2 / p.OW) % p.OH;
        const int fw = (idx / 2 / p.OW / p.OH) % p.FW;
        const int fh = (idx / 2 / p.OW / p.OH / p.FW) % p.FH;
        const int dg =
                (idx / 2 / p.OW / p.OH / p.FW / p.FH) % p.deformable_group;

        const int ih = oh * p.SH - p.PH;
        const int iw = ow * p.SW - p.PW;

        const float* offset_ptr = &offset[dg * 2 * p.FH * p.FW * p.OH * p.OW];
        const float* mask_ptr = &mask[dg * p.FH * p.FW * p.OH * p.OW];

        float* offset_grad_ptr =
                &offset_grad[dg * 2 * p.FH * p.FW * p.OH * p.OW];
        float* mask_grad_ptr = &mask_grad[dg * p.FH * p.FW * p.OH * p.OW];

        const int offset_h_idx =
                ((2 * (fh * p.FW + fw)) * p.OH + oh) * p.OW + ow;
        const int offset_w_idx =
                ((2 * (fh * p.FW + fw) + 1) * p.OH + oh) * p.OW + ow;
        const int mask_idx = ((fh * p.FW + fw) * p.OH + oh) * p.OW + ow;
        const int offset_grad_idx = (hw == 0) ? offset_h_idx : offset_w_idx;

        const float off_h = offset_ptr[offset_h_idx];
        const float off_w = offset_ptr[offset_w_idx];
        const float m = mask_ptr[mask_idx];

        float h = ih + fh * p.DH + off_h;
        float w = iw + fw * p.DW + off_w;

        const int ic_l = dg * p.icpdg, ic_r = (dg + 1) * p.icpdg;

        for (int ic = ic_l; ic < ic_r; ++ic) {
            const float* im_ptr = &im[ic * p.IH * p.IW];
            const int col_idx =
                    (((((ic * p.FH + fh) * p.FW + fw) * N + n) * p.OH + oh) *
                             p.OW +
                     ow);
            const float col_grad = col[col_idx];

            if (h <= -1 || w <= -1 || h >= p.IH || w >= p.IW) {
                h = w = -2;
            } else if (hw % 2 == 0) {
                mval += col_grad *
                        dmcn_im2col_bilinear(im_ptr, p.IW, p.IH, p.IW, h, w);
            }
            const float top_grad = col_grad * m;
            const float weight = dmcn_get_coordinate_weight(h, w, p.IH, p.IW,
                                                            im_ptr, p.IW, hw);
            val += weight * top_grad;
        }

        offset_grad_ptr[offset_grad_idx] = val;
        if (hw % 2 ==0) {
            mask_grad_ptr[mask_idx] = mval;
        }
    }
}

}  // namespace

namespace megdnn {
namespace cuda {
namespace deformable_conv {

void im2col(const float* dev_im, const float* dev_offset, const float* dev_mask,
            float* dev_col, const Param& p) {
    dim3 grid;
    size_t loops = p.IC * p.OH * p.OW;
    int nr_thds = query_blocksize_for_kernel(deformable_im2col);

    grid.x = DIVUP(loops, nr_thds), grid.y = p.batch_sz;

    deformable_im2col<<<grid, nr_thds, 0, p.stream>>>(p, dev_im, dev_offset,
                                                         dev_mask, dev_col);
    after_kernel_launch();
}

void col2im(const float* dev_col, const float* dev_offset,
            const float* dev_mask, float* dev_im_grad, const Param& p) {
    dim3 grid;
    size_t loops = p.FH * p.FW * p.OH * p.OW;
    int nr_thds = query_blocksize_for_kernel(deformable_col2im);

    grid.x = DIVUP(loops, nr_thds), grid.y = p.batch_sz * p.deformable_group;

    deformable_col2im<<<grid, nr_thds, 0, p.stream>>>(p, dev_col, dev_offset,
                                                         dev_mask, dev_im_grad);
    after_kernel_launch();
}

void col2im_coord(const float* dev_im, const float* dev_col,
                  const float* dev_offset, const float* dev_mask,
                  float* dev_offset_grad, float* dev_mask_grad,
                  const Param& p) {
    dim3 grid;
    size_t loops = 2 * p.FH * p.FW * p.OH * p.OW * p.deformable_group;
    int nr_thds = query_blocksize_for_kernel(deformable_col2coord);

    grid.x = DIVUP(loops, nr_thds);
    grid.y = p.batch_sz;

    deformable_col2coord<<<grid, nr_thds, 0, p.stream>>>(
            p, dev_im, dev_col, dev_offset, dev_mask, dev_offset_grad,
            dev_mask_grad);
    after_kernel_launch();
}

}  // namespace deformable_conv
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cuda syntax=cuda.doxygen
