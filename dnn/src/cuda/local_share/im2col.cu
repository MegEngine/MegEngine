/**
 * \file dnn/src/cuda/local_share/im2col.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./im2col.cuh"

using namespace megdnn;
using namespace cuda;
using namespace local_share;

namespace {
template <typename T>
__global__ void local_share_im2col(const T* __restrict__ img,
                                   T* __restrict__ col, int fh, int fw, int sh,
                                   int sw, int nr_groups, Param param) {
    const int in_ch_idx = threadIdx.x + blockIdx.y * blockDim.x;
    const int batch = threadIdx.y + blockIdx.z * blockDim.y;
    if (in_ch_idx >= param.ci || batch >= param.n)
        return;
    const int hw = blockIdx.x;
    const int wo = param.grp_wo * param.sgw;
    const int oh_idx = hw / wo;
    const int ow_idx = hw - oh_idx * wo;
    const int sgh_idx = oh_idx / param.grp_ho;
    const int sgw_idx = ow_idx / param.grp_wo;
    const int grp_oh_idx = oh_idx - sgh_idx * param.grp_ho;
    const int grp_ow_idx = ow_idx - sgw_idx * param.grp_wo;
    const int grp_sizes = param.grp_ho * param.grp_wo;
    const int icpg = param.ci / nr_groups;
    const int ch_grp_idx = in_ch_idx / icpg;
    const int grp_ch_idx = in_ch_idx - icpg * ch_grp_idx;

    const T* __restrict__ img_ptr = img +
                                    batch * param.ci * param.hi * param.wi +
                                    in_ch_idx * param.hi * param.wi;
    const int ld = icpg * fh * fw;
    T* __restrict__ col_ptr =
            col +
            ch_grp_idx * (param.sgh * param.sgw) * param.n * grp_sizes *
                    ld  // channel group stride
            + (sgh_idx * param.sgw + sgw_idx) * param.n * grp_sizes *
                      ld            // batch stride
            + grp_ch_idx * fh * fw  // input channel stride
            + (batch * grp_sizes + (grp_oh_idx * param.grp_wo + grp_ow_idx)) *
                      ld;  // row stride

    for (int kh = 0; kh < fh; kh++) {
        for (int kw = 0; kw < fw; kw++) {
            int ih_idx = oh_idx * sh - param.ph + kh;
            int iw_idx = ow_idx * sw - param.pw + kw;
            float val = 0.f;
            if (ih_idx < param.hi && ih_idx >= 0 && iw_idx < param.wi &&
                iw_idx >= 0) {
                val = img_ptr[ih_idx * param.wi + iw_idx];
            }
            *(col_ptr++) = val;
        }
    }
}

template <typename T>
__global__ void local_share_col2im(const T* __restrict__ col,
                                   T* __restrict__ img, int fh, int fw, int sh,
                                   int sw, int nr_groups, Param param) {
    const int batch = threadIdx.x + blockIdx.y * blockDim.x;
    const int in_ch_idx = threadIdx.y + blockIdx.z * blockDim.y;
    if (in_ch_idx >= param.ci || batch >= param.n)
        return;
    const int hw = blockIdx.x;
    const int ih_idx = hw / param.wi;
    const int iw_idx = hw - ih_idx * param.wi;
    const int ho = param.grp_ho * param.sgh;
    const int wo = param.grp_wo * param.sgw;
    const int icpg = param.ci / nr_groups;
    const int grp_sizes = param.grp_ho * param.grp_wo;
    const int filter_sizes = fh * fw;
    const int ch_filter_sizes = icpg * filter_sizes;
    const int nr_elems_per_grp = param.n * grp_sizes * ch_filter_sizes;
    const int ch_grp_idx = in_ch_idx / icpg;
    const int grp_ch_idx = in_ch_idx - icpg * ch_grp_idx;
    const T* __restrict__ col_ptr =
            col +
            ch_grp_idx * param.sgh * param.sgw * ch_filter_sizes * grp_sizes *
                    param.n  // channel group stride
            + batch          // batch stride
            +
            grp_ch_idx * filter_sizes * grp_sizes * param.n;  // channel stride

    T res(0);
    for (int kh = 0; kh < fh; ++kh) {
        uint32_t anchorh = ih_idx + param.ph - kh;
        if (anchorh < ho * sh && anchorh % sh == 0) {
            int oh_idx = anchorh / sh;
            int sgh_idx = oh_idx / param.grp_ho;
            int grp_oh_idx = oh_idx - sgh_idx * param.grp_ho;
            for (int kw = 0; kw < fw; ++kw) {
                uint32_t anchorw = iw_idx + param.pw - kw;
                if (anchorw < wo * sw && anchorw % sw == 0) {
                    int ow_idx = anchorw / sw;
                    int sgw_idx = ow_idx / param.grp_wo;
                    int grp_ow_idx = ow_idx - sgw_idx * param.grp_wo;
                    const T* __restrict__ sptr =
                            col_ptr +
                            (sgh_idx * param.sgw + sgw_idx) *
                                    nr_elems_per_grp  // spatial group stride
                            + (grp_oh_idx * param.grp_wo + grp_ow_idx) *
                                      param.n  // spatial stride
                            + (kh * fw + kw) * grp_sizes * param.n;
                    res += sptr[0];
                }
            }
        }
    }
    img[batch * param.ci * param.hi * param.wi +
        in_ch_idx * param.hi * param.wi + ih_idx * param.wi + iw_idx] = res;
}

}  // namespace

void megdnn::cuda::local_share::_do_local_share_im2col(
        const float* d_im, float* d_col, int fh, int fw, int sh, int sw,
        int nr_groups, const Param& param, cudaStream_t stream) {
    void (*kern)(const float* __restrict__, float* __restrict__, int, int, int,
                 int, int, Param);
    kern = local_share_im2col<float>;

    constexpr int threads_x = 256;
    uint32_t nr_threads =
            _get_kern_block_size(reinterpret_cast<const void*>(kern));
    uint32_t nr_threads_x = std::min(threads_x, param.ci);
    uint32_t nr_threads_y =
            std::min(static_cast<int>(nr_threads / nr_threads_x), param.n);
    uint32_t nr_blocks_x = param.sgw * param.sgh * param.grp_ho * param.grp_wo,
             nr_blocks_y = DIVUP(param.ci, nr_threads_x),
             nr_blocks_z = DIVUP(param.n, nr_threads_y);
    dim3 threads{nr_threads_x, nr_threads_y, 1};
    dim3 blocks{nr_blocks_x, nr_blocks_y, nr_blocks_z};
    kern<<<blocks, threads, 0, stream>>>(d_im, d_col, fh, fw, sh, sw, nr_groups,
                                         param);
    after_kernel_launch();
}

void megdnn::cuda::local_share::_do_local_share_col2im(
        const float* d_col, float* d_im, int fh, int fw, int sh, int sw,
        int nr_groups, const Param& param, cudaStream_t stream) {
    void (*kern)(const float* __restrict__, float* __restrict__, int, int, int,
                 int, int, Param);
    kern = local_share_col2im<float>;

    constexpr int threads_x = 256;
    uint32_t nr_threads =
            _get_kern_block_size(reinterpret_cast<const void*>(kern));
    uint32_t nr_threads_x = std::min(threads_x, param.n);
    uint32_t nr_threads_y =
            std::min(static_cast<int>(nr_threads / nr_threads_x), param.ci);
    uint32_t nr_blocks_x = param.hi * param.wi,
             nr_blocks_y = DIVUP(param.n, nr_threads_x),
             nr_blocks_z = DIVUP(param.ci, nr_threads_y);
    dim3 threads{nr_threads_x, nr_threads_y, 1};
    dim3 blocks{nr_blocks_x, nr_blocks_y, nr_blocks_z};
    kern<<<blocks, threads, 0, stream>>>(d_col, d_im, fh, fw, sh, sw, nr_groups,
                                         param);
    after_kernel_launch();
}

// vim: syntax=cuda.doxygen
