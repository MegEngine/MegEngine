/**
 * \file dnn/src/cuda/correlation/correlation_cuda.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/cuda/correlation/correlation_cuda.cuh"

#include <cfloat>
#include "megdnn/dtype.h"
#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/utils.cuh"
#define ROUND_OFF 50000

using namespace megdnn;
namespace megdnn {
namespace cuda {
namespace correlation {

#define CUDA_KERNEL_LOOP(vtid, vthreads)                                    \
    for (int vtid = blockIdx.x * blockDim.x + threadIdx.x; vtid < vthreads; \
         vtid += blockDim.x * gridDim.x)

template <typename T>
__global__ void forward_kernel(const int nthreads, const T* data1,
                               const T* data2, T* dst, const int bchannels,
                               const int bheight, const int bwidth,
                               const int tchannels, const int theight,
                               const int twidth, const int kernel_size,
                               const int max_displacement, const int stride1,
                               const int stride2, const int pad_size,
                               const bool is_multiply) {
    CUDA_KERNEL_LOOP(idx, nthreads) {
        int kernel_radius = (kernel_size - 1) / 2;
        int neighborhood_grid_radius = max_displacement / stride2;
        int neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;

        int x = idx % twidth;
        int y = (idx / twidth) % theight;
        int c = (idx / twidth / theight) % tchannels;
        int n = idx / twidth / theight / tchannels;

        // get src center position in image1
        int x1 = x * stride1 + kernel_radius + max_displacement - pad_size;
        int y1 = y * stride1 + kernel_radius + max_displacement - pad_size;

        // get offset of center in image2
        int s2o = (c % neighborhood_grid_width - neighborhood_grid_radius) *
                  stride2;
        int s2p = (c / neighborhood_grid_width - neighborhood_grid_radius) *
                  stride2;

        int x2 = x1 + s2o;
        int y2 = y1 + s2p;

        // compute kernel correlation
        T sum = T(0.f);
        for (int i = -kernel_radius; i <= kernel_radius; i++) {
            for (int j = -kernel_radius; j <= kernel_radius; j++) {
                int in_x1 = x1 + i;
                int in_y1 = y1 + j;
                int in_x2 = x2 + i;
                int in_y2 = y2 + j;

                for (int channel = 0; channel < bchannels; channel++) {
                    T tmp1 = T(0.f);
                    T tmp2 = T(0.f);
                    if (in_x1 >= 0 && in_x1 < bwidth && in_y1 >= 0 &&
                        in_y1 < bheight) {
                        int idx1 =
                                ((n * bchannels + channel) * bheight + in_y1) *
                                        bwidth +
                                in_x1;
                        tmp1 = data1[idx1];
                    }

                    if (in_x2 >= 0 && in_x2 < bwidth && in_y2 >= 0 &&
                        in_y2 < bheight) {
                        int idx2 =
                                ((n * bchannels + channel) * bheight + in_y2) *
                                        bwidth +
                                in_x2;
                        tmp2 = data2[idx2];
                    }
                    if (is_multiply) {
                        sum += tmp1 * tmp2;
                    } else {
                        sum += fabsf(tmp1 - tmp2);
                    }
                }
            }
        }

        const int sumelems =
                (kernel_radius * 2 + 1) * (kernel_radius * 2 + 1) * bchannels;
        dst[idx] = sum / sumelems;
    }
}

template <typename T>
__global__ void backward_kernel_data1(
        const int nthreads, const T* diff, const T* data1, const T* data2,
        T* grad1, const int bchannels, const int bheight, const int bwidth,
        const int tchannels, const int theight, const int twidth,
        const int kernel_size, const int max_displacement, const int stride1,
        const int stride2, const int pad_size, const bool is_multiply) {
    CUDA_KERNEL_LOOP(idx, nthreads) {
        int kernel_radius = (kernel_size - 1) / 2;
        int neighborhood_grid_radius = max_displacement / stride2;
        int neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;

        int x = idx % bwidth;
        int y = (idx / bwidth) % bheight;
        int c = (idx / bwidth / bheight) % bchannels;
        int n = idx / bwidth / bheight / bchannels;

        T tmp1 = data1[idx];
        // Get X,Y ranges and clamp
        // round_off is a trick to enable integer division with ceil, even for
        // negative numbers We use a large offset, for the inner part not to
        // become negative.
        const int round_off = ROUND_OFF;
        const int round_off_s1 = stride1 * round_off;

        // we show cal the x_min,y_min,x_max,y_max of diff for grad1(x,y)
        // for diff_x_min, diff_y_min, x,y at the position of right-down
        // ceil (l - 2*kernel_radius - max_displacement + pad_size) / stride1
        int xmin = (x + pad_size - 2 * kernel_radius - max_displacement +
                    round_off_s1 - 1) /
                           stride1 +
                   1 - round_off;
        int ymin = (y + pad_size - 2 * kernel_radius - max_displacement +
                    round_off_s1 - 1) /
                           stride1 +
                   1 - round_off;

        // floor (l - max_displacement + pad_size) / stride1
        int xmax = (x + pad_size - max_displacement + round_off_s1) / stride1 -
                   round_off;
        int ymax = (y + pad_size - max_displacement + round_off_s1) / stride1 -
                   round_off;

        T sum = T(0.f);

        if (xmax >= 0 && ymax >= 0 && (xmin <= twidth - 1) &&
            (ymin <= theight - 1)) {
            xmin = max(0, xmin);
            xmax = min(twidth - 1, xmax);

            ymin = max(0, ymin);
            ymax = min(theight - 1, ymax);

            for (int p = -neighborhood_grid_radius;
                 p <= neighborhood_grid_radius; p++) {
                for (int o = -neighborhood_grid_radius;
                     o <= neighborhood_grid_radius; o++) {
                    // Get bottom1 data:
                    int s2o = stride2 * o;
                    int s2p = stride2 * p;
                    int x2 = x + s2o, y2 = y + s2p;

                    int idx2 =
                            ((n * bchannels + c) * bheight + y2) * bwidth + x2;
                    T tmp2 = T(0.f);

                    if (x2 >= 0 && x2 < bwidth && y2 >= 0 && y2 < bheight) {
                        tmp2 = data2[idx2];
                    }

                    int op = (p + neighborhood_grid_radius) *
                                     neighborhood_grid_width +
                             (o + neighborhood_grid_radius);
                    int diff_channels_offset = (n * tchannels + op);
                    for (int diff_y = ymin; diff_y <= ymax; diff_y++) {
                        for (int diff_x = xmin; diff_x <= xmax; diff_x++) {
                            int idxtopdiff =
                                    (diff_channels_offset * theight + diff_y) *
                                            twidth +
                                    diff_x;

                            if (is_multiply) {
                                sum += diff[idxtopdiff] * tmp2;
                            } else {
                                T sign = (tmp1 >= tmp2) ? T(1.f) : T(-1.f);
                                sum += diff[idxtopdiff] * sign;
                            }
                        }
                    }
                }
            }
        }

        const int sumelems =
                (kernel_radius * 2 + 1) * (kernel_radius * 2 + 1) * bchannels;
        grad1[idx] = sum / sumelems;
    }
}

template <typename T>
__global__ void backward_kernel_data2(
        const int nthreads, const T* diff, const T* data1, const T* data2,
        T* grad2, const int bchannels, const int bheight, const int bwidth,
        const int tchannels, const int theight, const int twidth,
        const int kernel_size, const int max_displacement, const int stride1,
        const int stride2, const int pad_size, const bool is_multiply) {
    CUDA_KERNEL_LOOP(idx, nthreads) {
        int kernel_radius = (kernel_size - 1) / 2;
        int neighborhood_grid_radius = max_displacement / stride2;
        int neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;

        int x = idx % bwidth;
        int y = (idx / bwidth) % bheight;
        int c = (idx / bwidth / bheight) % bchannels;
        int n = idx / bwidth / bheight / bchannels;

        T tmp2 = data2[idx];

        T sum = T(0.f);

        for (int p = -neighborhood_grid_radius; p <= neighborhood_grid_radius;
             p++) {
            for (int o = -neighborhood_grid_radius;
                 o <= neighborhood_grid_radius; o++) {
                int s2o = o * stride2;
                int s2p = p * stride2;

                int x1 = x - s2o;
                int y1 = y - s2p;

                const int round_off = ROUND_OFF;
                const int round_off_s1 = stride1 * round_off;

                int xmin = (x1 + pad_size - 2 * kernel_radius -
                            max_displacement + round_off_s1 - 1) /
                                   stride1 +
                           1 - round_off;
                int ymin = (y1 + pad_size - 2 * kernel_radius -
                            max_displacement + round_off_s1 - 1) /
                                   stride1 +
                           1 - round_off;
                int xmax = (x1 + pad_size - max_displacement + round_off_s1) /
                                   stride1 -
                           round_off;
                int ymax = (y1 + pad_size - max_displacement + round_off_s1) /
                                   stride1 -
                           round_off;

                if (xmax >= 0 && ymax >= 0 && (xmin <= twidth - 1) &&
                    (ymin <= theight - 1)) {
                    xmin = max(0, xmin);
                    xmax = min(twidth - 1, xmax);

                    ymin = max(0, ymin);
                    ymax = min(theight - 1, ymax);

                    int idx1 =
                            ((n * bchannels + c) * bheight + y1) * bwidth + x1;
                    T tmp1 = T(0.f);
                    if (x1 >= 0 && x1 < bwidth && y1 >= 0 && y1 < bheight) {
                        tmp1 = data1[idx1];
                    }

                    int op = (p + neighborhood_grid_radius) *
                                     neighborhood_grid_width +
                             (o + neighborhood_grid_radius);
                    int diff_channels_offset = (n * tchannels + op);
                    for (int diff_y = ymin; diff_y <= ymax; diff_y++) {
                        for (int diff_x = xmin; diff_x <= xmax; diff_x++) {
                            int idxtopdiff =
                                    (diff_channels_offset * theight + diff_y) *
                                            twidth +
                                    diff_x;

                            if (is_multiply) {
                                sum += diff[idxtopdiff] * tmp1;
                            } else {
                                T sign = (tmp1 >= tmp2) ? T(-1.f) : T(1.f);
                                sum += diff[idxtopdiff] * sign;
                            }
                        }
                    }
                }
            }
        }

        const int sumelems =
                (kernel_radius * 2 + 1) * (kernel_radius * 2 + 1) * bchannels;
        grad2[idx] = sum / sumelems;
    }
}

template <typename T>
void forward_proxy(const int nthreads, const T* data1, const T* data2, T* dst,
                   const int bchannels, const int bheight, const int bwidth,
                   const int tchannels, const int theight, const int twidth,
                   const int kernel_size, const int max_displacement,
                   const int stride1, const int stride2, const int pad_size,
                   const bool is_multiply, cudaStream_t stream) {
    int threads_block = query_blocksize_for_kernel(forward_kernel<T>);
    forward_kernel<T>
            <<<DIVUP(nthreads, threads_block), threads_block, 0, stream>>>(
                    nthreads, data1, data2, dst, bchannels, bheight, bwidth,
                    tchannels, theight, twidth, kernel_size, max_displacement,
                    stride1, stride2, pad_size, is_multiply);
    after_kernel_launch();
}

template <typename T>
void backward_proxy_data1(const int nthreads, const T* diff, const T* data1,
                          const T* data2, T* grad1, const int bchannels,
                          const int bheight, const int bwidth,
                          const int tchannels, const int theight,
                          const int twidth, const int kernel_size,
                          const int max_displacement, const int stride1,
                          const int stride2, const int pad_size,
                          const bool is_multiply, cudaStream_t stream) {
    int threads_block = query_blocksize_for_kernel(backward_kernel_data1<T>);
    backward_kernel_data1<T>
            <<<DIVUP(nthreads, threads_block), threads_block, 0, stream>>>(
                    nthreads, diff, data1, data2, grad1, bchannels, bheight,
                    bwidth, tchannels, theight, twidth, kernel_size,
                    max_displacement, stride1, stride2, pad_size, is_multiply);
    after_kernel_launch();
}

template <typename T>
void backward_proxy_data2(const int nthreads, const T* diff, const T* data1,
                          const T* data2, T* grad2, const int bchannels,
                          const int bheight, const int bwidth,
                          const int tchannels, const int theight,
                          const int twidth, const int kernel_size,
                          const int max_displacement, const int stride1,
                          const int stride2, const int pad_size,
                          const bool is_multiply, cudaStream_t stream) {
    int threads_block = query_blocksize_for_kernel(backward_kernel_data2<T>);
    backward_kernel_data2<T>
            <<<DIVUP(nthreads, threads_block), threads_block, 0, stream>>>(
                    nthreads, diff, data1, data2, grad2, bchannels, bheight,
                    bwidth, tchannels, theight, twidth, kernel_size,
                    max_displacement, stride1, stride2, pad_size, is_multiply);
    after_kernel_launch();
}

#define INST(T)                                                                \
    template void forward_proxy<T>(                                            \
            const int, const T*, const T*, T* dst, const int, const int,       \
            const int, const int, const int, const int, const int, const int,  \
            const int, const int, const int, const bool, cudaStream_t);        \
    template void backward_proxy_data1<T>(                                     \
            const int, const T*, const T*, const T*, T*, const int, const int, \
            const int, const int, const int, const int, const int, const int,  \
            const int, const int, const int, const bool, cudaStream_t);        \
    template void backward_proxy_data2<T>(                                     \
            const int, const T*, const T*, const T*, T*, const int, const int, \
            const int, const int, const int, const int, const int, const int,  \
            const int, const int, const int, const bool, cudaStream_t);
INST(dt_float32)
INST(dt_float16)
INST(dt_bfloat16)
#undef INST

}  // namespace roi_align
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
