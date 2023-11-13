#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <stdio.h>
#include <algorithm>
#include <cfloat>
#include "../utils.cuh"
#include "../utils.h"
#include "./opr_impl.h"
#if MGB_CUSTOM_OP
#include "megbrain/custom/custom.h"
#include "megbrain/custom/platform/custom_cuda.h"

using namespace custom;

template <typename T>
__device__ int roundf_to_int(T val) {
    return (val > 0.0) ? (val + 0.5) : (val - 0.5);
}

// ==================  nearest implementation  ==============================

__device__ __forceinline__ static int nearest_neighbor_compute_source_index(
        const float scale, int dst_index, int input_size) {
    int src_index = min(static_cast<int>(floorf((dst_index)*scale)), input_size - 1);
    return src_index;
}

template <typename T>
__global__ void nearest2d_cropandpad(
        T* input, T* output, T* percent, T* pad_val, int N, int C, int H, int W) {
    size_t nc_iter = threadIdx.z + blockIdx.z * blockDim.z;
    int w2 = threadIdx.x + blockIdx.x * blockDim.x;
    int h2 = threadIdx.y + blockIdx.y * blockDim.y;
    int nc = N * C;
    int stride_n = C * H * W;
    int stride_c = H * W;
    int stride_h = W;

    if (w2 >= W || h2 >= H) {
        return;
    }

    while (nc_iter < nc) {
        int nc_stride = blockDim.z * gridDim.z;

        int dst_index = (nc_iter * H + h2) * W + w2;
        int dst_index_stride = nc_stride * W * H;

        int n_idx = static_cast<int>((dst_index) / (C * H * W * 1.0));

        T per = percent[n_idx];
        bool is_pad = true;
        if (per <= 0) {
            is_pad = false;
        }
        per = fabsf(per);
        int resize_val_H = roundf_to_int(H * per);
        int resize_val_W = roundf_to_int(W * per);

        T height_scale = static_cast<T>(0);
        T width_scale = static_cast<T>(0);

        if (is_pad) {
            height_scale = static_cast<T>(2 * resize_val_H + H) / H;
            width_scale = static_cast<T>(2 * resize_val_W + W) / W;
        } else {
            height_scale = static_cast<T>(H - 2 * resize_val_H) / H;
            width_scale = static_cast<T>(W - 2 * resize_val_W) / W;
        }

        size_t h1 = 0;
        size_t w1 = 0;
        size_t src_index = 0;
        size_t src_index_stride = 0;
        if (is_pad) {
            h1 = nearest_neighbor_compute_source_index(
                    height_scale, h2, 2 * resize_val_H + H);
            w1 = nearest_neighbor_compute_source_index(
                    width_scale, w2, 2 * resize_val_W + W);
            src_index =
                    (nc_iter * (2 * resize_val_H + H) + h1) * (2 * resize_val_W + W) +
                    w1;
            src_index_stride =
                    nc_stride * (2 * resize_val_W + W) * (2 * resize_val_H + H);
        } else {
            h1 = nearest_neighbor_compute_source_index(
                    height_scale, h2, H - 2 * resize_val_H);
            w1 = nearest_neighbor_compute_source_index(
                    width_scale, w2, W - 2 * resize_val_W);
            src_index =
                    (nc_iter * (H - 2 * resize_val_H) + h1) * (W - 2 * resize_val_W) +
                    w1;
            src_index_stride =
                    nc_stride * (W - 2 * resize_val_W) * (H - 2 * resize_val_H);
        }

        if (is_pad) {
            int n_id =
                    src_index / (C * (2 * resize_val_H + H) * (2 * resize_val_W + W));
            int h_id =
                    src_index % (C * (2 * resize_val_H + H) * (2 * resize_val_W + W));
            int c_id = h_id / ((2 * resize_val_H + H) * (2 * resize_val_W + W));
            h_id = h_id % ((2 * resize_val_H + H) * (2 * resize_val_W + W));
            h_id = h_id / (2 * resize_val_W + W);
            int w_id = src_index % (2 * resize_val_W + W);

            if (h_id < resize_val_H || h_id >= H + resize_val_H ||
                w_id < resize_val_W || w_id >= W + resize_val_W) {
                output[dst_index] = pad_val[n_id];
            } else {
                int index_temp = n_id * stride_n + c_id * stride_c +
                                 (h_id - resize_val_H) * stride_h +
                                 (w_id - resize_val_W);
                output[dst_index] = input[index_temp];
            }
        } else {
            int n_id =
                    src_index / (C * (H - 2 * resize_val_H) * (W - 2 * resize_val_W));
            int h_id =
                    src_index % (C * (H - 2 * resize_val_H) * (W - 2 * resize_val_W));
            int c_id = h_id / ((H - 2 * resize_val_H) * (W - 2 * resize_val_W));
            h_id = h_id % ((H - 2 * resize_val_H) * (W - 2 * resize_val_W));
            h_id = h_id / (W - 2 * resize_val_W);
            int w_id = src_index % (W - 2 * resize_val_W);
            int index_temp = n_id * stride_n + c_id * stride_c +
                             (h_id + resize_val_H) * stride_h + (w_id + resize_val_W);
            output[dst_index] = input[index_temp];
        }

        dst_index += dst_index_stride;
        src_index += src_index_stride;
        nc_iter += nc_stride;
    }
}

// ==================  bilinear implementation  ==============================
template <typename T>
__device__ __forceinline__ static T pixel_compute_scale(
        int input_size, int output_size, bool align_corners) {
    if (align_corners) {
        if (output_size > 1) {
            return (T)(input_size - 1) / (output_size - 1);
        } else {
            return static_cast<T>(0);
        }
    } else {
        return (T)input_size / output_size;
    }
}

template <typename T>
__device__ __forceinline__ static T area_pixel_compute_source_index(
        T scale, int dst_index, bool align_corners, bool cubic) {
    if (align_corners) {
        return scale * dst_index;
    } else {
        T src_idx = scale * (dst_index + static_cast<T>(0.5)) - static_cast<T>(0.5);
        // See Note[Follow Opencv resize logic]
        return (!cubic && src_idx < static_cast<T>(0)) ? static_cast<T>(0) : src_idx;
    }
}

template <typename T>
__global__ void bilinear2d_cropandpad(
        T* input, T* output, T* percent, T* pad_val, int N, int C, int H, int W,
        bool align_corners) {
    int xid = blockDim.x * blockIdx.x + threadIdx.x;
    int yid = blockDim.y * blockIdx.y + threadIdx.y;
    int b_idx = blockIdx.z;
    int c_idx = threadIdx.z;
    if (xid < H && yid < W) {
        int stride_b2 = C * H * W;
        int stride_c2 = H * W;
        int stride_h2 = W;
        const int h2 = xid;
        const int w2 = yid;
        T per = percent[b_idx];
        bool is_pad = true;
        if (per <= 0) {
            is_pad = false;
        }
        per = fabsf(per);
        int resize_val_H = roundf_to_int(H * per);
        int resize_val_W = roundf_to_int(W * per);
        T rheight = static_cast<T>(0);
        T rwidth = static_cast<T>(0);
        if (is_pad) {
            rheight = pixel_compute_scale<T>((2 * resize_val_H + H), H, align_corners);
            rwidth = pixel_compute_scale<T>((2 * resize_val_W + W), W, align_corners);
        } else {
            rheight = pixel_compute_scale<T>((H - 2 * resize_val_H), H, align_corners);
            rwidth = pixel_compute_scale<T>((W - 2 * resize_val_W), W, align_corners);
        }

        //
        const T h1r = area_pixel_compute_source_index<T>(
                rheight, h2, align_corners, /*cubic=*/false);
        int h1 = h1r;
        int h1p = 0;
        if (is_pad) {
            h1p = (h1 < (2 * resize_val_H + H) - 1) ? 1 : 0;
        } else {
            h1p = (h1 < (H - 2 * resize_val_H) - 1) ? 1 : 0;
        }
        const T h1lambda = h1r - h1;
        const T h0lambda = static_cast<T>(1) - h1lambda;
        //
        const T w1r = area_pixel_compute_source_index<T>(
                rwidth, w2, align_corners, /*cubic=*/false);
        int w1 = w1r;
        int w1p = 0;
        if (is_pad) {
            w1p = (w1 < (2 * resize_val_W + W) - 1) ? 1 : 0;
        } else {
            w1p = (w1 < (W - 2 * resize_val_W) - 1) ? 1 : 0;
        }
        const T w1lambda = w1r - w1;
        const T w0lambda = static_cast<T>(1) - w1lambda;

        T val = static_cast<T>(0);

        if (is_pad) {
            T val1 = static_cast<T>(0), val2 = static_cast<T>(0),
              val3 = static_cast<T>(0), val4 = static_cast<T>(0);
            if (h1 < resize_val_H || h1 >= resize_val_H + H || w1 < resize_val_W ||
                w1 >= resize_val_W + W) {
                val1 = pad_val[b_idx];
            } else {
                val1 =
                        input[b_idx * stride_b2 + c_idx * stride_c2 +
                              (h1 - resize_val_H) * stride_h2 + (w1 - resize_val_W)];
            }

            if (h1 < resize_val_H || h1 >= resize_val_H + H ||
                w1 + w1p < resize_val_W || w1 + w1p >= resize_val_W + W) {
                val2 = pad_val[b_idx];
            } else {
                val2 =
                        input[b_idx * stride_b2 + c_idx * stride_c2 +
                              (h1 - resize_val_H) * stride_h2 +
                              (w1 + w1p - resize_val_W)];
            }

            if (h1 + h1p < resize_val_H || h1 + h1p >= resize_val_H + H ||
                w1 < resize_val_W || w1 >= resize_val_W + W) {
                val3 = pad_val[b_idx];
            } else {
                val3 =
                        input[b_idx * stride_b2 + c_idx * stride_c2 +
                              (h1 + h1p - resize_val_H) * stride_h2 +
                              (w1 - resize_val_W)];
            }

            if (h1 + h1p < resize_val_H || h1 + h1p >= resize_val_H + H ||
                w1 + w1p < resize_val_W || w1 + w1p >= resize_val_W + W) {
                val4 = pad_val[b_idx];
            } else {
                val4 =
                        input[b_idx * stride_b2 + c_idx * stride_c2 +
                              (h1 + h1p - resize_val_H) * stride_h2 +
                              (w1 + w1p - resize_val_W)];
            }
            val = h0lambda * (w0lambda * val1 + w1lambda * val2) +
                  h1lambda * (w0lambda * val3 + w1lambda * val4);
        } else {
            val = h0lambda * (w0lambda * input[b_idx * stride_b2 + c_idx * stride_c2 +
                                               (h1 + resize_val_H) * stride_h2 +
                                               (w1 + resize_val_W)] +
                              w1lambda * input[b_idx * stride_b2 + c_idx * stride_c2 +
                                               (h1 + resize_val_H) * stride_h2 +
                                               (w1 + w1p + resize_val_W)]) +
                  h1lambda * (w0lambda * input[b_idx * stride_b2 + c_idx * stride_c2 +
                                               (h1 + h1p + resize_val_H) * stride_h2 +
                                               (w1 + resize_val_W)] +
                              w1lambda * input[b_idx * stride_b2 + c_idx * stride_c2 +
                                               (h1 + h1p + resize_val_H) * stride_h2 +
                                               (w1 + w1p + resize_val_W)]);
        }

        output[b_idx * stride_b2 + c_idx * stride_c2 + h2 * stride_h2 + w2] = val;
    }
}

void launch_cropandpad_kernel(
        const Tensor& inp, const Tensor& percent, const Tensor& pad_val, Tensor& output,
        bool align_corners, std::string mode) {
    auto inp_shape = inp.shape();
    size_t N = inp_shape[0];
    size_t C = inp_shape[1];
    size_t H = inp_shape[2];
    size_t W = inp_shape[3];
    auto stream = get_cuda_stream(inp.device());
    if (mode == "bilinear") {
        dim3 grid(ceil(H / (16 * 1.0)), ceil(W / (16 * 1.0)), N);
        dim3 block(16, 16, C);
        DISPATCH_INT_AND_FLOAT_TYPES(
                inp.dtype(), "cropandpad_bilinear_kernel_cuda", ([&]() {
                    bilinear2d_cropandpad<scalar_t><<<grid, block, 0, stream>>>(
                            inp.data<scalar_t>(), output.data<scalar_t>(),
                            percent.data<scalar_t>(), pad_val.data<scalar_t>(), N, C, H,
                            W, align_corners);
                }));
        after_kernel_launch();
    } else if (mode == "nearest") {
        int nc = N * C;
        int maxThreadsPerBlock = get_max_threads_per_block(inp.device());
        const int max_threads = std::min<int>(maxThreadsPerBlock, 512);
        const int* maxThreadsDim = get_max_threads_dim(inp.device());
        const int* maxGridSize = get_max_grid_size(inp.device());

        int block_x = std::min<int>(
                maxThreadsDim[0], std::min<int>(last_pow2(W), max_threads));
        int block_y = std::min<int>(
                maxThreadsDim[1], std::min<int>(last_pow2(H), max_threads / block_x));
        int block_z = std::min<int>(
                maxThreadsDim[2], std::min<int>(nc, max_threads / block_x / block_y));
        const dim3 block(block_x, block_y, block_z);

        int grid_x = ceil_div(static_cast<int>(W), block_x);
        int grid_y = ceil_div(static_cast<int>(H), block_y);
        int grid_z = std::min<int>(maxGridSize[2], ceil_div(nc, block_z * 4));
        const dim3 grid(grid_x, grid_y, grid_z);
        DISPATCH_INT_AND_FLOAT_TYPES(
                inp.dtype(), "cropandpad_nearst_kernel_cuda", ([&]() {
                    nearest2d_cropandpad<scalar_t><<<grid, block, 0, stream>>>(
                            inp.data<scalar_t>(), output.data<scalar_t>(),
                            percent.data<scalar_t>(), pad_val.data<scalar_t>(), N, C, H,
                            W);
                }));
        after_kernel_launch();
    }
}
#endif
