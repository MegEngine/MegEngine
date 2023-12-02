
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <stdio.h>
#include <algorithm>
#include <cfloat>
#include <limits>

#include "../utils.cuh"
#include "../utils.h"
#include "./opr_impl.h"

#if MGB_CUSTOM_OP

#include "megbrain/custom/custom.h"
#include "megbrain/custom/platform/custom_cuda.h"

using namespace custom;

enum BorderMode {
    BORDER_REPLICATE = 0,
    BORDER_REFLECT = 1,
    BORDER_REFLECT_101 = 2,
    BORDER_WRAP = 3,
    BORDER_CONSTANT = 4,
    BORDER_TRANSPARENT = 5,
    BORDER_ISOLATED = 6
};

BorderMode to_bordermode(const std::string& border_type) {
    BorderMode bmode;
    if (border_type == "BORDER_REPLICATE")
        bmode = BorderMode::BORDER_REPLICATE;
    else if (border_type == "BORDER_REFLECT")
        bmode = BorderMode::BORDER_REFLECT;
    else if (border_type == "BORDER_REFLECT_101")
        bmode = BorderMode::BORDER_REFLECT_101;
    else if (border_type == "BORDER_WRAP")
        bmode = BorderMode::BORDER_WRAP;
    else if (border_type == "BORDER_CONSTANT")
        bmode = BorderMode::BORDER_CONSTANT;
    else if (border_type == "BORDER_TRANSPARENT")
        bmode = BorderMode::BORDER_TRANSPARENT;
    else if (border_type == "BORDER_ISOLATED")
        bmode = BorderMode::BORDER_ISOLATED;
    return bmode;
}

__device__ int border_interpolate(int p, int len, BorderMode bmode) {
    if ((unsigned)p < (unsigned)len)
        ;
    else if (bmode == BorderMode::BORDER_REPLICATE)
        p = p < 0 ? 0 : len - 1;
    else if (
            bmode == BorderMode::BORDER_REFLECT ||
            bmode == BorderMode::BORDER_REFLECT_101) {
        int delta = (bmode == BorderMode::BORDER_REFLECT_101);
        if (len == 1)
            return 0;
        do {
            if (p < 0)
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        } while ((unsigned)p >= (unsigned)len);
    } else if (bmode == BorderMode::BORDER_WRAP) {
        assert(len > 0);
        if (p < 0)
            p -= ((p - len + 1) / len) * len;
        while (p >= len) {
            p -= len;
        }
    } else if (bmode == BorderMode::BORDER_CONSTANT)
        p = -1;
    else
        assert(false);
    return p;
}

template <typename T>
__device__ const T max(const T& lhs, const T& rhs) {
    return lhs > rhs ? lhs : rhs;
}

template <typename T>
__global__ void dilate(
        T* input, T* kernel, T* output, const int I_N, const int I_C, const int I_H,
        const int I_W, const int K_H, const int K_W, BorderMode border_type,
        const double border_value, T min_ele = std::numeric_limits<T>::min()) {
    size_t nc_iter = threadIdx.z + blockIdx.z * blockDim.z;
    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    const int NC = I_N * I_C;
    const size_t stride_c = I_H * I_W;
    const size_t HALF_KH = K_H / 2, HALF_KW = K_W / 2;

    if (w >= I_W || h >= I_H)
        return;
    const size_t stride_nc = blockDim.z * gridDim.z;
    for (int nc_pos = 0; nc_pos < NC; nc_pos += stride_nc) {
        T val = min_ele;
        for (int kh = 0; kh < K_H; kh++) {
            int h_pos = border_interpolate(h + kh - HALF_KH, I_H, border_type);
            for (int kw = 0; kw < K_W; kw++) {
                int w_pos = border_interpolate(w + kw - HALF_KW, I_W, border_type);
                T kernel_ele = kernel[kh * K_W + kw];
                if (kernel_ele) {
                    if (w_pos != -1 && h_pos != -1) {
                        T input_ele = input[nc_iter * (stride_c) + h_pos * I_W + w_pos];
                        val = max(input_ele, val);
                    } else {
                        if (nc_iter % I_C)
                            val = INFINITY;
                        else
                            val = max(static_cast<T>(border_value), val);
                    }
                }
            }
        }
        output[nc_iter * (stride_c) + h * I_W + w] = val;
    }
}

void launch_dilate_kernel(
        const Tensor& inp, const Tensor& kernel, Tensor& output, Tensor& workspace,
        const int iterations, const std::string& border_type,
        const double border_value) {
    const Shape& inp_shape = inp.shape();
    const size_t I_N = inp_shape[0];
    const size_t I_C = inp_shape[1];
    const size_t I_H = inp_shape[2];
    const size_t I_W = inp_shape[3];
    const size_t total_nr_elements = I_N * I_C * I_H * I_W;

    cudaStream_t stream = get_cuda_stream(inp.device());
    const int NC = I_N * I_C;
    const int max_threads_perblock = get_max_threads_per_block(inp.device());
    const int max_threads = std::min<int>(max_threads_perblock, 512);
    const int* max_threadsdim = get_max_threads_dim(inp.device());
    const int* max_gridsize = get_max_grid_size(inp.device());

    int block_x = std::min<int>(
            max_threadsdim[0], std::min<int>(last_pow2(I_W), max_threads));
    int block_y = std::min<int>(
            max_threadsdim[1], std::min<int>(last_pow2(I_H), max_threads / block_x));
    int block_z = std::min<int>(
            max_threadsdim[2], std::min<int>(NC, max_threads / block_x / block_y));
    const dim3 block(block_x, block_y, block_z);

    int grid_x = ceil_div(static_cast<int>(I_W), block_x);
    int grid_y = ceil_div(static_cast<int>(I_H), block_y);
    int grid_z = std::min<int>(max_gridsize[2], ceil_div(NC, block_z * 4));
    const dim3 grid(grid_x, grid_y, grid_z);

    const Shape& kernel_shape = kernel.shape();
    const int K_H = kernel_shape[0];
    const int K_W = kernel_shape[1];
    BorderMode bmode = to_bordermode(border_type);

    DISPATCH_INT_AND_FLOAT_TYPES(inp.dtype(), "dispatch_kernel_cuda", [&]() {
        scalar_t* inp_ptr = inp.data<scalar_t>();
        scalar_t* kernel_ptr = kernel.data<scalar_t>();
        scalar_t* output_ptr = output.data<scalar_t>();
        scalar_t* workspace_ptr = workspace.data<scalar_t>();
        dilate<scalar_t><<<grid, block, 0, stream>>>(
                inp_ptr, kernel_ptr, output_ptr, I_N, I_C, I_H, I_W, K_H, K_W, bmode,
                border_value);
        for (int idx = 1; idx < iterations; idx++) {
            cudaMemcpyAsync(
                    workspace_ptr, output_ptr, total_nr_elements * sizeof(scalar_t),
                    cudaMemcpyDeviceToDevice);
            dilate<scalar_t><<<grid, block, 0, stream>>>(
                    workspace_ptr, kernel_ptr, output_ptr, I_N, I_C, I_H, I_W, K_H, K_W,
                    bmode, border_value);
        }
    });
    after_kernel_launch();
}

#endif
