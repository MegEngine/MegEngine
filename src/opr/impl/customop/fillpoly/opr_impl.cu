// idea from
// https://github.com/opencv/opencv/blob/e20250139a72e1cb8f2b027d631ad0b07731f8c4/modules/imgproc/src/drawing.cpp#L2021

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <algorithm>
#include <cfloat>
#include "../utils.cuh"
#include "../utils.h"
#include "./opr_impl.h"
#if MGB_CUSTOM_OP
#include "megbrain/custom/custom.h"
#include "megbrain/custom/platform/custom_cuda.h"

using namespace custom;

struct PolyEdge {
    int y0, y1;
    long double x, dx;
    PolyEdge* next;
    __device__ bool operator<(const PolyEdge& other) const {
        if (this->y0 != other.y0)
            return this->y0 < other.y0;
        else if (this->x != other.x)
            return this->x < other.x;
        else
            return this->dx < other.dx;
    }
};

struct Rect {
    int y_max, y_min;
    int x_max, x_min;
};

template <typename T>
__device__ void dda_line(
        const int start_x, const int start_y, const int end_x, const int end_y, T* data,
        const T* color, const int N, const int C, const int H, const int W) {
    double xbase = start_x + 0.5, ybase = start_y + 0.5;
    double dx = end_x - start_x, dy = end_y - start_y;
    double length = (abs(dx) > abs(dy) ? abs(dx) : abs(dy));
    dx /= length;
    dy /= length;
    for (int i = 1; i <= length; i++) {
        int offset = 0;
        const int NC_stride = H * W;
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                int pos = offset + (int)(xbase) + (int)(ybase)*W;
                data[pos] = color[c];
                offset += NC_stride;
            }
        }
        xbase += dx;
        ybase += dy;
    }
}

template <typename T>
__global__ void collect_poly_edges(
        const int* points, T* img, const T* color, const int* lens, PolyEdge* edges,
        int* sizes, Rect* rects, const int N, const int C, const int H, const int W) {
    int thread_Idx = threadIdx.x;
    const size_t count = lens[thread_Idx];

    int pt_offset = 0, edge_offset = 0;
    for (int idx = 0; idx < thread_Idx; idx++) {
        pt_offset += lens[idx];
        edge_offset += lens[idx] + 1;
    }
    const size_t stride = 2;
    pt_offset *= stride;

    const int* points_begin = points + pt_offset;
    const int *pt0 = points_begin + (count - 1) * stride, *pt1;
    int size = 0;
    Rect rect;
    rect.y_max = rect.x_max = INT_MIN;
    rect.y_min = rect.x_min = INT_MAX;
    for (int idx = 0; idx < count; idx++, pt0 = pt1) {
        pt1 = points_begin + idx * stride;
        const int pt1_x = pt1[0], pt1_y = pt1[1];
        const int pt0_x = pt0[0], pt0_y = pt0[1];

        dda_line<T>(pt0_x, pt0_y, pt1_x, pt1_y, img, color, N, C, H, W);

        if (pt0_y == pt1_y)
            continue;
        PolyEdge edge;
        edge.dx = double(pt1_x - pt0_x) / (pt1_y - pt0_y);
        if (pt0_y < pt1_y) {
            edge.y0 = (int)(pt0_y);
            edge.y1 = (int)(pt1_y);
            edge.x = pt0_x;
        } else {
            edge.y0 = (int)(pt1_y);
            edge.y1 = (int)(pt0_y);
            edge.x = pt1_x;
        }
        int pos = size + edge_offset;
        edges[pos] = edge;
        size++;
        rect.y_max = max(rect.y_max, pt1_y);
        rect.y_min = min(rect.y_min, pt1_y);
        rect.x_max = max(rect.x_max, pt1_x);
        rect.x_min = min(rect.x_min, pt1_x);
    }
    rects[thread_Idx] = rect;
    sizes[thread_Idx] = size;
}

template <typename T>
__device__ void set_pixel(
        int x, int y, T* data, const T* color, const int N, const int C, const int H,
        const int W) {
    int offset = 0;
    const int nc_stride = H * W;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            int pos = offset + x + y * W;
            data[pos] = color[c];
            offset += nc_stride;
        }
    }
}
__device__ bool is_right(PolyEdge edge, int x, int y) {
    int x1 = edge.x + (y - edge.y0) * edge.dx;
    return x > x1;
}

template <typename T>
__global__ void my_fill_edge_collections(
        PolyEdge* edges, T* img, const T* color, const int* lens, int* sizes,
        Rect* rects, const int L, const int N, const int C, const int H, const int W) {
    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int h = threadIdx.y + blockIdx.y * blockDim.y;
    int l = threadIdx.z + blockIdx.z * blockDim.z;
    if (w >= W || h >= H)
        return;
    Rect rect = rects[l];
    if (rect.y_max < 0 || rect.x_max < 0 || rect.y_min >= H || rect.x_min >= W)
        return;
    if (h <= rect.y_min || h >= rect.y_max || w <= rect.x_min || w >= rect.x_max)
        return;

    int edge_offset = 0;

    const size_t stride_l = blockDim.z * gridDim.z;
    for (int l_pos = l; l_pos < L; l_pos += stride_l) {
        for (int idx = 0; idx < l_pos; idx++)
            edge_offset += lens[idx] + 1;
        edges += edge_offset;
        int draw = 0;
        for (int edge_idx = 0; edge_idx < sizes[l_pos]; edge_idx++) {
            PolyEdge edge = edges[edge_idx];
            if (edge.y1 <= h)
                continue;
            if (edge.y0 >= h)
                break;
            draw ^= is_right(edge, w, h);
        }
        if (draw)
            set_pixel(w, h, img, color, N, C, H, W);
    }
}

void launch_fillpoly_kernel(
        const Tensor& inp, const Tensor& points, const Tensor& lens,
        const Tensor& color, Tensor& output) {
    const Shape& inp_shape = inp.shape();
    const size_t N = inp_shape[0];
    const size_t C = inp_shape[1];
    const size_t H = inp_shape[2];
    const size_t W = inp_shape[3];

    const Shape& len_shape = lens.shape();
    const size_t L = len_shape[0];
    std::vector<int> host_point_lens(L, -1);
    int32_t* points_len_ptr = lens.data<int32_t>();
    cudaStream_t stream = get_cuda_stream(inp.device());
    cuda_check(cudaMemcpyAsync(
            &host_point_lens[0], points_len_ptr, sizeof(int32_t) * L,
            cudaMemcpyDeviceToHost, stream));
    DISPATCH_INT_AND_FLOAT_TYPES(
            inp.dtype(), "fillpoly_kernel", ([&] {
                cuda_check(cudaMemcpyAsync(
                        output.data<scalar_t>(), inp.data<scalar_t>(),
                        sizeof(scalar_t) * (N * C * H * W), cudaMemcpyDeviceToDevice,
                        stream));

                int total = 0;
                for (int idx = 0; idx < L; idx++)
                    total += host_point_lens[idx];
                // for saving  temp edge
                total += L;

                int32_t* points_ptr = points.data<int32_t>();
                scalar_t* color_ptr = color.data<scalar_t>();
                scalar_t* output_ptr = output.data<scalar_t>();

                thrust::device_vector<PolyEdge> edges(total);
                thrust::device_vector<int> device_edge_sizes(total);
                thrust::device_vector<Rect> rects(L);
                PolyEdge* edges_ptr = thrust::raw_pointer_cast(&edges[0]);
                Rect* rects_ptr = thrust::raw_pointer_cast(&rects[0]);
                int* device_size_ptr = thrust::raw_pointer_cast(&device_edge_sizes[0]);

                collect_poly_edges<<<1, L, 0, stream>>>(
                        points_ptr, output_ptr, color_ptr, points_len_ptr, edges_ptr,
                        device_size_ptr, rects_ptr, N, C, H, W);
                after_kernel_launch();

                int pos = 0;
                std::vector<int> host_edge_size(L, -1);
                cuda_check(cudaMemcpyAsync(
                        &host_edge_size[0], device_size_ptr, sizeof(int) * L,
                        cudaMemcpyDeviceToHost, stream));
                for (int idx = 0; idx < L; idx++) {
                    int len = host_point_lens[idx];
                    thrust::sort(
                            edges.begin() + pos,
                            edges.begin() + pos + host_edge_size[idx]);
                    pos += len;
                }

                const int max_threads_perblock =
                        get_max_threads_per_block(inp.device());
                const int max_threads = std::min<int>(max_threads_perblock, 512);
                const int* max_threadsdim = get_max_threads_dim(inp.device());
                const int* max_gridsize = get_max_grid_size(inp.device());

                int block_x = std::min<int>(
                        max_threadsdim[0], std::min<int>(last_pow2(W), max_threads));
                int block_y = std::min<int>(
                        max_threadsdim[1],
                        std::min<int>(last_pow2(H), max_threads / block_x));
                int block_z = std::min<int>(
                        max_threadsdim[2],
                        std::min<int>(L, max_threads / block_x / block_y));
                const dim3 block(block_x, block_y, block_z);

                int grid_x = ceil_div(static_cast<int>(W), block_x);
                int grid_y = ceil_div(static_cast<int>(H), block_y);
                int grid_z = std::min<int>(
                        max_gridsize[2], ceil_div(static_cast<int>(L), block_z));
                const dim3 grid(grid_x, grid_y, grid_z);

                my_fill_edge_collections<scalar_t><<<grid, block, 0, stream>>>(
                        edges_ptr, output_ptr, color_ptr, points_len_ptr,
                        device_size_ptr, rects_ptr, L, N, C, H, W);
                after_kernel_launch();
            }));
}

#endif