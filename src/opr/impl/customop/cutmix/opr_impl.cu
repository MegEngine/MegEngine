#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "../utils.cuh"
#include "./opr_impl.h"
#include "megbrain/custom/custom.h"
#include "megbrain/custom/platform/custom_cuda.h"

using namespace custom;

template <typename T>
__global__ void cutmix_kernel_cuda(
        T* inp1, T* inp2, const T* bbx1, const T* bbx2, const T* bby1, const T* bby2,
        T* out, size_t C, size_t H, size_t W) {
    int xid = blockDim.x * blockIdx.x + threadIdx.x;
    int yid = blockDim.y * blockIdx.y + threadIdx.y;
    int zid = blockDim.z * blockIdx.z + threadIdx.z;

    if ((int(bbx1[zid]) <= xid && xid < int(bbx2[zid])) &&
        (int(bby1[zid]) <= yid && yid < int(bby2[zid]))) {
        for (size_t i = 0; i < C; i++) {
            out[zid * (C * H * W) + i * (H * W) + xid * H + yid] =
                    inp2[zid * (C * H * W) + i * (H * W) + xid * H + yid];
        }
    } else {
        for (size_t i = 0; i < C; i++) {
            out[zid * (C * H * W) + i * (H * W) + xid * H + yid] =
                    inp1[zid * (C * H * W) + i * (H * W) + xid * H + yid];
        }
    }
}

template <typename T>
__device__ T cuda_clip(T value, T min_value, T max_value) {
    return min(max(value, min_value), max_value);
}

template <typename T>
__global__ void get_bbox_cuda(
        T* cx, T* cy, T* cut_h, T* cut_w, T* bbx1, T* bbx2, T* bby1, T* bby2, size_t H,
        size_t W) {
    int b = blockIdx.x;
    T cx_i = cx[b];
    T cy_i = cy[b];
    if (threadIdx.x == 0) {
        int cuh = cut_h[b] / 2;
        bbx1[b] = cuda_clip<T>((cx_i - cuh), 0, H);
        bbx2[b] = cuda_clip<T>((cx_i + cuh), 0, H);
    } else {
        int cuw = cut_w[b] / 2;
        bby1[b] = cuda_clip<T>((cy_i - cuw), 0, W);
        bby2[b] = cuda_clip<T>((cy_i + cuw), 0, W);
    }
}

void launch_cuda_kernel(
        const Tensor& inp1, const Tensor& inp2, const Tensor& cx, const Tensor& cy,
        const Tensor& cut_h, const Tensor& cut_w, Tensor& output, Tensor& bbx1,
        Tensor& bbx2, Tensor& bby1, Tensor& bby2) {
    auto inp_shape = inp1.shape();
    size_t b = inp_shape[0];
    size_t c = inp_shape[1];
    size_t h = inp_shape[2];
    size_t w = inp_shape[3];
    auto stream = get_cuda_stream(inp1.device());
    dim3 grid_cut(b);
    dim3 block_cut(2);
    DISPATCH_INT_AND_FLOAT_TYPES(
            cx.dtype(), "get_bbox_cuda", ([&]() {
                get_bbox_cuda<<<grid_cut, block_cut, 0, stream>>>(
                        cx.data<scalar_t>(), cy.data<scalar_t>(),
                        cut_h.data<scalar_t>(), cut_w.data<scalar_t>(),
                        bbx1.data<scalar_t>(), bbx2.data<scalar_t>(),
                        bby1.data<scalar_t>(), bby2.data<scalar_t>(), h, w);
            }));
    int grad_x = ceil(h / (16 * 1.0));
    int grad_y = ceil(w / (16 * 1.0));
    dim3 grid(grad_x, grad_y, b);
    dim3 block(16, 16, 1);
    DISPATCH_INT_AND_FLOAT_TYPES(
            inp1.dtype(), "cutmix_kernel_cuda", ([&]() {
                cutmix_kernel_cuda<<<grid, block, 0, stream>>>(
                        inp1.data<scalar_t>(), inp2.data<scalar_t>(),
                        bbx1.data<scalar_t>(), bbx2.data<scalar_t>(),
                        bby1.data<scalar_t>(), bby2.data<scalar_t>(),
                        output.data<scalar_t>(), c, h, w);
            }));
    after_kernel_launch();
}