#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "./matmul_scale.h"

using namespace custom;

// matmul_forward for Mat_mxk * Mat_k*n
template <typename T>
__global__ void matmul_forward_naive(
        const T* lhs, const T* rhs, T* res, size_t M, size_t K, size_t N, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    T acc = 0;
    for (int i = 0; i < K; ++i)
        acc += lhs[row * K + i] * rhs[i * N + col];
    res[row * N + col] = acc * scale;
}

// matmul_backward_lhs for Mat_mxk * Mat_k*n = Mat_mxn
// that is Mat_mxn * Mat_nxk
template <typename T>
__global__ void matmul_backward_lhs_naive(
        const T* rhs, const T* ograd, T* lhs_grad, size_t M, size_t K, size_t N,
        float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    T acc = 0;
    for (int i = 0; i < N; ++i)
        acc += ograd[row * N + i] * rhs[col * N + i];
    lhs_grad[row * K + col] = acc / scale;
}

// matmul_backward_rhs for Mat_mxk * Mat_k*n = Mat_mxn
// that is Mat_kxm * Mat_mxn
template <typename T>
__global__ void matmul_backward_rhs_naive(
        const T* lhs, const T* ograd, T* rhs_grad, size_t M, size_t K, size_t N,
        float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    T acc = 0;
    for (int i = 0; i < M; ++i)
        acc += lhs[i * K + row] * ograd[i * N + col];
    rhs_grad[row * N + col] = acc / scale;
}

void matmul_forward_helper(
        const Tensor& lhs, const Tensor& rhs, Tensor& res, size_t M, size_t K, size_t N,
        float scale) {
    dim3 block(1, 1);
    dim3 grid(N / block.x, M / block.y);

    DISPATCH_INT_AND_FLOAT_TYPES(res.dtype(), "matmul_forward", ([&]() {
                                     matmul_forward_naive<scalar_t><<<grid, block>>>(
                                             lhs.data<scalar_t>(), rhs.data<scalar_t>(),
                                             res.data<scalar_t>(), M, K, N, scale);
                                 }));
}

void matmul_backward_lhs_helper(
        const Tensor& rhs, const Tensor& ograd, Tensor& lhs_grad, size_t M, size_t K,
        size_t N, float scale) {
    dim3 block(1, 1);
    dim3 grid(K / block.x, M / block.y);
    DISPATCH_INT_AND_FLOAT_TYPES(
            lhs_grad.dtype(), "matmul_backward_lhs", ([&]() {
                matmul_backward_lhs_naive<scalar_t><<<grid, block>>>(
                        rhs.data<scalar_t>(), ograd.data<scalar_t>(),
                        lhs_grad.data<scalar_t>(), M, K, N, scale);
            }));
}

void matmul_backward_rhs_helper(
        const Tensor& lhs, const Tensor& ograd, Tensor& rhs_grad, size_t M, size_t K,
        size_t N, float scale) {
    dim3 block(1, 1);
    dim3 grid(N / block.x, K / block.y);
    DISPATCH_INT_AND_FLOAT_TYPES(
            rhs_grad.dtype(), "matmul_backward_rhs", ([&]() {
                matmul_backward_rhs_naive<scalar_t><<<grid, block>>>(
                        lhs.data<scalar_t>(), ograd.data<scalar_t>(),
                        rhs_grad.data<scalar_t>(), M, K, N, scale);
            }));
}
