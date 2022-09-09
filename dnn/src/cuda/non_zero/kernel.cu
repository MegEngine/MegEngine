#include "./kernel.cuh"
#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/utils.cuh"

using namespace megdnn;
using namespace megdnn::cuda;
using namespace megdnn::cuda::non_zero;

namespace {

__global__ void multi_div_size(
        dt_int32* div_size, const size_t* shape_arr, size_t ndim) {
    div_size[0] = 1;
    for (int div_index = 1; div_index < ndim; div_index++) {
        div_size[div_index] = shape_arr[ndim - div_index] * div_size[div_index - 1];
    }
}
__global__ void expansion(
        dt_int32* pt, dt_int32* div_size, const size_t* shape_arr, int loop_count,
        int index_size, int ndim) {
    int dim_idx = blockIdx.x;
    int offset_from_each_dim = (blockDim.x * threadIdx.y + threadIdx.x) +
                               (loop_count * (blockDim.x * blockDim.y));
    if (offset_from_each_dim >= index_size)
        return;
    int offset = dim_idx * (index_size) + offset_from_each_dim;
    dt_int32* target_pt = pt + offset;
    int dim_pos_of_ele = *target_pt / div_size[ndim - 1 - dim_idx];
    dt_int32 dim_index_of_ele = dim_pos_of_ele % shape_arr[dim_idx];
    target_pt[0] = dim_index_of_ele;
}

__global__ void copy_kern(dt_int32* dest_idx, const dt_int32* src_idx, uint32_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size && src_idx[tid] > src_idx[tid - 1]) {
        uint32_t v = src_idx[tid] - 1;
        dest_idx[v] = tid;
    }
}
__global__ void set_zero(dt_int32* dest) {
    dest[0] = 0;
}
}  // namespace

void megdnn::cuda::non_zero::copy_idx(
        dt_int32* dest_idx, dt_int32* src_idx, uint32_t size, cudaStream_t stream) {
    int nr_thread = query_blocksize_for_kernel(copy_kern);
    int nr_block = DIVUP(size, nr_thread);
    // Todo  Set block and thread to 1 to set an element to 0. Need to consider
    // optimization
    set_zero<<<1, 1, 0, stream>>>(src_idx);
    after_kernel_launch();
    copy_kern<<<nr_block, nr_thread, 0, stream>>>(dest_idx, src_idx + 1, size);
    after_kernel_launch();
}

void megdnn::cuda::non_zero::expansion_index(
        dt_int32* dst_pt, size_t index_size, const size_t* src_shape,
        size_t* src_shape_workspace_pt, size_t src_ndim, dt_int32* div_workspace_pt,
        cudaStream_t stream) {
    cuda_check(cudaMemcpyAsync(
            src_shape_workspace_pt, src_shape, sizeof(size_t) * 7,
            cudaMemcpyHostToDevice, stream));
    // Todo  change the cuda kernel to cpu for loop, or make the number of threads and
    // blocks more reasonable
    multi_div_size<<<1, 1, 0, stream>>>(
            div_workspace_pt, src_shape_workspace_pt, src_ndim);
    after_kernel_launch();
    dim3 threadsPerBlock(
            std::min<int>(NR_THREADS_X, index_size),
            std::min<int>(NR_THREADS_Y, DIVUP(index_size, NR_THREADS_X)));
    int loop_size = DIVUP(index_size, (NR_THREADS_X * NR_THREADS_Y));
    for (int loop_idx = 0; loop_idx < loop_size; loop_idx++) {
        expansion<<<src_ndim, threadsPerBlock, 0, stream>>>(
                dst_pt, div_workspace_pt, src_shape_workspace_pt, loop_idx, index_size,
                src_ndim);
        after_kernel_launch();
    }
}