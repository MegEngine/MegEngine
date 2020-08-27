#include "nms_kern.cuh"

#include <cassert>
#include <algorithm>

namespace {

// each thread computs one bit
const int THREADS_PER_BLOCK = 64;

const int WARP_SIZE = 32;

// use aligned structure for large memory transaction
struct __align__(16) Box {
    float x0, y0, x1, y1;
};

//! return whether IoU(a, b) > thresh
__device__ __forceinline__ bool box_iou(Box a, Box b, float thresh) {
    float left = max(a.x0, b.x0), right = min(a.x1, b.x1);
    float top = max(a.y0, b.y0), bottom = min(a.y1, b.y1);
    float width = max(right - left, 0.f),
          height = max(bottom - top, 0.f);
    float interS = width * height;
    float Sa = (a.x1 - a.x0) * (a.y1 - a.y0);
    float Sb = (b.x1 - b.x0) * (b.y1 - b.y0);
    return interS > (Sa + Sb - interS) * thresh;
}

//! store uint64_t with cache streaming
__device__ __forceinline__ void store_u64_cs(uint64_t *ptr, uint64_t val) {
    asm volatile("st.cs.u64 [%0], %1;" :  : "l"(ptr), "l"(val));
}

//! load uint64_t with cache streaming
__device__ __forceinline__ uint64_t load_u64_cs(const uint64_t *ptr) {
    uint64_t val;
    asm volatile("ld.cs.u64 %0, [%1];" : "=l"(val) : "l"(ptr));
    return val;
}

__global__ void kern_gen_mask(
        const int nr_boxes, const float nms_overlap_thresh,
        const Box *dev_boxes, const int dev_mask_width, uint64_t *dev_mask) {
    const int
        box_group_row = blockIdx.y,
        box_group_col = blockIdx.x;

    if (box_group_row > box_group_col)
        return;

    const int
        row_nr_boxes = min(
                nr_boxes - box_group_row * THREADS_PER_BLOCK,
                THREADS_PER_BLOCK),
        col_nr_boxes = min(
                nr_boxes - box_group_col * THREADS_PER_BLOCK,
                THREADS_PER_BLOCK);

    __shared__ Box block_boxes[THREADS_PER_BLOCK];

    if (threadIdx.x < col_nr_boxes) {
        block_boxes[threadIdx.x] = dev_boxes[
            THREADS_PER_BLOCK * box_group_col + threadIdx.x];
    }
    __syncthreads();

    if (threadIdx.x < row_nr_boxes) {
        const int cur_box_idx = THREADS_PER_BLOCK * box_group_row + threadIdx.x;
        Box cur_box = dev_boxes[cur_box_idx];

        uint64_t result = 0;
        const int start = (box_group_row == box_group_col) ?
            threadIdx.x + 1 : // blocks on diagnal
            0;
        for (int i = start; i < col_nr_boxes; ++ i) {
            result |= static_cast<uint64_t>(
                    box_iou(cur_box, block_boxes[i],
                        nms_overlap_thresh)) << i;
        }
        store_u64_cs(
                &dev_mask[cur_box_idx * dev_mask_width + box_group_col],
                result);
    }
}

//! true -> ~0, false -> 0
__device__ __forceinline__ uint32_t bool_as_u32_mask(bool v) {
    return (!v) - 1;
}

//! return min value of val in current warp
__device__ __forceinline__ uint32_t warp_reduce_min_brdcst(uint32_t val) {
    __shared__ uint32_t ans;
    static_assert(WARP_SIZE == 32, "warp size != 32");
#pragma unroll
    for (uint32_t offset = WARP_SIZE / 2; offset; offset /= 2)
        val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));

    if (!threadIdx.x)
        ans = val;
    __syncthreads();
    return ans;
}

struct BitwiseOrArgs {
    uint64_t *dst;
    const uint64_t *src;
    uint32_t size;
};

__device__ __forceinline__ void bitwise_or_single_warp(BitwiseOrArgs args) {
    uint64_t * __restrict__ dst = args.dst;
    const uint64_t * __restrict__ src = args.src;
    uint32_t size = args.size;
    for (uint32_t i = threadIdx.x; i < size; i += WARP_SIZE) {
        dst[i] |= load_u64_cs(&src[i]);
    }
}

__global__ void kern_gen_indices(
        uint32_t nr_boxes, uint32_t max_output, uint32_t overlap_mask_width,
        const uint64_t * __restrict__ overlap_mask, uint64_t *__restrict__ rm_mask,
        uint32_t * __restrict__ out_idx, uint32_t * __restrict__ out_size) {
    __shared__ uint32_t out_pos;
    __shared__ BitwiseOrArgs bitwise_or_args;

    const uint32_t nr_box_blocks = DIVUP(nr_boxes, 64);

    if (!threadIdx.x) {
        uint32_t cnt = nr_box_blocks * 64 - nr_boxes;
        // mark the padded boxes as having been removed
        rm_mask[nr_box_blocks - 1] = ((1ull << cnt) - 1) << (64 - cnt);
        out_pos = 0;
    }
    __syncthreads();

    uint32_t
        box_block_id = threadIdx.x,
        th0_box_block_id = 0;

    while (th0_box_block_id < nr_box_blocks) {
        bool in_range = box_block_id < nr_box_blocks;
        uint64_t cur_mask = ~rm_mask[box_block_id & bool_as_u32_mask(in_range)];
        uint32_t min_box_block_id = warp_reduce_min_brdcst(
                box_block_id | bool_as_u32_mask(!(in_range && cur_mask)));

        if (min_box_block_id + 1) {
            // min_box_block_id != UINT32_MAX, so at least one thread finds a
            // un-removed box
            if (min_box_block_id == box_block_id) {
                // exactly one thread can take this path
                uint32_t box_id_in_block = __ffsll(cur_mask) - 1,
                         box_id = box_block_id * 64 + box_id_in_block;

                // so this box would not be processed again
                rm_mask[box_block_id] |= 1ull << box_id_in_block;

                bitwise_or_args.dst = &rm_mask[box_block_id];
                bitwise_or_args.src =
                    &overlap_mask[box_id * overlap_mask_width + box_block_id];
                bitwise_or_args.size = nr_box_blocks - box_block_id;
                out_idx[out_pos ++] = box_id;
            }
            __syncthreads();
            if (out_pos == max_output)
                break;
            bitwise_or_single_warp(bitwise_or_args);

            // skip the blocks before min_box_block_id
            th0_box_block_id = min_box_block_id;
            box_block_id = min_box_block_id + threadIdx.x;
        } else {
            th0_box_block_id += WARP_SIZE;
            box_block_id += WARP_SIZE;
        }
    }

    if (out_pos < max_output) {
        // fill the values after out_pos
        uint32_t val = out_idx[out_pos - 1];
        for (uint32_t i = out_pos + threadIdx.x; i < max_output; i += WARP_SIZE) {
            out_idx[i] = val;
        }
    }
    if (!threadIdx.x) {
        *out_size = out_pos;
    }
}

} // anonymous namespace

void mgb::opr::standalone::nms::launch_gen_mask(
        const int nr_boxes, const float nms_overlap_thresh,
        const float *dev_boxes, const int dev_mask_width,
        uint64_t *dev_mask, cudaStream_t stream) {
    dim3 blocks(DIVUP(nr_boxes, THREADS_PER_BLOCK),
                DIVUP(nr_boxes, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    kern_gen_mask<<<blocks, threads, 0, stream>>>(
            nr_boxes, nms_overlap_thresh,
            reinterpret_cast<const Box*>(dev_boxes), dev_mask_width, dev_mask);
}

void mgb::opr::standalone::nms::launch_gen_indices(
        int nr_boxes, int max_output, int overlap_mask_width,
        const uint64_t *overlap_mask, uint64_t *rm_mask,
        uint32_t *out_idx, uint32_t *out_size,
        cudaStream_t stream) {
    kern_gen_indices<<<1, WARP_SIZE, 0, stream>>>(
            nr_boxes, max_output, overlap_mask_width,
            overlap_mask, rm_mask,
            out_idx, out_size);
}

// vim: ft=cuda syntax=cuda.doxygen
