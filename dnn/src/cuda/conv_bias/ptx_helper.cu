#include "src/cuda/conv_bias/ptx_helper.cuh"
#include "src/cuda/integer_subbyte_utils.cuh"
#include "src/cuda/query_blocksize.cuh"

using namespace megdnn;
using namespace cuda;
using namespace ptx;

namespace {
template <uint32_t size_bits, uint32_t interleaved>
__device__ __forceinline__ void reorder_imma_filter_func(
        int8_t* dst, const int8_t* src, uint32_t OC, uint32_t IC, uint32_t FH,
        uint32_t FW, uint32_t lane) {
    static constexpr uint32_t elements_per_lane = 128 / size_bits;
    uint32_t elements = lane * elements_per_lane;
    uint32_t row = elements / (IC * FH * FW);
    uint32_t col = elements - row * IC * FH * FW;
    uint32_t sec = row / 4;
    uint32_t res = col & (interleaved - 1);
    uint32_t sec_sec = row & 3;
    uint32_t sec_res = (row & 15) / 4;
    uint32_t crosswise_offset = ((sec_sec >> 1) * 2 * interleaved) +
                                (((sec_sec & 1) ^ (sec_res >> 1)) * interleaved);
    uint32_t residue_offset =
            ((res / elements_per_lane) ^ (sec_res & 1)) * elements_per_lane;
    uint32_t dst_offset =
            (sec / 2) * 8 * FH * FW * IC + (col / interleaved) * (8 * interleaved) +
            (sec & 1) * (4 * interleaved) + crosswise_offset + residue_offset;
    static constexpr uint32_t instruction_shape_col = 8;
    // 4 threads per Quad
    static constexpr uint32_t elements_per_thread = instruction_shape_col / 4;
    // 4 threads per Quad
    static constexpr uint32_t reordered_elements_per_thread = interleaved / 4;

    uint32_t elem_in_interleaved = row % interleaved;
    uint32_t elem_in_interleaved_pack = elem_in_interleaved / elements_per_thread;
    int elem_new = (row / interleaved * interleaved +
                    elem_in_interleaved_pack % 4 * reordered_elements_per_thread +
                    elem_in_interleaved_pack / 4 * elements_per_thread +
                    elem_in_interleaved % elements_per_thread) *
                           (IC * FH * FW) +
                   col;

    *(reinterpret_cast<int4*>(dst + (dst_offset * size_bits / 8))) =
            *(reinterpret_cast<const int4*>(src + (elem_new * size_bits / 8)));
}

template <uint32_t interleaved>
__device__ __forceinline__ void reorder_imma_bias_func(
        float* __restrict__ dst, float src_value, uint32_t OC, uint32_t lane) {
    dst[lane] = src_value;
}

template <uint32_t size_bits, uint32_t interleaved>
__global__ void reorder_imma_filter_bias_kernel(
        int8_t* __restrict__ dst_filter, float* __restrict__ dst_bias,
        const int8_t* __restrict__ src_filter, const int32_t* __restrict__ src_bias,
        float bias_scale, uint32_t OC, uint32_t IC, uint32_t FH, uint32_t FW) {
    static constexpr uint32_t elements_per_lane = 128 / size_bits;
    const uint32_t size1 = OC * IC * FH * FW / elements_per_lane;
    const uint32_t size2 = OC;
    uint32_t lane = threadIdx.x + blockIdx.x * blockDim.x;
    if (lane < size1) {
        reorder_imma_filter_func<size_bits, interleaved>(
                dst_filter, src_filter, OC, IC, FH, FW, lane);
    } else if (lane < size1 + size2) {
        lane = lane - size1;
        float src_bias_value = src_bias[lane] * bias_scale;
        reorder_imma_bias_func<interleaved>(dst_bias, src_bias_value, OC, lane);
    }
}

template <uint32_t size_bits, uint32_t interleaved>
__global__ void reorder_imma_filter_bias_fusion_zero_point_kernel(
        int8_t* __restrict__ dst_filter, float* __restrict__ dst_bias,
        const int8_t* __restrict__ src_filter, const int32_t* __restrict__ src_bias,
        float bias_scale, const int32_t* reduce_filter, float zero_point, uint32_t OC,
        uint32_t IC, uint32_t FH, uint32_t FW) {
    static constexpr uint32_t elements_per_lane = 128 / size_bits;
    const uint32_t size1 = OC * IC * FH * FW / elements_per_lane;
    const uint32_t size2 = OC;
    uint32_t lane = threadIdx.x + blockIdx.x * blockDim.x;
    if (lane < size1) {
        reorder_imma_filter_func<size_bits, interleaved>(
                dst_filter, src_filter, OC, IC, FH, FW, lane);
    } else if (lane < size1 + size2) {
        lane = lane - size1;
        // fusion bias and zero_point
        // zero_point = zero_point * src_scale * filter_scale
        float src_bias_value =
                src_bias[lane] * bias_scale - reduce_filter[lane] * zero_point;
        reorder_imma_bias_func<interleaved>(dst_bias, src_bias_value, OC, lane);
    }
}

}  // namespace

template <uint32_t size_bits, uint32_t interleaved>
void megdnn::cuda::ptx::reorder_imma_filter_bias(
        int8_t* dst_filter, float* dst_bias, const int8_t* src_filter,
        const int32_t* src_bias, float bias_scale, uint32_t OC, uint32_t IC,
        uint32_t FH, uint32_t FW, cudaStream_t stream) {
    static constexpr uint32_t elements_per_lane = 128 / size_bits;
    uint32_t nr_threads = query_blocksize_for_kernel(reinterpret_cast<const void*>(
            reorder_imma_filter_bias_kernel<size_bits, interleaved>));
    uint32_t vthreads = DIVUP(OC * IC * FH * FW, elements_per_lane) + OC;
    nr_threads = std::min(nr_threads, vthreads);
    uint32_t nr_blocks = DIVUP(vthreads, nr_threads);
    reorder_imma_filter_bias_kernel<size_bits, interleaved>
            <<<nr_blocks, nr_threads, 0, stream>>>(
                    dst_filter, dst_bias, src_filter, src_bias, bias_scale, OC, IC, FH,
                    FW);
    after_kernel_launch();
}

template <uint32_t size_bits, uint32_t interleaved>
void megdnn::cuda::ptx::reorder_imma_filter_bias_fusion_zero_point(
        int8_t* dst_filter, float* dst_bias, const int8_t* src_filter,
        const int32_t* src_bias, float bias_scale, const int32_t* reduce_filter,
        float zero_point, uint32_t OC, uint32_t IC, uint32_t FH, uint32_t FW,
        cudaStream_t stream) {
    static constexpr uint32_t elements_per_lane = 128 / size_bits;
    uint32_t nr_threads = query_blocksize_for_kernel(reinterpret_cast<const void*>(
            reorder_imma_filter_bias_fusion_zero_point_kernel<size_bits, interleaved>));
    uint32_t vthreads = DIVUP(OC * IC * FH * FW, elements_per_lane) + OC;
    nr_threads = std::min(nr_threads, vthreads);
    uint32_t nr_blocks = DIVUP(vthreads, nr_threads);
    reorder_imma_filter_bias_fusion_zero_point_kernel<size_bits, interleaved>
            <<<nr_blocks, nr_threads, 0, stream>>>(
                    dst_filter, dst_bias, src_filter, src_bias, bias_scale,
                    reduce_filter, zero_point, OC, IC, FH, FW);
    after_kernel_launch();
}

#define INST(_size_bits, _interleaved)                                           \
    template void                                                                \
    megdnn::cuda::ptx::reorder_imma_filter_bias<_size_bits, _interleaved>(       \
            int8_t * dst_filter, float* dst_bias, const int8_t* src_filter,      \
            const int32_t* src_bias, float bias_scale, uint32_t OC, uint32_t IC, \
            uint32_t FH, uint32_t FW, cudaStream_t stream);

INST(8, 32)
INST(4, 64)
#undef INST

#define INST(_size_bits, _interleaved)                                               \
    template void megdnn::cuda::ptx::reorder_imma_filter_bias_fusion_zero_point<     \
            _size_bits, _interleaved>(                                               \
            int8_t * dst_filter, float* dst_bias, const int8_t* src_filter,          \
            const int32_t* src_bias, float bias_scale, const int32_t* reduce_filter, \
            float zero_point, uint32_t OC, uint32_t IC, uint32_t FH, uint32_t FW,    \
            cudaStream_t stream);
INST(4, 64)
#undef INST

// vim: syntax=cuda.doxygen
