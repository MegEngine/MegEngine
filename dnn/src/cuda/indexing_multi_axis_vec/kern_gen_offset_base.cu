#include "./kern.cuh"
#include "megdnn/internal/defs.h"
#include "src/cuda/query_blocksize.cuh"

using namespace megdnn;
using namespace cuda;
using namespace indexing_multi_axis_vec;

namespace {
template <int nidx, int idx_ndim>
__global__ void kgen_offset_base(GenOffsetBaseParam<nidx, idx_ndim> param) {
    int oidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (oidx < param.size) {
        int offset = 0;
#pragma unroll
        for (int i = 0; i < nidx; ++i) {
            auto& indexer = param.indexer[i];
            // index in index
            int idx_flat = 0, coidx = oidx;
#pragma unroll
            for (int j = idx_ndim - 1; j >= 0; --j) {
                int ax_idx;
                if (j) {
                    int next_coidx = coidx / indexer.shape[j - 1];
                    ax_idx = coidx - (next_coidx * indexer.shape[j - 1].divisor());
                    coidx = next_coidx;
                } else {
                    ax_idx = coidx;
                }
                idx_flat += indexer.stride[j] * ax_idx;
            }
            int data_idx = indexer.ptr[idx_flat];
            data_idx += (data_idx < 0 ? param.data_shape[i] : 0);
            if (static_cast<uint32_t>(data_idx) >= param.data_shape[i]) {
                // cast to uint32 to handle both negative and overflow
                set_async_error_info(
                        param.error_info, param.error_tracker,
                        "invalid advanced indexing: "
                        "input index %d is out of bounds for axis %d with size %d",
                        data_idx, i, param.data_shape[i]);
                data_idx = 0;
            }
            // calculate offset from current index
            offset += data_idx * param.data_stride[i];
        }
        // sum offsets and store at offset table
        param.output[oidx] = offset;
    }
}
}  // namespace

template <int nidx, int idx_ndim>
void indexing_multi_axis_vec::gen_offset_base(
        const GenOffsetBaseParam<nidx, idx_ndim>& param, cudaStream_t stream) {
    void (*kptr)(GenOffsetBaseParam<nidx, idx_ndim>) = kgen_offset_base<nidx, idx_ndim>;
    int bsize = query_blocksize_for_kernel(kptr);
    (*kptr)<<<DIVUP(param.size, bsize), bsize, 0, stream>>>(param);
}

namespace megdnn {
namespace cuda {
namespace indexing_multi_axis_vec {

#define INST(_m, _n) \
    template void gen_offset_base(const GenOffsetBaseParam<_m, _n>&, cudaStream_t);

MEGDNN_FOREACH_TENSOR_NDIM(INST, 1)
MEGDNN_FOREACH_TENSOR_NDIM(INST, 2)
MEGDNN_FOREACH_TENSOR_NDIM(INST, 3)
MEGDNN_FOREACH_TENSOR_NDIM(INST, 4)
MEGDNN_FOREACH_TENSOR_NDIM(INST, 5)
MEGDNN_FOREACH_TENSOR_NDIM(INST, 6)
MEGDNN_FOREACH_TENSOR_NDIM(INST, 7)

#undef INST

}  // namespace indexing_multi_axis_vec
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cuda syntax=cpp.doxygen
