/**
 * \file dnn/src/rocm/indexing_multi_axis_vec/kern_gen_offset_base.hip.cpp
 *
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "hip_header.h"
#include "./kern.h.hip"
#include "megdnn/internal/defs.h"
#include "src/rocm/utils.h.hip"

using namespace megdnn;
using namespace rocm;
using namespace indexing_multi_axis_vec;

namespace {
    template<int nidx, int idx_ndim>
    __global__ void kgen_offset_base(GenOffsetBaseParam<nidx, idx_ndim> param) {
        int oidx = threadIdx.x + blockDim.x * blockIdx.x;
        if (oidx < param.size) {
            int offset = 0;
#pragma unroll
            for (int i = 0; i < nidx; ++ i) {
                auto& indexer = param.indexer[i];
                int offset2 = 0, coidx = oidx;
#pragma unroll
                for (int j = idx_ndim-1; j >= 0; --j) {
                    int ax_idx;
                    if (j) {
                        int next_coidx = coidx / indexer.shape[j-1];
                        ax_idx = coidx - (next_coidx * indexer.shape[j-1].divisor());
                        coidx = next_coidx;
                    } else {
                        ax_idx = coidx;
                    }
                    offset2 += indexer.stride[j] * ax_idx;
                }
                int data_idx = indexer.ptr[offset2];
                data_idx += (data_idx < 0 ? param.data_shape[i] : 0);
                if (static_cast<uint32_t>(data_idx) >= param.data_shape[i]) {
                    // cast to uint32 to handle both negative and overflow
                    set_async_error_info(param.error_info, param.error_tracker,
                            "invalid advanced indexing: "
                            "indexer=%d idx=%d shape=%d",
                            i, data_idx, param.data_shape[i]);
                    data_idx = 0;
                }
                offset += data_idx * param.data_stride[i];
            }
            param.output[oidx] = offset;
        }
    }
}

namespace megdnn {
namespace rocm {
namespace indexing_multi_axis_vec {

#define INST(_m, _n) \
    template void gen_offset_base( \
            const GenOffsetBaseParam<_m, _n> &, hipStream_t);

    MEGDNN_FOREACH_TENSOR_NDIM(INST, 1)
    MEGDNN_FOREACH_TENSOR_NDIM(INST, 2)
    MEGDNN_FOREACH_TENSOR_NDIM(INST, 3)
    MEGDNN_FOREACH_TENSOR_NDIM(INST, 4)
    MEGDNN_FOREACH_TENSOR_NDIM(INST, 5)
    MEGDNN_FOREACH_TENSOR_NDIM(INST, 6)
    MEGDNN_FOREACH_TENSOR_NDIM(INST, 7)

#undef INST

} // namespace indexing_multi_axis_vec
} // namespace rocm
} // namespace megdnn

template<int nidx, int idx_ndim>
void indexing_multi_axis_vec::gen_offset_base(
        const GenOffsetBaseParam<nidx, idx_ndim> &param, hipStream_t stream) {
    void (*kptr)(GenOffsetBaseParam<nidx, idx_ndim>) = kgen_offset_base<nidx, idx_ndim>;
    int bsize = 256; 
    hipLaunchKernelGGL(kptr,
		       DIVUP(param.size, bsize), bsize, 0, stream,
		       param);
}

// vim: ft=cuda syntax=cpp.doxygen

