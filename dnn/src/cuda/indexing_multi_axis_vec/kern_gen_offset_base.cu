/**
 * \file dnn/src/cuda/indexing_multi_axis_vec/kern_gen_offset_base.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./kern.cuh"
#include "megdnn/internal/defs.h"
#include "src/cuda/query_blocksize.cuh"

using namespace megdnn;
using namespace cuda;
using namespace indexing_multi_axis_vec;

namespace {
    template<int nidx>
    __global__ void kgen_offset_base(GenOffsetBaseParam<nidx> param) {
        int oidx = threadIdx.x + blockDim.x * blockIdx.x;
        if (oidx < param.size) {
            int offset = 0;
#pragma unroll
            for (int i = 0; i < nidx; ++ i) {
                int data_idx = param.indexer[i].ptr[
                         param.indexer[i].stride * oidx];
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

template<int nidx>
void indexing_multi_axis_vec::gen_offset_base(
        const GenOffsetBaseParam<nidx> &param, cudaStream_t stream) {
    void (*kptr)(GenOffsetBaseParam<nidx>) = kgen_offset_base<nidx>;
    int bsize = query_blocksize_for_kernel(kptr);
    (*kptr) <<<DIVUP(param.size, bsize), bsize, 0, stream>>> (param);
}

namespace megdnn {
namespace cuda {
namespace indexing_multi_axis_vec {

#define INST(_n) \
    template void gen_offset_base( \
            const GenOffsetBaseParam<_n> &, cudaStream_t);
    MEGDNN_FOREACH_TENSOR_NDIM(INST)
#undef INST

} // namespace indexing_multi_axis_vec
} // namespace cuda
} // namespace megdnn

// vim: ft=cuda syntax=cpp.doxygen

