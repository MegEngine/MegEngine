/**
 * \file src/rocm/indexing_multi_axis_vec/kern_gen_offset_base.hip.cpp
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
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

namespace megdnn {
namespace rocm {
namespace indexing_multi_axis_vec {

#define INST(_n) \
    template void gen_offset_base( \
            const GenOffsetBaseParam<_n> &, hipStream_t);
    MEGDNN_FOREACH_TENSOR_NDIM(INST)
#undef INST

} // namespace indexing_multi_axis_vec
} // namespace rocm
} // namespace megdnn

template<int nidx>
void indexing_multi_axis_vec::gen_offset_base(
        const GenOffsetBaseParam<nidx> &param, hipStream_t stream) {
    void (*kptr)(GenOffsetBaseParam<nidx>) = kgen_offset_base<nidx>;
    int bsize = 256; 
    hipLaunchKernelGGL(kptr,
		       DIVUP(param.size, bsize), bsize, 0, stream,
		       param);
}

// vim: ft=cuda syntax=cpp.doxygen

