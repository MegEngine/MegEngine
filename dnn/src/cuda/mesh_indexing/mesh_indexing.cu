/**
 * \file dnn/src/cuda/mesh_indexing/mesh_indexing.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/basic_types.h"
#include "megdnn/dtype.h"
#include "src/common/indexing_multi_axis_vec_kdef.h"
#include "src/cuda/indexing_multi_axis_vec/kern.cuh"
#include "src/cuda/mesh_indexing/mesh_indexing.cuh"
#include "src/cuda/utils.cuh"

#define KERN_APPLY_OPR_INDEXING ::megdnn::indexing_multi_axis_vec_kdef::OprFwd

#define KERN_APPLY_OPR_INCR \
    ::megdnn::cuda::indexing_multi_axis_vec::OprAtomicIncr

#define KERN_APPLY_OPR_SET ::megdnn::indexing_multi_axis_vec_kdef::OprSet

namespace {

using namespace megdnn;
using namespace cuda;
using namespace mesh_indexing;

template <typename T, class Opr>
__global__ void mesh_indexing_general_kernel(T* src, T* dst,
                                             const KernIndexer indexer) {
    uint32_t dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_idx < indexer.size) {
        int src_idx = indexer.convert_indxer(dst_idx);
        Opr::apply(src[src_idx], dst[dst_idx]);
    }
}
}  // namespace

namespace megdnn {
namespace cuda {
namespace mesh_indexing {

template <typename T, class Opr>
void mesh_indexing_proxy(T* src, T* dst, KernIndexer* indexer,
                         cudaStream_t stream) {
    mesh_indexing_general_kernel<T, Opr>
            <<<DIVUP(indexer->size, NR_THREADS), NR_THREADS, 0, stream>>>(
                    src, dst, *indexer);
}

#define INST(_ctype)                                                    \
    template void mesh_indexing_proxy<_ctype, KERN_APPLY_OPR_INDEXING>( \
            _ctype * src, _ctype * dst, KernIndexer * indexer,          \
            cudaStream_t stream);                                       \
                                                                        \
    template void mesh_indexing_proxy<_ctype, KERN_APPLY_OPR_SET>(      \
            _ctype * src, _ctype * dst, KernIndexer * indexer,          \
            cudaStream_t stream);

#define INST_ATOMIC_ADD(_ctype)                                     \
    template void mesh_indexing_proxy<_ctype, KERN_APPLY_OPR_INCR>( \
            _ctype * src, _ctype * dst, KernIndexer * indexer,      \
            cudaStream_t stream);

#define cb(_dtype) INST(DTypeTrait<_dtype>::ctype)

MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb

#define cb(_dtype) INST_ATOMIC_ADD(DTypeTrait<_dtype>::ctype)

cb(dtype::Float32);
cb(dtype::Int32)
#undef cb

#undef INST
}  // namespace mesh_indexing
}  // namespace cuda
}  // namespace megdnn
