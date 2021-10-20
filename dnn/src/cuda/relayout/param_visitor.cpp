/**
 * \file dnn/src/cuda/relayout/param_visitor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/utils.h"
#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/relayout/kern.cuh"
#include "src/cuda/relayout/kern_contiguous.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
template <int ndim, typename ctype>
void ParamElemVisitor<ndim, ctype, CONTIG_OTHER>::host_init(
        const TensorND& rv, int /*grid_size*/, int /*block_size*/) {
    megdnn_assert(rv.layout.ndim && rv.layout.ndim <= ndim);
    m_ptr = rv.ptr<ctype>();
    for (size_t i = 0; i < rv.layout.ndim; ++i) {
        m_stride[i] = rv.layout.stride[i];
        if (i + 1 < rv.layout.ndim)
            m_shape_highdim[i] = rv.layout.shape[i + 1];
    }
    for (int i = rv.layout.ndim - 1; i < ndim - 1; ++i) {
        m_shape_highdim[i] = 1;
    }
    for (int i = rv.layout.ndim; i < ndim; ++i) {
        m_stride[i] = 0;
    }
}
#pragma GCC diagnostic pop

template <int ndim, typename ctype>
void ParamElemVisitor<ndim, ctype, CONTIG_FULL>::host_init(
        const TensorND& rv, int /*grid_size*/, int /*block_size*/) {
    megdnn_assert_contiguous(rv.layout);
    m_ptr = rv.ptr<ctype>();
}

#define INST(ndim, ctype, ctg) template class ParamElemVisitor<ndim, ctype, ctg>
#define INST_FOR_CTYPE         MEGDNN_FOREACH_TENSOR_NDIM(ndim_cb)

#define ndim_cb(_ndim)             \
    INST(_ndim, ct, CONTIG_OTHER); \
    INST(_ndim, ct, CONTIG_FULL);

#define ct dt_byte
INST_FOR_CTYPE
#undef ct
#define ct dt_int32
INST_FOR_CTYPE
#undef ct
#define ct dt_float16
INST_FOR_CTYPE
#undef ct

#undef ndim_cb

#undef INST_FOR_CTYPE
#undef INST

template <int ndim>
void ParamElemVisitor<ndim, dt_quint4, CONTIG_OTHER>::host_init(
        const TensorND& rv, int /*grid_size*/, int /*block_size*/) {
    megdnn_assert(rv.layout.ndim && rv.layout.ndim <= ndim);
    m_ptr = reinterpret_cast<Storage*>(rv.raw_ptr);
    ptrdiff_t min_stride = std::numeric_limits<ptrdiff_t>::max();
    for (size_t i = 0; i < rv.layout.ndim; ++i) {
        m_stride[i] = rv.layout.stride[i];
        m_shape[i] = rv.layout.shape[i];
        if (i + 1 < rv.layout.ndim) {
            m_shape_highdim[i] = rv.layout.shape[i + 1];
            if (rv.layout.stride[i + 1] == 1)
                m_align_shape_highdim[i] =
                        (uint32_t)round_up((int)rv.layout.shape[i + 1], 2);
            else
                m_align_shape_highdim[i] = rv.layout.shape[i + 1];
        }
        // \remark: stride=0 means this dimension should be broadcast, so here
        // we skip dimension with stride that equals 0
        if (rv.layout.stride[i] != 0 && min_stride > rv.layout.stride[i]) {
            min_stride = rv.layout.stride[i];
        }
    }
    megdnn_assert(min_stride == 1 || min_stride == 2);
    m_is_min_stride_2 = (min_stride == 2);
    for (size_t i = rv.layout.ndim - 1; i < ndim - 1; ++i) {
        m_shape_highdim[i] = 1;
        m_align_shape_highdim[i] = 1;
    }
    for (size_t i = rv.layout.ndim; i < ndim; ++i) {
        m_stride[i] = 0;
        m_shape[i] = 1;
    }
    m_is_physical_contiguous = rv.layout.is_physical_contiguous();
    m_is_contiguous = rv.layout.is_contiguous();
}

#define INST(ndim, ctg) template class ParamElemVisitor<ndim, dt_quint4, ctg>
#define ndim_cb(_ndim)  INST(_ndim, CONTIG_OTHER);

MEGDNN_FOREACH_TENSOR_NDIM(ndim_cb)

#undef ndim_cb
#undef INST

}  // namespace cuda
}  // namespace megdnn
// vim: ft=cpp syntax=cpp.doxygen
