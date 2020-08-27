/**
 * \file dnn/src/rocm/relayout/relayout_contiguous.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "src/rocm/relayout/relayout_contiguous.h.hip"
#include "src/common/utils.h"

namespace megdnn {
namespace rocm {

namespace contiguous_intl {

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

template <int ndim, typename ctype>
void ParamElemVisitor<ndim, ctype, CONTIG_FULL>::host_init(const TensorND& rv,
                                                           int /*grid_size*/,
                                                           int /*block_size*/) {
    megdnn_assert_contiguous(rv.layout);
    m_ptr = rv.ptr<ctype>();
}

#define INST(ndim, ctype, ctg) template class ParamElemVisitor<ndim, ctype, ctg>
#define INST_FOR_CTYPE MEGDNN_FOREACH_TENSOR_NDIM(ndim_cb)

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

}  // namespace contiguous_intl

void get_last_contiguous_launch_spec(const void* /*kern*/, size_t size,
                                     size_t contiguous_size, int* grid_size,
                                     int* block_size) {
    safe_size_in_kern(size);
    const uint32_t blocks = 256;
    *block_size = blocks;

    int a = size / (blocks * (contiguous_size - 1)),
        b = (size - 1) / (blocks * contiguous_size) + 1;
    *grid_size = std::max(a, b);

    if (!*grid_size) {
        *block_size = std::min<int>(std::max<int>(size / 64, 1) * 32, 1024);
        *grid_size = std::max<int>(size / *block_size, 1);
    }

    // because we unroll contiguous_size times in the kernel
    megdnn_assert(static_cast<size_t>(*block_size) * *grid_size *
                          contiguous_size >=
                  size);
}

}  // namespace rocm
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
