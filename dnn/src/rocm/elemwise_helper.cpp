/**
 * \file dnn/src/rocm/elemwise_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "src/rocm/utils.h"
#include "src/rocm/elemwise_helper.h.hip"
#include "megcore_cdefs.h"

#include "src/common/utils.h"

#include <limits>
#include <mutex>
#include <unordered_map>

#define _cb_check_ndim(n) megdnn::TensorShape::MAX_NDIM == n ||
static_assert(MEGDNN_FOREACH_TENSOR_NDIM(_cb_check_ndim) false,
              "bad foreach ndim");
#undef _cb_check_ndim

namespace megdnn {
namespace rocm {

// ParamElemVisitor::init impls
namespace elemwise_intl {

template <int ndim, typename ctype>
void ParamElemVisitor<ndim, ctype, BCAST_OTHER>::host_init(const TensorND& rv,
                                                           int /*grid_size*/,
                                                           int /*block_size*/) {
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

template <typename ctype>
void ParamElemVisitor<3, ctype, BCAST_101>::host_init(const TensorND& rv,
                                                      int grid_size,
                                                      int block_size) {
    uint32_t shape2, shape1;
    int stride1;
    if (rv.layout.ndim == 3) {
        megdnn_assert(!rv.layout.stride[0] && !rv.layout.stride[2]);
        shape1 = rv.layout[1];
        shape2 = rv.layout[2];
        stride1 = rv.layout.stride[1];
    } else {
        megdnn_assert(rv.layout.ndim == 2 && !rv.layout.stride[1]);
        shape1 = rv.layout[0];
        shape2 = rv.layout[1];
        stride1 = rv.layout.stride[0];
    }
    m_ptr = rv.ptr<ctype>();
    m_stride1 = stride1;
    m_shape12.host_init(grid_size * block_size, shape2, shape1);
}

template <typename ctype>
void ParamElemVisitor<2, ctype, BCAST_10>::host_init(const TensorND& rv,
                                                     int grid_size,
                                                     int block_size) {
    megdnn_assert(rv.layout.ndim == NDIM && !rv.layout.stride[0]);
    m_ptr = rv.ptr<ctype>();
    m_stride1 = rv.layout.stride[1];
    m_shape1.host_init(grid_size * block_size, rv.layout.shape[1]);
}

template <typename ctype>
void ParamElemVisitor<2, ctype, BCAST_01>::host_init(const TensorND& rv,
                                                     int grid_size,
                                                     int block_size) {
    megdnn_assert(rv.layout.ndim == NDIM && !rv.layout.stride[1]);
    m_ptr = rv.ptr<ctype>();
    m_stride0 = rv.layout.stride[0];
    m_shape1.host_init(grid_size * block_size, rv.layout.shape[1]);
}

template <typename ctype>
void ParamElemVisitor<1, ctype, BCAST_FULL>::host_init(const TensorND& rv,
                                                       int /*grid_size*/,
                                                       int /*block_size*/) {
    megdnn_assert(rv.layout.ndim == NDIM && !rv.layout.stride[0]);
    m_ptr = rv.ptr<ctype>();
}

#define INST(ndim, ctype, brd) template class ParamElemVisitor<ndim, ctype, brd>
#define INST_FOR_CTYPE                  \
    MEGDNN_FOREACH_TENSOR_NDIM(ndim_cb) \
    INST(3, ct, BCAST_101);             \
    INST(2, ct, BCAST_10);              \
    INST(2, ct, BCAST_01);              \
    INST(1, ct, BCAST_FULL);

#define ndim_cb(_ndim) INST(_ndim, ct, BCAST_OTHER);

#define ct dt_byte
INST_FOR_CTYPE
#undef ct
#define ct dt_int32
INST_FOR_CTYPE
#undef ct
#define ct dt_float32
INST_FOR_CTYPE
#undef ct
#if !MEGDNN_DISABLE_FLOAT16
#define ct dt_float16
INST_FOR_CTYPE
#undef ct
#endif
#define ct dt_bfloat16
INST_FOR_CTYPE
#undef ct
#define ct dt_int8
INST_FOR_CTYPE
#undef ct
#define ct dt_uint8
INST_FOR_CTYPE
#undef ct
#define ct dt_int16
INST_FOR_CTYPE
#undef ct
#define ct dt_quint8
INST_FOR_CTYPE
#undef ct
#define ct dt_qint8
INST_FOR_CTYPE
#undef ct
#define ct dt_qint32
INST_FOR_CTYPE
#undef ct
#define ct dt_bool
INST_FOR_CTYPE
#undef ct

#undef ndim_cb

#undef INST_FOR_CTYPE
#undef INST

}  // namespace elemwise_intl

void elemwise_intl::get_launch_spec(const void* /*kern*/, size_t size,
                                    int* grid_size, int* block_size) {
    safe_size_in_kern(size);
    const uint32_t blocks = 256;
    *block_size = blocks;
    int a = size / (blocks * 2), b = (size - 1) / (blocks * 3) + 1;
    *grid_size = std::max(a, b);
    if (!*grid_size) {
        *block_size = std::min<int>(std::max<int>(size / 64, 1) * 32, 1024);
        *grid_size = std::max<int>(size / *block_size, 1);
    }
    // because we unroll 3 times in the kernel
    megdnn_assert(static_cast<size_t>(*block_size) * *grid_size * 3 >= size);
}

void elemwise_intl::on_bad_ndim(int ndim) {
    megdnn_throw(ssprintf("invalid ndim: %d", ndim));
    MEGDNN_MARK_USED_VAR(ndim);
}
}  // namespace rocm
}  // namespace megdnn


// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

