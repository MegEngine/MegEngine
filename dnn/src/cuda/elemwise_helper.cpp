/**
 * \file dnn/src/cuda/elemwise_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/elemwise_helper.cuh"
#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/utils.h"

#include "src/common/utils.h"

#include <limits>
#include <mutex>
#include <unordered_map>

#define _cb_check_ndim(n) megdnn::TensorShape::MAX_NDIM == n ||
static_assert(MEGDNN_FOREACH_TENSOR_NDIM(_cb_check_ndim) false,
              "bad foreach ndim");
#undef _cb_check_ndim

namespace megdnn {
namespace cuda {

// ParamElemVisitor::init impls
namespace elemwise_intl {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
template <int ndim, typename ctype>
void ParamVisitorBase<ndim, ctype, BCAST_OTHER>::host_init(
        const TensorND& rv, int /*grid_size*/, int /*block_size*/,
        int /*packed_size*/) {
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

template <typename ctype>
void ParamVisitorBase<3, ctype, BCAST_101>::host_init(const TensorND& rv,
                                                      int grid_size,
                                                      int block_size,
                                                      int packed_size) {
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
    m_shape12.host_init(packed_size * grid_size * block_size, shape2, shape1);
}

template <typename ctype>
void ParamVisitorBase<2, ctype, BCAST_10>::host_init(const TensorND& rv,
                                                     int grid_size,
                                                     int block_size,
                                                     int packed_size) {
    megdnn_assert(rv.layout.ndim == NDIM && !rv.layout.stride[0]);
    m_ptr = rv.ptr<ctype>();
    m_stride1 = rv.layout.stride[1];
    m_shape1.host_init(packed_size * grid_size * block_size,
                       rv.layout.shape[1]);
}

template <typename ctype>
void ParamVisitorBase<2, ctype, BCAST_01>::host_init(const TensorND& rv,
                                                     int grid_size,
                                                     int block_size,
                                                     int packed_size) {
    megdnn_assert(rv.layout.ndim == NDIM && !rv.layout.stride[1]);
    m_ptr = rv.ptr<ctype>();
    m_stride0 = rv.layout.stride[0];
    m_shape1.host_init(packed_size * grid_size * block_size,
                       rv.layout.shape[1]);
}

template <typename ctype>
void ParamVisitorBase<1, ctype, BCAST_FULL>::host_init(const TensorND& rv,
                                                       int /*grid_size*/,
                                                       int /*block_size*/,
                                                       int /*packed_size*/) {
    megdnn_assert(rv.layout.ndim == NDIM && !rv.layout.stride[0]);
    m_ptr = rv.ptr<ctype>();
}

template <typename ctype>
void ParamVectVisitor<4, ctype, BCAST_1010>::host_init(const TensorND& rv,
                                                       int grid_size,
                                                       int block_size) {
    megdnn_assert(rv.layout.ndim == NDIM && !rv.layout.stride[0] &&
                  !rv.layout.stride[2]);
    m_ptr = rv.ptr<ctype>();
    m_stride1 = rv.layout.stride[1];
    m_stride3 = rv.layout.stride[3];
    uint32_t shape1 = rv.layout.shape[1];
    uint32_t shape2 = rv.layout.shape[2];
    uint32_t shape3 = rv.layout.shape[3];
    m_shape123.host_init(packed_size * grid_size * block_size, shape2 * shape3,
                         shape1);
    m_shape3.host_init(packed_size * grid_size * block_size, shape3);
}

#define INST(ndim, ctype, brd) template class ParamVisitorBase<ndim, ctype, brd>
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
#define ct dt_float16
INST_FOR_CTYPE
#undef ct
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

#undef INST_FOR_CTYPE
#undef INST

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
#define ct dt_float16
INST_FOR_CTYPE
#undef ct
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

#define INST(dt_ibyte) template class ParamVectVisitor<4, dt_ibyte, BCAST_1010>
INST(dt_int8);
INST(dt_uint8);
INST(dt_bool);
INST(dt_qint8);
INST(dt_quint8);
#undef dt_ibyte

}  // namespace elemwise_intl

void elemwise_intl::get_launch_spec(const void* kern, size_t size,
                                    int* grid_size, int* block_size) {
    safe_size_in_kern(size);
    auto config = query_launch_config_for_kernel(kern);
    *block_size = config.block_size;
    int a = size / (config.block_size * 2),
        b = (size - 1) / (config.block_size * 3) + 1;
    if (current_device_prop().major <= 3) {
        // for Kepler, less blocks (more work per thread) is faster
        *grid_size = b;
    } else {
        *grid_size = std::max(a, b);
    }
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
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
