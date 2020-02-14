/**
 * \file dnn/src/cuda/relayout/param_visitor.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/basic_types.h"
#include "src/cuda/int_fastdiv.cuh"
#include "src/cuda/utils.cuh"


#pragma once

namespace megdnn {
namespace cuda {
#define devfunc __device__ __forceinline__

/*!
 * \brief contiguous type
 * If the layout is contiguous, then the type is CONTIG_FULL, CONTIG_OTHER
 * otherwise.
 */
enum ContigType { CONTIG_OTHER, CONTIG_FULL };

/* f{{{ ParamElemVisitor specialization */
/*!
* \brief visitor to access an element in a tensor at given logic index
* \tparam ctype plain element ctype (i.e. ctype in DTypeTrait)
* \tparam contig_mask bit mask for contig of params;
*
* host interface:
*      void host_init(
*              const TensorND &tensor, int grid_size, int block_size)
*
* device interface:
*      void thread_init(uint32_t idx)
*          called on thread entrance, with logical indexing; the index
y
*          go beyond buffer range
*
*      ctype* ptr()
*          return buffer pointer; can be used by specialized OpCaller
*
*      int offset(uint32_t idx)
*          get physical offset from logical index
*
*      ctype& at(uint32_t idx)
*          ptr()[offset(idx)]
*
*/
template <int ndim, typename ctype, ContigType contig_type>
class ParamElemVisitor;
#define PARAM_ELEM_VISITOR_COMMON_DEV      \
    devfunc ctype *ptr() { return m_ptr; } \
    devfunc ctype &at(uint32_t idx) { return m_ptr[offset(idx)]; }

//! specialization for CONTIG_OTHER
template <int ndim, typename ctype>
class ParamElemVisitor<ndim, ctype, CONTIG_OTHER> {
    ctype *__restrict m_ptr;
    int m_stride[ndim];

    //! m_shape_highdim[i] = original_shape[i + 1]
#ifdef _MSC_VER
    Uint32Fastdiv m_shape_highdim[ndim > 1 ? ndim - 1 : 1];
#else
    Uint32Fastdiv m_shape_highdim[ndim - 1];
#endif

public:
    static const int NDIM = ndim;

    void host_init(const TensorND &rv, int grid_size, int block_size);

#if MEGDNN_CC_CUDA
    devfunc void thread_init(uint32_t) {}

    devfunc void next() {}

    devfunc int offset(uint32_t idx) {
        int offset = 0;
#pragma unroll
        for (int i = ndim - 1; i >= 1; --i) {
            Uint32Fastdiv &shp = m_shape_highdim[i - 1];
            uint32_t idx_div = idx / shp;
            offset += (idx - idx_div * shp.divisor()) * m_stride[i];
            idx = idx_div;
        }
        offset += idx * m_stride[0];
        return offset;
    }

    PARAM_ELEM_VISITOR_COMMON_DEV
#endif
};

//! specialization for CONTIG_FULL
template <int ndim, typename ctype>
class ParamElemVisitor<ndim, ctype, CONTIG_FULL> {
    ctype *__restrict m_ptr;

public:
    static const int NDIM = ndim;

    void host_init(const TensorND &rv, int grid_size, int block_size);

#if MEGDNN_CC_CUDA
    devfunc void thread_init(uint32_t) {}

    devfunc void next() {}

    devfunc int offset(uint32_t idx) { return idx; }

    PARAM_ELEM_VISITOR_COMMON_DEV
#endif
};

#undef PARAM_ELEM_VISITOR_COMMON_DEV

} // namespace cuda
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
