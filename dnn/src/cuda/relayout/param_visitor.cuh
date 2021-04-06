/**
 * \file dnn/src/cuda/relayout/param_visitor.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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

template <int ndim>
class ParamElemVisitor<ndim, dt_quint4, CONTIG_OTHER> {
    using Storage = uint8_t;

protected:
    Storage* __restrict m_ptr;
    int m_stride[ndim];
    int m_shape[ndim];
    bool m_is_contiguous;
    bool m_is_physical_contiguous;

    //! m_shape_highdim[i] = original_shape[i + 1]
#ifdef _MSC_VER
    Uint32Fastdiv m_shape_highdim[ndim > 1 ? ndim - 1 : 1];
    Uint32Fastdiv m_align_shape_highdim[ndim > 1 ? ndim - 1 : 1];
#else
    Uint32Fastdiv m_shape_highdim[ndim];
    Uint32Fastdiv m_align_shape_highdim[ndim];
#endif

public:
    static const Storage kMask = 0xf;
    static const Storage kBits = 4;
    static const int NDIM = ndim;
    void host_init(const TensorND& rv, int grid_size, int block_size);

#if MEGDNN_CC_CUDA
    devfunc void thread_init(uint32_t) {}

    devfunc void next() {}

    devfunc void get_shape_from_access(uint32_t access_idx,
                                       int (&shape_idx)[ndim]) {
#pragma unroll
        for (int i = ndim - 1; i >= 1; --i) {
            Uint32Fastdiv& align_shp = m_align_shape_highdim[i - 1];
            uint32_t access_idx_div = access_idx / align_shp;
            shape_idx[i] = access_idx - access_idx_div * align_shp.divisor();
            access_idx = access_idx_div;
        }
        shape_idx[0] = access_idx;
    }

    devfunc int offset(uint32_t idx) {
        int offset = 0;
#pragma unroll
        for (int i = ndim - 1; i >= 1; --i) {
            Uint32Fastdiv& shp = m_shape_highdim[i - 1];
            uint32_t idx_div = idx / shp;
            offset += (idx - idx_div * shp.divisor()) * m_stride[i];
            idx = idx_div;
        }
        offset += idx * m_stride[0];
        return offset;
    }

    devfunc int offset_from_access(uint32_t access_idx) {
        int offset = 0;
        if (m_is_contiguous) {
            offset = access_idx;
        } else {
            int shape_idx[ndim];
            get_shape_from_access(access_idx, shape_idx);
    #pragma unroll
            for (int i = ndim - 1; i >= 0; --i) {
                offset += shape_idx[i] * m_stride[i];
            }
        }
        return offset;
    }

    devfunc int idx(uint32_t access_idx) {
        int idx = 0;
        if (m_is_physical_contiguous) {
            idx = access_idx;
        } else {
            int shape_idx[ndim];
            bool valid = true;
            get_shape_from_access(access_idx, shape_idx);
#pragma unroll
            for (int i = 0; i < ndim; ++i) {
                valid &= (shape_idx[i] < m_shape[i]);
            }
#pragma unroll
            for (int i = 0; i < ndim - 1; ++i) {
                idx = (idx + shape_idx[i]) * m_shape[i + 1];
            }
            idx = valid ? idx + shape_idx[ndim - 1] : -1;
        }
        return idx;
    }

    devfunc Storage* ptr() { return m_ptr; }

    devfunc Storage at(uint32_t idx) {
        int offset_ = offset(idx);
        int vec_idx = offset_ >> 1;
        int lane_idx = offset_ & 0x1;

        Storage item = Storage(unpack_integer_4bits<false>(
                *(Storage*)&m_ptr[vec_idx], lane_idx * 4));

        return item;
    }

    using rwtype = typename elemwise_intl::VectTypeTrait<dt_quint4>::vect_type;

    devfunc rwtype make_vector(Storage x, Storage y) {
        return elemwise_intl::VectTypeTrait<dt_quint4>::make_vector(x, y);
    }
#endif
};

} // namespace cuda
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
