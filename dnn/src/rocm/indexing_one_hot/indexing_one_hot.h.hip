/**
 * \file src/rocm/indexing_one_hot/indexing_one_hot.h.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */

#pragma once

#include "src/rocm/error_info.h.hip"
#include "src/rocm/int_fastdiv.h.hip"

namespace megdnn {
namespace rocm {
namespace indexing_one_hot {

struct KernParam {
    //! stride[axis], also prod(shape[axis+1:ndim])
    Uint32Fastdiv shape_lo;
    //! stride[axis-1]
    uint32_t stride_hi;

    //! max value that user provide index array can give
    uint32_t max_mid_index;
    void* error_tracker;
    AsyncErrorInfo* error_info;

    template <typename idx_type>
    __device__ uint32_t get_idx(uint32_t offset, const idx_type* idx) const {
        uint32_t idx0, idx1, idx2;
        idx0 = offset / shape_lo;
        idx2 = offset - idx0 * shape_lo.divisor();
        idx1 = idx[offset];
        if (idx1 >= max_mid_index) {
            set_async_error_info(error_info, error_tracker,
                                 "invalid IndexingOneHot: "
                                 "offset=%d idx0=%d indexer=%d idx2=%d",
                                 offset, idx0, idx1, idx2);
            idx1 = 0;
        }
        return idx0 * stride_hi + idx1 * shape_lo.divisor() + idx2;
    }
};

template <typename data_type, typename idx_type>
struct OpGet {
    const data_type* m_src;
    const idx_type* m_idx;
    data_type* m_dst;
    KernParam m_param;

    __device__ void operator()(uint32_t offset) {
        m_dst[offset] = m_src[m_param.get_idx(offset, m_idx)];
    }
};

template <typename data_type, typename idx_type>
struct OpSet {
    data_type* m_data;
    const idx_type* m_idx;
    const data_type* m_sub;
    KernParam m_param;

    __device__ void operator()(uint32_t offset) {
        m_data[m_param.get_idx(offset, m_idx)] = m_sub[offset];
    }
};

}  // namespace indexing_one_hot
}  // namespace rocm
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
