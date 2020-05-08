/**
 * \file dnn/src/common/relayout_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
namespace relayout {

static inline bool is_contig(const TensorLayout& layout) {
    return layout.ndim == 1 && layout.stride[0] == 1;
}

//! [b][m][n][c] to [b][n][m][c]
struct TransposeParam {
    size_t batch, m, n, c;
};

/**
 * \brief whether the relayout can be formulated as TransposeParam
 *
 * Note that \p src and \p dst should have been processed by
 * RelayoutForward::check_layout_and_canonize
 */
bool is_transpose(const TensorLayout& src, const TensorLayout& dst,
                  TransposeParam& p);

namespace transpose_fallback {

#if MEGDNN_X86
constexpr size_t BLOCK_LINE_SIZE_BYTES = 64;
#elif MEGDNN_AARCH64 || MEGDNN_ARMV7 
constexpr size_t BLOCK_LINE_SIZE_BYTES = 32;
#else
#error "unknown megdnn arch"
#endif

/**
 * \brief transpose traits
 * \tparam T element type
 */
template <typename T>
struct transpose_traits {
    static constexpr size_t block_size = BLOCK_LINE_SIZE_BYTES / sizeof(T);
};

template <typename T>
void transpose_block_fallback(const T* src, T* dst, const size_t src_stride,
                              const size_t dst_stride, size_t block_h,
                              size_t block_w) {
    constexpr size_t block_size = transpose_traits<T>::block_size;
    T block[block_size][block_size];

    for (size_t i = 0; i < block_h; ++i) {
        auto src_ptr = src + i * src_stride;
        for (size_t j = 0; j < block_w; ++j) {
            block[j][i] = src_ptr[j];
        }
    }
    for (size_t i = 0; i < block_w; ++i) {
        auto dst_ptr = dst + i * dst_stride;
        for (size_t j = 0; j < block_h; ++j) {
            dst_ptr[j] = block[i][j];
        }
    }
}

template <typename T>
void transpose_block(const T* src, T* dst, const size_t src_stride,
                     const size_t dst_stride, size_t block_h, size_t block_w) {
    transpose_block_fallback(src, dst, src_stride, dst_stride, block_h,
                             block_w);
}

/*!
 * \brief transpose a single block whose size is transpose_traits<T>::block_size
 *
 * This function and transpose_traits can be specialized to implement optimized
 * block transpose
 */
template <typename T>
void transpose_block(const T* src, T* dst, const size_t src_stride,
                     const size_t dst_stride) {
    constexpr size_t block_size = transpose_traits<T>::block_size;
    transpose_block_fallback(src, dst, src_stride, dst_stride, block_size,
                             block_size);
}

/*!
 * \brief transpose contiguous (batch, m, n) to (batch, n, m)
 */
template <typename T>
void transpose(size_t batch, size_t m, size_t n, T* src, T* dst) {
    auto batch_src = src;
    auto batch_dst = dst;
    constexpr size_t B = transpose_traits<T>::block_size;

    auto work_block = [m, n, &batch_src, &batch_dst](
                              const size_t i, const size_t j, const size_t h,
                              const size_t w) {

        auto src = batch_src + i * n + j, dst = batch_dst + j * m + i;
        if (h == B && w == B) {
            transpose_block(src, dst, n, m);
        } else {
            transpose_block(src, dst, n, m, h, w);
        }
    };
    auto work_row = [&work_block, n](size_t i, size_t h) {
        size_t j = 0;
        for (; j + B <= n; j += B) {
            work_block(i, j, h, B);
        }
        if (j < n) {
            work_block(i, j, h, n - j);
        }
    };

    for (size_t b = 0; b < batch; ++b) {
        size_t i = 0;
        for (; i + B <= m; i += B) {
            work_row(i, B);
        }
        if (i < m) {
            work_row(i, m - i);
        }
        batch_src += m * n;
        batch_dst += m * n;
    }
}
}  // namespace transpose_fallback

}  // namespace relayout
}  // namespace megdnn

// vim: syntax=cpp.doxygen
