/**
 * \file dnn/src/fallback/matrix_mul/gemm_common.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * ---------------------------------------------------------------
 *  Part of the following code in this file refs to ComputeLibrary
 *
 *  MIT License
 *
 *  Copyright (c) 2017-2020 ARM Software
 * ---------------------------------------------------------------
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include "src/common/utils.h"

namespace megdnn {
namespace matmul {

/**
 * \brief Generic pack function.
 *
 * Assuming the untransposed case, this works by first reading <block_w>
 * consecutive values from the first input row.  This same number of values
 * are then read from the next <block_h-1> rows.  Now return to the first
 * input row and repeat.
 *
 * Need to cope with the work requested in either dimension not actually
 * being a multiple of the block sizes.
 */
template <size_t block_h, size_t block_w, bool transposed, typename TOut,
          typename TIn>
void pack(TOut* out, const TIn* const in, const size_t stride,
          const size_t h_start, const size_t h_end, const size_t w_start,
          const size_t w_end) {
    const size_t n_whole_h_blocks = (h_end - h_start) / block_h;
    const size_t h_remainders = (h_end - h_start) % block_h;
    const size_t n_h_blocks = n_whole_h_blocks + (h_remainders ? 1 : 0);

    const size_t n_whole_w_blocks = (w_end - w_start) / block_w;
    const size_t w_remainders = (w_end - w_start) % block_w;
    const size_t n_w_blocks = n_whole_w_blocks + (w_remainders ? 1 : 0);

    //! "h" loop: advance down the rows of the source block_h rows at a time.
    //! Set up fill_rows to show the number rows to copy from, and blank_rows
    //! for the number of blank rows to add.
    for (size_t h_block = 0; h_block < n_h_blocks; h_block++) {
        size_t fill_rows =
                (h_block < n_whole_h_blocks) ? block_h : h_remainders;
        size_t blank_rows = block_h - fill_rows;

        size_t h_base = h_start + (h_block * block_h);

        //! So now advance along this block of rows, block_w columns at a
        //! time.
        for (size_t w_block = 0; w_block < n_w_blocks; w_block++) {
            size_t fill_cols =
                    (w_block < n_whole_w_blocks) ? block_w : w_remainders;
            size_t blank_cols = block_w - fill_cols;

            size_t w_base = w_start + (w_block * block_w);

            for (size_t row = 0; row < fill_rows; row++) {
                for (size_t col = 0; col < fill_cols; col++) {
                    //! In-range copy.  If it's transposed, we reverse the
                    //! sense of rows and columns here.
                    if (transposed) {
                        *out++ = static_cast<TOut>(
                                in[(w_base + col) * stride + h_base + row]);
                    } else {
                        *out++ = static_cast<TOut>(
                                in[(h_base + row) * stride + w_base + col]);
                    }
                }
                //! "col" tail - row is in range but column is out of range.
                for (size_t col = 0; col < blank_cols; col++) {
                    *out++ = static_cast<TOut>(0);
                }
            }
            //! "row" tail - row is out of range so fill with zeros always.
            for (size_t row = 0; row < blank_rows; row++) {
                for (size_t col = 0; col < (fill_cols + blank_cols); col++) {
                    *out++ = static_cast<TOut>(0);
                }
            }
        }
    }
}

/**
 * This is illustrated in this picture:
 *
 *                             B_interleave
 *                        <----------------->
 *                        +-----------------+ ^
 *                        |        B        | | unroll_k
 *                        +-----------------+ v
 *                 ^ +--+ +-----------------+
 *                 | |  | |                 |
 *   A_interleave  | |A | |      Result     |
 *                 | |  | |                 |
 *                 v +--+ +-----------------+
 *                   <-->
 *                 unroll_k
 *
 *  The kern function calc  block_m * block_n result, each subblock calc
 *  kernel_h * kernel_w result.
 */

template <typename Strategy, typename Tout, typename Tin>
void gemm_kern(const Tin* packA, const Tin* packB, size_t M, size_t N, size_t K,
               Tout* C, size_t LDC, bool is_first_k, const Strategy& strategy) {
    size_t block_m = strategy.block_m;
    size_t block_n = strategy.block_n;
    size_t block_k = strategy.block_k;
    size_t kernel_h = strategy.KERNEL_H;
    size_t kernel_w = strategy.KERNEL_W;
    size_t unroll_k = strategy.UNROLL_K;
    megdnn_assert(block_m % kernel_h == 0 && block_n % kernel_w == 0 &&
                  block_k % unroll_k == 0);
    size_t ablocks = block_m / kernel_h;
    size_t bblocks = block_n / kernel_w;
    size_t kblocks = (K + unroll_k - 1) / unroll_k;

    for (size_t a_bidx = 0; a_bidx < ablocks; a_bidx++) {
        for (size_t b_bidx = 0; b_bidx < bblocks; b_bidx++) {
            for (size_t a_idx = 0; a_idx < kernel_h; a_idx++) {
                for (size_t b_idx = 0; b_idx < kernel_w; b_idx++) {
                    size_t r = a_bidx * kernel_h + a_idx;
                    size_t c = b_bidx * kernel_w + b_idx;

                    if (r < M && c < N) {
                        if (is_first_k) {
                            C[r * LDC + c] = 0;
                        }
                        for (size_t bk = 0; bk < kblocks; bk++) {
                            /**
                             * The index of packA ((a_bidx, bk, a_idx, k),
                             * (kernel_h * block_k, kernel_h * unroll_k,
                             * unroll_k, 1))
                             * The index of packB ((b_bidx, bk, a_idx, k),
                             * (kernel_w * block_k, kernel_w * unroll_k,
                             * unroll_k, 1))
                             */
                            for (size_t k = 0; k < unroll_k; k++) {
                                C[r * LDC + c] +=
                                        packA[a_bidx * kernel_h * block_k +
                                              bk * kernel_h * unroll_k +
                                              a_idx * unroll_k + k] *
                                        packB[b_bidx * kernel_w * block_k +
                                              bk * kernel_w * unroll_k +
                                              b_idx * unroll_k + k];
                            }
                        }
                    }
                }
            }
        }
    }
}
#define MEGDNN_REG_GEMM_STRATEGY_WITH_PACK_A_TYPE(                             \
        _stype, _pack_a_type, _dtype, _ctype, _L1_block_m, _L1_block_n,        \
        _L1_block_k, _A_transpose, _B_transpose, _strategy_cls_name)           \
    class _strategy_cls_name {                                                 \
    public:                                                                    \
        using stype = _stype;                                                  \
        using pack_a_type = _pack_a_type;                                      \
        using dst_type = _dtype;                                               \
        using compute_type = _ctype;                                           \
        constexpr static size_t A_INTERLEAVE = _L1_block_m;                    \
        constexpr static size_t A_BLOCK = _L1_block_k;                         \
        constexpr static bool A_TRANSPOSE = _A_transpose;                      \
        constexpr static size_t B_INTERLEAVE = _L1_block_n;                    \
        constexpr static size_t B_BLOCK = _L1_block_k;                         \
        constexpr static bool B_TRANSPOSE = _B_transpose;                      \
        constexpr static size_t KERNEL_H = _L1_block_m;                        \
        constexpr static size_t KERNEL_W = _L1_block_n;                        \
        constexpr static size_t UNROLL_K = _L1_block_k;                        \
        const size_t block_m;                                                  \
        const size_t block_n;                                                  \
        const size_t block_k;                                                  \
        const DType A_dtype;                                                   \
        const DType B_dtype;                                                   \
        const DType C_dtype;                                                   \
        _strategy_cls_name(size_t m, size_t n, size_t k, DType dtype_a,        \
                           DType dtype_b, DType dtype_c);                      \
        void pack_A(pack_a_type* out, const _stype* in, int ldin, int y0,      \
                    int ymax, int k0, int kmax,                                \
                    bool transpose_A = false) const;                           \
        void pack_B(_stype* out, const _stype* in, int ldin, int x0, int xmax, \
                    int k0, int kmax, bool transpose_B = false) const;         \
        void kern(const pack_a_type* packA, const _stype* packB, size_t M,     \
                  size_t N, size_t K, _dtype* C, size_t LDC, bool is_first_k,  \
                  const _ctype* bias = nullptr,                                \
                  _ctype* workspace = nullptr) const;                          \
        size_t get_workspace_size() const { return 0; }                        \
    }

#define MEGDNN_REG_GEMM_STRATEGY(_stype, _dtype, _ctype, _L1_block_m,          \
                                 _L1_block_n, _L1_block_k, _A_transpose,       \
                                 _B_transpose, _strategy_cls_name)             \
    class _strategy_cls_name {                                                 \
    public:                                                                    \
        using stype = _stype;                                                  \
        using pack_a_type = stype;                                             \
        using dst_type = _dtype;                                               \
        using compute_type = _ctype;                                           \
        constexpr static size_t A_INTERLEAVE = _L1_block_m;                    \
        constexpr static size_t A_BLOCK = _L1_block_k;                         \
        constexpr static bool A_TRANSPOSE = _A_transpose;                      \
        constexpr static size_t B_INTERLEAVE = _L1_block_n;                    \
        constexpr static size_t B_BLOCK = _L1_block_k;                         \
        constexpr static bool B_TRANSPOSE = _B_transpose;                      \
        constexpr static size_t KERNEL_H = _L1_block_m;                        \
        constexpr static size_t KERNEL_W = _L1_block_n;                        \
        constexpr static size_t UNROLL_K = _L1_block_k;                        \
        const size_t block_m;                                                  \
        const size_t block_n;                                                  \
        const size_t block_k;                                                  \
        const DType A_dtype;                                                   \
        const DType B_dtype;                                                   \
        const DType C_dtype;                                                   \
        _strategy_cls_name(size_t m, size_t n, size_t k, DType dtype_a,        \
                           DType dtype_b, DType dtype_c);                      \
        void pack_A(pack_a_type* out, const _stype* in, int ldin, int y0,      \
                    int ymax, int k0, int kmax,                                \
                    bool transpose_A = false) const;                           \
        void pack_B(_stype* out, const _stype* in, int ldin, int x0, int xmax, \
                    int k0, int kmax, bool transpose_B = false) const;         \
        void kern(const pack_a_type* packA, const _stype* packB, size_t M,     \
                  size_t N, size_t K, _dtype* C, size_t LDC, bool is_first_k,  \
                  const _ctype* bias = nullptr,                                \
                  _ctype* workspace = nullptr) const;                          \
        size_t get_workspace_size() const { return 0; }                        \
    }

#define MEGDNN_REG_GEMM_STRATEGY_WITH_WRITEBACK(                               \
        _stype, _dtype, _ctype, _L1_block_m, _L1_block_n, _L1_block_k,         \
        _A_transpose, _B_transpose, _strategy_cls_name)                        \
    class _strategy_cls_name {                                                 \
    public:                                                                    \
        using stype = _stype;                                                  \
        using pack_a_type = stype;                                             \
        using dst_type = _dtype;                                               \
        using compute_type = _ctype;                                           \
        constexpr static size_t A_INTERLEAVE = _L1_block_m;                    \
        constexpr static size_t A_BLOCK = _L1_block_k;                         \
        constexpr static bool A_TRANSPOSE = _A_transpose;                      \
        constexpr static size_t B_INTERLEAVE = _L1_block_n;                    \
        constexpr static size_t B_BLOCK = _L1_block_k;                         \
        constexpr static bool B_TRANSPOSE = _B_transpose;                      \
        constexpr static size_t KERNEL_H = _L1_block_m;                        \
        constexpr static size_t KERNEL_W = _L1_block_n;                        \
        constexpr static size_t UNROLL_K = _L1_block_k;                        \
        const size_t block_m;                                                  \
        const size_t block_n;                                                  \
        const size_t block_k;                                                  \
        const DType A_dtype;                                                   \
        const DType B_dtype;                                                   \
        const DType C_dtype;                                                   \
        _strategy_cls_name(size_t m, size_t n, size_t k, DType dtype_a,        \
                           DType dtype_b, DType dtype_c);                      \
        void pack_A(pack_a_type* out, const _stype* in, int ldin, int y0,      \
                    int ymax, int k0, int kmax,                                \
                    bool transpose_A = false) const;                           \
        void pack_B(_stype* out, const _stype* in, int ldin, int x0, int xmax, \
                    int k0, int kmax, bool transpose_B = false) const;         \
        void kern(const pack_a_type* packA, const _stype* packB, size_t M,     \
                  size_t N, size_t K, _dtype* C, size_t LDC, bool is_first_k,  \
                  const _ctype* bias = nullptr,                                \
                  _ctype* workspace = nullptr) const;                          \
        /**                                                                    \
         * \brief get the workspace which needed for inner output storage.     \
         *                                                                     \
         * \warning default is 0, otherwise _L1_block_m * _L1_block_n *        \
         * sizeof(ctype)                                                       \
         **/                                                                   \
        size_t get_workspace_size() const;                                     \
    }

#define MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(_cls, _super)                 \
    class _cls : public _super {                                          \
    public:                                                               \
        using _super::_super;                                             \
        using stype = _super::stype;                                      \
        using pack_a_type = stype;                                        \
        using dst_type = _super::dst_type;                                \
        using compute_type = _super::compute_type;                        \
        void kern(const pack_a_type* packA, const stype* packB, size_t M, \
                  size_t N, size_t K, dst_type* C, size_t LDC,            \
                  bool is_first_k, const compute_type* bias = nullptr,    \
                  compute_type* workspace = nullptr) const;               \
    }

#define MEGDNN_REG_GEMM_STRATEGY_IMPL(_strategy_cls_name)                      \
    constexpr size_t _strategy_cls_name::A_INTERLEAVE;                         \
    constexpr size_t _strategy_cls_name::A_BLOCK;                              \
    constexpr bool _strategy_cls_name::A_TRANSPOSE;                            \
    constexpr size_t _strategy_cls_name::B_INTERLEAVE;                         \
    constexpr size_t _strategy_cls_name::B_BLOCK;                              \
    constexpr bool _strategy_cls_name::B_TRANSPOSE;                            \
    constexpr size_t _strategy_cls_name::KERNEL_H;                             \
    constexpr size_t _strategy_cls_name::KERNEL_W;                             \
    constexpr size_t _strategy_cls_name::UNROLL_K;                             \
    _strategy_cls_name::_strategy_cls_name(size_t m, size_t n, size_t k,       \
                                           DType dtype_a, DType dtype_b,       \
                                           DType dtype_c)                      \
            : block_m(round_up(m, KERNEL_H)),                                  \
              block_n(round_up(n, KERNEL_W)),                                  \
              block_k(round_up(k, UNROLL_K)),                                  \
              A_dtype(dtype_a),                                                \
              B_dtype(dtype_b),                                                \
              C_dtype(dtype_c) {                                               \
        megdnn_assert(block_m % KERNEL_H == 0 && block_n % KERNEL_W == 0 &&    \
                              block_k % UNROLL_K == 0,                         \
                      "L2 blocking size(%zu, %zu, %zu) should be multiply of " \
                      "L1 blocking(%zu, %zu, %zu)",                            \
                      block_m, block_n, block_k, KERNEL_H, KERNEL_W,           \
                      UNROLL_K);                                               \
    }

#define MEGDNN_REG_GEMM_STRATEGY_NOPACK(                                       \
        _stype, _dtype, _ctype, _L1_block_m, _L1_block_n, _L1_block_k,         \
        _A_transpose, _B_transpose, _strategy_cls_name)                        \
    class _strategy_cls_name {                                                 \
    public:                                                                    \
        using stype = _stype;                                                  \
        using dst_type = _dtype;                                               \
        using compute_type = _ctype;                                           \
        const DType A_dtype;                                                   \
        const DType B_dtype;                                                   \
        const DType C_dtype;                                                   \
        _strategy_cls_name(DType dtype_a, DType dtype_b, DType dtype_c);       \
        void kern(const _stype* A, size_t LDA, const _stype* B, size_t LDB,    \
                  _dtype* C, size_t LDC, size_t M, size_t K, size_t N,         \
                  const compute_type* bias, void* workspace, bool transpose_A, \
                  bool transpose_B) const;                                     \
        size_t get_workspace_size() const { return 0; }                        \
    }

#define MEGDNN_REG_GEMM_STRATEGY_IMPL_NOPACK(_strategy_cls_name)         \
    _strategy_cls_name::_strategy_cls_name(DType dtype_a, DType dtype_b, \
                                           DType dtype_c)                \
            : A_dtype(dtype_a), B_dtype(dtype_b), C_dtype(dtype_c) {}

#define MEGDNN_OVERRIDE_MATMUL_DESC(_m, _n, _k, _packa_type_size, _data_type, \
                                    _format)                                  \
    MatmulDescription matmul_description() const override {                   \
        MatmulDescription mdesc;                                              \
        mdesc.packmode = packmode();                                          \
        mdesc.innerblocksize = {_m, _n, _k};                                  \
        mdesc.packa_type_size = _packa_type_size;                             \
        mdesc.algo_type = {_data_type, Param::Format::_format};               \
        return mdesc;                                                         \
    }

#define MEGDNN_REG_GEMM_FUNC_FOR_IM2COL()                             \
    WorkspaceBundle get_bundle(const KernSizeParam&) const override;  \
    kern_naked_t get_kern_naked(const KernSizeParam&) const override; \
    void pack_A(const KernParam& kern_param, void* out, size_t index, \
                size_t stride) const override;                        \
    void pack_B(const KernParam& kern_param, void* out, size_t x0,    \
                size_t xmax) const override;                          \
    InnerBlockSize get_inner_block_size() const override;             \
    MatmulDescription matmul_description() const override;

#define MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL_DETAIL(                          \
        _algo_name, _midout_name, _mid_index, _strategy, _i_type, _c_type,    \
        _packa_type, _support_data_type, _format)                             \
                                                                              \
    MatrixMulImpl::kern_naked_t MatrixMulImpl::_algo_name::get_kern_naked(    \
            const KernSizeParam&) const {                                     \
        auto kern = [](const MatrixMulImpl::KernParam& kern_param,            \
                       const void* packed_a, const void* packed_b) {          \
            MIDOUT_BEGIN(_midout_name, midout_iv(_mid_index),                 \
                         midout_iv("get_kern_naked"_hash)) {                  \
                auto M = kern_param.M, N = kern_param.N, K = kern_param.K;    \
                auto trA = kern_param.trA, trB = kern_param.trB;              \
                auto LDC = kern_param.LDC;                                    \
                auto A_type = kern_param.A_type, B_type = kern_param.B_type,  \
                     C_type = kern_param.C_type;                              \
                auto Cptr = kern_param.C<_c_type>();                          \
                                                                              \
                _strategy strategy(M, N, K, A_type, B_type, C_type);          \
                megdnn::matmul::GemmInterleaved<_strategy>(M, N, K, trA, trB, \
                                                           strategy)          \
                        .execute_naked(Cptr, LDC, packed_a, packed_b);        \
            }                                                                 \
            MIDOUT_END();                                                     \
        };                                                                    \
        return kern;                                                          \
    }                                                                         \
                                                                              \
    void MatrixMulImpl::_algo_name::pack_A(const KernParam& kern_param,       \
                                           void* out, size_t index,           \
                                           size_t stride) const {             \
        MIDOUT_BEGIN(_midout_name, midout_iv(_mid_index),                     \
                     midout_iv("pack_A"_hash)) {                              \
            auto M = kern_param.M, N = kern_param.N, K = kern_param.K;        \
            auto A_type = kern_param.A_type, B_type = kern_param.B_type,      \
                 C_type = kern_param.C_type;                                  \
                                                                              \
            auto trA = kern_param.trA, trB = kern_param.trB;                  \
            auto LDA = kern_param.LDA;                                        \
            const auto Aptr = kern_param.A<_i_type>();                        \
            _strategy strategy(M, N, K, A_type, B_type, C_type);              \
            size_t start_index = index * stride;                              \
            size_t end_index = start_index + stride;                          \
            end_index = std::min(end_index, M);                               \
            megdnn::matmul::GemmInterleaved<_strategy>(M, N, K, trA, trB,     \
                                                       strategy)              \
                    .pack_A(reinterpret_cast<_packa_type*>(out), Aptr, LDA,   \
                            start_index, end_index);                          \
        }                                                                     \
        MIDOUT_END();                                                         \
    }                                                                         \
                                                                              \
    void MatrixMulImpl::_algo_name::pack_B(const KernParam& kern_param,       \
                                           void* out, const size_t x0,        \
                                           size_t xmax) const {               \
        MIDOUT_BEGIN(_midout_name, midout_iv(_mid_index),                     \
                     midout_iv("pack_B"_hash)) {                              \
            auto M = kern_param.M, N = kern_param.N, K = kern_param.K;        \
            auto A_type = kern_param.A_type, B_type = kern_param.B_type,      \
                 C_type = kern_param.C_type;                                  \
                                                                              \
            auto trA = kern_param.trA, trB = kern_param.trB;                  \
            auto LDB = kern_param.LDB;                                        \
            const auto Bptr = kern_param.B<_i_type>();                        \
            _strategy strategy(M, N, K, A_type, B_type, C_type);              \
            megdnn::matmul::GemmInterleaved<_strategy>(M, N, K, trA, trB,     \
                                                       strategy)              \
                    .pack_B(reinterpret_cast<_i_type*>(out), Bptr, LDB, x0,   \
                            xmax);                                            \
        }                                                                     \
        MIDOUT_END();                                                         \
    }                                                                         \
                                                                              \
    WorkspaceBundle MatrixMulImpl::_algo_name::get_bundle(                    \
            const KernSizeParam& kern_size_param) const {                     \
        MIDOUT_BEGIN(_midout_name, midout_iv(_mid_index),                     \
                     midout_iv("get_bundle"_hash)) {                          \
            auto M = kern_size_param.M, N = kern_size_param.N,                \
                 K = kern_size_param.K;                                       \
            auto trA = kern_size_param.trA, trB = kern_size_param.trB;        \
            auto A_type = kern_size_param.A_type,                             \
                 B_type = kern_size_param.B_type,                             \
                 C_type = kern_size_param.C_type;                             \
            _strategy strategy(M, N, K, A_type, B_type, C_type);              \
            return megdnn::matmul::GemmInterleaved<_strategy>(M, N, K, trA,   \
                                                              trB, strategy)  \
                    .get_bundle();                                            \
        }                                                                     \
        MIDOUT_END();                                                         \
    }                                                                         \
                                                                              \
    MatrixMulImpl::_algo_name::InnerBlockSize                                 \
    MatrixMulImpl::_algo_name::get_inner_block_size() const {                 \
        return {_strategy::KERNEL_H, _strategy::KERNEL_W,                     \
                _strategy::UNROLL_K};                                         \
    }                                                                         \
                                                                              \
    MatrixMulImpl::_algo_name::MatmulDescription                              \
    MatrixMulImpl::_algo_name::matmul_description() const {                   \
        MatmulDescription mdesc;                                              \
        mdesc.packmode = PackMode();                                          \
        mdesc.innerblocksize = {_strategy::KERNEL_H, _strategy::KERNEL_W,     \
                                _strategy::UNROLL_K};                         \
        mdesc.packa_type_size = sizeof(_packa_type);                          \
        mdesc.algo_type = {_support_data_type, Param::Format::_format};       \
        return mdesc;                                                         \
    }

#define MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(                                  \
        _algo_name, _midout_name, _mid_index, _strategy, _i_type, _c_type,     \
        _support_data_type, _format)                                           \
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL_DETAIL(                               \
            _algo_name, _midout_name, _mid_index, _strategy, _i_type, _c_type, \
            _i_type, _support_data_type, _format)
}  // namespace matmul
}  // namespace megdnn

// vim: syntax=cpp.doxygen
