/**
 * \file dnn/src/x86/matrix_mul/int8/avx2_strategy_4x16x2.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/common/utils.h"
#include "src/x86/matrix_mul/int8/kernel_avx2_4x16x2.h"
#include "src/x86/matrix_mul/int8/strategy.h"
#include "src/x86/utils.h"

using namespace megdnn;
using namespace x86;
using namespace x86::matmul;

static inline void gemm_packa(dt_int16* out, const dt_int8* in, int ldin,
                              int y0, int ymax, int k0, int kmax,
                              bool transpose) {
    if (transpose) {
        matmul_avx2_4x16x2::gemm_s8s8s32_avx2_4x16x2_pack_at(out, in, ldin, y0,
                                                             ymax, k0, kmax);
    } else {
        matmul_avx2_4x16x2::gemm_s8s8s32_avx2_4x16x2_pack_an(out, in, ldin, y0,
                                                             ymax, k0, kmax);
    }
}
static inline void gemm_packb(dt_int8* out, const dt_int8* in, int ldin, int x0,
                              int xmax, int k0, int kmax, bool transpose) {
    if (transpose) {
        matmul_avx2_4x16x2::gemm_s8s8s32_avx2_4x16x2_pack_bt(out, in, ldin, x0,
                                                             xmax, k0, kmax);
    } else {
        matmul_avx2_4x16x2::gemm_s8s8s32_avx2_4x16x2_pack_bn(out, in, ldin, x0,
                                                             xmax, k0, kmax);
    }
}
template <typename CType>
static inline void gemm_kern(const dt_int16* pack_a_ptr,
                             const dt_int8* pack_b_ptr, size_t m, size_t n,
                             size_t k, CType* c_ptr, size_t ldc,
                             bool is_first_k) {
    constexpr size_t m_tile = 4;
    constexpr size_t n_tile = 16;
    constexpr size_t k_tile = 2;
    const size_t roundup_k = round_up(k, k_tile);

    const size_t m_end = m / m_tile * m_tile;
    const size_t n_end = n / n_tile * n_tile;
    const size_t m_remain = m - m_end;
    const size_t n_remain = n - n_end;

    for (size_t m_offset = 0; m_offset < m_end; m_offset += m_tile) {
        auto iter_a_ptr = pack_a_ptr + m_offset * roundup_k;
        for (size_t n_offset = 0; n_offset < n_end; n_offset += n_tile) {
            auto iter_b_ptr = pack_b_ptr + n_offset * roundup_k;
            auto iter_c_ptr = c_ptr + m_offset * n + n_offset;
            matmul_avx2_4x16x2::kern_gemm_s8s8s32_avx2_4x16x2(
                    iter_a_ptr, iter_b_ptr, iter_c_ptr, ldc, k);
        }
        if (n_remain > 0) {
            auto iter_b_ptr = pack_b_ptr + n_end * roundup_k;
            auto iter_c_ptr = c_ptr + m_offset * n + n_end;
            if (n_remain <= 8) {
                matmul_avx2_4x16x2::kern_gemm_s8s8s32_avx2_4x16x2_n8_remain_n(
                        iter_a_ptr, iter_b_ptr, iter_c_ptr, ldc, k, n_remain);
            } else {
                matmul_avx2_4x16x2::kern_gemm_s8s8s32_avx2_4x16x2_remain_n(
                        iter_a_ptr, iter_b_ptr, iter_c_ptr, ldc, k, n_remain);
            }
        }
    }
    if (m_remain > 0) {
        auto iter_a_ptr = pack_a_ptr + m_end * roundup_k;
        for (size_t n_offset = 0; n_offset < n_end; n_offset += n_tile) {
            auto iter_b_ptr = pack_b_ptr + n_offset * roundup_k;
            auto iter_c_ptr = c_ptr + m_end * n + n_offset;
            matmul_avx2_4x16x2::kern_gemm_s8s8s32_avx2_4x16x2_remain_m(
                    iter_a_ptr, iter_b_ptr, iter_c_ptr, ldc, k, m_remain);
        }
        if (n_remain > 0) {
            auto iter_b_ptr = pack_b_ptr + n_end * roundup_k;
            auto iter_c_ptr = c_ptr + m_end * n + n_end;
            if (n_remain <= 8) {
                matmul_avx2_4x16x2::kern_gemm_s8s8s32_avx2_4x16x2_n8_remain_m_n(
                        iter_a_ptr, iter_b_ptr, iter_c_ptr, ldc, k, m_remain,
                        n_remain);
            } else {
                matmul_avx2_4x16x2::kern_gemm_s8s8s32_avx2_4x16x2_remain_m_n(
                        iter_a_ptr, iter_b_ptr, iter_c_ptr, ldc, k, m_remain,
                        n_remain);
            }
        }
    }
}

MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_avx2_s8s8s32_4x16x2);
void gemm_avx2_s8s8s32_4x16x2::pack_A(dt_int16* out, const dt_int8* in,
                                      int ldin, int y0, int ymax, int k0,
                                      int kmax, bool transpose) const {
    gemm_packa(out, in, ldin, y0, ymax, k0, kmax, transpose);
}

void gemm_avx2_s8s8s32_4x16x2::pack_B(dt_int8* out, const dt_int8* in, int ldin,
                                      int x0, int xmax, int k0, int kmax,
                                      bool transpose) const {
    gemm_packb(out, in, ldin, x0, xmax, k0, kmax, transpose);
}

void gemm_avx2_s8s8s32_4x16x2::kern(const dt_int16* pack_a_ptr,
                                    const dt_int8* pack_b_ptr, size_t m,
                                    size_t n, size_t k, dt_int32* c_ptr,
                                    size_t ldc, bool is_first_k,
                                    const dt_int32*, dt_int32*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                          ((A_dtype.enumv() == DTypeEnum::Int8 &&
                            C_dtype.enumv() == DTypeEnum::Int32) ||
                           (A_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                            C_dtype.enumv() == DTypeEnum::QuantizedS32)),
                  "A: %s B: %s C: %s", A_dtype.name(), B_dtype.name(),
                  C_dtype.name());
    megdnn_assert(is_first_k == true);
    gemm_kern(pack_a_ptr, pack_b_ptr, m, n, k, c_ptr, ldc, is_first_k);
}

MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_avx2_s8s8s16_4x16x2);
void gemm_avx2_s8s8s16_4x16x2::pack_A(dt_int16* out, const dt_int8* in,
                                      int ldin, int y0, int ymax, int k0,
                                      int kmax, bool transpose) const {
    gemm_packa(out, in, ldin, y0, ymax, k0, kmax, transpose);
}

void gemm_avx2_s8s8s16_4x16x2::pack_B(dt_int8* out, const dt_int8* in, int ldin,
                                      int x0, int xmax, int k0, int kmax,
                                      bool transpose) const {
    gemm_packb(out, in, ldin, x0, xmax, k0, kmax, transpose);
}

void gemm_avx2_s8s8s16_4x16x2::kern(const dt_int16* pack_a_ptr,
                                    const dt_int8* pack_b_ptr, size_t m,
                                    size_t n, size_t k, dt_int16* c_ptr,
                                    size_t ldc, bool is_first_k,
                                    const dt_int32*, dt_int32*) const {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv() &&
                          ((A_dtype.enumv() == DTypeEnum::Int8 &&
                            C_dtype.enumv() == DTypeEnum::Int16) ||
                           (A_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                            C_dtype.enumv() == DTypeEnum::QuantizedS16)),
                  "A: %s B: %s C: %s", A_dtype.name(), B_dtype.name(),
                  C_dtype.name());
    megdnn_assert(is_first_k == true);
    gemm_kern(pack_a_ptr, pack_b_ptr, m, n, k, c_ptr, ldc, is_first_k);
}
// vim: syntax=cpp.doxygen
