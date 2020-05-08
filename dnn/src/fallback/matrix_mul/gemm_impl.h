/**
 * \file dnn/src/fallback/matrix_mul/gemm_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <stdio.h>
#include <cassert>
#include "src/common/utils.h"
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace matmul {

/**
 * \brief implementation of the GemmCommon abstract class, for normal gemm
 */
template <typename Strategy, bool with_pack = true>
class GemmInterleaved;

template <typename Strategy>
class GemmInterleaved<Strategy, true> {
    using compute_type = typename Strategy::compute_type;
    using stype = typename Strategy::stype;
    using pack_a_type = typename Strategy::pack_a_type;
    using dtype = typename Strategy::dst_type;

    const size_t m_M;
    const size_t m_N;
    const size_t m_K;

    const bool m_transpose_A;
    const bool m_transpose_B;

    Strategy m_strategy;
    static constexpr size_t CACHELINE_SIZE = 64;
    //! align must be 2^n, default is 16
    const size_t m_align_size = 16;

    size_t get_a_workspace_size() const {
        size_t M = round_up(m_strategy.block_m, m_strategy.KERNEL_H);
        size_t K = round_up(m_strategy.block_k, m_strategy.UNROLL_K);
        return round_up(sizeof(pack_a_type) * M * K, CACHELINE_SIZE) +
               m_align_size;
    }

    size_t get_b_workspace_size() const {
        size_t N = round_up(m_strategy.block_n, m_strategy.KERNEL_W);
        size_t K = round_up(m_strategy.block_k, m_strategy.UNROLL_K);
        return round_up(sizeof(stype) * N * K, CACHELINE_SIZE) + m_align_size;
    }

    //! temporary storage for output, post process such as add bias or relu will
    //! be processed
    size_t get_c_workspace_size() const {
        size_t ret = m_strategy.get_workspace_size();
        if (ret == 0) {
            return ret;
        }
        ret = round_up(ret, CACHELINE_SIZE);
        return ret;
    }

public:
    size_t get_workspace_size() const {
        return get_bundle().total_size_in_bytes();
    }

    WorkspaceBundle get_bundle() const {
        return {nullptr,
                {get_a_workspace_size(), get_b_workspace_size(),
                 get_c_workspace_size()}};
    }

    GemmInterleaved(const size_t M, const size_t N, const size_t K,
                    const bool trA, const bool trB, const Strategy& strategy,
                    size_t align_size = 16)
            : m_M(M),
              m_N(N),
              m_K(K),
              m_transpose_A(trA),
              m_transpose_B(trB),
              m_strategy(strategy),
              m_align_size(align_size) {}

    // Actually execute the GEMM.
    void execute(const stype* A, const size_t LDA, const stype* B,
                 const size_t LDB, dtype* C, const size_t LDC, void* workspace,
                 const compute_type* bias = nullptr) const {
        megdnn_assert(workspace);
        int8_t* workspace_bytes = reinterpret_cast<int8_t*>(workspace);
        intptr_t workspace_int = reinterpret_cast<intptr_t>(workspace_bytes);
        size_t diff = 0;

        //! get the diff to align to m_align_size
        if (workspace_int & (m_align_size - 1)) {
            diff = m_align_size - (workspace_int & (m_align_size - 1));
        }

        pack_a_type* a_panel =
                reinterpret_cast<pack_a_type*>(workspace_bytes + diff);
        stype* b_panel = reinterpret_cast<stype*>(
                workspace_bytes + get_a_workspace_size() + diff);

        compute_type* c_panel = reinterpret_cast<compute_type*>(
                workspace_bytes + get_a_workspace_size() +
                get_b_workspace_size() + diff);

        for (size_t k = 0; k < m_K; k += m_strategy.block_k) {
            size_t kmax = std::min(k + m_strategy.block_k, m_K);
            for (size_t m = 0; m < m_M; m += m_strategy.block_m) {
                size_t mmax = std::min(m + m_strategy.block_m, m_M);
                m_strategy.pack_A(a_panel, A, LDA, m, mmax, k, kmax,
                                  m_transpose_A);

                for (size_t n = 0; n < m_N; n += m_strategy.block_n) {
                    size_t nmax = std::min(n + m_strategy.block_n, m_N);
                    m_strategy.pack_B(b_panel, B, LDB, n, nmax, k, kmax,
                                      m_transpose_B);

                    m_strategy.kern(a_panel, b_panel, mmax - m, nmax - n,
                                    kmax - k, C + m * LDC + n, LDC, k == 0,
                                    bias, c_panel);
                }
            }
        }
    }
    void pack_A(pack_a_type* out, const stype* in, int ldin, int y0, int ymax) {
        megdnn_assert(out);
        megdnn_assert(m_M <= m_strategy.block_m && m_N <= m_strategy.block_n &&
                              m_K <= m_strategy.block_k,
                      "currently we only support 1-level blocking");
        m_strategy.pack_A(out, in, ldin, y0, ymax, 0,
                          std::min(m_strategy.block_k, m_K), m_transpose_A);
    }

    void pack_B(stype* out, const stype* in, int ldin, int x0, int xmax) {
        megdnn_assert(out);
        megdnn_assert(m_M <= m_strategy.block_m && m_N <= m_strategy.block_n &&
                              m_K <= m_strategy.block_k,
                      "currently we only support 1-level blocking");
        m_strategy.pack_B(out, in, ldin, x0, xmax, 0,
                          std::min(m_strategy.block_k, m_K), m_transpose_B);
    }

    void execute_naked(dtype* C, const size_t LDC, /* void* workspace,*/
                       const void* packed_a, const void* packed_b) const {
        megdnn_assert(packed_a);
        megdnn_assert(packed_b);
        megdnn_assert(m_M <= m_strategy.block_m && m_N <= m_strategy.block_n &&
                              m_K <= m_strategy.block_k,
                      "currently we only support 1-level blocking");
        pack_a_type* a_panel =
                static_cast<pack_a_type*>(const_cast<void*>(packed_a));
        stype* b_panel = static_cast<stype*>(const_cast<void*>(packed_b));
        for (size_t k = 0; k < m_K; k += m_strategy.block_k) {
            size_t kmax = std::min(k + m_strategy.block_k, m_K);
            for (size_t m = 0; m < m_M; m += m_strategy.block_m) {
                size_t mmax = std::min(m + m_strategy.block_m, m_M);
                for (size_t n = 0; n < m_N; n += m_strategy.block_n) {
                    size_t nmax = std::min(n + m_strategy.block_n, m_N);
                    m_strategy.kern(a_panel, b_panel, mmax - m, nmax - n,
                                    kmax - k, C + m * LDC + n, LDC, k == 0);
                }
            }
        }
    }
};

template <typename Strategy>
class GemmInterleaved<Strategy, false> {
    using compute_type = typename Strategy::compute_type;
    using stype = typename Strategy::stype;
    using dtype = typename Strategy::dst_type;

    const size_t m_M;
    const size_t m_N;
    const size_t m_K;

    const bool m_transpose_A;
    const bool m_transpose_B;

    Strategy m_strategy;

public:
    size_t get_workspace_size() const {
        return m_strategy.get_workspace_size();
    }

    GemmInterleaved(const size_t M, const size_t N, const size_t K,
                    const bool trA, const bool trB, const Strategy& strategy)
            : m_M(M),
              m_N(N),
              m_K(K),
              m_transpose_A(trA),
              m_transpose_B(trB),
              m_strategy(strategy) {}

    // Actually execute the GEMM.
    void execute(const stype* A, const size_t LDA, const stype* B,
                 const size_t LDB, dtype* C, const size_t LDC, void* workspace,
                 const compute_type* bias = nullptr) const {
        m_strategy.kern(A, LDA, B, LDB, C, LDC, m_M, m_K, m_N, bias, workspace,
                        m_transpose_A, m_transpose_B);
    }
};

}  // namespace matmul
}  // namespace megdnn

// vim: syntax=cpp.doxygen
