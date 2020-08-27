/**
 * \file src/rocm/int_fastdiv.h.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \brief fast integer division for constant divisor
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */

#pragma once

#include "src/common/utils.cuh"
#include "hip_header.h"

#include <stdint.h>
#include <cstdlib>

namespace megdnn {
namespace rocm {

/*!
 * \brief fast division for uint32
 */
class Uint32Fastdiv {
    uint32_t m_mul, m_divisor, m_divisor_is_not_1, m_inc_dividend, m_shift;

public:
    Uint32Fastdiv();

    Uint32Fastdiv(uint32_t d) { operator=(d); }

    //! set the divisor to be d
    Uint32Fastdiv& operator=(uint32_t d);

    //! caller must ensure that dividend would not exceed this number
    static MEGDNN_CONSTEXPR uint32_t MAX_DIVIDEND = ~0u - 1;

    __device__ __forceinline__ uint32_t divisor() const { return m_divisor; }

    __device__ __forceinline__ uint32_t divide(uint32_t dividend) const {
        uint32_t ans_for_one = dividend & ~m_divisor_is_not_1,
                 dfix = dividend + m_inc_dividend,
#if MEGDNN_CC_CUDA
                 hi32 = __umulhi(dfix, m_mul),
#else
                 hi32 = ((uint64_t)dfix * m_mul) >> 32,
#endif
                 ans = hi32 >> m_shift;

        return (ans & m_divisor_is_not_1) | ans_for_one;
    }
};

static __forceinline__ __device__ uint32_t operator/(uint32_t a,
                                                     const Uint32Fastdiv& d) {
    return d.divide(a);
}

static __forceinline__ __device__ uint32_t operator%(uint32_t a,
                                                     const Uint32Fastdiv& d) {
    return a - d.divisor() * d.divide(a);
}

/*!
 * \brief maintain (a + k * x) / b and (a + k * x) % b for x >= 0
 * \tparam need_quotient whether quotient need to be maintained
 */
template <bool need_quotient>
class StridedDivSeq;

template <>
class StridedDivSeq<false> {
    Uint32Fastdiv m_b;

    //! k % b
    uint32_t m_kr;

    //! current (a + k * x) % b
    uint32_t m_r;

public:
    void host_init(uint32_t k, uint32_t b) {
        m_b = b;
        m_kr = k % b;
    }

    //! init to k == 0
    __device__ __forceinline__ void device_init(uint32_t a) { m_r = a % m_b; }

    //! perform x += 1
    __device__ __forceinline__ void next() {
        uint32_t b = m_b.divisor(), r1 = m_r + m_kr, carry_mask = (r1 < b) - 1;
        m_r = r1 - (b & carry_mask);
    }

    //! current remainder
    __device__ __forceinline__ uint32_t r() const { return m_r; }
};

template <>
class StridedDivSeq<true> {
    Uint32Fastdiv m_b;

    //! k / b, k % b
    uint32_t m_kq, m_kr;

    //! current (a + k * x) / b and (a + k * x) % b
    uint32_t m_q, m_r;

public:
    void host_init(uint32_t k, uint32_t b) {
        m_b = b;
        m_kq = k / b;
        m_kr = k % b;
    }

    //! init to k == 0
    __device__ __forceinline__ void device_init(uint32_t a) {
        //! fix operator/() defined but not used error
        m_q = a / m_b;
        m_r = a - m_b.divisor() * m_q;
    }

    //! perform x += 1
    __device__ __forceinline__ void next() {
        uint32_t b = m_b.divisor(), r1 = m_r + m_kr, carry_mask = (r1 < b) - 1;
        m_q += m_kq + (r1 >= b);
        m_r = r1 - (b & carry_mask);
    }

    //! current quotient
    __device__ __forceinline__ uint32_t q() const { return m_q; }

    //! current remainder
    __device__ __forceinline__ uint32_t r() const { return m_r; }
};

/*!
 * \brief maintain (a + k * x) / b % c for x >= 0
 */
class StridedDivSeq2 {
    Uint32Fastdiv m_b, m_c;

    //! k / b, k % b, k / b % c
    uint32_t m_qkb, m_rkb, m_rkbc;

    //! current (a + k * x) % b and (a + k * x) / b % c
    uint32_t m_cur_rkb, m_cur_ans;

public:
    void host_init(uint32_t k, uint32_t b, uint32_t c) {
        m_b = b;
        m_c = c;
        m_qkb = k / b;
        m_rkb = k % b;
        m_rkbc = m_qkb % c;
    }

    //! init to k == 0
    __device__ __forceinline__ void device_init(uint32_t a) {
        uint32_t q = m_b.divide(a);
        m_cur_rkb = a - m_b.divisor() * q;
        m_cur_ans = q % m_c;
    }

    //! perform x += 1
    __device__ __forceinline__ void next() {
        uint32_t b = m_b.divisor(), c = m_c.divisor(), rkb = m_cur_rkb + m_rkb,
                 carry0 = (rkb < b) - 1,
                 next_ans = m_cur_ans + m_rkbc + (rkb >= b),
                 carry1 = (next_ans < c) - 1;
        m_cur_rkb = rkb - (b & carry0);
        m_cur_ans = next_ans - (c & carry1);
    }

    __device__ __forceinline__ uint32_t get() const { return m_cur_ans; }
};

}  // namespace rocm
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
