/**
 * \file dnn/src/armv7/matrix_mul/asm/common.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include <arm_neon.h>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

namespace megdnn {
namespace armv7 {

/* ======================== Prefetch ======================== */
#define ASM_PREFETCH(address) "PLD " address "\n"

static inline void prefetch_6x(const void* pfp) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[pfp]]")
                 ASM_PREFETCH("[%[pfp], #64]")
                 ASM_PREFETCH("[%[pfp], #128]")
                 ASM_PREFETCH("[%[pfp], #192]")
                 ASM_PREFETCH("[%[pfp], #256]")
                 ASM_PREFETCH("[%[pfp], #320]")
                 :
                 : [pfp] "r"(pfp)
                 : "memory");
    // clang-format on
}

static inline void prefetch_5x(const void* pfp) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[pfp]]")
                 ASM_PREFETCH("[%[pfp], #64]")
                 ASM_PREFETCH("[%[pfp], #128]")
                 ASM_PREFETCH("[%[pfp], #192]")
                 ASM_PREFETCH("[%[pfp], #256]")
                 :
                 : [pfp] "r"(pfp)
                 : "memory");
    // clang-format on
}

static inline void prefetch_4x(const void* pfp) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[pfp]]")
                 ASM_PREFETCH("[%[pfp], #64]")
                 ASM_PREFETCH("[%[pfp], #128]")
                 ASM_PREFETCH("[%[pfp], #192]")
                 :
                 : [pfp] "r"(pfp)
                 : "memory");
    // clang-format on
}

static inline void prefetch_3x(const void* pfp) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[pfp]]")
                 ASM_PREFETCH("[%[pfp], #64]")
                 ASM_PREFETCH("[%[pfp], #128]")
                 :
                 : [pfp] "r"(pfp)
                 : "memory");
    // clang-format on
}

static inline void prefetch_2x(const void* pfp) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[pfp]]")
                 ASM_PREFETCH("[%[pfp], #64]")
                 :
                 : [pfp] "r"(pfp)
                 : "memory");
    // clang-format on
}

static inline void prefetch_1x(const void* pfp) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[pfp]]") : : [pfp] "r"(pfp) : "memory");
    // clang-format on
}

/* ======================== transform ======================== */
/**
 * interleave_INTERLEAVE_UNROLLK_BATCH_type
 *
 * BATCH means process BATCH * UNROLL_K cols once, BATCH * sizeof(TYPE) *
 * UNROLL_K = 16bytes(128bits, a vector size).
 *
 * the elements traverse order:
 * rep(j, 0, INTERLEAVE) rep(i, 0, UNROLL_K) *ouptr++ = inptr[j, i]
 */

static inline void interleave_4x1_2_d(const int64_t*& inptr0,
                                      const int64_t*& inptr1,
                                      const int64_t*& inptr2,
                                      const int64_t*& inptr3,
                                      int64_t*& outptr) {
    asm volatile(
            "vld1.32 {d0, d1}, [%[inptr0]]!\n"  // A0A1
            "vld1.32 {d2, d3}, [%[inptr1]]!\n"  // B0B1
            "vld1.32 {d4, d5}, [%[inptr2]]!\n"  // C0C1
            "vld1.32 {d6, d7}, [%[inptr3]]!\n"  // D0D1

            "vst1.32 {d0}, [%[outptr]]!\n"
            "vst1.32 {d2}, [%[outptr]]!\n"
            "vst1.32 {d4}, [%[outptr]]!\n"
            "vst1.32 {d6}, [%[outptr]]!\n"
            "vst1.32 {d1}, [%[outptr]]!\n"
            "vst1.32 {d3}, [%[outptr]]!\n"
            "vst1.32 {d5}, [%[outptr]]!\n"
            "vst1.32 {d7}, [%[outptr]]!\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q3", "cc", "memory");
}

static inline void interleave_2x1_4_s(const int32_t*& inptr0,
                                      const int32_t*& inptr1,
                                      int32_t*& outptr) {
    asm volatile(
            "vld1.32 {d0, d1},  [%[inptr0]]!\n"  // A0A1A2A3
            "vld1.32 {d2, d3},  [%[inptr1]]!\n"  // A0A1A2A3
            "vst1.32 {d0, d1},   [%[outptr]]!\n"
            "vst1.32 {d2, d3},   [%[outptr]]!\n"
            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "d3", "cc", "memory");
}

template <typename T>
static inline void interleave_8x8_1_b(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      const T*& inptr4, const T*& inptr5,
                                      const T*& inptr6, const T*& inptr7,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_8x8_1_b only support uint8_t and int8_t");
    asm volatile(
            "vld1.32 {d0},  [%[inptr0]]!\n"  // A1A2A3A4A5A6A7A8
            "vld1.32 {d1},  [%[inptr1]]!\n"  // B1B2B3B4B5B6B7B8
            "vld1.32 {d2},  [%[inptr2]]!\n"  // C1C2C3C4C5C6C7C8
            "vld1.32 {d3},  [%[inptr3]]!\n"  // D1D2D3D4D5D6D7D8
            "vld1.32 {d4},  [%[inptr4]]!\n"  // E1E2E3E4E5E6E7E8
            "vld1.32 {d5},  [%[inptr5]]!\n"  // F1F2F3F4F5F6F7F8
            "vld1.32 {d6},  [%[inptr6]]!\n"  // G1G2G3G4G5G6G7G8
            "vld1.32 {d7},  [%[inptr7]]!\n"  // H1H2H3H4H5H6H7H8

            "vst1.32 {d0},  [%[outptr]]!\n"  // A1A2A3A4A5A6A7A8
            "vst1.32 {d1},  [%[outptr]]!\n"  // B1B2B3B4B5B6B7B8
            "vst1.32 {d2},  [%[outptr]]!\n"  // C1C2C3C4C5C6C7C8
            "vst1.32 {d3},  [%[outptr]]!\n"  // D1D2D3D4D5D6D7D8
            "vst1.32 {d4},  [%[outptr]]!\n"  // E1E2E3E4E5E6E7E8
            "vst1.32 {d5},  [%[outptr]]!\n"  // F1F2F3F4F5F6F7F8
            "vst1.32 {d6},  [%[outptr]]!\n"  // G1G2G3G4G5G6G7G8
            "vst1.32 {d7},  [%[outptr]]!\n"  // H1H2H3H4H5H6H7H8
            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q3", "memory");
}

template <typename T>
static inline void interleave_4x4_4_b(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_4x4_4_b only support uint8_t and int8_t");
    asm volatile(
            "vld1.32 {d0, d1},  [%[inptr0]]!\n"  // A0A1A2A3
            "vld1.32 {d2, d3},  [%[inptr1]]!\n"  // B0B1B2B3
            "vld1.32 {d4, d5},  [%[inptr2]]!\n"  // C0C1C2C3
            "vld1.32 {d6, d7},  [%[inptr3]]!\n"  // D0D1D2D3
            "vtrn.32 q0, q1\n"                   // A0B0A2B2 A1B1A3B3
            "vtrn.32 q2, q3\n"                   // C0D0C2D2 C1D1C3D3
            "vswp     d1, d4    \n"  // q0=A0,B0,C0,D0 q2=A2,B2,C2,D2
            "vswp     d3, d6    \n"  // q1=A1,B1,C1,D1 q3=A3,B3,C3,D3
            "vst1.32 {d0-d1},[%[outptr]]!\n"
            "vst1.32 {d2-d3},[%[outptr]]!\n"
            "vst1.32 {d4-d5},[%[outptr]]!\n"
            "vst1.32 {d6-d7},[%[outptr]]!\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q3", "memory");
}

template <typename T>
static inline void interleave_2x4_4_b(const T*& inptr0, const T*& inptr1,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_2x4_4_b only support uint8_t and int8_t");
    interleave_2x1_4_s(reinterpret_cast<const int32_t*&>(inptr0),
                       reinterpret_cast<const int32_t*&>(inptr1),
                       reinterpret_cast<int32_t*&>(outptr));
}

template <typename T>
static inline void interleave_6x4_4_b(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      const T*& inptr4, const T*& inptr5,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_6x4_4_b only support uint8_t and int8_t");
    asm volatile(
            "vld1.32 {d0, d1},  [%[inptr0]]!\n"    // A0A1A2A3
            "vld1.32 {d2, d3},  [%[inptr1]]!\n"    // B0B1B2B3
            "vld1.32 {d4, d5},  [%[inptr2]]!\n"    // C0C1C2C3
            "vld1.32 {d6, d7},  [%[inptr3]]!\n"    // D0D1D2D3
            "vld1.32 {d8, d9},  [%[inptr4]]!\n"    // E0E1E2E3
            "vld1.32 {d10, d11},  [%[inptr5]]!\n"  // F0F1F2F3
            "vtrn.32 q0, q1\n"                     // A0B0A2B2 A1B1A3B3
            "vtrn.32 q2, q3\n"                     // C0D0C2D2 C1D1C3D3
            "vtrn.32 q4, q5\n"                     // E0F0E2F2 E1F1E3F3
            "vswp     d1, d4    \n"  // q0=A0,B0,C0,D0 q2=A2,B2,C2,D2
            "vswp     d3, d6    \n"  // q1=A1,B1,C1,D1 q3=A3,B3,C3,D3
            "vst1.32 {d0-d1},[%[outptr]]!\n"
            "vst1.32 {d8},   [%[outptr]]!\n"

            "vst1.32 {d2-d3},[%[outptr]]!\n"
            "vst1.32 {d10},  [%[outptr]]!\n"

            "vst1.32 {d4-d5},[%[outptr]]!\n"
            "vst1.32 {d9},   [%[outptr]]!\n"

            "vst1.32 {d6-d7},[%[outptr]]!\n"
            "vst1.32 {d11},  [%[outptr]]!\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
              [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "memory");
}

template <typename T>
static inline void interleave_8x4_4_b(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      const T*& inptr4, const T*& inptr5,
                                      const T*& inptr6, const T*& inptr7,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_8x4_4_b only support uint8_t and int8_t");
    asm volatile(
            "vld1.32 {d0, d1},  [%[inptr0]]!\n"    // A0A1A2A3
            "vld1.32 {d2, d3},  [%[inptr1]]!\n"    // B0B1B2B3
            "vld1.32 {d4, d5},  [%[inptr2]]!\n"    // C0C1C2C3
            "vld1.32 {d6, d7},  [%[inptr3]]!\n"    // D0D1D2D3
            "vld1.32 {d8, d9},  [%[inptr4]]!\n"    // E0E1E2E3
            "vld1.32 {d10, d11},  [%[inptr5]]!\n"  // F0F1F2F3
            "vld1.32 {d12, d13},  [%[inptr6]]!\n"  // G0G1G2G3
            "vld1.32 {d14, d15},  [%[inptr7]]!\n"  // H0H1H2H3
            "vtrn.32 q0, q1\n"                     // A0B0A2B2 A1B1A3B3
            "vtrn.32 q2, q3\n"                     // C0D0C2D2 C1D1C3D3
            "vtrn.32 q4, q5\n"                     // E0F0E2F2 E1F1E3F3
            "vtrn.32 q6, q7\n"                     // G0H0G2H2 G1H1G3H3

            "vswp     d1, d4    \n"  // q0=A0,B0,C0,D0 q2=A2,B2,C2,D2
            "vswp     d3, d6    \n"  // q1=A1,B1,C1,D1 q3=A3,B3,C3,D3

            "vswp     d9, d12    \n"   // q4=E0,F0,G0,H0 q6=E2,F2,G2,H2
            "vswp     d11, d14    \n"  // q5=E1,F1,G1,H1 q7=E3,F3,G3,H3

            "vst1.32 {d0-d1},[%[outptr]]!\n"
            "vst1.32 {d8-d9},[%[outptr]]!\n"

            "vst1.32 {d2-d3},[%[outptr]]!\n"
            "vst1.32 {d10-d11},[%[outptr]]!\n"

            "vst1.32 {d4-d5},[%[outptr]]!\n"
            "vst1.32 {d12-d13},[%[outptr]]!\n"

            "vst1.32 {d6-d7},[%[outptr]]!\n"
            "vst1.32 {d14-d15},[%[outptr]]!\n"
            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "memory");
}

template <typename T>
static inline void interleave_6x4_8_b(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      const T*& inptr4, const T*& inptr5,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_6x8_4_b only support uint8_t and int8_t");
    asm volatile(
            "vld4.32  {d0-d3}, [%[inptr0]]! \n"  // q0,q1=r00,r04,r01,r05,r02,r06,r03,r07
            "vld4.32  {d4-d7}, [%[inptr1]]! \n"  // q2,q3=r10,r14,r11,r15,r12,r16,r13,r17
            "vld4.32  {d8-d11}, [%[inptr2]]!\n"  // q4,q5=r20,r24,r21,r25,r22,r26,r23,r27
            "vld4.32  {d12-d15}, [%[inptr3]]!\n"  // q6,q7=r30,r34,r31,r35,r32,r36,r33,r37
            "vld4.32  {d16-d19}, [%[inptr4]]!\n"  // q8,q9=r40,r44,r41,r45,r42,r46,r43,r47
            "vld4.32  {d20-d23}, [%[inptr5]]!\n"  // q10,q11=r50,r54,r51,r55,r52,r56,r53,r5

            "vtrn.32  q0, q2    \n"  // q0=r00,r10,r01,r11 q2=r04,r14,r05,r15
            "vtrn.32  q4, q6    \n"  // q4=r20,r30,r21,r31 q6=r24,r34,r25,r35
            "vtrn.32  q8, q10   \n"  // q8=r40,r50,r41,r51 q10=r44,r54,r45,r55
            "vswp     d1, d8    \n"  // q0=r00,r10,r20,r30 q4=r01,r11,r21,r31
            "vtrn.32  q1, q3    \n"  // q1=r02,r12,r03,r13 q3=r06,r16,r07,r17
            "vtrn.32  q5, q7    \n"  // q5=r22,r32,r23,r33 q7=r26,r36,r27,r37
            "vtrn.32  q9, q11   \n"  // q9=r42,r52,r43,r53 q11=r46,r56,r47,r57
            "vst1.32  {d0-d1},  [%[outptr]]! \n"
            "vst1.32  {d16},    [%[outptr]]! \n"
            "vswp     d3, d10   \n"  //  q1=r02,r12,r22,r32 q5=r03,r13,r23,r33
            "vst1.32  {d8-d9},  [%[outptr]]! \n"
            "vst1.32  {d17},    [%[outptr]]! \n"
            "vst1.32  {d2-d3},  [%[outptr]]!\n"
            "vst1.32  {d18},    [%[outptr]]!\n"
            "vswp     d5, d12   \n"  // q2=r04,r14,r24,r34 q6=r05,r15,r25,r35
            "vst1.32  {d10-d11},[%[outptr]]!\n"
            "vst1.32  {d19},    [%[outptr]]!\n"
            "vst1.32  {d4-d5},  [%[outptr]]! \n"
            "vst1.32  {d20},    [%[outptr]]! \n"
            "vswp     d7, d14   \n"  // q3=r06,r16,r26,r36 q7=r07,r17,r27,r37
            "vst1.32  {d12-d13},[%[outptr]]! \n"
            "vst1.32  {d21},    [%[outptr]]! \n"
            "vst1.32  {d6-d7},  [%[outptr]]! \n"
            "vst1.32  {d22},    [%[outptr]]! \n"
            "vst1.32  {d14-d15},[%[outptr]]! \n"
            "vst1.32  {d23},    [%[outptr]]! \n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
              [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
              "q11", "cc", "memory");
}

template <typename T>
static inline void interleave_4x16_1_b(const T*& inptr0, const T*& inptr1,
                                       const T*& inptr2, const T*& inptr3,
                                       T*& outptr) {
    static_assert(sizeof(T) == 1, "only support size == 1");
    asm volatile(
            "vld1.32 {d0, d1}, [%[inptr0]]!\n"  // d0 = A0A1A2A3
            "vld1.32 {d2, d3}, [%[inptr1]]!\n"  // d1 = B0B1B2B3
            "vld1.32 {d4, d5}, [%[inptr2]]!\n"  // d2 = C0C1C2C3
            "vld1.32 {d6, d7}, [%[inptr3]]!\n"  // d3 = D0D1D2D3
            "vst1.32 {d0, d1}, [%[outptr]]!\n"
            "vst1.32 {d2, d3}, [%[outptr]]!\n"
            "vst1.32 {d4, d5}, [%[outptr]]!\n"
            "vst1.32 {d6, d7}, [%[outptr]]!\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q3", "cc", "memory");
}

template <typename T>
static inline void interleave_4x8_2_b(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_4x8_2_b only support uint8_t and int8_t");
    interleave_4x1_2_d(reinterpret_cast<const int64_t*&>(inptr0),
                       reinterpret_cast<const int64_t*&>(inptr1),
                       reinterpret_cast<const int64_t*&>(inptr2),
                       reinterpret_cast<const int64_t*&>(inptr3),
                       reinterpret_cast<int64_t*&>(outptr));
}

template <typename T>
static inline void interleave_2x16_1_b(const T*& inptr0, const T*& inptr1,
                                       T*& outptr) {
    static_assert(sizeof(T) == 1, "only support size == 2");
    asm volatile(
            "vld1.32 {d0, d1}, [%[inptr0]]!\n"
            "vld1.32 {d2, d3}, [%[inptr1]]!\n"
            "vst1.32 {d0, d1}, [%[outptr]]!\n"
            "vst1.32 {d2, d3}, [%[outptr]]!\n"

            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [outptr] "+r"(outptr)
            :
            : "q0", "q1", "cc", "memory");
}

template <typename T>
static inline void interleave_4x4_1_h(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      T*& outptr) {
    static_assert(sizeof(T) == 2,
                  "interleave_4x16_1_h only support sizeof(T) == 2");
    asm volatile(
            "vld1.16 {d0},  [%[inptr0]]!\n"
            "vld1.16 {d1},  [%[inptr1]]!\n"
            "vld1.16 {d2},  [%[inptr2]]!\n"
            "vld1.16 {d3},  [%[inptr3]]!\n"

            "vst1.16 {d0},  [%[outptr]]!\n"
            "vst1.16 {d1},  [%[outptr]]!\n"
            "vst1.16 {d2},  [%[outptr]]!\n"
            "vst1.16 {d3},  [%[outptr]]!\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "d3", "d4", "memory");
}

template <typename T>
static inline void interleave_4x12_1_h(const T*& inptr0, const T*& inptr1,
                                       const T*& inptr2, const T*& inptr3,
                                       T*& outptr) {
    static_assert(sizeof(T) == 2,
                  "interleave_4x12_1_h only support sizeof(T) == 2");
    asm volatile(
            "pld [%[inptr0],#192]\n"
            "vld1.16 {d0},  [%[inptr0]]!\n"  // A0A1A2A3
            "vld1.16 {d1},  [%[inptr0]]!\n"  // B0B1B2B3
            "vld1.16 {d2},  [%[inptr0]]!\n"  // C0C1C2C3
            "pld [%[inptr1],#192]\n"
            "vld1.16 {d3},  [%[inptr1]]!\n"  // A0A1A2A3
            "vld1.16 {d4},  [%[inptr1]]!\n"  // B0B1B2B3
            "vld1.16 {d5},  [%[inptr1]]!\n"  // C0C1C2C3
            "pld [%[inptr2],#192]\n"
            "vld1.16 {d6},  [%[inptr2]]!\n"  // A0A1A2A3
            "vld1.16 {d7},  [%[inptr2]]!\n"  // B0B1B2B3
            "vld1.16 {d8},  [%[inptr2]]!\n"  // C0C1C2C3
            "pld [%[inptr3],#192]\n"
            "vld1.16 {d9},  [%[inptr3]]!\n"  // A0A1A2A3
            "vld1.16 {d10}, [%[inptr3]]!\n"  // B0B1B2B3
            "vld1.16 {d11}, [%[inptr3]]!\n"  // C0C1C2C3

            "vst1.16 {d0},  [%[outptr]]!\n"   // A0B0C0D0
            "vst1.16 {d1},  [%[outptr]]!\n"   // E0F0G0H0
            "vst1.16 {d2},  [%[outptr]]!\n"   // I0J0K0L0
            "vst1.16 {d3},  [%[outptr]]!\n"   // D0D1D2D3
            "vst1.16 {d4},  [%[outptr]]!\n"   // E0E1E2E3
            "vst1.16 {d5},  [%[outptr]]!\n"   // F0F1F2F3
            "vst1.16 {d6},  [%[outptr]]!\n"   // G0G1G2G3
            "vst1.16 {d7},  [%[outptr]]!\n"   // H0H1H2H3
            "vst1.16 {d8},  [%[outptr]]!\n"   // H0H1H2H3
            "vst1.16 {d9},  [%[outptr]]!\n"   // G0G1G2G3
            "vst1.16 {d10},  [%[outptr]]!\n"  // H0H1H2H3
            "vst1.16 {d11},  [%[outptr]]!\n"  // H0H1H2H3
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
              "d11", "memory");
}

template <typename T>
static inline void interleave_4x16_1_h(const T*& inptr0, const T*& inptr1,
                                       const T*& inptr2, const T*& inptr3,
                                       T*& outptr) {
    static_assert(sizeof(T) == 2,
                  "interleave_4x16_1_h only support sizeof(T) == 2");
    asm volatile(
            "vld1.16 {d0, d1, d2, d3},  [%[inptr0]]!\n"
            "vld1.16 {d4, d5, d6, d7},  [%[inptr1]]!\n"
            "vld1.16 {d8, d9, d10, d11},  [%[inptr2]]!\n"
            "vld1.16 {d12, d13, d14, d15},  [%[inptr3]]!\n"

            "vst1.16 {d0, d1, d2, d3},  [%[outptr]]!\n"
            "vst1.16 {d4, d5, d6, d7},  [%[outptr]]!\n"
            "vst1.16 {d8, d9, d10, d11},  [%[outptr]]!\n"
            "vst1.16 {d12, d13, d14, d15},  [%[outptr]]!\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
              "d11", "d12", "d13", "d14", "d15", "memory");
}

template <typename T>
static inline void interleave_4x4_1_s(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      T*& outptr) {
    static_assert(sizeof(T) == 4,
                  "interleave_4x4_1_s only support sizeof(T) == 4");
    asm volatile(
            "vld1.32 {d0, d1},  [%[inptr0]]!\n"  // A0A1A2A3
            "vld1.32 {d2, d3},  [%[inptr1]]!\n"  // A0A1A2A3
            "vld1.32 {d4, d5},  [%[inptr2]]!\n"  // A0A1A2A3
            "vld1.32 {d6, d7},  [%[inptr3]]!\n"  // A0A1A2A3

            "vst1.32 {d0, d1},   [%[outptr]]!\n"  // A0B0C0D0
            "vst1.32 {d2, d3},   [%[outptr]]!\n"  // E0F0G0H0
            "vst1.32 {d4, d5},   [%[outptr]]!\n"  // I0J0K0L0
            "vst1.32 {d6, d7},   [%[outptr]]!\n"  // D0D1D2D3
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "memory");
}

template <typename T>
static inline void interleave_1x4_1_h(const T*& inptr0, T*& outptr) {
    static_assert(sizeof(T) == 2,
                  "transpose_1x4_1_h only support sizeof(T) == 2");
    asm volatile(
            "vld1.16 {d0},  [%[inptr0]]!\n"  // A01234567
            "vst1.16 {d0},  [%[outptr]]!\n"
            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "d0", "memory");
}

template <typename T>
static inline void interleave_4x12_1_s(const T*& inptr0, const T*& inptr1,
                                       const T*& inptr2, const T*& inptr3,
                                       T*& outptr) {
    static_assert(sizeof(T) == 4,
                  "interleave_4x12_1_s only support sizeof(T) == 4");
    asm volatile(
            "vld1.32 {d0, d1},  [%[inptr0]]!\n"    // A0A1A2A3
            "vld1.32 {d2, d3},  [%[inptr0]]!\n"    // B0B1B2B3
            "vld1.32 {d4, d5},  [%[inptr0]]!\n"    // C0C1C2C3
            "vld1.32 {d6, d7},  [%[inptr1]]!\n"    // A0A1A2A3
            "vld1.32 {d8, d9},  [%[inptr1]]!\n"    // B0B1B2B3
            "vld1.32 {d10, d11},  [%[inptr1]]!\n"  // C0C1C2C3
            "vld1.32 {d12, d13},  [%[inptr2]]!\n"  // A0A1A2A3
            "vld1.32 {d14, d15},  [%[inptr2]]!\n"  // B0B1B2B3
            "vld1.32 {d16, d17},  [%[inptr2]]!\n"  // C0C1C2C3
            "vld1.32 {d18, d19},  [%[inptr3]]!\n"  // A0A1A2A3
            "vld1.32 {d20, d21}, [%[inptr3]]!\n"   // B0B1B2B3
            "vld1.32 {d22, d23}, [%[inptr3]]!\n"   // C0C1C2C3

            "vst1.32 {d0, d1},   [%[outptr]]!\n"   // A0B0C0D0
            "vst1.32 {d2, d3},   [%[outptr]]!\n"   // E0F0G0H0
            "vst1.32 {d4, d5},   [%[outptr]]!\n"   // I0J0K0L0
            "vst1.32 {d6, d7},   [%[outptr]]!\n"   // D0D1D2D3
            "vst1.32 {d8, d9},   [%[outptr]]!\n"   // E0E1E2E3
            "vst1.32 {d10, d11}, [%[outptr]]!\n"   // F0F1F2F3
            "vst1.32 {d12, d13}, [%[outptr]]!\n"   // G0G1G2G3
            "vst1.32 {d14, d15}, [%[outptr]]!\n"   // H0H1H2H3
            "vst1.32 {d16, d17}, [%[outptr]]!\n"   // H0H1H2H3
            "vst1.32 {d18, d19}, [%[outptr]]!\n"   // G0G1G2G3
            "vst1.32 {d20, d21},  [%[outptr]]!\n"  // H0H1H2H3
            "vst1.32 {d22, d23},  [%[outptr]]!\n"  // H0H1H2H3
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
              "d11", "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19",
              "d20", "d21", "d22", "d23", "memory");
}

template <typename T>
static inline void interleave_1x12_1_h(const T*& inptr0, T*& outptr) {
    static_assert(sizeof(T) == 2,
                  "transpose_1x12_1_h only support sizeof(T) == 2");
    asm volatile(
            "vld1.16 {d0,d1},  [%[inptr0]]!\n"  // A01234567
            "vld1.16 {d2}   ,  [%[inptr0]]!\n"  // A891011
            "vst1.16 {d0,d1},  [%[outptr]]!\n"
            "vst1.16 {d2}   ,  [%[outptr]]\n"
            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "memory");
}

template <typename T>
static inline void interleave_1x12_1_s(const T*& inptr0, T*& outptr) {
    static_assert(sizeof(T) == 4,
                  "interleave_1x12_1_s only support sizeof(T) == 4");
    asm volatile(
            "vld1.32 {d0, d1}, [%[inptr0]]!\n"
            "vld1.32 {d2, d3}, [%[inptr0]]!\n"
            "vld1.32 {d4, d5}, [%[inptr0]]!\n"
            "vst1.32 {d0, d1}, [%[outptr]]!\n"
            "vst1.32 {d2, d3}, [%[outptr]]!\n"
            "vst1.32 {d4, d5}, [%[outptr]]!\n"
            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "memory");
}

template <typename T>
static inline void interleave_1x16_1_h(const T*& inptr0, T*& outptr) {
    static_assert(sizeof(T) == 2,
                  "transpose_1x12_1_h only support sizeof(T) == 2");
    asm volatile(
            "vld1.16 {d0,d1, d2, d3},  [%[inptr0]]!\n"
            "vst1.16 {d0,d1, d2, d3},  [%[outptr]]!\n"
            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "d0", "d1", "d2", "d3", "memory");
}

template <typename T>
static inline void interleave_1x4_1_s(const T*& inptr0, T*& outptr) {
    static_assert(sizeof(T) == 4,
                  "interleave_1x4_1_s only support sizeof(T) == 4");
    asm volatile(
            "vld1.32 {d0, d1},  [%[inptr0]]!\n"
            "vst1.32 {d0, d1},  [%[outptr]]\n"
            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "d0", "d1", "memory");
}

template <typename T>
static inline void interleave_helper(const T*& inptr, T*& outptr, int unroll_k,
                                     int ksize, T val = 0) {
    int k = 0;
    for (; k < ksize; k++) {
        *outptr++ = *inptr++;
    }
    for (; k < unroll_k; k++) {
        *outptr++ = val;
    }
}

template <typename T>
static inline void interleave_1(const T*& inptr0, T*& outptr, int unroll_k,
                                int ksize, T val = 0) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        interleave_helper(inptr0, outptr, unroll_k, size, val);
    }
}

template <typename T>
static inline void interleave_2(const T*& inptr0, const T*& inptr1, T*& outptr,
                                int unroll_k, int ksize, T val = 0) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        interleave_helper(inptr0, outptr, unroll_k, size, val);
        interleave_helper(inptr1, outptr, unroll_k, size, val);
    }
}

template <typename T>
static inline void interleave_4(const T*& inptr0, const T*& inptr1,
                                const T*& inptr2, const T*& inptr3, T*& outptr,
                                int unroll_k, int ksize, T val = 0) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        interleave_helper(inptr0, outptr, unroll_k, size, val);
        interleave_helper(inptr1, outptr, unroll_k, size, val);
        interleave_helper(inptr2, outptr, unroll_k, size, val);
        interleave_helper(inptr3, outptr, unroll_k, size, val);
    }
}

template <typename T>
static inline void interleave_6(const T*& inptr0, const T*& inptr1,
                                const T*& inptr2, const T*& inptr3,
                                const T*& inptr4, const T*& inptr5, T*& outptr,
                                int unroll_k, int ksize, T val = 0) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        interleave_helper(inptr0, outptr, unroll_k, size, val);
        interleave_helper(inptr1, outptr, unroll_k, size, val);
        interleave_helper(inptr2, outptr, unroll_k, size, val);
        interleave_helper(inptr3, outptr, unroll_k, size, val);
        interleave_helper(inptr4, outptr, unroll_k, size, val);
        interleave_helper(inptr5, outptr, unroll_k, size, val);
    }
}
template <typename T>
static inline void interleave_8(const T*& inptr0, const T*& inptr1,
                                const T*& inptr2, const T*& inptr3,
                                const T*& inptr4, const T*& inptr5,
                                const T*& inptr6, const T*& inptr7, T*& outptr,
                                int unroll_k, int ksize, T val = 0) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        interleave_helper(inptr0, outptr, unroll_k, size, val);
        interleave_helper(inptr1, outptr, unroll_k, size, val);
        interleave_helper(inptr2, outptr, unroll_k, size, val);
        interleave_helper(inptr3, outptr, unroll_k, size, val);
        interleave_helper(inptr4, outptr, unroll_k, size, val);
        interleave_helper(inptr5, outptr, unroll_k, size, val);
        interleave_helper(inptr6, outptr, unroll_k, size, val);
        interleave_helper(inptr7, outptr, unroll_k, size, val);
    }
}

/* ======================== transpose pack B ======================== */
/**
 * transpose_INTERLEAVE_UNROLLK_BATCH_type
 *
 * BATCH means process BATCH * INTERLEAVE cols once, BATCH * sizeof(TYPE) *
 * INTERLEAVE = 16bytes(128bits, a vector size).
 *
 * the elements traverse order:
 * rep(j, 0, INTERLEAVE) rep(i, 0, UNROLL_K) *ouptr++ = inptr[i, j]
 */
template <typename T>
static inline void transpose_8x8_1_b(const T*& inptr0, const T*& inptr1,
                                     const T*& inptr2, const T*& inptr3,
                                     const T*& inptr4, const T*& inptr5,
                                     const T*& inptr6, const T*& inptr7,
                                     T* outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "transpose_8x8_1_b only support uint8_t and int8_t");
    asm volatile(
            "vld1.32 {d0},  [%[inptr0]]!\n"  // A1A2A3A4A5A6A7A8
            "vld1.32 {d1},  [%[inptr1]]!\n"  // B1B2B3B4B5B6B7B8
            "vld1.32 {d2},  [%[inptr2]]!\n"  // C1C2C3C4C5C6C7C8
            "vld1.32 {d3},  [%[inptr3]]!\n"  // D1D2D3D4D5D6D7D8
            "vld1.32 {d4},  [%[inptr4]]!\n"  // E1E2E3E4E5E6E7E8
            "vld1.32 {d5},  [%[inptr5]]!\n"  // F1F2F3F4F5F6F7F8
            "vld1.32 {d6},  [%[inptr6]]!\n"  // G1G2G3G4G5G6G7G8
            "vld1.32 {d7},  [%[inptr7]]!\n"  // H1H2H3H4H5H6H7H8

            "vzip.8 d0, d1\n"  // A1B1A2B2A3B3A4B4 A5B5A6B6A7B7A8B8
            "vzip.8 d2, d3\n"  // C1D1C2D2C3D3C4D4 C5D5C6D6C7D7C8D8
            "vzip.8 d4, d5\n"  // E1F1E2F2E3F3E4F4 E5F5E6F6E7F7E8F8
            "vzip.8 d6, d7\n"  // G1H1G2H2G3H3G4H4 G5H5G6H6G7H7G8H8

            "vzip.16 d0, d2\n"  // A1B1C1D1A2B2C2D2 A3B3C3D3A4B4C4D4
            "vzip.16 d4, d6\n"  // E1F1G1H1E2F2G2H2 E3F3G3H3E4F4G4H4
            "vzip.16 d1, d3\n"  // A5B5C5D5A6B6C6D6 A7B7C7D7A8B8C8D8
            "vzip.16 d5, d7\n"  // E5F5G5H5E6F6G6H6 E7F7G7H7E8F8G8H8

            "vzip.32 d0, d4\n"  // A1B1C1D1E1F1G1H1 A2B2C2D2E2F2G2H2
            "vzip.32 d1, d5\n"  // A5B5C5D5E5F5G5H5 A6B6C6D6E6F6G6H6
            "vzip.32 d2, d6\n"  // A3B3C3D3E3F3G3H3 A4B4C4D4E4F4G4H4
            "vzip.32 d3, d7\n"  // A7B7C7D7E7F7G7H7 A8B8C8D8E8F8G8H8

            "vst1.32 {d0},  [%[outptr]]!\n"  // A1B1C1D1E1F1G1H1
            "vst1.32 {d4},  [%[outptr]]!\n"  // A2B2C2D2E2F2G2H2
            "vst1.32 {d2},  [%[outptr]]!\n"  // A3B3C3D3E3F3G3H3
            "vst1.32 {d6},  [%[outptr]]!\n"  // A4B4C4D4E4F4G4H4
            "vst1.32 {d1},  [%[outptr]]!\n"  // A5B5C5D5E5F5G5H5
            "vst1.32 {d5},  [%[outptr]]!\n"  // A6B6C6D6E6F6G6H6
            "vst1.32 {d3},  [%[outptr]]!\n"  // A7B7C7D7E7F7G7H7
            "vst1.32 {d7},  [%[outptr]]!\n"  // A8B8C8D8E8F8G8H8
            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q3", "cc", "memory");
}

template <typename T>
static inline void transpose_8x4_1_b(const T*& inptr0, const T*& inptr1,
                                     const T*& inptr2, const T*& inptr3,
                                     T* outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "transpose_8x4_1_b only support uint8_t and int8_t");
    asm volatile(
            "vld1.32 {d0},  [%[inptr0]]!\n"  // A1A2A3A4A5A6A7A8
            "vld1.32 {d1},  [%[inptr1]]!\n"  // B1B2B3B4B5B6B7B8
            "vld1.32 {d2},  [%[inptr2]]!\n"  // C1C2C3C4C5C6C7C8
            "vld1.32 {d3},  [%[inptr3]]!\n"  // D1D2D3D4D5D6D7D8

            "vtrn.8 d0, d1\n"  // A1B1A3B3A5B5A7B7 A2B2A4B4A6B6A8B8
            "vtrn.8 d2, d3\n"  // C1D1C3D3C5D5C7D7 C2D2C4D4C6D6C8D8

            "vtrn.16 d0, d2\n"  // A1B1C1D1A5B5C5D5 A3B3C3D3A7B7C7D7
            "vtrn.16 d1, d3\n"  // A2B2C2D2A6B6C6D6 A4B4C4D4A8B8C8D8

            //! ABCD=E then
            //! d0: E1E5 d1: E2E6 d2: E3E7 d3: E4E8
            "vzip.32 d0, d1\n"  // E1E2 E5E6
            "vzip.32 d2, d3\n"  // E3E4 E7E8

            "vst1.32 {d0}, [%[outptr]]!\n"
            "vst1.32 {d2}, [%[outptr]]!\n"
            "vst1.32 {d1}, [%[outptr]]!\n"
            "vst1.32 {d3}, [%[outptr]]!\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "q0", "q1", "memory");
}

template <typename T>
static inline void transpose_12x4_1_h(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      const T*& inptr4, const T*& inptr5,
                                      const T*& inptr6, const T*& inptr7,
                                      const T*& inptr8, const T*& inptr9,
                                      const T*& inptr10, const T*& inptr11,
                                      int ldin, T*& outptr) {
    static_assert(
            std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value,
            "interleave_12x4_1_h only support uint16_t and int16_t");
    auto ldin_asm = ldin << 1;
    asm volatile(
            "vld1.16 {d0},  [%[inptr0]]!\n"  // A0A1A2A3
            "vld1.16 {d1},  [%[inptr1]]!\n"  // B0B1B2B3
            "vld1.16 {d2},  [%[inptr2]]!\n"  // C0C1C2C3
            "vld1.16 {d3},  [%[inptr3]]!\n"  // D0D1D2D3
            "vld1.16 {d4},  [%[inptr4]]!\n"  // E0E1E2E3
            "vld1.16 {d5},  [%[inptr5]]!\n"  // F0F1F2F3
            "vld1.16 {d6},  [%[inptr6]]!\n"  // G0G1G2G3
            "vld1.16 {d7},  [%[inptr7]]!\n"  // H0H1H2H3
            "vld1.16 {d8},  [%[inptr8]]!\n"  // I0I1I2I3
            "vld1.16 {d9},  [%[inptr9]]\n"   // J0J1J2J3
            "add %[inptr9], %[inptr9], %[ldin_asm]\n"
            "vld1.16 {d10}, [%[inptr9]]\n"  // K0K1K2K3
            "add %[inptr9], %[inptr9], %[ldin_asm]\n"
            "vld1.16 {d11}, [%[inptr9]]\n"  // L0L1L2L3

            "vtrn.16 d0, d1\n"  // A0B0A2B2A1B1A3B3
            "vtrn.16 d2, d3\n"  // C0D0C2D2C1D1C3D3
            "vtrn.16 d4, d5\n"  // E0F0E2F2E1F1E3F3
            "vtrn.16 d6, d7\n"  // G0H0G2H2G1H1G3H3

            "vtrn.16 d8, d9\n"    // I0J0I2J2I1J1I3J3
            "vtrn.16 d10, d11\n"  // K0L0K2L2K1L1K3L3

            "vtrn.32 q0, q1\n"  // A0B0C0D0 A1B1C1D1 A2B2C2D2 A3B3C3D3
            "vtrn.32 q2, q3\n"  // E0F0G0H0 E1F1G1G1 E2F2G2H2 E3F3G3H3
            "vtrn.32 q4, q5\n"  // I0J0K0L0 I1J1K1L1 I2J2K2L2 I3J3K3L3

            "vst1.16 {d0},  [%[outptr]]!\n"  // A0B0C0D0
            "vst1.16 {d4},  [%[outptr]]!\n"  // E0F0G0H0
            "vst1.16 {d8},  [%[outptr]]!\n"  // I0J0K0L0
            "vst1.16 {d1},  [%[outptr]]!\n"  // D0D1D2D3
            "vst1.16 {d5},  [%[outptr]]!\n"  // E0E1E2E3
            "vst1.16 {d9},  [%[outptr]]!\n"  // F0F1F2F3
            "vst1.16 {d2},  [%[outptr]]!\n"  // G0G1G2G3
            "vst1.16 {d6},  [%[outptr]]!\n"  // H0H1H2H3
            "vst1.16 {d10}, [%[outptr]]!\n"  // H0H1H2H3
            "vst1.16 {d3},  [%[outptr]]!\n"  // G0G1G2G3
            "vst1.16 {d7},  [%[outptr]]!\n"  // H0H1H2H3
            "vst1.16 {d11}, [%[outptr]]!\n"  // H0H1H2H3
            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [inptr8] "+r"(inptr8),
            [inptr9] "+r"(inptr9), [outptr] "+r"(outptr)
            : [ldin_asm] "r"(ldin_asm)
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10",
              "d11", "memory");
    inptr9 -= ldin_asm;
    inptr9 += 4;
    inptr10 += 4;
    inptr11 += 4;
}

template <typename T>
static inline void transpose_2x16_1_b_helper(const T*& inptr0, const T*& inptr1,
                                             const T*& inptr2, const T*& inptr3,
                                             const T*& inptr4, const T*& inptr5,
                                             const T*& inptr6, const T*& inptr7,
                                             T* outptr) {
    static_assert(sizeof(T) == 1, "only support size == 1");
    static uint8x8_t shuffle_idx = {0, 2, 4, 6, 1, 3, 5, 7};
    asm volatile(
            "vld1.16 {d0[0]}, [%[inptr0]]!\n"
            "vld1.16 {d0[1]}, [%[inptr1]]!\n"
            "vld1.16 {d0[2]}, [%[inptr2]]!\n"
            "vld1.16 {d0[3]}, [%[inptr3]]!\n"
            "vld1.16 {d2[0]}, [%[inptr4]]!\n"
            "vld1.16 {d2[1]}, [%[inptr5]]!\n"
            "vld1.16 {d2[2]}, [%[inptr6]]!\n"
            "vld1.16 {d2[3]}, [%[inptr7]]!\n"
            "mov r0, #16\n"

            "vtbl.8 d1, {d0}, %[shuffle_idx]\n"
            "vtbl.8 d3, {d2}, %[shuffle_idx]\n"

            "vzip.32 d1, d3\n"

            "vst1.64 d1, [%[outptr]], r0\n"
            "vst1.64 d3, [%[outptr]]\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
              [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7),
              [outptr] "+r"(outptr), [shuffle_idx] "+w"(shuffle_idx)
            :
            : "q0", "q1", "q2", "r0", "memory");
}

template <typename T>
static inline void transpose_4x8_1_b(const T*& inptr0, const T*& inptr1,
                                     const T*& inptr2, const T*& inptr3,
                                     const T*& inptr4, const T*& inptr5,
                                     const T*& inptr6, const T*& inptr7,
                                     T* outptr) {
    static uint8x8_t shuffle_idx = {0, 4, 1, 5, 2, 6, 3, 7};
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "transpose_4x8_1_b only support uint8_t and int8_t");
    asm volatile(
            "vld1.32 {d0[0]},  [%[inptr0]]!\n"  // A1A2A3A4
            "vld1.32 {d0[1]},  [%[inptr1]]!\n"  // B1B2B3B4
            "vld1.32 {d1[0]},  [%[inptr2]]!\n"  // C1C2C3C4
            "vld1.32 {d1[1]},  [%[inptr3]]!\n"  // D1D2D3D4
            "vld1.32 {d2[0]},  [%[inptr4]]!\n"  // E1E2E3E4
            "vld1.32 {d2[1]},  [%[inptr5]]!\n"  // F1F2F3F4
            "vld1.32 {d3[0]},  [%[inptr6]]!\n"  // G1G2G3G4
            "vld1.32 {d3[1]},  [%[inptr7]]!\n"  // H1H2H3H4

            "vtbl.8 d4, {d0}, %[shuffle_idx]\n"  // A1B1A2B2A3B3A4B4
            "vtbl.8 d5, {d1}, %[shuffle_idx]\n"  // C1D1C2D2C3D3C4D4
            "vtbl.8 d6, {d2}, %[shuffle_idx]\n"  // E1F1E2F2E3F3E4F4
            "vtbl.8 d7, {d3}, %[shuffle_idx]\n"  // G1H1G2H2G3H3G4H4

            "vzip.16 d4, d5\n"  // A1B1C1D1A2B2C2D2 A3B3C3D3A4B4C4D4
            "vzip.16 d6, d7\n"  // E1F1G1H1E2F2G2H2 E3F3G3H3E4F4G4H4
            "vzip.32 d4, d6\n"  // A1B1C1D1E1F1G1H1 A2B2C2D2E2F2G2H2
            "vzip.32 d5, d7\n"  // A3B3C3D3E3F3G3H3 A4B4C4D4E4F4G4H4

            "vst1.32 {d4},  [%[outptr]]!\n"  // A1B1C1D1E1F1G1H1
            "vst1.32 {d6},  [%[outptr]]!\n"  // A2B2C2D2E2F2G2H2
            "vst1.32 {d5},  [%[outptr]]!\n"  // A3B3C3D3E3F3G3H3
            "vst1.32 {d7},  [%[outptr]]!\n"  // A4B4C4D4E4F4G4H4
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
              [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7),
              [outptr] "+r"(outptr), [shuffle_idx] "+w"(shuffle_idx)
            :
            : "q0", "q1", "q2", "q3", "cc", "memory");
}

template <typename T>
static inline void transpose_4x16_1_b_helper(const T*& inptr0, const T*& inptr1,
                                             const T*& inptr2, const T*& inptr3,
                                             const T*& inptr4, const T*& inptr5,
                                             const T*& inptr6, const T*& inptr7,
                                             T* outptr) {
    static_assert(sizeof(T) == 1, "only support size == 1");
    static uint8x8_t shuffle_idx = {0, 4, 1, 5, 2, 6, 3, 7};
    asm volatile(
            "vld1.32 {d0[0]}, [%[inptr0]]!\n"
            "vld1.32 {d0[1]}, [%[inptr1]]!\n"
            "vld1.32 {d1[0]}, [%[inptr2]]!\n"
            "vld1.32 {d1[1]}, [%[inptr3]]!\n"
            "vld1.32 {d2[0]}, [%[inptr4]]!\n"
            "vld1.32 {d2[1]}, [%[inptr5]]!\n"
            "vld1.32 {d3[0]}, [%[inptr6]]!\n"
            "vld1.32 {d3[1]}, [%[inptr7]]!\n"
            "mov r0, #16\n"

            "vtbl.8 d4, {d0}, %[shuffle_idx]\n"
            "vtbl.8 d5, {d1}, %[shuffle_idx]\n"
            "vtbl.8 d6, {d2}, %[shuffle_idx]\n"
            "vtbl.8 d7, {d3}, %[shuffle_idx]\n"

            "vzip.16 d4, d5\n"
            "vzip.16 d6, d7\n"
            "vzip.32 d4, d6\n"
            "vzip.32 d5, d7\n"

            "vst1.64 d4, [%[outptr]], r0\n"
            "vst1.64 d6, [%[outptr]], r0\n"
            "vst1.64 d5, [%[outptr]], r0\n"
            "vst1.64 d7, [%[outptr]]\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
              [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7),
              [outptr] "+r"(outptr), [shuffle_idx] "+w"(shuffle_idx)
            :
            : "q0", "q1", "q2", "q3", "q4", "r0", "memory");
}

template <typename T>
static inline void transpose_4x4_1_h(const T*& inptr0, const T*& inptr1,
                                     const T*& inptr2, const T*& inptr3,
                                     T*& outptr, int stride = 8) {
    static_assert(sizeof(T) == 2,
                  "transpose_4x4_1_h only support sizeof(T) == 2");

    asm volatile(
            "vld1.16 {d0},  [%[inptr0]]!\n"  // A0A1A2A3
            "vld1.16 {d1},  [%[inptr1]]!\n"  // B0B1B2B3
            "vld1.16 {d2},  [%[inptr2]]!\n"  // C0C1C2C3
            "vld1.16 {d3},  [%[inptr3]]!\n"  // D0D1D2D3
            "vtrn.16 d0, d1\n"               // A0B0A2B2A1B1A3B3
            "vtrn.16 d2, d3\n"               // C0D0C2D2C1D1C3D3
            "vtrn.32 q0, q1\n"  // A0B0C0D0 A1B1C1D1 A2B2C2D2 A3B3C3D3
            "vst1.16 {d0},  [%[outptr]], %[stride]\n"  // A0B0C0D0
            "vst1.16 {d1},  [%[outptr]], %[stride]\n"  // A1B1C1D1
            "vst1.16 {d2},  [%[outptr]], %[stride]\n"  // A2B2C2D2
            "vst1.16 {d3},  [%[outptr]], %[stride]\n"  // A3B3C3D3
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            : [stride] "r"(stride)
            : "d0", "d1", "d2", "d3", "memory");
}

template <typename T>
static inline void transpose_4x4_1_s(const T*& inptr0, const T*& inptr1,
                                     const T*& inptr2, const T*& inptr3,
                                     T*& outptr, int stride = 16) {
    static_assert(sizeof(T) == 4,
                  "transpose_4x4_1_s only support sizeof(T) == 4");

    stride -= 8;
    asm volatile(
            "vld1.32 {d0, d1},  [%[inptr0]]!\n"  // A0A1A2A3
            "vld1.32 {d2, d3},  [%[inptr1]]!\n"  // B0B1B2B3
            "vld1.32 {d4, d5},  [%[inptr2]]!\n"  // C0C1C2C3
            "vld1.32 {d6, d7},  [%[inptr3]]!\n"  // D0D1D2D3
            "vtrn.32 q0, q1\n"                   // A0B0A2B2 A1B1A3B3
            "vtrn.32 q2, q3\n"                   // C0D0C2D2 C1D1C3D3
            "vst1.32 {d0},  [%[outptr]]!\n"
            "vst1.32 {d4},  [%[outptr]], %[stride]\n"
            "vst1.32 {d2},  [%[outptr]]!\n"
            "vst1.32 {d6},  [%[outptr]], %[stride]\n"
            "vst1.32 {d1},  [%[outptr]]!\n"
            "vst1.32 {d5},  [%[outptr]], %[stride]\n"
            "vst1.32 {d3},  [%[outptr]]!\n"
            "vst1.32 {d7},  [%[outptr]], %[stride]\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr), [stride] "+r"(stride)
            :
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "memory");
}

template <typename T>
static inline void transpose_4x2_1_s(const T*& inptr0, const T*& inptr1,
                                     const T*& inptr2, const T*& inptr3,
                                     T* outptr, int stride = 8) {
    static_assert(sizeof(T) == 4,
                  "transpose_4x2_1_s only support sizeof(T) == 4");

    stride -= 8;
    asm volatile(
            "vld1.32 {d0},  [%[inptr0]]!\n"  // A0A1
            "vld1.32 {d1},  [%[inptr1]]!\n"  // B0B1
            "vld1.32 {d2},  [%[inptr2]]!\n"  // C0C1
            "vld1.32 {d3},  [%[inptr3]]!\n"  // D0D1
            "vtrn.32 d0, d1\n"               // A0B0 A1B1
            "vtrn.32 d2, d3\n"               // C0D0 C1D1
            "vst1.32 {d0},  [%[outptr]]!\n"
            "vst1.32 {d2},  [%[outptr]]!\n"
            "vst1.32 {d1},  [%[outptr]]!\n"
            "vst1.32 {d3},  [%[outptr]]!\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr), [stride] "+r"(stride)
            :
            : "d0", "d1", "d2", "d3", "memory");
}

template <typename T>
static inline void transpose_6x4_1_b(const T*& inptr0, const T*& inptr1,
                                     const T*& inptr2, const T*& inptr3,
                                     T* outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_6x4_1_b only support uint8_t and int8_t");
    asm volatile(
            "vld1.8 {d0},  [%[inptr0]]\n"  // A0A1A2A3A4A5 A6A7
            "vld1.8 {d1},  [%[inptr1]]\n"  // B0B1B2B3B4B5 B6B7
            "vld1.8 {d2},  [%[inptr2]]\n"  // C0C1C2C3C4C5 C6C7
            "vld1.8 {d3},  [%[inptr3]]\n"  // D0D1D2D3D4D5 D6D7
            "vtrn.8 d0, d1\n"              // A0B0A2B2A4B4A6B6 A1B1A3B3A5B5A7B7
            "vtrn.8 d2, d3\n"              // C0D0C2D2C4D4C6D6 C1D1C3D3C5D5C7D7

            "add %[inptr0],%[inptr0],#6  \n"
            "add %[inptr1],%[inptr1],#6  \n"
            "add %[inptr2],%[inptr2],#6  \n"
            "add %[inptr3],%[inptr3],#6  \n"

            "vtrn.16 d0, d2\n"  // A0B0 C0D0 A4B4 C4D4---A2B2 C2D2 A6B6 C6D6
            "vtrn.16 d1, d3\n"  // A1B1 C1D1 A5B5 C5D5---A3B3 C3D3 A7B7 C7D7

            "vst1.32 {d0[0]},[%[outptr]]!\n"
            "vst1.32 {d1[0]},[%[outptr]]!\n"

            "vst1.32 {d2[0]},[%[outptr]]!\n"
            "vst1.32 {d3[0]},[%[outptr]]!\n"

            "vst1.32 {d0[1]},[%[outptr]]!\n"
            "vst1.32 {d1[1]},[%[outptr]]!\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "memory");
}

template <typename T>
static inline void transpose_4x4_1_b(const T*& inptr0, const T*& inptr1,
                                     const T*& inptr2, const T*& inptr3,
                                     T* outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_4x4_1_b only support uint8_t and int8_t");
    asm volatile(
            "vld1.8 {d0},  [%[inptr0]]\n"  // A0A1A2A3A4A5 A6A7
            "vld1.8 {d1},  [%[inptr1]]\n"  // B0B1B2B3B4B5 B6B7
            "vld1.8 {d2},  [%[inptr2]]\n"  // C0C1C2C3C4C5 C6C7
            "vld1.8 {d3},  [%[inptr3]]\n"  // D0D1D2D3D4D5 D6D7
            "vtrn.8 d0, d1\n"              // A0B0A2B2A4B4A6B6 A1B1A3B3A5B5A7B7
            "vtrn.8 d2, d3\n"              // C0D0C2D2C4D4C6D6 C1D1C3D3C5D5C7D7

            "add %[inptr0],%[inptr0],#4  \n"
            "add %[inptr1],%[inptr1],#4  \n"
            "add %[inptr2],%[inptr2],#4  \n"
            "add %[inptr3],%[inptr3],#4  \n"

            "vtrn.16 d0, d2\n"  // A0B0 C0D0 A4B4 C4D4---A2B2 C2D2 A6B6 C6D6
            "vtrn.16 d1, d3\n"  // A1B1 C1D1 A5B5 C5D5---A3B3 C3D3 A7B7 C7D7

            "vst1.32 {d0[0]},[%[outptr]]!\n"
            "vst1.32 {d1[0]},[%[outptr]]!\n"

            "vst1.32 {d2[0]},[%[outptr]]!\n"
            "vst1.32 {d3[0]},[%[outptr]]!\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "memory");
}

template <typename T>
static inline void transpose_1x12_4_s(const T*& inptr0, T* outptr) {
    static_assert(sizeof(T) == 4,
                  "transpose_1x12_4_s only support sizeof(T) == 4");

    asm volatile(
            "vld4.32 {d0-d3},  [%[inptr0]]!\n"
            "vld4.32 {d4-d7},  [%[inptr0]]!\n"
            "vld4.32 {d8-d11},  [%[inptr0]]!\n"
            "vld4.32 {d12-d15},  [%[inptr0]]!\n"
            "vld4.32 {d16-d19},  [%[inptr0]]!\n"
            "vld4.32 {d20-d23},  [%[inptr0]]!\n"
            "vswp d1, d4\n"
            "vswp d3, d6\n"
            "vswp d9, d12\n"
            "vswp d11, d14\n"
            "vswp d17, d20\n"
            "vswp d19, d22\n"

            "vst1.32 {d0-d1}, [%[outptr]]! \n"
            "vst1.32 {d8-d9}, [%[outptr]]! \n"
            "vst1.32 {d16-d17}, [%[outptr]]! \n"
            "vst1.32 {d4-d5}, [%[outptr]]! \n"
            "vst1.32 {d12-d13}, [%[outptr]]! \n"
            "vst1.32 {d20-d21}, [%[outptr]]! \n"
            "vst1.32 {d2-d3}, [%[outptr]]! \n"
            "vst1.32 {d10-d11}, [%[outptr]]! \n"
            "vst1.32 {d18-d19}, [%[outptr]]! \n"
            "vst1.32 {d6-d7}, [%[outptr]]! \n"
            "vst1.32 {d14-d15}, [%[outptr]]! \n"
            "vst1.32 {d22-d23}, [%[outptr]]! \n"
            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
              "q11", "memory");
}

template <typename T>
static inline void transpose_1x4_4_s(const T*& inptr0, T* outptr) {
    static_assert(sizeof(T) == 4,
                  "transpose_1x4_4_s only support sizeof(T) == 4");
    asm volatile(
            "vld4.32 {d0-d3},  [%[inptr0]]!\n"
            "vld4.32 {d4-d7},  [%[inptr0]]!\n"
            "vswp d1, d4\n"
            "vswp d3, d6\n"
            "vst1.32 {d0-d1}, [%[outptr]]! \n"
            "vst1.32 {d4-d5}, [%[outptr]]! \n"
            "vst1.32 {d2-d3}, [%[outptr]]! \n"
            "vst1.32 {d6-d7}, [%[outptr]]! \n"
            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "q0", "q1", "q2", "q3", "memory");
}

template <typename T>
static inline void transpose_4(const T*& inptr0, const T*& inptr1,
                               const T*& inptr2, const T*& inptr3, T* outptr,
                               int interleave, int size, T val = 0) {
    megdnn_assert(size <= interleave);
    int i = 0;
    for (; i < size; i++) {
        *outptr++ = *inptr0++;
        *outptr++ = *inptr1++;
        *outptr++ = *inptr2++;
        *outptr++ = *inptr3++;
    }
    for (; i < interleave; i++) {
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
    }
}

template <typename T>
static inline void transpose_8(const T*& inptr0, const T*& inptr1,
                               const T*& inptr2, const T*& inptr3,
                               const T*& inptr4, const T*& inptr5,
                               const T*& inptr6, const T*& inptr7, T* outptr,
                               int interleave, int size, T val = 0) {
    megdnn_assert(size <= interleave);
    int i = 0;
    for (; i < size; i++) {
        *outptr++ = *inptr0++;
        *outptr++ = *inptr1++;
        *outptr++ = *inptr2++;
        *outptr++ = *inptr3++;
        *outptr++ = *inptr4++;
        *outptr++ = *inptr5++;
        *outptr++ = *inptr6++;
        *outptr++ = *inptr7++;
    }
    for (; i < interleave; i++) {
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
    }
}

template <typename T>
static inline void transpose_4x1(const T*& inptr0, const T*& inptr1,
                                 const T*& inptr2, const T*& inptr3,
                                 T*& outptr) {
    *outptr++ = *inptr0++;
    *outptr++ = *inptr1++;
    *outptr++ = *inptr2++;
    *outptr++ = *inptr3++;
}

template <typename T>
static inline void transpose_12x1(const T*& inptr0, const T*& inptr1,
                                  const T*& inptr2, const T*& inptr3,
                                  const T*& inptr4, const T*& inptr5,
                                  const T*& inptr6, const T*& inptr7,
                                  const T*& inptr8, const T*& inptr9,
                                  const T*& inptr10, const T*& inptr11,
                                  T*& outptr) {
    *outptr++ = *inptr0++;
    *outptr++ = *inptr1++;
    *outptr++ = *inptr2++;
    *outptr++ = *inptr3++;
    *outptr++ = *inptr4++;
    *outptr++ = *inptr5++;
    *outptr++ = *inptr6++;
    *outptr++ = *inptr7++;
    *outptr++ = *inptr8++;
    *outptr++ = *inptr9++;
    *outptr++ = *inptr10++;
    *outptr++ = *inptr11++;
}

/***********************************Transpose interleave *************/
//! pack form {1, 4(icb), 4(ic), 4(oc)} to {1, 1, 4(oc), 16(ic)}
template <typename T>
static inline void transpose_interleave_4x4_4_b(const T*& inptr0,
                                                const T*& inptr1,
                                                const T*& inptr2,
                                                const T*& inptr3, T* outptr,
                                                int stride = 64) {
    static_assert(sizeof(T) == 1,
                  "transpose_interleave_4x4_4_b only support sizeof(T) == 1");

    asm volatile(
            "add r1, %[outptr], %[stride]\n"
            "vld4.8 {d0-d3},[%[inptr0]]!\n"
            "vld4.8 {d4-d7},[%[inptr0]]!\n"
            "add r2, r1, %[stride]\n"
            "vld4.8 {d8-d11},[%[inptr1]]!\n"
            "vld4.8 {d12-d15},[%[inptr1]]!\n"
            "vld4.8 {d16-d19},[%[inptr2]]!\n"
            "add r3, r2, %[stride]\n"
            "vld4.8 {d20-d23},[%[inptr2]]!\n"
            "vld4.8 {d24-d27},[%[inptr3]]!\n"
            "vld4.8 {d28-d31},[%[inptr3]]!\n"

            "vst1.8 d0, [%[outptr]]!\n"
            "vst1.8 d4, [%[outptr]]!\n"
            "vst1.8 d1, [%[outptr]]!\n"
            "vst1.8 d5, [%[outptr]]!\n"
            "vst1.8 d2, [%[outptr]]!\n"
            "vst1.8 d6, [%[outptr]]!\n"
            "vst1.8 d3, [%[outptr]]!\n"
            "vst1.8 d7, [%[outptr]]!\n"

            "vst1.8 d8, [r1]!\n"
            "vst1.8 d12,[r1]!\n"
            "vst1.8 d9, [r1]!\n"
            "vst1.8 d13,[r1]!\n"
            "vst1.8 d10,[r1]!\n"
            "vst1.8 d14,[r1]!\n"
            "vst1.8 d11,[r1]!\n"
            "vst1.8 d15,[r1]!\n"

            "vst1.8 d16,[r2]!\n"
            "vst1.8 d20,[r2]!\n"
            "vst1.8 d17,[r2]!\n"
            "vst1.8 d21,[r2]!\n"
            "vst1.8 d18,[r2]!\n"
            "vst1.8 d22,[r2]!\n"
            "vst1.8 d19,[r2]!\n"
            "vst1.8 d23,[r2]!\n"

            "vst1.8 d24,[r3]!\n"
            "vst1.8 d28,[r3]!\n"
            "vst1.8 d25,[r3]!\n"
            "vst1.8 d29,[r3]!\n"
            "vst1.8 d26,[r3]!\n"
            "vst1.8 d30,[r3]!\n"
            "vst1.8 d27,[r3]!\n"
            "vst1.8 d31,[r3]!\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr), [stride] "+r"(stride)
            :
            : "r1", "r2", "r3", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
              "q8", "q9", "q10", "q11", "q12", "q14", "q15", "memory");
}

template <typename T>
static inline void transpose_interleave_1x4_4_b(const T*& inptr0, T* outptr,
                                                int stride = 64) {
    static_assert(sizeof(T) == 1,
                  "transpose_interleave_1x4_4_b only support sizeof(T) == 1");

    asm volatile(
            "vld4.8 {d0-d3},[%[inptr0]]!\n"
            "vld4.8 {d4-d7},[%[inptr0]]!\n"

            "vst1.8 d0, [%[outptr]]!\n"
            "vst1.8 d4, [%[outptr]]!\n"
            "vst1.8 d1, [%[outptr]]!\n"
            "vst1.8 d5, [%[outptr]]!\n"
            "vst1.8 d2, [%[outptr]]!\n"
            "vst1.8 d6, [%[outptr]]!\n"
            "vst1.8 d3, [%[outptr]]!\n"
            "vst1.8 d7, [%[outptr]]!\n"
            :
            [inptr0] "+r"(inptr0), [outptr] "+r"(outptr), [stride] "+r"(stride)
            :
            : "q0", "q1", "q2", "q3", "memory");
}

static inline void interleave_4x4_8x4_s8_s16(const int8_t* inptr0,
                                             const int8_t* inptr1,
                                             int16_t* outptr) {
    int8x16_t row0 = vld1q_s8(inptr0);
    int16x8_t row0_01 = vmovl_low_s8(row0);
    int16x8_t row0_23 = vmovl_high_s8(row0);
    int16x4_t row0_0 = vget_low_s16(row0_01);
    int16x4_t row0_1 = vget_high_s16(row0_01);
    int16x4_t row0_2 = vget_low_s16(row0_23);
    int16x4_t row0_3 = vget_high_s16(row0_23);

    int8x16_t row1 = vld1q_s8(inptr1);
    int16x8_t row1_01 = vmovl_low_s8(row1);
    int16x8_t row1_23 = vmovl_high_s8(row1);
    int16x4_t row1_0 = vget_low_s16(row1_01);
    int16x4_t row1_1 = vget_high_s16(row1_01);
    int16x4_t row1_2 = vget_low_s16(row1_23);
    int16x4_t row1_3 = vget_high_s16(row1_23);

    vst1_s16(outptr, row0_0);
    vst1_s16(outptr + 1 * 4, row1_0);
    vst1_s16(outptr + 2 * 4, row0_1);
    vst1_s16(outptr + 3 * 4, row1_1);
    vst1_s16(outptr + 4 * 4, row0_2);
    vst1_s16(outptr + 5 * 4, row1_2);
    vst1_s16(outptr + 6 * 4, row0_3);
    vst1_s16(outptr + 7 * 4, row1_3);
};

static inline void transpos_8x4_int8(const int8_t* inptr0, int8_t* outptr) {
    int8x8x4_t input = vld4_s8(inptr0);
    vst1_s8(outptr, input.val[0]);
    vst1_s8(outptr + 1 * 8, input.val[1]);
    vst1_s8(outptr + 2 * 8, input.val[2]);
    vst1_s8(outptr + 3 * 8, input.val[3]);
}
static inline void memcpy_s8_s16(const int8_t* inptr, int16_t* outptr,
                                 int count) {
    for (; count >= 32; count -= 32) {
        int8x8_t in0 = vld1_s8(inptr);
        int8x8_t in1 = vld1_s8(inptr + 1 * 8);
        int8x8_t in2 = vld1_s8(inptr + 2 * 8);
        int8x8_t in3 = vld1_s8(inptr + 3 * 8);
        vst1q_s16(outptr, vmovl_s8(in0));
        vst1q_s16(outptr + 1 * 8, vmovl_s8(in1));
        vst1q_s16(outptr + 2 * 8, vmovl_s8(in2));
        vst1q_s16(outptr + 3 * 8, vmovl_s8(in3));
        inptr += 32;
        outptr += 32;
    }
    for (; count >= 8; count -= 8) {
        int8x8_t in0 = vld1_s8(inptr);
        vst1q_s16(outptr, vmovl_s8(in0));
        inptr += 8;
        outptr += 8;
    }
    for (; count > 0; --count) {
        *outptr++ = (int16_t)(*inptr++);
    }
}

}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
