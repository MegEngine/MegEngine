#pragma once

#include "src/fallback/general_intrinsic/gi_float.h"

namespace megdnn {
namespace matmul {
namespace fallback {

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

template <typename T>
static GI_FORCEINLINE void interleave_4x4_1_s(
        const T*& inptr0, const T*& inptr1, const T*& inptr2, const T*& inptr3,
        T*& outptr) {
    static_assert(sizeof(T) == 4, "interleave_4x4_1_s only support sizeof(T) == 4");
    GI_FLOAT32_t d0d1 = GiLoadFloat32(inptr0);
    GI_FLOAT32_t d2d3 = GiLoadFloat32(inptr1);
    GI_FLOAT32_t d4d5 = GiLoadFloat32(inptr2);
    GI_FLOAT32_t d6d7 = GiLoadFloat32(inptr3);
    inptr0 += 4;
    inptr1 += 4;
    inptr2 += 4;
    inptr3 += 4;

    GiStoreFloat32(outptr, d0d1);
    outptr += 4;
    GiStoreFloat32(outptr, d2d3);
    outptr += 4;
    GiStoreFloat32(outptr, d4d5);
    outptr += 4;
    GiStoreFloat32(outptr, d6d7);
    outptr += 4;
}

template <typename T>
static GI_FORCEINLINE void interleave_4x12_1_s(
        const T*& inptr0, const T*& inptr1, const T*& inptr2, const T*& inptr3,
        T*& outptr) {
    static_assert(sizeof(T) == 4, "interleave_4x12_1_s only support sizeof(T) == 4");
    GI_FLOAT32_t d0d1 = GiLoadFloat32(inptr0);
    inptr0 += 4;
    GI_FLOAT32_t d2d3 = GiLoadFloat32(inptr0);
    inptr0 += 4;
    GI_FLOAT32_t d4d5 = GiLoadFloat32(inptr0);
    inptr0 += 4;

    GI_FLOAT32_t d6d7 = GiLoadFloat32(inptr1);
    inptr1 += 4;
    GI_FLOAT32_t d8d9 = GiLoadFloat32(inptr1);
    inptr1 += 4;
    GI_FLOAT32_t d10d11 = GiLoadFloat32(inptr1);
    inptr1 += 4;

    GI_FLOAT32_t d12d13 = GiLoadFloat32(inptr2);
    inptr2 += 4;
    GI_FLOAT32_t d14d15 = GiLoadFloat32(inptr2);
    inptr2 += 4;
    GI_FLOAT32_t d16d17 = GiLoadFloat32(inptr2);
    inptr2 += 4;

    GI_FLOAT32_t d18d19 = GiLoadFloat32(inptr3);
    inptr3 += 4;
    GI_FLOAT32_t d20d21 = GiLoadFloat32(inptr3);
    inptr3 += 4;
    GI_FLOAT32_t d22d23 = GiLoadFloat32(inptr3);
    inptr3 += 4;

    GiStoreFloat32(outptr, d0d1);
    outptr += 4;
    GiStoreFloat32(outptr, d2d3);
    outptr += 4;
    GiStoreFloat32(outptr, d4d5);
    outptr += 4;
    GiStoreFloat32(outptr, d6d7);
    outptr += 4;
    GiStoreFloat32(outptr, d8d9);
    outptr += 4;
    GiStoreFloat32(outptr, d10d11);
    outptr += 4;
    GiStoreFloat32(outptr, d12d13);
    outptr += 4;
    GiStoreFloat32(outptr, d14d15);
    outptr += 4;
    GiStoreFloat32(outptr, d16d17);
    outptr += 4;
    GiStoreFloat32(outptr, d18d19);
    outptr += 4;
    GiStoreFloat32(outptr, d20d21);
    outptr += 4;
    GiStoreFloat32(outptr, d22d23);
    outptr += 4;
}

template <typename T>
static GI_FORCEINLINE void interleave_1x12_1_s(const T*& inptr0, T*& outptr) {
    static_assert(sizeof(T) == 4, "interleave_1x12_1_s only support sizeof(T) == 4");
    GI_FLOAT32_t d0d1 = GiLoadFloat32(inptr0);
    inptr0 += 4;
    GI_FLOAT32_t d2d3 = GiLoadFloat32(inptr0);
    inptr0 += 4;
    GI_FLOAT32_t d4d5 = GiLoadFloat32(inptr0);
    inptr0 += 4;

    GiStoreFloat32(outptr, d0d1);
    outptr += 4;
    GiStoreFloat32(outptr, d2d3);
    outptr += 4;
    GiStoreFloat32(outptr, d4d5);
    outptr += 4;
}

template <typename T>
static GI_FORCEINLINE void interleave_1x4_1_s(const T*& inptr0, T*& outptr) {
    static_assert(sizeof(T) == 4, "interleave_1x4_1_s only support sizeof(T) == 4");
    GI_FLOAT32_t d0d1 = GiLoadFloat32(inptr0);
    inptr0 += 4;

    GiStoreFloat32(outptr, d0d1);
    outptr += 4;
}

template <typename T>
static GI_FORCEINLINE void interleave_helper(
        const T*& inptr, T*& outptr, int unroll_k, int ksize, T val = 0) {
    int k = 0;
    for (; k < ksize; k++) {
        *outptr++ = *inptr++;
    }
    for (; k < unroll_k; k++) {
        *outptr++ = val;
    }
}

template <typename T>
static GI_FORCEINLINE void interleave_1(
        const T*& inptr0, T*& outptr, int unroll_k, int ksize, T val = 0) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        interleave_helper(inptr0, outptr, unroll_k, size, val);
    }
}

template <typename T>
static GI_FORCEINLINE void interleave_4(
        const T*& inptr0, const T*& inptr1, const T*& inptr2, const T*& inptr3,
        T*& outptr, int unroll_k, int ksize, T val = 0) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        interleave_helper(inptr0, outptr, unroll_k, size, val);
        interleave_helper(inptr1, outptr, unroll_k, size, val);
        interleave_helper(inptr2, outptr, unroll_k, size, val);
        interleave_helper(inptr3, outptr, unroll_k, size, val);
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
static GI_FORCEINLINE void transpose_4x4_1_s(
        const T*& inptr0, const T*& inptr1, const T*& inptr2, const T*& inptr3,
        T*& outptr, int stride = 16) {
    static_assert(sizeof(T) == 4, "transpose_4x4_1_s only support sizeof(T) == 4");

    stride = stride / sizeof(float);
    stride -= 2;
    GI_FLOAT32_t d0d1 = GiLoadFloat32(inptr0);
    GI_FLOAT32_t d2d3 = GiLoadFloat32(inptr1);
    GI_FLOAT32_t d4d5 = GiLoadFloat32(inptr2);
    GI_FLOAT32_t d6d7 = GiLoadFloat32(inptr3);
    inptr0 += 4;
    inptr1 += 4;
    inptr2 += 4;
    inptr3 += 4;

    GI_FLOAT32_V2_t q0q1 = GiZipqFloat32(d0d1, d2d3);
    GI_FLOAT32_V2_t q2q3 = GiZipqFloat32(d4d5, d6d7);

    GiSt1Float32(outptr, GiGetLowFloat32(GiGetSubVectorFloat32V2(q0q1, 0)));
    outptr += 2;
    GiSt1Float32(outptr, GiGetLowFloat32(GiGetSubVectorFloat32V2(q2q3, 0)));
    outptr += stride;

    GiSt1Float32(outptr, GiGetHighFloat32(GiGetSubVectorFloat32V2(q0q1, 0)));
    outptr += 2;
    GiSt1Float32(outptr, GiGetHighFloat32(GiGetSubVectorFloat32V2(q2q3, 0)));
    outptr += stride;

    GiSt1Float32(outptr, GiGetLowFloat32(GiGetSubVectorFloat32V2(q0q1, 1)));
    outptr += 2;
    GiSt1Float32(outptr, GiGetLowFloat32(GiGetSubVectorFloat32V2(q2q3, 1)));
    outptr += stride;

    GiSt1Float32(outptr, GiGetHighFloat32(GiGetSubVectorFloat32V2(q0q1, 1)));
    outptr += 2;
    GiSt1Float32(outptr, GiGetHighFloat32(GiGetSubVectorFloat32V2(q2q3, 1)));
    outptr += stride;
}

}  // namespace fallback
}  // namespace matmul
}  // namespace megdnn

// vim: syntax=cpp.doxygen
