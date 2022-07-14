#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"

#ifdef __GNUC__
#ifndef __has_warning
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#else
#if __has_warning("-Wmaybe-uninitialized")
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#endif
#endif

#include "src/fallback/matrix_mul/generic_strategy.h"
#include "src/fallback/matrix_mul/gi/fp32/common.h"

using namespace megdnn;
using namespace matmul::fallback;

namespace {

#undef PREFER_VF
#if defined(GI_TARGET_X86) || defined(GI_RVV_INTRINSICS)
#define PREFER_VF
#endif

#if defined(PREFER_VF)
#define MLA(a, b, c, d) GiMultiplyAddScalarFloat32(a, b, *(c + d))
#else
#define MLA(a, b, c, d) GiSimdFmaLane(a, b, c, d)
#endif

void kern_4x12(
        const float* packA, const float* packB, int K, float* output, int LDC,
        bool is_first_k, int m_remain) {
    const float* a_ptr = packA;
    const float* b_ptr = packB;
    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;

    float* r0 = output;
    float* r1 = r0 + LDC;
    float* r2 = r1 + LDC;
    float* r3 = r2 + LDC;

#if defined(PREFER_VF)
    const float* d0d1;
#else
    GI_FLOAT32_t d0d1;
#endif
    GI_FLOAT32_t d2d3, d4d5, d6d7, d8d9, d10d11, d12d13, d14d15, d16d17, d18d19, d20d21,
            d22d23, d24d25, d26d27, d28d29, d30d31;

    if (is_first_k) {
        d8d9 = GiBroadcastFloat32(0.0f);
        d10d11 = GiBroadcastFloat32(0.0f);
        d12d13 = GiBroadcastFloat32(0.0f);
        d14d15 = GiBroadcastFloat32(0.0f);
        d16d17 = GiBroadcastFloat32(0.0f);
        d18d19 = GiBroadcastFloat32(0.0f);
        d20d21 = GiBroadcastFloat32(0.0f);
        d22d23 = GiBroadcastFloat32(0.0f);
        d24d25 = GiBroadcastFloat32(0.0f);
        d26d27 = GiBroadcastFloat32(0.0f);
        d28d29 = GiBroadcastFloat32(0.0f);
        d30d31 = GiBroadcastFloat32(0.0f);
    } else {
        if (m_remain == 4) {
            d8d9 = GiLoadFloat32(r0);
            d10d11 = GiLoadFloat32(r0 + 4);
            d12d13 = GiLoadFloat32(r0 + 8);

            d14d15 = GiLoadFloat32(r1);
            d16d17 = GiLoadFloat32(r1 + 4);
            d18d19 = GiLoadFloat32(r1 + 8);

            d20d21 = GiLoadFloat32(r2);
            d22d23 = GiLoadFloat32(r2 + 4);
            d24d25 = GiLoadFloat32(r2 + 8);

            d26d27 = GiLoadFloat32(r3);
            d28d29 = GiLoadFloat32(r3 + 4);
            d30d31 = GiLoadFloat32(r3 + 8);
        } else if (m_remain == 3) {
            d8d9 = GiLoadFloat32(r0);
            d10d11 = GiLoadFloat32(r0 + 4);
            d12d13 = GiLoadFloat32(r0 + 8);

            d14d15 = GiLoadFloat32(r1);
            d16d17 = GiLoadFloat32(r1 + 4);
            d18d19 = GiLoadFloat32(r1 + 8);

            d20d21 = GiLoadFloat32(r2);
            d22d23 = GiLoadFloat32(r2 + 4);
            d24d25 = GiLoadFloat32(r2 + 8);
        } else if (m_remain == 2) {
            d8d9 = GiLoadFloat32(r0);
            d10d11 = GiLoadFloat32(r0 + 4);
            d12d13 = GiLoadFloat32(r0 + 8);

            d14d15 = GiLoadFloat32(r1);
            d16d17 = GiLoadFloat32(r1 + 4);
            d18d19 = GiLoadFloat32(r1 + 8);
        } else if (m_remain == 1) {
            d8d9 = GiLoadFloat32(r0);
            d10d11 = GiLoadFloat32(r0 + 4);
            d12d13 = GiLoadFloat32(r0 + 8);
        }
    }
    d2d3 = GiLoadFloat32(b_ptr);
    b_ptr = b_ptr + 4;
    d4d5 = GiLoadFloat32(b_ptr);
    b_ptr = b_ptr + 4;
    d6d7 = GiLoadFloat32(b_ptr);
    b_ptr = b_ptr + 4;

    for (; K > 0; K--) {
#if defined(PREFER_VF)
        d0d1 = a_ptr;
#else
        d0d1 = GiLoadFloat32(a_ptr);
#endif
        a_ptr = a_ptr + 4;

        d8d9 = MLA(d8d9, d2d3, d0d1, 0);
        d10d11 = MLA(d10d11, d4d5, d0d1, 0);
        d12d13 = MLA(d12d13, d6d7, d0d1, 0);
        d14d15 = MLA(d14d15, d2d3, d0d1, 1);
        d16d17 = MLA(d16d17, d4d5, d0d1, 1);
        d18d19 = MLA(d18d19, d6d7, d0d1, 1);
        d20d21 = MLA(d20d21, d2d3, d0d1, 2);
        d22d23 = MLA(d22d23, d4d5, d0d1, 2);
        d24d25 = MLA(d24d25, d6d7, d0d1, 2);
        d26d27 = MLA(d26d27, d2d3, d0d1, 3);
        d28d29 = MLA(d28d29, d4d5, d0d1, 3);
        d30d31 = MLA(d30d31, d6d7, d0d1, 3);

#if defined(PREFER_VF)
        d0d1 = a_ptr;
#else
        d0d1 = GiLoadFloat32(a_ptr);
#endif
        a_ptr = a_ptr + 4;
        d2d3 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d4d5 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;

        d8d9 = MLA(d8d9, d2d3, d0d1, 0);
        d10d11 = MLA(d10d11, d4d5, d0d1, 0);
        d12d13 = MLA(d12d13, d6d7, d0d1, 0);
        d14d15 = MLA(d14d15, d2d3, d0d1, 1);
        d16d17 = MLA(d16d17, d4d5, d0d1, 1);
        d18d19 = MLA(d18d19, d6d7, d0d1, 1);
        d20d21 = MLA(d20d21, d2d3, d0d1, 2);
        d22d23 = MLA(d22d23, d4d5, d0d1, 2);
        d24d25 = MLA(d24d25, d6d7, d0d1, 2);
        d26d27 = MLA(d26d27, d2d3, d0d1, 3);
        d28d29 = MLA(d28d29, d4d5, d0d1, 3);
        d30d31 = MLA(d30d31, d6d7, d0d1, 3);

        d2d3 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d4d5 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
    }

    if (1 == oddk) {
#if defined(PREFER_VF)
        d0d1 = a_ptr;
#else
        d0d1 = GiLoadFloat32(a_ptr);
#endif
        a_ptr = a_ptr + 4;

        d8d9 = MLA(d8d9, d2d3, d0d1, 0);
        d10d11 = MLA(d10d11, d4d5, d0d1, 0);
        d12d13 = MLA(d12d13, d6d7, d0d1, 0);
        d14d15 = MLA(d14d15, d2d3, d0d1, 1);
        d16d17 = MLA(d16d17, d4d5, d0d1, 1);
        d18d19 = MLA(d18d19, d6d7, d0d1, 1);
        d20d21 = MLA(d20d21, d2d3, d0d1, 2);
        d22d23 = MLA(d22d23, d4d5, d0d1, 2);
        d24d25 = MLA(d24d25, d6d7, d0d1, 2);
        d26d27 = MLA(d26d27, d2d3, d0d1, 3);
        d28d29 = MLA(d28d29, d4d5, d0d1, 3);
        d30d31 = MLA(d30d31, d6d7, d0d1, 3);

    } else {
#if defined(PREFER_VF)
        d0d1 = a_ptr;
#else
        d0d1 = GiLoadFloat32(a_ptr);
#endif
        a_ptr = a_ptr + 4;

        d8d9 = MLA(d8d9, d2d3, d0d1, 0);
        d10d11 = MLA(d10d11, d4d5, d0d1, 0);
        d12d13 = MLA(d12d13, d6d7, d0d1, 0);
        d14d15 = MLA(d14d15, d2d3, d0d1, 1);
        d16d17 = MLA(d16d17, d4d5, d0d1, 1);
        d18d19 = MLA(d18d19, d6d7, d0d1, 1);
        d20d21 = MLA(d20d21, d2d3, d0d1, 2);
        d22d23 = MLA(d22d23, d4d5, d0d1, 2);
        d24d25 = MLA(d24d25, d6d7, d0d1, 2);
        d26d27 = MLA(d26d27, d2d3, d0d1, 3);
        d28d29 = MLA(d28d29, d4d5, d0d1, 3);
        d30d31 = MLA(d30d31, d6d7, d0d1, 3);

#if defined(PREFER_VF)
        d0d1 = a_ptr;
#else
        d0d1 = GiLoadFloat32(a_ptr);
#endif
        a_ptr = a_ptr + 4;
        d2d3 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d4d5 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;

        d8d9 = MLA(d8d9, d2d3, d0d1, 0);
        d10d11 = MLA(d10d11, d4d5, d0d1, 0);
        d12d13 = MLA(d12d13, d6d7, d0d1, 0);
        d14d15 = MLA(d14d15, d2d3, d0d1, 1);
        d16d17 = MLA(d16d17, d4d5, d0d1, 1);
        d18d19 = MLA(d18d19, d6d7, d0d1, 1);
        d20d21 = MLA(d20d21, d2d3, d0d1, 2);
        d22d23 = MLA(d22d23, d4d5, d0d1, 2);
        d24d25 = MLA(d24d25, d6d7, d0d1, 2);
        d26d27 = MLA(d26d27, d2d3, d0d1, 3);
        d28d29 = MLA(d28d29, d4d5, d0d1, 3);
        d30d31 = MLA(d30d31, d6d7, d0d1, 3);
    }

    if (m_remain == 4) {
        GiStoreFloat32(r0, d8d9);
        GiStoreFloat32(r0 + 4, d10d11);
        GiStoreFloat32(r0 + 8, d12d13);

        GiStoreFloat32(r1, d14d15);
        GiStoreFloat32(r1 + 4, d16d17);
        GiStoreFloat32(r1 + 8, d18d19);

        GiStoreFloat32(r2, d20d21);
        GiStoreFloat32(r2 + 4, d22d23);
        GiStoreFloat32(r2 + 8, d24d25);

        GiStoreFloat32(r3, d26d27);
        GiStoreFloat32(r3 + 4, d28d29);
        GiStoreFloat32(r3 + 8, d30d31);
    } else if (m_remain == 3) {
        GiStoreFloat32(r0, d8d9);
        GiStoreFloat32(r0 + 4, d10d11);
        GiStoreFloat32(r0 + 8, d12d13);

        GiStoreFloat32(r1, d14d15);
        GiStoreFloat32(r1 + 4, d16d17);
        GiStoreFloat32(r1 + 8, d18d19);

        GiStoreFloat32(r2, d20d21);
        GiStoreFloat32(r2 + 4, d22d23);
        GiStoreFloat32(r2 + 8, d24d25);
    } else if (m_remain == 2) {
        GiStoreFloat32(r0, d8d9);
        GiStoreFloat32(r0 + 4, d10d11);
        GiStoreFloat32(r0 + 8, d12d13);

        GiStoreFloat32(r1, d14d15);
        GiStoreFloat32(r1 + 4, d16d17);
        GiStoreFloat32(r1 + 8, d18d19);
    } else if (m_remain == 1) {
        GiStoreFloat32(r0, d8d9);
        GiStoreFloat32(r0 + 4, d10d11);
        GiStoreFloat32(r0 + 8, d12d13);
    }
}

void kern_4x4(
        const float* packA, const float* packB, int K, float* output, int LDC,
        bool is_first_k, int m_remain, int n_remain) {
    const float* a_ptr = packA;
    const float* b_ptr = packB;
    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;

    float* r0 = output;
    float* r1 = r0 + LDC;
    float* r2 = r1 + LDC;
    float* r3 = r2 + LDC;
    size_t d_size = sizeof(float);

#if defined(PREFER_VF)
    const float* d0d1;
    const float* d2d3;
#else
    GI_FLOAT32_t d0d1, d2d3;
#endif
    GI_FLOAT32_t d4d5, d6d7, d8d9, d10d11, d12d13, d14d15;
    float tmp[4];
    if (is_first_k) {
        d8d9 = GiBroadcastFloat32(0.0f);
        d10d11 = GiBroadcastFloat32(0.0f);
        d12d13 = GiBroadcastFloat32(0.0f);
        d14d15 = GiBroadcastFloat32(0.0f);
    } else {
        if (m_remain == 4) {
            if (n_remain == 4) {
                d8d9 = GiLoadFloat32(r0);
                d10d11 = GiLoadFloat32(r1);
                d12d13 = GiLoadFloat32(r2);
                d14d15 = GiLoadFloat32(r3);
            } else if (n_remain == 3) {
                memcpy(tmp, r0, d_size * 3);
                r0 += 3;
                d8d9 = GiLoadFloat32(tmp);

                memcpy(tmp, r1, d_size * 3);
                r1 += 3;
                d10d11 = GiLoadFloat32(tmp);

                memcpy(tmp, r2, d_size * 3);
                r2 += 3;
                d12d13 = GiLoadFloat32(tmp);

                memcpy(tmp, r3, d_size * 3);
                r3 += 3;
                d14d15 = GiLoadFloat32(tmp);
            } else if (n_remain == 2) {
                memcpy(tmp, r0, d_size * 2);
                r0 += 2;
                d8d9 = GiLoadFloat32(tmp);

                memcpy(tmp, r1, d_size * 2);
                r1 += 2;
                d10d11 = GiLoadFloat32(tmp);

                memcpy(tmp, r2, d_size * 2);
                r2 += 2;
                d12d13 = GiLoadFloat32(tmp);

                memcpy(tmp, r3, d_size * 2);
                r3 += 2;
                d14d15 = GiLoadFloat32(tmp);
            } else if (n_remain == 1) {
                tmp[0] = *r0;
                r0++;
                d8d9 = GiLoadFloat32(tmp);

                tmp[0] = *r1;
                r1++;
                d10d11 = GiLoadFloat32(tmp);

                tmp[0] = *r2;
                r2++;
                d12d13 = GiLoadFloat32(tmp);

                tmp[0] = *r3;
                r3++;
                d14d15 = GiLoadFloat32(tmp);
            }
        } else if (m_remain == 3) {
            if (n_remain == 4) {
                d8d9 = GiLoadFloat32(r0);
                d10d11 = GiLoadFloat32(r1);
                d12d13 = GiLoadFloat32(r2);
            } else if (n_remain == 3) {
                memcpy(tmp, r0, d_size * 3);
                r0 += 3;
                d8d9 = GiLoadFloat32(tmp);

                memcpy(tmp, r1, d_size * 3);
                r1 += 3;
                d10d11 = GiLoadFloat32(tmp);

                memcpy(tmp, r2, d_size * 3);
                r2 += 3;
                d12d13 = GiLoadFloat32(tmp);
            } else if (n_remain == 2) {
                memcpy(tmp, r0, d_size * 2);
                r0 += 2;
                d8d9 = GiLoadFloat32(tmp);

                memcpy(tmp, r1, d_size * 2);
                r1 += 2;
                d10d11 = GiLoadFloat32(tmp);

                memcpy(tmp, r2, d_size * 2);
                r2 += 2;
                d12d13 = GiLoadFloat32(tmp);
            } else if (n_remain == 1) {
                tmp[0] = *r0;
                r0++;
                d8d9 = GiLoadFloat32(tmp);

                tmp[0] = *r1;
                r1++;
                d10d11 = GiLoadFloat32(tmp);

                tmp[0] = *r2;
                r2++;
                d12d13 = GiLoadFloat32(tmp);
            }
        } else if (m_remain == 2) {
            if (n_remain == 4) {
                d8d9 = GiLoadFloat32(r0);
                d10d11 = GiLoadFloat32(r1);
            } else if (n_remain == 3) {
                memcpy(tmp, r0, d_size * 3);
                r0 += 3;
                d8d9 = GiLoadFloat32(tmp);

                memcpy(tmp, r1, d_size * 3);
                r1 += 3;
                d10d11 = GiLoadFloat32(tmp);
            } else if (n_remain == 2) {
                memcpy(tmp, r0, d_size * 2);
                r0 += 2;
                d8d9 = GiLoadFloat32(tmp);

                memcpy(tmp, r1, d_size * 2);
                r1 += 2;
                d10d11 = GiLoadFloat32(tmp);
            } else if (n_remain == 1) {
                tmp[0] = *r0;
                r0++;
                d8d9 = GiLoadFloat32(tmp);

                tmp[0] = *r1;
                r1++;
                d10d11 = GiLoadFloat32(tmp);
            }
        } else if (m_remain == 1) {
            if (n_remain == 4) {
                d8d9 = GiLoadFloat32(r0);
            } else if (n_remain == 3) {
                memcpy(tmp, r0, d_size * 3);
                r0 += 3;
                d8d9 = GiLoadFloat32(tmp);
            } else if (n_remain == 2) {
                memcpy(tmp, r0, d_size * 2);
                r0 += 2;
                d8d9 = GiLoadFloat32(tmp);
            } else if (n_remain == 1) {
                tmp[0] = *r0;
                r0++;
                d8d9 = GiLoadFloat32(tmp);
            }
        }
    }

#if defined(PREFER_VF)
    d0d1 = a_ptr;
#else
    d0d1 = GiLoadFloat32(a_ptr);
#endif
    a_ptr = a_ptr + 4;
    d4d5 = GiLoadFloat32(b_ptr);
    b_ptr = b_ptr + 4;

    for (; K > 0; K--) {
#if defined(PREFER_VF)
        d2d3 = a_ptr;
#else
        d2d3 = GiLoadFloat32(a_ptr);
#endif
        a_ptr = a_ptr + 4;
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;

        d8d9 = MLA(d8d9, d4d5, d0d1, 0);
        d10d11 = MLA(d10d11, d4d5, d0d1, 1);
        d12d13 = MLA(d12d13, d4d5, d0d1, 2);
        d14d15 = MLA(d14d15, d4d5, d0d1, 3);

#if defined(PREFER_VF)
        d0d1 = a_ptr;
#else
        d0d1 = GiLoadFloat32(a_ptr);
#endif
        a_ptr = a_ptr + 4;
        d4d5 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;

        d8d9 = MLA(d8d9, d6d7, d2d3, 0);
        d10d11 = MLA(d10d11, d6d7, d2d3, 1);
        d12d13 = MLA(d12d13, d6d7, d2d3, 2);
        d14d15 = MLA(d14d15, d6d7, d2d3, 3);
    }

    if (1 == oddk) {
        d8d9 = MLA(d8d9, d4d5, d0d1, 0);
        d10d11 = MLA(d10d11, d4d5, d0d1, 1);
        d12d13 = MLA(d12d13, d4d5, d0d1, 2);
        d14d15 = MLA(d14d15, d4d5, d0d1, 3);

    } else {
#if defined(PREFER_VF)
        d2d3 = a_ptr;
#else
        d2d3 = GiLoadFloat32(a_ptr);
#endif
        a_ptr = a_ptr + 4;
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;

        d8d9 = MLA(d8d9, d4d5, d0d1, 0);
        d10d11 = MLA(d10d11, d4d5, d0d1, 1);
        d12d13 = MLA(d12d13, d4d5, d0d1, 2);
        d14d15 = MLA(d14d15, d4d5, d0d1, 3);

        d8d9 = MLA(d8d9, d6d7, d2d3, 0);
        d10d11 = MLA(d10d11, d6d7, d2d3, 1);
        d12d13 = MLA(d12d13, d6d7, d2d3, 2);
        d14d15 = MLA(d14d15, d6d7, d2d3, 3);
    }

    if (m_remain == 4) {
        if (n_remain == 4) {
            GiStoreFloat32(r0, d8d9);
            r0 = r0 + 4;
            GiStoreFloat32(r1, d10d11);
            r1 = r1 + 4;
            GiStoreFloat32(r2, d12d13);
            r2 = r2 + 4;
            GiStoreFloat32(r3, d14d15);
            r3 = r3 + 4;
        } else if (n_remain == 3) {
            GiStoreFloat32(tmp, d8d9);
            memcpy(r0, tmp, d_size * 3);
            r0 += 3;

            GiStoreFloat32(tmp, d10d11);
            memcpy(r1, tmp, d_size * 3);
            r1 += 3;

            GiStoreFloat32(tmp, d12d13);
            memcpy(r2, tmp, d_size * 3);
            r2 += 3;

            GiStoreFloat32(tmp, d14d15);
            memcpy(r3, tmp, d_size * 3);
            r3 += 3;
        } else if (n_remain == 2) {
            GiStoreFloat32(tmp, d8d9);
            memcpy(r0, tmp, d_size * 2);
            r0 += 2;

            GiStoreFloat32(tmp, d10d11);
            memcpy(r1, tmp, d_size * 2);
            r1 += 2;

            GiStoreFloat32(tmp, d12d13);
            memcpy(r2, tmp, d_size * 2);
            r2 += 2;

            GiStoreFloat32(tmp, d14d15);
            memcpy(r3, tmp, d_size * 2);
            r3 += 2;
        } else if (n_remain == 1) {
            GiStoreFloat32(tmp, d8d9);
            *r0 = tmp[0];
            r0++;

            GiStoreFloat32(tmp, d10d11);
            *r1 = tmp[0];
            r1++;

            GiStoreFloat32(tmp, d12d13);
            *r2 = tmp[0];
            r2++;

            GiStoreFloat32(tmp, d14d15);
            *r3 = tmp[0];
            r3++;
        }
    } else if (m_remain == 3) {
        if (n_remain == 4) {
            GiStoreFloat32(r0, d8d9);
            r0 = r0 + 4;
            GiStoreFloat32(r1, d10d11);
            r1 = r1 + 4;
            GiStoreFloat32(r2, d12d13);
            r2 = r2 + 4;
        } else if (n_remain == 3) {
            GiStoreFloat32(tmp, d8d9);
            memcpy(r0, tmp, d_size * 3);
            r0 += 3;

            GiStoreFloat32(tmp, d10d11);
            memcpy(r1, tmp, d_size * 3);
            r1 += 3;

            GiStoreFloat32(tmp, d12d13);
            memcpy(r2, tmp, d_size * 3);
            r2 += 3;
        } else if (n_remain == 2) {
            GiStoreFloat32(tmp, d8d9);
            memcpy(r0, tmp, d_size * 2);
            r0 += 2;

            GiStoreFloat32(tmp, d10d11);
            memcpy(r1, tmp, d_size * 2);
            r1 += 2;

            GiStoreFloat32(tmp, d12d13);
            memcpy(r2, tmp, d_size * 2);
            r2 += 2;
        } else if (n_remain == 1) {
            GiStoreFloat32(tmp, d8d9);
            *r0 = tmp[0];
            r0++;

            GiStoreFloat32(tmp, d10d11);
            *r1 = tmp[0];
            r1++;

            GiStoreFloat32(tmp, d12d13);
            *r2 = tmp[0];
            r2++;
        }
    } else if (m_remain == 2) {
        if (n_remain == 4) {
            GiStoreFloat32(r0, d8d9);
            r0 = r0 + 4;
            GiStoreFloat32(r1, d10d11);
            r1 = r1 + 4;
        } else if (n_remain == 3) {
            GiStoreFloat32(tmp, d8d9);
            memcpy(r0, tmp, d_size * 3);
            r0 += 3;

            GiStoreFloat32(tmp, d10d11);
            memcpy(r1, tmp, d_size * 3);
            r1 += 3;
        } else if (n_remain == 2) {
            GiStoreFloat32(tmp, d8d9);
            memcpy(r0, tmp, d_size * 2);
            r0 += 2;

            GiStoreFloat32(tmp, d10d11);
            memcpy(r1, tmp, d_size * 2);
            r1 += 2;
        } else if (n_remain == 1) {
            GiStoreFloat32(tmp, d8d9);
            *r0 = tmp[0];
            r0++;

            GiStoreFloat32(tmp, d10d11);
            *r1 = tmp[0];
            r1++;
        }
    } else if (m_remain == 1) {
        if (n_remain == 4) {
            GiStoreFloat32(r0, d8d9);
            r0 = r0 + 4;
        } else if (n_remain == 3) {
            GiStoreFloat32(tmp, d8d9);
            memcpy(r0, tmp, d_size * 3);
            r0 += 3;
        } else if (n_remain == 2) {
            GiStoreFloat32(tmp, d8d9);
            memcpy(r0, tmp, d_size * 2);
            r0 += 2;
        } else if (n_remain == 1) {
            GiStoreFloat32(tmp, d8d9);
            *r0 = tmp[0];
            r0++;
        }
    }
}

void gi_sgemm_4x12_pack_A_n(
        float* outptr, const float* inptr, int ldin, int y0, int ymax, int k0,
        int kmax) {
    float zerobuff[4];
    std::memset(zerobuff, 0, sizeof(float) * 4);

    int y = y0;
    for (; y + 3 < ymax; y += 4) {
        const float* inptr0 = inptr + y * ldin + k0;
        const float* inptr1 = inptr0 + ldin;
        const float* inptr2 = inptr1 + ldin;
        const float* inptr3 = inptr2 + ldin;

        int K = (kmax - k0);
        for (; K > 3; K -= 4) {
            transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr);
        }

        interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 1, K);
    }

    for (; y < ymax; y += 4) {
        const float* inptr0 = inptr + y * ldin + k0;
        const float* inptr1 = inptr0 + ldin;
        const float* inptr2 = inptr1 + ldin;
        const float* inptr3 = inptr2 + ldin;

        int K = (kmax - k0);
        for (; K > 3; K -= 4) {
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    /* Everything falls through in here */
                    case 2:
                        inptr1 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 1:
                        inptr2 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr);
        }

        if (K > 0) {
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    /* Everything falls through in here */
                    case 2:
                        inptr1 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 1:
                        inptr2 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 1, K);
        }
    }
}

void gi_sgemm_4x12_pack_A_t(
        float* out, const float* in, int ldin, int x0, int xmax, int k0, int kmax) {
    int ksize = kmax - k0;
    int ksize4 = (ksize << 2);
    float* outptr_base = out;

    int k = k0;
    for (; k + 3 < kmax; k += 4) {
        const float* inptr = in + k * ldin + x0;
        const float* inptr1 = inptr + ldin;
        const float* inptr2 = inptr1 + ldin;
        const float* inptr3 = inptr2 + ldin;

        int x = x0;
        auto outptr = outptr_base;
        for (; x + 4 <= xmax; x += 4) {
            auto outptr_interleave = outptr;
            interleave_4x4_1_s(inptr, inptr1, inptr2, inptr3, outptr_interleave);
            outptr += ksize4;
        }

        if (x < xmax) {
            interleave_4(inptr, inptr1, inptr2, inptr3, outptr, 4, xmax - x);
        }

        outptr_base += 4 * 4;
    }

    for (; k < kmax; k++) {
        const float* inptr = in + k * ldin + x0;
        int x = x0;
        auto outptr = outptr_base;
        for (; x + 4 <= xmax; x += 4) {
            auto outptr_interleave = outptr;
            interleave_1x4_1_s(inptr, outptr_interleave);
            outptr += ksize4;
        }

        if (x < xmax) {
            interleave_1(inptr, outptr, 4, xmax - x);
        }

        outptr_base += 4;
    }
}

void gi_sgemm_4x12_pack_B_n(
        float* out, const float* in, int ldin, int x0, int xmax, int k0, int kmax) {
    int ksize = kmax - k0;
    int ksize12 = ksize * 12;
    int ksize4 = (ksize << 2);
    float* outptr_base = out;
    float* outptr_base4 = outptr_base + (xmax - x0) / 12 * ksize12;

    int k = k0;
    for (; k + 3 < kmax; k += 4) {
        const float* inptr = in + k * ldin + x0;
        const float* inptr1 = inptr + ldin;
        const float* inptr2 = inptr1 + ldin;
        const float* inptr3 = inptr2 + ldin;

        int x = x0;
        auto outptr = outptr_base;
        for (; x + 12 <= xmax; x += 12) {
            auto outptr_interleave = outptr;
            interleave_4x12_1_s(inptr, inptr1, inptr2, inptr3, outptr_interleave);
            outptr += ksize12;
        }
        outptr = outptr_base4;
        for (; x + 4 <= xmax; x += 4) {
            auto outptr_interleave = outptr;
            interleave_4x4_1_s(inptr, inptr1, inptr2, inptr3, outptr_interleave);
            outptr += ksize4;
        }

        if (x < xmax) {
            interleave_4(inptr, inptr1, inptr2, inptr3, outptr, 4, xmax - x);
        }

        outptr_base += 12 * 4;
        outptr_base4 += 4 * 4;
    }

    for (; k < kmax; k++) {
        const float* inptr = in + k * ldin + x0;
        int x = x0;
        auto outptr = outptr_base;
        for (; x + 12 <= xmax; x += 12) {
            auto outptr_interleave = outptr;
            interleave_1x12_1_s(inptr, outptr_interleave);
            outptr += ksize12;
        }
        outptr = outptr_base4;
        for (; x + 4 <= xmax; x += 4) {
            auto outptr_interleave = outptr;
            interleave_1x4_1_s(inptr, outptr_interleave);
            outptr += ksize4;
        }

        if (x < xmax) {
            interleave_1(inptr, outptr, 4, xmax - x);
        }

        outptr_base += 12;
        outptr_base4 += 4;
    }
}

void gi_sgemm_4x12_pack_B_t(
        float* out, const float* in, int ldin, int y0, int ymax, int k0, int kmax) {
    float* outptr = out;
    const float* inptr = in;
    float zerobuff[4];
    std::memset(zerobuff, 0, sizeof(float) * 4);
    int K12 = 12 * (kmax - k0);

    int y = y0;

    for (; y + 12 <= ymax; y += 12) {
        int yi = y;
        for (; yi < y + 12; yi += 4) {
            const float* inptr0 = inptr + yi * ldin + k0;
            const float* inptr1 = inptr0 + ldin;
            const float* inptr2 = inptr1 + ldin;
            const float* inptr3 = inptr2 + ldin;
            float* outptr_inner = outptr + yi - y;

            int x = (kmax - k0);
            for (; x > 3; x -= 4) {
                transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr_inner, 48);
            }
            for (; x > 0; x--) {
                *outptr_inner++ = *inptr0++;
                *outptr_inner++ = *inptr1++;
                *outptr_inner++ = *inptr2++;
                *outptr_inner++ = *inptr3++;
                outptr_inner += 8;
            }
        }
        outptr += K12;
    }

    for (; y < ymax; y += 4) {
        const float* inptr0 = inptr + y * ldin + k0;
        const float* inptr1 = inptr0 + ldin;
        const float* inptr2 = inptr1 + ldin;
        const float* inptr3 = inptr2 + ldin;

        /* Cope with ragged cases by copying from a buffer of zeroes instead
         */
        int x = (kmax - k0);
        for (; x > 3; x -= 4) {
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    /* Everything falls through in here */
                    case 2:
                        inptr1 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 1:
                        inptr2 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }

            transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr);
        }

        if (x > 0) {
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    /* Everything falls through in here */
                    case 2:
                        inptr1 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 1:
                        inptr2 = zerobuff;
                        MEGDNN_FALLTHRU
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        megdnn_assert(0);
                }
            }
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 1, x);
        }
    }
}

#undef MLA
}  // namespace

MEGDNN_REG_GEMM_STRATEGY_IMPL(gi_sgemm_4x12);

void gi_sgemm_4x12::pack_A(
        float* out, const float* in, int ldin, int y0, int ymax, int k0, int kmax,
        bool transpose_A) const {
    if (transpose_A) {
        gi_sgemm_4x12_pack_A_t(out, in, ldin, y0, ymax, k0, kmax);
    } else {
        gi_sgemm_4x12_pack_A_n(out, in, ldin, y0, ymax, k0, kmax);
    }
}

void gi_sgemm_4x12::pack_B(
        float* out, const float* in, int ldin, int x0, int xmax, int k0, int kmax,
        bool transpose_B) const {
    if (transpose_B) {
        gi_sgemm_4x12_pack_B_t(out, in, ldin, x0, xmax, k0, kmax);
    } else {
        gi_sgemm_4x12_pack_B_n(out, in, ldin, x0, xmax, k0, kmax);
    }
}

void gi_sgemm_4x12::kern(
        const float* packA, const float* packB, size_t M, size_t N, size_t K, float* C,
        size_t LDC, bool is_first_k, const float*, float*) const {
    megdnn_assert(
            A_dtype.enumv() == B_dtype.enumv() && A_dtype.enumv() == C_dtype.enumv() &&
            A_dtype.enumv() == DTypeEnum::Float32);
    MEGDNN_MARK_USED_VAR(A_dtype);
    MEGDNN_MARK_USED_VAR(B_dtype);
    MEGDNN_MARK_USED_VAR(C_dtype);

    constexpr size_t A_INTERLEAVE = 4;
    constexpr size_t B_INTERLEAVE = 12;
    const int K12 = K * 12;
    const int K4 = K * 4;

    size_t m = 0;
    for (; m < M; m += A_INTERLEAVE) {
        float* output = C + (m * LDC);

        size_t n = 0;
        const float* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            kern_4x12(
                    packA, cur_packB, K, output, LDC, is_first_k,
                    std::min<size_t>(M - m, 4));
            output += B_INTERLEAVE;
            cur_packB += K12;
        }

        for (; n < N; n += 4) {
            kern_4x4(
                    packA, cur_packB, K, output, LDC, is_first_k,
                    std::min<size_t>(M - m, 4), std::min<size_t>(N - n, 4));
            output += 4;
            cur_packB += K4;
        }

        packA += K4;
    }
}

// vim: syntax=cpp.doxygen
