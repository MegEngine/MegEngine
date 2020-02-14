/**
 * \file dnn/src/x86/simd_macro/sse_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include <stdint.h>
#include <xmmintrin.h>  // SSE
#include "src/x86/simd_macro/sse_helper_typedef.h"

#define MEGDNN_SIMD_NAME SSE
#define MEGDNN_SIMD_TARGET sse
#define MEGDNN_SIMD_ATTRIBUTE_TARGET MEGDNN_ATTRIBUTE_TARGET("sse")
#define MEGDNN_SIMD_LAMBDA_ATTRIBUTE_TARGET \
    MEGDNN_LAMBDA_ATTRIBUTE_TARGET("sse")
#define MEGDNN_SIMD_WIDTH 4
#define MEGDNN_SIMD_TYPE __m128
#define MEGDNN_SIMD_LOADU(addr) _mm_loadu_ps(addr)
#define MEGDNN_SIMD_STOREU(addr, reg) _mm_storeu_ps(addr, reg)
#define MEGDNN_SIMD_SETZERO() _mm_setzero_ps()
#define MEGDNN_SIMD_SET1(num) _mm_set1_ps(num)
#define MEGDNN_SIMD_FMADD(a, b, c) _mm_add_ps(c, _mm_mul_ps(a, b))
#define MEGDNN_SIMD_MAX(a, b) _mm_max_ps(a, b)
#define MEGDNN_SIMD_UZP(s0, s1, d0, d1)                       \
    do {                                                      \
        d0 = _mm_shuffle_ps(s0, s1, _MM_SHUFFLE(2, 0, 2, 0)); \
        d1 = _mm_shuffle_ps(s0, s1, _MM_SHUFFLE(3, 1, 3, 1)); \
    } while (0)
// I cannot find a good way to perform UZP for 256-bit SIMD.

#define MEGDNN_SIMD_TYPE2 float32x4x2_t

#define _INSERTPS_NDX(srcField, dstField) \
    (((srcField) << 6) | ((dstField) << 4))
#define _M64(out, inp) _mm_storel_epi64((__m128i*)&(out), inp)
#define _pM128i(a) _mm_loadl_epi64((__m128i*)&(a))
#define _pM128(a) _mm_castsi128_ps(_pM128i(a))
#define _M128i(a) _mm_castps_si128(a)
#define _M128(a) _mm_castsi128_ps(a)
#define _M64f(out, inp) out.m64_i64[0] = _mm_cvtsi128_si64(_M128i(inp));

#define MEGDNN_SIMD_LOAD2(addr)                                              \
    ({                                                                       \
        float32x4x2_t v;                                                     \
        v.val[0] = _mm_loadu_ps(addr);                                       \
        v.val[1] = _mm_loadu_ps(addr + 4);                                   \
        float32x4x2_t ret__;                                                 \
        ret__.val[0] =                                                       \
                _mm_shuffle_ps(v.val[0], v.val[1], _MM_SHUFFLE(2, 0, 2, 0)); \
        ret__.val[1] =                                                       \
                _mm_shuffle_ps(v.val[0], v.val[1], _MM_SHUFFLE(3, 1, 3, 1)); \
        ret__;                                                               \
    })

#define MEGDNN_SIMD_EXT(a, b, c)                                   \
    ({                                                             \
        auto tmp__ = _mm_alignr_epi8(_M128i(b), _M128i(a), c * 4); \
        auto ret__ = _mm_castsi128_ps(tmp__);                      \
        ret__;                                                     \
    })

#define MEGDNN_SIMD_MUL(a, b) _mm_mul_ps(a, b)
#define MEGDNN_SIMD_SET_LANE(a, b, c)                         \
    ({                                                        \
        __m128 ret__ = _mm_set1_ps(a);                        \
        ret__ = _mm_insert_ps(b, ret__, _INSERTPS_NDX(0, c)); \
        ret__;                                                \
    })

#define MEGDNN_SIMD_FMA_LANE(a, b, c, d)                             \
    ({                                                               \
        int32_t tmp__ = _mm_extract_ps(c, d);                        \
        auto ret__ = _mm_set1_ps(*reinterpret_cast<float*>(&tmp__)); \
        ret__ = _mm_add_ps(a, _mm_mul_ps(b, ret__));                 \
        ret__;                                                       \
    })

#define MEGDNN_SIMD_ADD(a, b) _mm_add_ps(a, b)
#define MEGDNN_SIMD_SUB(a, b) _mm_sub_ps(a, b)
