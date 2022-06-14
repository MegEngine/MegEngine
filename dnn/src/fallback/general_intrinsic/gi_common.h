#pragma once

#include "math.h"
#include "stdint.h"
#include "string.h"

#if defined(_WIN32)
#include <intrin.h>
#include <windows.h>
#else
#if defined(__arm__) || defined(__aarch64__)
#include "src/arm_common/simd_macro/marm_neon.h"
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <immintrin.h>
#endif
#endif

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
#define GI_TARGET_X86
#endif

#if defined(__arm__) || defined(__aarch64__)
#define GI_TARGET_ARM
#endif

#ifdef _WIN32
//! GI stand for general intrinsic
#define _GI_ALIGN_16                           __declspec(align(16))
#define GI_DECLSPEC_ALIGN(variable, alignment) DECLSPEC_ALIGN(alignment) variable
#else
#define _GI_ALIGN_16 __attribute__((aligned(16)))
#define GI_DECLSPEC_ALIGN(variable, alignment) \
    variable __attribute__((aligned(alignment)))
#endif

#if defined(_MSC_VER)
#define GI_FORCEINLINE __forceinline
#else
#define GI_FORCEINLINE __attribute__((always_inline)) inline
#endif

#if defined(_MSC_VER)
#define GI_INTERNAL_DATA extern "C"
#else
#define GI_INTERNAL_DATA extern "C" __attribute((visibility("hidden")))
#endif

#if defined(GI_TARGET_ARM) && defined(__ARM_NEON)
#define GI_NEON_INTRINSICS
#if defined(__aarch64__)
#define GI_NEON64_INTRINSICS
#define GI_NEON32_INTRINSICS
#else
#define GI_NEON32_INTRINSICS
#endif
#elif defined(GI_TARGET_X86)
//#if defined(__FMA__)
//#define GI_FMA_INTRINSICS
//#define GI_AVX2_INTRINSICS
//#define GI_AVX_INTRINSICS
//#elif defined(__AVX2__)
//#define GI_AVX2_INTRINSICS
//#define GI_AVX_INTRINSICS
//#elif defined(__AVX__)
//#define GI_AVX_INTRINSICS
#if defined(__SSE4_2__)
#define GI_SSE42_INTRINSICS
#define GI_SSE2_INTRINSICS
#elif defined(__SSE2__)
#define GI_SSE2_INTRINSICS
#endif
#endif
#if defined(__riscv_vector)
#define GI_RVV_INTRINSICS
#endif

#if defined(GI_TEST_NAIVE)
#undef GI_NEON_INTRINSICS
#undef GI_NEON64_INTRINSICS
#undef GI_NEON32_INTRINSICS
#undef GI_FMA_INTRINSICS
#undef GI_AVX2_INTRINSICS
#undef GI_AVX_INTRINSICS
#undef GI_SSE42_INTRINSICS
#undef GI_SSE2_INTRINSICS
#undef GI_RVV_INTRINSICS
#endif

//! general intrinsic support dynamic length simd, if avx or avx2 the simd
//! length is 256
#if defined(GI_AVX_INTRINSICS) || defined(GI_AVX2_INTRINSICS) || \
        defined(GI_FMA_INTRINSICS)
//! if neon and sse the simd lenght is 128
#define GI_SIMD_LEN      256
#define GI_SIMD_LEN_BYTE 32
#elif defined(GI_NEON_INTRINSICS) || defined(GI_SSE2_INTRINSICS) || \
        defined(GI_SSE42_INTRINSICS)
#define GI_SIMD_LEN      128
#define GI_SIMD_LEN_BYTE 16
#elif defined(GI_RVV_INTRINSICS)
//! TODO: make gi algo usable for other GI_SIMD_LEN/GI_SIMD_LEN_BYTE
#define GI_SIMD_LEN      128
#define GI_SIMD_LEN_BYTE 16
#else
//! if no simd hardware support, the simd is implemented by C, default set to
//! 128
#define GI_SIMD_LEN      128
#define GI_SIMD_LEN_BYTE 16
#endif

#define gi_trap() __builtin_trap()

//! for ci test now
enum GiSimdType {
    GI_UNKNOWN,
    GI_NAIVE,
    GI_AVX,
    GI_SSE42,
    GI_SSE2,
    GI_NEON,
    GI_RVV,
};

#if defined(GI_AVX_INTRINSICS) || defined(GI_AVX2_INTRINSICS) || \
        defined(GI_FMA_INTRINSICS)
#define __gi_simd_type GI_AVX
typedef __m256 GI_FLOAT32_t;
typedef __m256i GI_UINT8_t;
typedef __m256i GI_INT8_t;
typedef __m256i GI_INT16_t;
typedef __m256i GI_INT32_t;
typedef __m256i GI_UINT32_t;
#elif defined(GI_NEON_INTRINSICS)
#define __gi_simd_type GI_NEON
typedef float32x4_t GI_FLOAT32_t;
typedef uint8x16_t GI_UINT8_t;
typedef int8x16_t GI_INT8_t;
typedef int16x8_t GI_INT16_t;
typedef int32x4_t GI_INT32_t;
typedef uint32x4_t GI_UINT32_t;
typedef float32x4x2_t GI_FLOAT32_V2_t;
typedef float32x4x3_t GI_FLOAT32_V3_t;
typedef float32x4x4_t GI_FLOAT32_V4_t;
typedef int32x4x2_t GI_INT32_V2_t;
typedef int32x4x4_t GI_INT32_V4_t;
typedef int16x8x2_t GI_INT16_V2_t;
typedef int8x16x2_t GI_INT8_V2_t;
typedef int64x2_t GI_INT64_t;
#elif defined(GI_SSE2_INTRINSICS) || defined(GI_SSE42_INTRINSICS)

#if defined(GI_SSE42_INTRINSICS)
#define __gi_simd_type GI_SSE42
#elif defined(GI_SSE2_INTRINSICS)
#define __gi_simd_type GI_SSE2
#else
#define __gi_simd_type GI_UNKNOWN
#error "code issue happened!!"
#endif

typedef __m128 GI_FLOAT32_t;
typedef __m128i GI_UINT8_t;
typedef __m128i GI_INT8_t;
typedef __m128i GI_INT16_t;
typedef __m128i GI_INT32_t;
typedef __m128i GI_UINT32_t;
typedef __m128i GI_INT64_t;
#define _SWAP_HI_LOW32                    (2 | (3 << 2) | (0 << 4) | (1 << 6))
#define _INSERTPS_NDX(srcField, dstField) (((srcField) << 6) | ((dstField) << 4))
#define _M64(out, inp)                    _mm_storel_epi64((__m128i*)&(out), inp)
#define _pM128i(a)                        _mm_loadl_epi64((__m128i*)&(a))
#define _pM128(a)                         _mm_castsi128_ps(_pM128i(a))
#define _M128i(a)                         _mm_castps_si128(a)
#define _M128(a)                          _mm_castsi128_ps(a)
#if defined(__x86_64__)
#define _M64f(out, inp) out.m64_i64[0] = _mm_cvtsi128_si64(_M128i(inp));
#else
#define _M64f(out, inp) _mm_storel_epi64((__m128i*)&(out), _M128i(inp))
#endif
#define _SSE_SWITCH16(NAME, a, b, LANE) \
    switch (LANE) {                     \
        case 0:                         \
            return NAME(a b, 0);        \
        case 1:                         \
            return NAME(a b, 1);        \
        case 2:                         \
            return NAME(a b, 2);        \
        case 3:                         \
            return NAME(a b, 3);        \
        case 4:                         \
            return NAME(a b, 4);        \
        case 5:                         \
            return NAME(a b, 5);        \
        case 6:                         \
            return NAME(a b, 6);        \
        case 7:                         \
            return NAME(a b, 7);        \
        case 8:                         \
            return NAME(a b, 8);        \
        case 9:                         \
            return NAME(a b, 9);        \
        case 10:                        \
            return NAME(a b, 10);       \
        case 11:                        \
            return NAME(a b, 11);       \
        case 12:                        \
            return NAME(a b, 12);       \
        case 13:                        \
            return NAME(a b, 13);       \
        case 14:                        \
            return NAME(a b, 14);       \
        case 15:                        \
            return NAME(a b, 15);       \
        default:                        \
            gi_trap();                  \
            return NAME(a b, 0);        \
    }
#if !defined(__SSE3__)
GI_FORCEINLINE __m128i _sse2_mm_alignr_epi8(__m128i b, __m128i a, int imm8) {
    int imm2 = sizeof(__m128i) - imm8;
    return _mm_or_si128(_mm_srli_si128(a, imm8), _mm_slli_si128(b, imm2));
}
#endif

#define _SSE_COMMA ,
GI_FORCEINLINE __m128i _MM_ALIGNR_EPI8(__m128i a, __m128i b, int LANE) {
#if !defined(__SSE3__)
    _SSE_SWITCH16(_sse2_mm_alignr_epi8, a, _SSE_COMMA b, LANE)
#else
    _SSE_SWITCH16(_mm_alignr_epi8, a, _SSE_COMMA b, LANE)
#endif
}
typedef float float32_t;
typedef double float64_t;
typedef union __m64_128 {
    uint64_t m64_u64[1];
    int64_t m64_i64[1];
    float64_t m64_d64[1];
    uint32_t m64_u32[2];
    int32_t m64_i32[2];
    float32_t m64_f32[2];
    int16_t m64_i16[4];
    uint16_t m64_u16[4];
    int8_t m64_i8[8];
    uint8_t m64_u8[8];
} __m64_128;
typedef __m64_128 float32x2_t;

#define return64(a) \
    _M64(res64, a); \
    return res64;
#define return64f(a) \
    _M64f(res64, a); \
    return res64;
#define _sse_vextq_s32(a, b, c)       _MM_ALIGNR_EPI8(b, a, c * 4)
#define _sse_vget_lane_f32(vec, lane) vec.m64_f32[lane]
#elif defined(GI_RVV_INTRINSICS)
#define __gi_simd_type GI_RVV
typedef vfloat32m1_t GI_FLOAT32_t;
typedef vuint8m1_t GI_UINT8_t;
typedef vint8m1_t GI_INT8_t;
typedef vint16m1_t GI_INT16_t;
typedef vint32m1_t GI_INT32_t;
typedef vuint32m1_t GI_UINT32_t;
//! FIXME: nezha D1 do not support vmv.x.s instruct
//! as a workaround, define GI_INT64_t to naive
typedef int64_t GI_INT64_RVV_WORKAROUND_t
        __attribute__((vector_size(GI_SIMD_LEN_BYTE)));
typedef GI_INT64_RVV_WORKAROUND_t GI_INT64_t;
typedef vfloat32m1x2_t GI_FLOAT32_V2_t;
typedef vfloat32m1x3_t GI_FLOAT32_V3_t;
typedef vfloat32m1x4_t GI_FLOAT32_V4_t;
typedef vint32m1x2_t GI_INT32_V2_t;
typedef vint32m1x4_t GI_INT32_V4_t;
typedef vint16m1x2_t GI_INT16_V2_t;
typedef vint8m1x2_t GI_INT8_V2_t;
//! vfloat32mf2_t usable at RVV1.0, now we support 0.7, as
//! a workaround, we use vfloat32m1_t instead
typedef vfloat32m1_t float32x2_t;

#else
#define __gi_simd_type GI_NAIVE
typedef float GI_FLOAT32_t __attribute__((vector_size(GI_SIMD_LEN_BYTE)));
typedef uint8_t GI_UINT8_t __attribute__((vector_size(GI_SIMD_LEN_BYTE)));
typedef int8_t GI_INT8_t __attribute__((vector_size(GI_SIMD_LEN_BYTE)));
typedef int16_t GI_INT16_t __attribute__((vector_size(GI_SIMD_LEN_BYTE)));
typedef int32_t GI_INT32_t __attribute__((vector_size(GI_SIMD_LEN_BYTE)));
typedef uint32_t GI_UINT32_t __attribute__((vector_size(GI_SIMD_LEN_BYTE)));
typedef int64_t GI_INT64_t __attribute__((vector_size(GI_SIMD_LEN_BYTE)));
#if !defined(__arm__) && !defined(__aarch64__) || !defined(__ARM_NEON)
typedef float float32x2_t __attribute__((vector_size(GI_SIMD_LEN_BYTE / 2)));
#endif
typedef float float32_t;
#endif

//! some GI api do not support full GiSimdType
//! for example: GiAbsInt32 do not imp SSE2 case
//! when *_t will define as _m128*(may be long long)
//! vector index do not have same logic as naive vector
typedef float GI_FLOAT32_NAIVE_t __attribute__((vector_size(GI_SIMD_LEN_BYTE)));
typedef uint8_t GI_UINT8_NAIVE_t __attribute__((vector_size(GI_SIMD_LEN_BYTE)));
typedef int8_t GI_INT8_NAIVE_t __attribute__((vector_size(GI_SIMD_LEN_BYTE)));
typedef int16_t GI_INT16_NAIVE_t __attribute__((vector_size(GI_SIMD_LEN_BYTE)));
typedef int32_t GI_INT32_NAIVE_t __attribute__((vector_size(GI_SIMD_LEN_BYTE)));
typedef uint32_t GI_UINT32_NAIVE_t __attribute__((vector_size(GI_SIMD_LEN_BYTE)));
typedef int64_t GI_INT64_NAIVE_t __attribute__((vector_size(GI_SIMD_LEN_BYTE)));
typedef float float32x2_NAIVE_t __attribute__((vector_size(GI_SIMD_LEN_BYTE / 2)));
typedef struct {
    GI_INT32_NAIVE_t val[2];
} GI_INT32_V2_NAIVE_t;

typedef struct {
    GI_INT32_NAIVE_t val[4];
} GI_INT32_V4_NAIVE_t;

typedef struct {
    GI_FLOAT32_NAIVE_t val[2];
} GI_FLOAT32_V2_NAIVE_t;

typedef struct {
    GI_FLOAT32_NAIVE_t val[3];
} GI_FLOAT32_V3_NAIVE_t;

typedef struct {
    GI_FLOAT32_NAIVE_t val[4];
} GI_FLOAT32_V4_NAIVE_t;

typedef struct {
    GI_INT16_NAIVE_t val[2];
} GI_INT16_V2_NAIVE_t;

typedef struct {
    GI_INT8_NAIVE_t val[2];
} GI_INT8_V2_NAIVE_t;

#if !defined(GI_NEON_INTRINSICS) && !defined(GI_RVV_INTRINSICS)
typedef struct {
    GI_INT32_t val[2];
} GI_INT32_V2_t;

typedef struct {
    GI_INT32_t val[4];
} GI_INT32_V4_t;

typedef struct {
    GI_FLOAT32_t val[2];
} GI_FLOAT32_V2_t;

typedef struct {
    GI_FLOAT32_t val[3];
} GI_FLOAT32_V3_t;

typedef struct {
    GI_FLOAT32_t val[4];
} GI_FLOAT32_V4_t;

typedef struct {
    GI_INT16_t val[2];
} GI_INT16_V2_t;

typedef struct {
    GI_INT8_t val[2];
} GI_INT8_V2_t;

#endif
//! variable length type intrinsic can not be a member of c++ class
//! caused by can not do sizeof at build stage, for example RVV and SVE
//! so we define a type_CLASS to solve this case
//! some variable length type intrinsic can not do array subscript, for
//! example RVV, so we define a GiGetSubVector_xx function to solve this
//! case. when fix-len type in fact will do nothing
#if defined(GI_RVV_INTRINSICS)
typedef GI_FLOAT32_NAIVE_t GI_FLOAT32_FIXLEN_t;
typedef GI_FLOAT32_V2_NAIVE_t GI_FLOAT32_FIXLEN_V2_t;
typedef GI_UINT8_NAIVE_t GI_UINT8_FIXLEN_t;
typedef GI_INT8_NAIVE_t GI_INT8_FIXLEN_t;
typedef GI_INT16_NAIVE_t GI_INT16_FIXLEN_t;
typedef GI_INT32_NAIVE_t GI_INT32_FIXLEN_t;
typedef GI_UINT32_NAIVE_t GI_UINT32_FIXLEN_t;

//! get subvector
#define GiGetSubVectorFloat32V2(s, index) vget_f32m1x2_f32m1(s, index)
#define GiGetSubVectorFloat32V3(s, index) vget_f32m1x3_f32m1(s, index)
#define GiGetSubVectorFloat32V4(s, index) vget_f32m1x4_f32m1(s, index)

#define GiGetSubVectorInt32V2(s, index) vget_i32m1x2_i32m1(s, index)
#define GiGetSubVectorInt32V4(s, index) vget_i32m1x4_i32m1(s, index)

#define GiGetSubVectorInt16V2(s, index) vget_i16m1x2_i16m1(s, index)

#define GiGetSubVectorInt8V2(s, index) vget_i8m1x2_i8m1(s, index)

//! insert subvector
#define GiSetSubVectorFloat32V2(d, index, s) d = vset_f32m1x2(d, index, s)
#define GiSetSubVectorFloat32V3(d, index, s) d = vset_f32m1x3(d, index, s)
#define GiSetSubVectorFloat32V4(d, index, s) d = vset_f32m1x4(d, index, s)

#define GiSetSubVectorInt32V2(d, index, s) d = vset_i32m1x2(d, index, s)
#define GiSetSubVectorInt32V4(d, index, s) d = vset_i32m1x4(d, index, s)

#define GiSetSubVectorInt16V2(d, index, s) d = vset_i16m1x2(d, index, s)

#define GiSetSubVectorInt8V2(d, index, s) d = vset_i8m1x2(d, index, s)

//! convert
#define GiFloat32Type2FixLenType(s)                                     \
    __extension__({                                                     \
        GI_FLOAT32_FIXLEN_t d;                                          \
        vse32_v_f32m1((float*)&d, s, GI_SIMD_LEN_BYTE / sizeof(float)); \
        d;                                                              \
    })

#define GiFixLenType2GiFloat32Type(s)                                    \
    __extension__({                                                      \
        GI_FLOAT32_t d;                                                  \
        d = vle32_v_f32m1((float*)&s, GI_SIMD_LEN_BYTE / sizeof(float)); \
        d;                                                               \
    })

#define GiFloat32Type2FixLenV2Type(s)                                       \
    __extension__({                                                         \
        GI_FLOAT32_FIXLEN_V2_t d;                                           \
        d.val[0] = GiFloat32Type2FixLenType(GiGetSubVectorFloat32V2(s, 0)); \
        d.val[1] = GiFloat32Type2FixLenType(GiGetSubVectorFloat32V2(s, 1)); \
        d;                                                                  \
    })

#define GiFixLenType2GiFloat32V2Type(s)                                      \
    __extension__({                                                          \
        GI_FLOAT32_V2_t d;                                                   \
        GiSetSubVectorFloat32V2(d, 0, GiFixLenType2GiFloat32Type(s.val[0])); \
        GiSetSubVectorFloat32V2(d, 1, GiFixLenType2GiFloat32Type(s.val[1])); \
        d;                                                                   \
    })

#define GiUint8Type2FixLenType(s)                                         \
    __extension__({                                                       \
        GI_UINT8_FIXLEN_t d;                                              \
        vse8_v_u8m1((uint8_t*)&d, s, GI_SIMD_LEN_BYTE / sizeof(uint8_t)); \
        d;                                                                \
    })

#define GiFixLenType2GiUint8Type(s)                                        \
    __extension__({                                                        \
        GI_UINT8_t d;                                                      \
        d = vle8_v_u8m1((uint8_t*)&s, GI_SIMD_LEN_BYTE / sizeof(uint8_t)); \
        d;                                                                 \
    })

#define GiInt8Type2FixLenType(s)                                        \
    __extension__({                                                     \
        GI_INT8_FIXLEN_t d;                                             \
        vse8_v_i8m1((int8_t*)&d, s, GI_SIMD_LEN_BYTE / sizeof(int8_t)); \
        d;                                                              \
    })

#define GiFixLenType2GiInt8Type(s)                                       \
    __extension__({                                                      \
        GI_INT8_t d;                                                     \
        d = vle8_v_i8m1((int8_t*)&s, GI_SIMD_LEN_BYTE / sizeof(int8_t)); \
        d;                                                               \
    })

#define GiInt16Type2FixLenType(s)                                           \
    __extension__({                                                         \
        GI_INT16_FIXLEN_t d;                                                \
        vse16_v_i16m1((int16_t*)&d, s, GI_SIMD_LEN_BYTE / sizeof(int16_t)); \
        d;                                                                  \
    })

#define GiFixLenType2GiInt16Type(s)                                          \
    __extension__({                                                          \
        GI_INT16_t d;                                                        \
        d = vle16_v_i16m1((int16_t*)&s, GI_SIMD_LEN_BYTE / sizeof(int16_t)); \
        d;                                                                   \
    })

#define GiInt32Type2FixLenType(s)                                           \
    __extension__({                                                         \
        GI_INT32_FIXLEN_t d;                                                \
        vse32_v_i32m1((int32_t*)&d, s, GI_SIMD_LEN_BYTE / sizeof(int32_t)); \
        d;                                                                  \
    })

#define GiFixLenType2GiInt32Type(s)                                          \
    __extension__({                                                          \
        GI_INT32_t d;                                                        \
        d = vle32_v_i32m1((int32_t*)&s, GI_SIMD_LEN_BYTE / sizeof(int32_t)); \
        d;                                                                   \
    })

#define GiUint32Type2FixLenType(s)                                            \
    __extension__({                                                           \
        GI_UINT32_FIXLEN_t d;                                                 \
        vse32_v_u32m1((uint32_t*)&d, s, GI_SIMD_LEN_BYTE / sizeof(uint32_t)); \
        d;                                                                    \
    })

#define GiFixLenType2GiUint32Type(s)                                           \
    __extension__({                                                            \
        GI_UINT32_t d;                                                         \
        d = vle32_v_u32m1((uint32_t*)&s, GI_SIMD_LEN_BYTE / sizeof(uint32_t)); \
        d;                                                                     \
    })
#else
typedef GI_FLOAT32_t GI_FLOAT32_FIXLEN_t;
typedef GI_FLOAT32_V2_t GI_FLOAT32_FIXLEN_V2_t;
typedef GI_UINT8_t GI_UINT8_FIXLEN_t;
typedef GI_INT8_t GI_INT8_FIXLEN_t;
typedef GI_INT16_t GI_INT16_FIXLEN_t;
typedef GI_INT32_t GI_INT32_FIXLEN_t;
typedef GI_UINT32_t GI_UINT32_FIXLEN_t;
#define GiFloat32Type2FixLenType(s)   s
#define GiFixLenType2GiFloat32Type(s) s

#define GiFloat32Type2FixLenV2Type(s)   s
#define GiFixLenType2GiFloat32V2Type(s) s

#define GiUint8Type2FixLenType(s)   s
#define GiFixLenType2GiUint8Type(s) s

#define GiInt8Type2FixLenType(s)   s
#define GiFixLenType2GiInt8Type(s) s

#define GiInt16Type2FixLenType(s)   s
#define GiFixLenType2GiInt16Type(s) s

#define GiInt32Type2FixLenType(s)   s
#define GiFixLenType2GiInt32Type(s) s

#define GiUint32Type2FixLenType(s)        s
#define GiFixLenType2GiUint32Type(s)      s

//! get subvector
#define GiGetSubVectorFloat32V2(s, index) s.val[index]
#define GiGetSubVectorFloat32V3(s, index) s.val[index]
#define GiGetSubVectorFloat32V4(s, index) s.val[index]

#define GiGetSubVectorInt32V2(s, index) s.val[index]
#define GiGetSubVectorInt32V4(s, index) s.val[index]

#define GiGetSubVectorInt16V2(s, index) s.val[index]

#define GiGetSubVectorInt8V2(s, index)       s.val[index]

//! insert subvector
#define GiSetSubVectorFloat32V2(d, index, s) d.val[index] = s
#define GiSetSubVectorFloat32V3(d, index, s) d.val[index] = s
#define GiSetSubVectorFloat32V4(d, index, s) d.val[index] = s

#define GiSetSubVectorInt32V2(d, index, s) d.val[index] = s
#define GiSetSubVectorInt32V4(d, index, s) d.val[index] = s

#define GiSetSubVectorInt16V2(d, index, s) d.val[index] = s

#define GiSetSubVectorInt8V2(d, index, s) d.val[index] = s
#endif

#define Max(a, b) (a) > (b) ? (a) : (b)
#define Min(a, b) (a) < (b) ? (a) : (b)

#if defined(GI_NEON_INTRINSICS)
#if defined(__ARM_FEATURE_FMA) && defined(GI_NEON64_INTRINSICS)
#define v_fma_ps_f32(c, b, a)         vfmaq_f32((c), (b), (a))
#define v_fma_n_f32(c, b, a)          vfmaq_n_f32((c), (b), (a))
#define v_fma_lane_f32(c, b, a, lane) vfmaq_lane_f32((c), (b), (a), (lane))
#else
#define v_fma_ps_f32(c, b, a)         vmlaq_f32((c), (b), (a))
#define v_fma_n_f32(c, b, a)          vmlaq_n_f32((c), (b), (a))
#define v_fma_lane_f32(c, b, a, lane) vmlaq_lane_f32((c), (b), (a), (lane))
#endif
#endif

GI_FORCEINLINE
enum GiSimdType GiGetSimdType() {
    //! override by special macro to insure ci have test naive and sse2
    //! now we do not imp GI_AVX to now and x64 ci device will test GI_SSE42
    //! now arm ci device will test GI_NEON
    //! insure test GI_SSE2 by command:
    //! --copt -march=core2 --copt -mno-sse4.2
    //! --copt -mno-sse3 --copt -DGI_TEST_SSE2
    //! insure test GI_NAIVE by command:
    //! --copt -DGI_TEST_SSE2
    //! DNN code at least need sse2 at x86
    //! so we can not test GI_NAIVE by
    //! --copt -march=core2 --copt -mno-sse4.2
    //! --copt -mno-sse3 --copt -mno-sse2
    //! --copt -DGI_TEST_NAIVE
    //! about CMake, can override build flags to CMAKE_CXX_FLAGS/CMAKE_C_FLAGS by
    //! EXTRA_CMAKE_ARGS when use scripts/cmake-build/*.sh
#if defined(GI_TEST_NAIVE)
#undef __gi_simd_type
#define __gi_simd_type GI_NAIVE
#elif defined(GI_TEST_SSE2)
#undef __gi_simd_type
#define __gi_simd_type GI_SSE2
#endif

    return __gi_simd_type;
}

GI_FORCEINLINE
GI_FLOAT32_t GiBroadcastFloat32(float Value) {
#if defined(GI_NEON_INTRINSICS)
    return vdupq_n_f32(Value);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_set1_ps(Value);
#elif defined(GI_RVV_INTRINSICS)
    return vfmv_v_f_f32m1(Value, GI_SIMD_LEN_BYTE / sizeof(float));
#else
    GI_FLOAT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret[i] = Value;
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT8_t GiBroadcastInt8(int8_t Value) {
#if defined(GI_NEON_INTRINSICS)
    return vdupq_n_s8(Value);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_set1_epi8(Value);
#elif defined(GI_RVV_INTRINSICS)
    return vmv_v_x_i8m1(Value, GI_SIMD_LEN_BYTE / sizeof(int8_t));
#else
    GI_INT8_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++) {
        ret[i] = Value;
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiBroadcastInt32(int32_t Value) {
#if defined(GI_NEON_INTRINSICS)
    return vdupq_n_s32(Value);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_set1_epi32(Value);
#elif defined(GI_RVV_INTRINSICS)
    return vmv_v_x_i32m1(Value, GI_SIMD_LEN_BYTE / sizeof(int32_t));
#else
    GI_INT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        ret[i] = Value;
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiAndInt32(GI_INT32_t Vector1, GI_INT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vandq_s32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_and_si128(Vector1, Vector2);
#elif defined(GI_RVV_INTRINSICS)
    return vand_vv_i32m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int32_t));
#else
    return Vector1 & Vector2;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiOrInt32(GI_INT32_t Vector1, GI_INT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vorrq_s32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_or_si128(Vector1, Vector2);
#elif defined(GI_RVV_INTRINSICS)
    return vor_vv_i32m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int32_t));
#else
    return Vector1 | Vector2;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiAndNotInt32(GI_INT32_t VectorNot, GI_INT32_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vandq_s32(vmvnq_s32(VectorNot), Vector);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_andnot_si128(VectorNot, Vector);
#elif defined(GI_RVV_INTRINSICS)
    GI_INT32_t not_v = vnot_v_i32m1(VectorNot, GI_SIMD_LEN_BYTE / sizeof(int32_t));
    return vand_vv_i32m1(not_v, Vector, GI_SIMD_LEN_BYTE / sizeof(int32_t));
#else
    return (~VectorNot) & Vector;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiXorInt32(GI_INT32_t Vector1, GI_INT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return veorq_s32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_xor_si128(Vector1, Vector2);
#elif defined(GI_RVV_INTRINSICS)
    return vxor_vv_i32m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int32_t));
#else
    return Vector1 ^ Vector2;
#endif
}
// vim: syntax=cpp.doxygen
