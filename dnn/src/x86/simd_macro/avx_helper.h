#include <immintrin.h>
#include <xmmintrin.h>

#define MEGDNN_SIMD_NAME              AVX
#define MEGDNN_SIMD_TARGET            avx
#define MEGDNN_SIMD_ATTRIBUTE_TARGET  MEGDNN_ATTRIBUTE_TARGET("avx")
#define MEGDNN_SIMD_WIDTH             8
#define MEGDNN_SIMD_TYPE              __m256
#define MEGDNN_SIMD_LOADU(addr)       _mm256_loadu_ps(addr)
#define MEGDNN_SIMD_STOREU(addr, reg) _mm256_storeu_ps(addr, reg)
#define MEGDNN_SIMD_SETZERO()         _mm256_setzero_ps()
#define MEGDNN_SIMD_SET1(num)         _mm256_set1_ps(num)
#define MEGDNN_SIMD_FMADD(a, b, c)    _mm256_add_ps(c, _mm256_mul_ps(a, b))
