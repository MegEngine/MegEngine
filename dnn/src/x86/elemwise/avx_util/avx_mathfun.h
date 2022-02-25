#pragma once
#include <immintrin.h>
#include "megdnn/arch.h"
#include "megdnn/basic_types.h"
#ifdef WIN32
#include <avx2intrin.h>
#include <avxintrin.h>
#include <fmaintrin.h>
#include <smmintrin.h>
#endif

#include <cstddef>

namespace megdnn {
namespace x86 {
namespace detail {

__m256 log256_ps(__m256 x) MEGDNN_ATTRIBUTE_TARGET("avx2");

__m256 exp256_ps(__m256 x) MEGDNN_ATTRIBUTE_TARGET("avx2");

__m256 sin256_ps(__m256 x) MEGDNN_ATTRIBUTE_TARGET("avx2");

__m256 cos256_ps(__m256 x) MEGDNN_ATTRIBUTE_TARGET("avx2");

void sincos256_ps(__m256 x, __m256* s, __m256* c) MEGDNN_ATTRIBUTE_TARGET("avx2");

}  // namespace detail
}  // namespace x86
}  // namespace megdnn
