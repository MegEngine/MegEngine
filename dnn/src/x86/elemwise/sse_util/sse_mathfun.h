#pragma once
#include <xmmintrin.h>
#include "megdnn/arch.h"
#include "megdnn/basic_types.h"

#include <cstddef>

namespace megdnn {
namespace x86 {
namespace detail {

__m128 log_ps(__m128 x) MEGDNN_ATTRIBUTE_TARGET("sse2");

__m128 exp_ps(__m128 x) MEGDNN_ATTRIBUTE_TARGET("sse2");

__m128 sin_ps(__m128 x) MEGDNN_ATTRIBUTE_TARGET("sse2");

__m128 cos_ps(__m128 x) MEGDNN_ATTRIBUTE_TARGET("sse2");

void sincos_ps(__m128 x, __m128* s, __m128* c) MEGDNN_ATTRIBUTE_TARGET("sse2");

}  // namespace detail
}  // namespace x86
}  // namespace megdnn
