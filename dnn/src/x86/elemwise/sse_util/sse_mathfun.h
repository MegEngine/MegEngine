/**
 * \file dnn/src/x86/elemwise/sse_util/sse_mathfun.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/arch.h"
#include "megdnn/basic_types.h"
#include <xmmintrin.h>

#include <cstddef>

namespace megdnn {
namespace x86 {
namespace detail {

__m128 log_ps(__m128 x) MEGDNN_ATTRIBUTE_TARGET("sse2");

__m128 exp_ps(__m128 x) MEGDNN_ATTRIBUTE_TARGET("sse2");

__m128 sin_ps(__m128 x) MEGDNN_ATTRIBUTE_TARGET("sse2");

__m128 cos_ps(__m128 x) MEGDNN_ATTRIBUTE_TARGET("sse2");

void sincos_ps(__m128 x, __m128 *s, __m128 *c) MEGDNN_ATTRIBUTE_TARGET("sse2");


} // namespace detail
} // namespace x86
} // namespace megdnn
