/**
 * \file dnn/src/x86/elemwise/avx_util/avx_mathfun.h
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
#include <immintrin.h>
#ifdef WIN32
#include <avxintrin.h>
#include <smmintrin.h>
#include <avx2intrin.h>
#include <fmaintrin.h>
#endif

#include <cstddef>

namespace megdnn {
namespace x86 {
namespace detail {

__m256 log256_ps(__m256 x) MEGDNN_ATTRIBUTE_TARGET("avx2");

__m256 exp256_ps(__m256 x) MEGDNN_ATTRIBUTE_TARGET("avx2");

__m256 sin256_ps(__m256 x) MEGDNN_ATTRIBUTE_TARGET("avx2");

__m256 cos256_ps(__m256 x) MEGDNN_ATTRIBUTE_TARGET("avx2");

void sincos256_ps(__m256 x, __m256* s, __m256* c)
        MEGDNN_ATTRIBUTE_TARGET("avx2");

}  // namespace detail
}  // namespace x86
}  // namespace megdnn
