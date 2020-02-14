/**
 * \file dnn/src/x86/avx_helper.h
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

#include <immintrin.h>
#include <avxintrin.h>
#include <avx2intrin.h>
#include <fmaintrin.h>

namespace megdnn {
namespace x86 {

MEGDNN_ATTRIBUTE_TARGET("avx")
static inline __m256 _mm256_loadu2_m128_emulate(
        const float *hiaddr, const float *loaddr) {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(loaddr)),
            _mm_loadu_ps(hiaddr), 1);
}

} // namespace x86
} // namespace megdnn

// vim: syntax=cpp.doxygen
