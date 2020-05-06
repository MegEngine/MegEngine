/**
 * \file dnn/src/x86/utils.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cstddef>
#include <vector>
#include "src/common/utils.h"

#if MEGDNN_X86_WITH_MKL
#include <mkl.h>
//! As INTEL_MKL_VERSION >= 20190001 support SUPPORT_MKL_PACKED_GEMM
#if INTEL_MKL_VERSION >= 20190001
#define SUPPORT_MKL_PACKED_GEMM 1
#else
#define SUPPORT_MKL_PACKED_GEMM 0
#endif

#endif

namespace megdnn {
namespace x86 {

enum class SIMDType {
    SSE,
    SSE2,
    SSE3,
    SSE4_1,
    SSE4_2,
    AVX,
    AVX2,
    FMA,
    VNNI,
    NONE,
    __NR_SIMD_TYPE  //! total number of SIMD types; used for testing
};

bool is_supported(SIMDType type);

//! disable a particular and more advanced SIMD types, for testing
void disable_simd_type(SIMDType type);

template <typename T>
T find_nearest_elem(T val, const std::vector<T>& vec) {
    megdnn_assert_internal(!vec.empty());
    T res = vec[0];
    typedef typename std::make_signed<T>::type S;
    S opt_cost = (val - res > 0 ? val - res : res - val);
    for (auto&& cand : vec) {
        S cur_cost = (val - cand > 0 ? val - cand : cand - val);
        if (cur_cost < opt_cost) {
            opt_cost = cur_cost;
            res = cand;
        }
    }
    return res;
}

// Treat the denormalized numbers as zero.
void disable_denorm();

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
