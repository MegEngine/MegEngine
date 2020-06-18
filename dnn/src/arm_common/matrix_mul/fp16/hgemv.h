/**
 * \file dnn/src/arm_common/matrix_mul/fp16/hgemv.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <stddef.h>
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

namespace megdnn {
namespace arm_common {

bool is_hgemv_preferred(bool transposeA, bool transposeB, size_t M, size_t N,
                        size_t K, size_t /*LDA*/, size_t LDB, size_t /*LDC*/);

void gemv_like(const __fp16* __restrict A, const __fp16* __restrict B,
               __fp16* __restrict C, size_t M, size_t N, size_t K,
               size_t Astride, size_t Bstride, size_t Cstride);


}  // namespace aarch64
}  // namespace megdnn

#endif
// vim: syntax=cpp.doxygen
