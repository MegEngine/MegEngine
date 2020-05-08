/**
 * \file dnn/src/arm_common/pooling/do_max_pooling_w4x4_s2x2.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "src/fallback/pooling/opr_impl.h"

namespace megdnn {
namespace arm_common {

void do_max_pooling_w4x4_s2x2_float_NEON(const dt_float32* src, dt_float32* dst,
                                         DType src_dtype, const int IH,
                                         const int IW, const int OH,
                                         const int OW, const int PH,
                                         const int PW);
void do_max_pooling_w4x4_s2x2_int8_NEON(const int8_t* src, int8_t* dst,
                                        DType src_dtype, const int IH,
                                        const int IW, const int OH,
                                        const int OW, const int PH,
                                        const int PW);
void do_max_pooling_w4x4_s2x2_uint8_NEON(const uint8_t* src, uint8_t* dst,
                                         DType src_dtype, const int IH,
                                         const int IW, const int OH,
                                         const int OW, const int PH,
                                         const int PW);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void do_max_pooling_w4x4_s2x2_float16_NEON(const __fp16* src, __fp16* dst,
                                           DType src_dtype, const int IH,
                                           const int IW, const int OH,
                                           const int OW, const int PH,
                                           const int PW);
#endif
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen

