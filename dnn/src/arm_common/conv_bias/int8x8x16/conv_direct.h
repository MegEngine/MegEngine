/**
 * \file dnn/src/arm_common/conv_bias/int8x8x16/conv_direct.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/arm_common/conv_bias/opr_impl.h"

#include <cstddef>
#include <cstdint>

namespace megdnn {
namespace arm_common {
namespace conv_bias {

template <bool add_to_dst>
void conv_direct_2x2_sc_int8_int8_int16(const int8_t* src, const int8_t* filter,
                                        int16_t* dst, size_t IH, size_t IW,
                                        size_t OH, size_t OW, size_t PH,
                                        size_t PW);
template <bool add_to_dst>
void conv_direct_3x3_sc_int8_int8_int16(const int8_t* src, const int8_t* filter,
                                        int16_t* dst, size_t IH, size_t IW,
                                        size_t OH, size_t OW, size_t PH,
                                        size_t PW);
template <bool add_to_dst>
void conv_direct_5x5_sc_int8_int8_int16(const int8_t* src, const int8_t* filter,
                                        int16_t* dst, size_t IH, size_t IW,
                                        size_t OH, size_t OW, size_t PH,
                                        size_t PW);
}  // namespace conv_bias
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
