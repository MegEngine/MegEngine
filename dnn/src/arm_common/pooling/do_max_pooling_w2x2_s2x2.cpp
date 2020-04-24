/**
 * \file dnn/src/arm_common/pooling/do_max_pooling_w2x2_s2x2.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/arm_common/pooling/do_max_pooling_w2x2_s2x2.h"
#include "src/arm_common/pooling/pooling_helper.h"

namespace megdnn {
namespace arm_common {

void pooling_max_w2x2_s2x2(const int8_t* src, int8_t* dst, size_t N, size_t C,
                           size_t IH, size_t IW, size_t OH, size_t OW) {
    for (size_t nc = 0; nc < N * C; ++nc) {
        for (size_t oh = 0; oh < OH; ++oh) {
            size_t ih = oh << 1;
            const int8_t* __restrict sptr0 = src + (ih + 0) * IW;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW;
            int8_t* __restrict dptr = dst + oh * OW;
            size_t ow = 0;
            for (; ow + 8 <= OW; ow += 8) {
                // process 2x16 to produce 1x8 elements at a time.
                int8x16_t vs0 = vld1q_s8(sptr0), vs1 = vld1q_s8(sptr1);
                int8x16_t vi = vmaxq_s8(vs0, vs1);
                int8x8_t vd = vpmax_s8(vget_low_s8(vi), vget_high_s8(vi));
                vst1_s8(dptr, vd);
                sptr0 += 16;
                sptr1 += 16;
                dptr += 8;
            }
            for (; ow < OW; ++ow) {
                dptr[0] = std::max(std::max(sptr0[0], sptr0[1]),
                                   std::max(sptr1[0], sptr1[1]));
                sptr0 += 2;
                sptr1 += 2;
                dptr += 1;
            }
        }
        src += IH * IW;
        dst += OH * OW;
    }
}
}  // namespace arm_common
}  // namespace megdnn
// vim: syntax=cpp.doxygen

