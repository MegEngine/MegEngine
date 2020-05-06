/**
 * \file dnn/src/arm_common/pooling/do_max_pooling_2x2_nchw44.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/arm_common/pooling/do_pooling_2x2_nchw44.h"
#include "src/arm_common/pooling/algo.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/unroll_macro.h"

namespace megdnn {
namespace arm_common {

void do_max_pooling_2x2_stride1_int8_nchw44_NEON(const int8_t* src, int8_t* dst,
                                                 size_t IH, size_t IW,
                                                 size_t OH, size_t OW,
                                                 size_t PH, size_t PW,
                                                 const WorkspaceBundle& ws) {
    const int8_t* sptr = nullptr;
    size_t IH2, IW2;
    sptr = handle_padding(src, IH, IW, IH2, IW2, PH, PW, ws);
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh;
        const int8_t* __restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* __restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        int8_t* __restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int8x16_t src0123 = vld1q_s8(sptr0);
            int8x16_t src1234 = vld1q_s8(sptr0 + 4);
            int8x16_t max0 = vmaxq_s8(src0123, src1234);

            src0123 = vld1q_s8(sptr1);
            src1234 = vld1q_s8(sptr1 + 4);
            int8x16_t max1 = vmaxq_s8(src0123, src1234);

            int8x16_t max_out = vmaxq_s8(max0, max1);

            vst1q_s8(dptr, max_out);

            sptr0 += 16;
            sptr1 += 16;
            dptr += 16;
        }
        for (; ow < OW; ++ow) {
            int8x8_t src001 = vld1_s8(sptr0);
            int8x8_t src101 = vld1_s8(sptr1);

            int8x8_t max_out = vmax_s8(src001, src101);
#define store(i) *(dptr + i) = std::max(max_out[i], max_out[i + 4]);
            UNROLL_CALL_NOWRAPPER(4, store)
#undef store
            sptr0 += 4;
            sptr1 += 4;
            dptr += 4;
        }
    }
}
void do_max_pooling_2x2_stride2_int8_nchw44_NEON(const int8_t* src, int8_t* dst,
                                                 size_t IH, size_t IW,
                                                 size_t OH, size_t OW,
                                                 size_t PH, size_t PW,
                                                 const WorkspaceBundle& ws) {
    const int8_t* sptr = nullptr;
    size_t IH2, IW2;
    sptr = handle_padding(src, IH, IW, IH2, IW2, PH, PW, ws);
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh << 1;
        const int8_t* __restrict sptr0 = sptr + (ih + 0) * IW2 * 4;
        const int8_t* __restrict sptr1 = sptr + (ih + 1) * IW2 * 4;
        int8_t* __restrict dptr = dst + oh * OW * 4;
        size_t ow = 0;
        for (; ow + 3 < OW; ow += 4) {
            int8x16_t src00 = vld1q_s8(sptr0);
            int8x16_t src04 = vld1q_s8(sptr0 + 4 * 4);
            int32x4x2_t src_tmp = vuzpq_s32(vreinterpretq_s32_s8(src00),
                                            vreinterpretq_s32_s8(src04));
            int32x4_t src0246 = src_tmp.val[0];
            int32x4_t src1357 = src_tmp.val[1];
            int8x16_t max0 = vmaxq_s8(vreinterpretq_s8_s32(src0246),
                                      vreinterpretq_s8_s32(src1357));

            src00 = vld1q_s8(sptr1);
            src04 = vld1q_s8(sptr1 + 4 * 4);
            src_tmp = vuzpq_s32(vreinterpretq_s32_s8(src00),
                                vreinterpretq_s32_s8(src04));
            src0246 = src_tmp.val[0];
            src1357 = src_tmp.val[1];
            int8x16_t max1 = vmaxq_s8(vreinterpretq_s8_s32(src0246),
                                      vreinterpretq_s8_s32(src1357));

            int8x16_t max_out = vmaxq_s8(max0, max1);

            vst1q_s8(dptr, max_out);

            sptr0 += 32;
            sptr1 += 32;
            dptr += 16;
        }
        for (; ow < OW; ++ow) {
            int8x8_t src001 = vld1_s8(sptr0);
            int8x8_t src101 = vld1_s8(sptr1);

            int8x8_t max_out = vmax_s8(src001, src101);
#define store(i) *(dptr + i) = std::max(max_out[i], max_out[i + 4]);
            UNROLL_CALL_NOWRAPPER(4, store)
#undef store
            sptr0 += 8;
            sptr1 += 8;
            dptr += 4;
        }
    }
}

}  // namespace arm_common
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
