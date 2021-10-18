/**
 * \file dnn/src/aarch64/relayout/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/relayout_helper.h"
#include "src/common/utils.h"

#include "src/aarch64/handle.h"
#include "src/aarch64/relayout/opr_impl.h"
#include "src/arm_common/simd_macro/marm_neon.h"

using namespace megdnn;
using namespace relayout;

namespace {

struct TransposeByte {
    uint8_t v;
};

void trans_16x16_u8(
        const void* src, void* dst, const size_t src_step, const size_t dst_step) {
    asm volatile(
            "\n"
            "ld1 {v0.16b}, [%[src]], %[src_step] \n"
            "ld1 {v1.16b}, [%[src]], %[src_step] \n"
            "ld1 {v2.16b}, [%[src]], %[src_step] \n"
            "ld1 {v3.16b}, [%[src]], %[src_step] \n"
            "ld1 {v4.16b}, [%[src]], %[src_step] \n"
            "ld1 {v5.16b}, [%[src]], %[src_step] \n"
            "ld1 {v6.16b}, [%[src]], %[src_step] \n"
            "ld1 {v7.16b}, [%[src]], %[src_step] \n"
            "ld1 {v8.16b}, [%[src]], %[src_step] \n"
            "ld1 {v9.16b}, [%[src]], %[src_step] \n"
            "ld1 {v10.16b}, [%[src]], %[src_step] \n"
            "ld1 {v11.16b}, [%[src]], %[src_step] \n"
            "ld1 {v12.16b}, [%[src]], %[src_step] \n"
            "ld1 {v13.16b}, [%[src]], %[src_step] \n"
            "ld1 {v14.16b}, [%[src]], %[src_step] \n"
            "ld1 {v15.16b}, [%[src]], %[src_step] \n"
            "trn1 v16.16b, v0.16b, v1.16b \n"
            "trn2 v17.16b, v0.16b, v1.16b \n"
            "trn1 v18.16b, v2.16b, v3.16b \n"
            "trn2 v19.16b, v2.16b, v3.16b \n"
            "trn1 v20.16b, v4.16b, v5.16b \n"
            "trn2 v21.16b, v4.16b, v5.16b \n"
            "trn1 v22.16b, v6.16b, v7.16b \n"
            "trn2 v23.16b, v6.16b, v7.16b \n"
            "trn1 v24.16b, v8.16b, v9.16b \n"
            "trn2 v25.16b, v8.16b, v9.16b \n"
            "trn1 v26.16b, v10.16b, v11.16b \n"
            "trn2 v27.16b, v10.16b, v11.16b \n"
            "trn1 v28.16b, v12.16b, v13.16b \n"
            "trn2 v29.16b, v12.16b, v13.16b \n"
            "trn1 v30.16b, v14.16b, v15.16b \n"
            "trn2 v31.16b, v14.16b, v15.16b \n"
            "trn1 v0.8h, v16.8h, v18.8h \n"
            "trn2 v2.8h, v16.8h, v18.8h \n"
            "trn1 v4.8h, v20.8h, v22.8h \n"
            "trn2 v6.8h, v20.8h, v22.8h \n"
            "trn1 v8.8h, v24.8h, v26.8h \n"
            "trn2 v10.8h, v24.8h, v26.8h \n"
            "trn1 v12.8h, v28.8h, v30.8h \n"
            "trn2 v14.8h, v28.8h, v30.8h \n"
            "trn1 v1.8h, v17.8h, v19.8h \n"
            "trn2 v3.8h, v17.8h, v19.8h \n"
            "trn1 v5.8h, v21.8h, v23.8h \n"
            "trn2 v7.8h, v21.8h, v23.8h \n"
            "trn1 v9.8h, v25.8h, v27.8h \n"
            "trn2 v11.8h, v25.8h, v27.8h \n"
            "trn1 v13.8h, v29.8h, v31.8h \n"
            "trn2 v15.8h, v29.8h, v31.8h \n"
            "trn1 v16.4s, v0.4s, v4.4s \n"
            "trn2 v20.4s, v0.4s, v4.4s \n"
            "trn1 v24.4s, v8.4s, v12.4s \n"
            "trn2 v28.4s, v8.4s, v12.4s \n"
            "trn1 v17.4s, v1.4s, v5.4s \n"
            "trn2 v21.4s, v1.4s, v5.4s \n"
            "trn1 v25.4s, v9.4s, v13.4s \n"
            "trn2 v29.4s, v9.4s, v13.4s \n"
            "trn1 v18.4s, v2.4s, v6.4s \n"
            "trn2 v22.4s, v2.4s, v6.4s \n"
            "trn1 v26.4s, v10.4s, v14.4s \n"
            "trn2 v30.4s, v10.4s, v14.4s \n"
            "trn1 v19.4s, v3.4s, v7.4s \n"
            "trn2 v23.4s, v3.4s, v7.4s \n"
            "trn1 v27.4s, v11.4s, v15.4s \n"
            "trn2 v31.4s, v11.4s, v15.4s \n"
            "trn1 v0.2d, v16.2d, v24.2d \n"
            "trn2 v8.2d, v16.2d, v24.2d \n"
            "trn1 v1.2d, v17.2d, v25.2d \n"
            "trn2 v9.2d, v17.2d, v25.2d \n"
            "trn1 v2.2d, v18.2d, v26.2d \n"
            "trn2 v10.2d, v18.2d, v26.2d \n"
            "trn1 v3.2d, v19.2d, v27.2d \n"
            "trn2 v11.2d, v19.2d, v27.2d \n"
            "trn1 v4.2d, v20.2d, v28.2d \n"
            "trn2 v12.2d, v20.2d, v28.2d \n"
            "trn1 v5.2d, v21.2d, v29.2d \n"
            "trn2 v13.2d, v21.2d, v29.2d \n"
            "trn1 v6.2d, v22.2d, v30.2d \n"
            "trn2 v14.2d, v22.2d, v30.2d \n"
            "trn1 v7.2d, v23.2d, v31.2d \n"
            "trn2 v15.2d, v23.2d, v31.2d \n"
            "st1 {v0.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v1.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v2.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v3.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v4.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v5.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v6.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v7.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v8.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v9.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v10.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v11.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v12.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v13.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v14.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v15.16b}, [%[dst]], %[dst_step] \n"
            : [src] "+r"(src), [dst] "+r"(dst)
            : [src_step] "r"(src_step), [dst_step] "r"(dst_step)
            : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11",
              "d12", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20", "d21",
              "d22", "d23", "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31");
}

struct Transpose4Byte {
    uint32_t v;
};

static inline void trans_8x8_u32(
        const void* src, void* dst, const size_t src_step, const size_t dst_step) {
    uint32_t* src_ptr = (uint32_t*)src;
    uint32_t* dst_ptr = (uint32_t*)dst;
    uint32x4x2_t src0 = vld1q_u32_x2(src_ptr + 0 * src_step);  // A0A1A2A3
    uint32x4x2_t src1 = vld1q_u32_x2(src_ptr + 1 * src_step);  // B0B1B2B3
    uint32x4x2_t src2 = vld1q_u32_x2(src_ptr + 2 * src_step);  // C0C1C2C3
    uint32x4x2_t src3 = vld1q_u32_x2(src_ptr + 3 * src_step);  // D0D1D2D3
    uint32x4x2_t src4 = vld1q_u32_x2(src_ptr + 4 * src_step);  // E0E1E2E3
    uint32x4x2_t src5 = vld1q_u32_x2(src_ptr + 5 * src_step);  // F0F1F2F3
    uint32x4x2_t src6 = vld1q_u32_x2(src_ptr + 6 * src_step);  // G0G1G2G3
    uint32x4x2_t src7 = vld1q_u32_x2(src_ptr + 7 * src_step);  // H0H1H2H3

    uint32x4_t ab_low = vzip1q_u32(src0.val[0], src1.val[0]);   // A0B0A1B1
    uint32x4_t ab_high = vzip2q_u32(src0.val[0], src1.val[0]);  // A2B2A3B3
    uint32x4_t cd_low = vzip1q_u32(src2.val[0], src3.val[0]);   // C0D0C1D1
    uint32x4_t cd_high = vzip2q_u32(src2.val[0], src3.val[0]);  // C2D2C3D3
    uint32x4_t ef_low = vzip1q_u32(src4.val[0], src5.val[0]);   // E0F0E1F1
    uint32x4_t ef_high = vzip2q_u32(src4.val[0], src5.val[0]);  // E2F2E3F3
    uint32x4_t gh_low = vzip1q_u32(src6.val[0], src7.val[0]);   // G0H0G1H1
    uint32x4_t gh_high = vzip2q_u32(src6.val[0], src7.val[0]);  // G2H2G3H3

    uint32x4_t abcd_0 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ab_low), vreinterpretq_u64_u32(cd_low)));  // A0B0C0D0
    uint32x4_t abcd_1 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ab_low), vreinterpretq_u64_u32(cd_low)));  // A1B1C1D1
    uint32x4_t abcd_2 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ab_high),
            vreinterpretq_u64_u32(cd_high)));  // A2B2C2D2
    uint32x4_t abcd_3 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ab_high),
            vreinterpretq_u64_u32(cd_high)));  // A3B3C3D3
    uint32x4_t efgh_0 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ef_low), vreinterpretq_u64_u32(gh_low)));  // E0F0G0H0
    uint32x4_t efgh_1 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ef_low), vreinterpretq_u64_u32(gh_low)));  // E1F1G1H1
    uint32x4_t efgh_2 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ef_high),
            vreinterpretq_u64_u32(gh_high)));  // E2F2G2H2
    uint32x4_t efgh_3 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ef_high),
            vreinterpretq_u64_u32(gh_high)));  // E3F3G3H3

    vst1q_u32(dst_ptr + 0 * dst_step, abcd_0);
    vst1q_u32(dst_ptr + 0 * dst_step + 4, efgh_0);
    vst1q_u32(dst_ptr + 1 * dst_step, abcd_1);
    vst1q_u32(dst_ptr + 1 * dst_step + 4, efgh_1);
    vst1q_u32(dst_ptr + 2 * dst_step, abcd_2);
    vst1q_u32(dst_ptr + 2 * dst_step + 4, efgh_2);
    vst1q_u32(dst_ptr + 3 * dst_step, abcd_3);
    vst1q_u32(dst_ptr + 3 * dst_step + 4, efgh_3);

    ab_low = vzip1q_u32(src0.val[1], src1.val[1]);   // A0B0A1B1
    ab_high = vzip2q_u32(src0.val[1], src1.val[1]);  // A2B2A3B3
    cd_low = vzip1q_u32(src2.val[1], src3.val[1]);   // C0D0C1D1
    cd_high = vzip2q_u32(src2.val[1], src3.val[1]);  // C2D2C3D3
    ef_low = vzip1q_u32(src4.val[1], src5.val[1]);   // E0F0E1F1
    ef_high = vzip2q_u32(src4.val[1], src5.val[1]);  // E2F2E3F3
    gh_low = vzip1q_u32(src6.val[1], src7.val[1]);   // G0H0G1H1
    gh_high = vzip2q_u32(src6.val[1], src7.val[1]);  // G2H2G3H3

    abcd_0 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ab_low), vreinterpretq_u64_u32(cd_low)));  // A0B0C0D0
    abcd_1 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ab_low), vreinterpretq_u64_u32(cd_low)));  // A1B1C1D1
    abcd_2 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ab_high),
            vreinterpretq_u64_u32(cd_high)));  // A2B2C2D2
    abcd_3 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ab_high),
            vreinterpretq_u64_u32(cd_high)));  // A3B3C3D3
    efgh_0 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ef_low), vreinterpretq_u64_u32(gh_low)));  // E0F0G0H0
    efgh_1 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ef_low), vreinterpretq_u64_u32(gh_low)));  // E1F1G1H1
    efgh_2 = vreinterpretq_u32_u64(vzip1q_u64(
            vreinterpretq_u64_u32(ef_high),
            vreinterpretq_u64_u32(gh_high)));  // E2F2G2H2
    efgh_3 = vreinterpretq_u32_u64(vzip2q_u64(
            vreinterpretq_u64_u32(ef_high),
            vreinterpretq_u64_u32(gh_high)));  // E3F3G3H3

    vst1q_u32(dst_ptr + 4 * dst_step, abcd_0);
    vst1q_u32(dst_ptr + 4 * dst_step + 4, efgh_0);
    vst1q_u32(dst_ptr + 5 * dst_step, abcd_1);
    vst1q_u32(dst_ptr + 5 * dst_step + 4, efgh_1);
    vst1q_u32(dst_ptr + 6 * dst_step, abcd_2);
    vst1q_u32(dst_ptr + 6 * dst_step + 4, efgh_2);
    vst1q_u32(dst_ptr + 7 * dst_step, abcd_3);
    vst1q_u32(dst_ptr + 7 * dst_step + 4, efgh_3);
}

struct Transpose2Byte {
    uint16_t v;
};
static inline void trans_8x8_u16(
        const void* src, void* dst, const size_t src_step, const size_t dst_step) {
    uint16_t* src_ptr = (uint16_t*)src;
    uint16_t* dst_ptr = (uint16_t*)dst;
    uint16x8_t src0 = vld1q_u16(src_ptr + 0 * src_step);  // A0A1A2A3A4A5A6A7
    uint16x8_t src1 = vld1q_u16(src_ptr + 1 * src_step);  // B0B1B2B3B4B5B6B7
    uint16x8_t src2 = vld1q_u16(src_ptr + 2 * src_step);  // C0C1C2C3C4C5C6C7
    uint16x8_t src3 = vld1q_u16(src_ptr + 3 * src_step);  // D0D1D2D3D4D5D6D7
    uint16x8_t src4 = vld1q_u16(src_ptr + 4 * src_step);  // E0E1E2E3E4E5E6E7
    uint16x8_t src5 = vld1q_u16(src_ptr + 5 * src_step);  // F0F1F2F3F4F5F6F7
    uint16x8_t src6 = vld1q_u16(src_ptr + 6 * src_step);  // G0G1G2G3G4G5G6G7
    uint16x8_t src7 = vld1q_u16(src_ptr + 7 * src_step);  // H0H1H2H3H4H5H6H7

    uint16x8_t ab_low = vzip1q_u16(src0, src1);   // A0B0A1B1A2B2A3B3
    uint16x8_t ab_high = vzip2q_u16(src0, src1);  // A4B4A5B5A6B6A7B7
    uint16x8_t cd_low = vzip1q_u16(src2, src3);   // C0D0C1D1C2D2C3D3
    uint16x8_t cd_high = vzip2q_u16(src2, src3);  // C4D4C5D5C6D6C7D7
    uint16x8_t ef_low = vzip1q_u16(src4, src5);   // E0F0E1F1E2F2E3F3
    uint16x8_t ef_high = vzip2q_u16(src4, src5);  // E4F4E5F5E6F6E7F7
    uint16x8_t gh_low = vzip1q_u16(src6, src7);   // G0H0G1H1G2H2G3H3
    uint16x8_t gh_high = vzip2q_u16(src6, src7);  // G4H4G5H5G6H6G7H7

    uint16x8_t abcd_0 = vreinterpretq_u16_u32(vzip1q_u32(
            vreinterpretq_u32_u16(ab_low),
            vreinterpretq_u32_u16(cd_low)));  // A0B0C0D0A1B1C1D1
    uint16x8_t abcd_2 = vreinterpretq_u16_u32(vzip2q_u32(
            vreinterpretq_u32_u16(ab_low),
            vreinterpretq_u32_u16(cd_low)));  // A2B2C2D2A3B3C3D3
    uint16x8_t abcd_4 = vreinterpretq_u16_u32(vzip1q_u32(
            vreinterpretq_u32_u16(ab_high),
            vreinterpretq_u32_u16(cd_high)));  // A4B4C4D4A5B5C5D5
    uint16x8_t abcd_6 = vreinterpretq_u16_u32(vzip2q_u32(
            vreinterpretq_u32_u16(ab_high),
            vreinterpretq_u32_u16(cd_high)));  // A6B6C6D6A7B7C7D7
    uint16x8_t efgh_0 = vreinterpretq_u16_u32(vzip1q_u32(
            vreinterpretq_u32_u16(ef_low),
            vreinterpretq_u32_u16(gh_low)));  // E0F0G0H0E1F1G1H1
    uint16x8_t efgh_2 = vreinterpretq_u16_u32(vzip2q_u32(
            vreinterpretq_u32_u16(ef_low),
            vreinterpretq_u32_u16(gh_low)));  // E2F2G2H2E3F3G3H3
    uint16x8_t efgh_4 = vreinterpretq_u16_u32(vzip1q_u32(
            vreinterpretq_u32_u16(ef_high),
            vreinterpretq_u32_u16(gh_high)));  // E4F4G4H4E5F5G5H5
    uint16x8_t efgh_6 = vreinterpretq_u16_u32(vzip2q_u32(
            vreinterpretq_u32_u16(ef_high),
            vreinterpretq_u32_u16(gh_high)));  // E6F6G6H6E7F7G7H7

    uint16x8_t row_0 = vreinterpretq_u16_u64(
            vzip1q_u64(vreinterpretq_u64_u16(abcd_0), vreinterpretq_u64_u16(efgh_0)));
    uint16x8_t row_1 = vreinterpretq_u16_u64(
            vzip2q_u64(vreinterpretq_u64_u16(abcd_0), vreinterpretq_u64_u16(efgh_0)));
    uint16x8_t row_2 = vreinterpretq_u16_u64(
            vzip1q_u64(vreinterpretq_u64_u16(abcd_2), vreinterpretq_u64_u16(efgh_2)));
    uint16x8_t row_3 = vreinterpretq_u16_u64(
            vzip2q_u64(vreinterpretq_u64_u16(abcd_2), vreinterpretq_u64_u16(efgh_2)));
    uint16x8_t row_4 = vreinterpretq_u16_u64(
            vzip1q_u64(vreinterpretq_u64_u16(abcd_4), vreinterpretq_u64_u16(efgh_4)));
    uint16x8_t row_5 = vreinterpretq_u16_u64(
            vzip2q_u64(vreinterpretq_u64_u16(abcd_4), vreinterpretq_u64_u16(efgh_4)));
    uint16x8_t row_6 = vreinterpretq_u16_u64(
            vzip1q_u64(vreinterpretq_u64_u16(abcd_6), vreinterpretq_u64_u16(efgh_6)));
    uint16x8_t row_7 = vreinterpretq_u16_u64(
            vzip2q_u64(vreinterpretq_u64_u16(abcd_6), vreinterpretq_u64_u16(efgh_6)));

    vst1q_u16(dst_ptr + 0 * dst_step, row_0);
    vst1q_u16(dst_ptr + 1 * dst_step, row_1);
    vst1q_u16(dst_ptr + 2 * dst_step, row_2);
    vst1q_u16(dst_ptr + 3 * dst_step, row_3);
    vst1q_u16(dst_ptr + 4 * dst_step, row_4);
    vst1q_u16(dst_ptr + 5 * dst_step, row_5);
    vst1q_u16(dst_ptr + 6 * dst_step, row_6);
    vst1q_u16(dst_ptr + 7 * dst_step, row_7);
}

}  // anonymous namespace

namespace megdnn {
namespace relayout {
namespace transpose_fallback {
template <>
struct transpose_traits<TransposeByte> {
    static constexpr size_t block_size = 16;
};

template <>
void transpose_block<TransposeByte>(
        const TransposeByte* src, TransposeByte* dst, const size_t src_stride,
        const size_t dst_stride) {
    trans_16x16_u8(src, dst, src_stride, dst_stride);
}

template <>
struct transpose_traits<Transpose4Byte> {
    static constexpr size_t block_size = 8;
};

template <>
void transpose_block<Transpose4Byte>(
        const Transpose4Byte* src, Transpose4Byte* dst, const size_t src_stride,
        const size_t dst_stride) {
    trans_8x8_u32(src, dst, src_stride, dst_stride);
}

template <>
struct transpose_traits<Transpose2Byte> {
    static constexpr size_t block_size = 8;
};

template <>
void transpose_block<Transpose2Byte>(
        const Transpose2Byte* src, Transpose2Byte* dst, const size_t src_stride,
        const size_t dst_stride) {
    trans_8x8_u16(src, dst, src_stride, dst_stride);
}

}  // namespace transpose_fallback
}  // namespace relayout
}  // namespace megdnn

void aarch64::RelayoutForwardImpl::exec(
        _megdnn_tensor_in src0, _megdnn_tensor_out dst0, Handle* src_handle) {
    check_cpu_handle(src_handle);
    TensorND src = src0, dst = dst0;
    check_layout_and_canonize(src.layout, dst.layout);

    // FIXME: optimize for lowbit cases
    if (src.layout.dtype.enumv() == DTypeEnum::QuantizedS4 ||
        src.layout.dtype.enumv() == DTypeEnum::Quantized4Asymm) {
        fallback::RelayoutForwardImpl::exec(src0, dst0, src_handle);
        return;
    }
    relayout::TransposeParam trans_param;
    bool trans = relayout::is_transpose(src.layout, dst.layout, trans_param, true);
    if (trans && trans_param.c == 1 && src0.layout.dtype.size() == 1) {
        auto sptr = static_cast<TransposeByte*>(src.raw_ptr),
             dptr = static_cast<TransposeByte*>(dst.raw_ptr);
        MEGDNN_DISPATCH_CPU_KERN_OPR(transpose_fallback::transpose<TransposeByte>(
                trans_param.batch, trans_param.m, trans_param.n, sptr, dptr,
                trans_param.stride_m));
        return;
    } else if (trans && trans_param.c == 1 && src0.layout.dtype.size() == 2) {
        auto sptr = static_cast<Transpose2Byte*>(src.raw_ptr),
             dptr = static_cast<Transpose2Byte*>(dst.raw_ptr);

        MEGDNN_DISPATCH_CPU_KERN_OPR(transpose_fallback::transpose<Transpose2Byte>(
                trans_param.batch, trans_param.m, trans_param.n, sptr, dptr,
                trans_param.stride_m));
        return;
    } else if (trans && trans_param.c == 1 && src0.layout.dtype.size() == 4) {
        auto sptr = static_cast<Transpose4Byte*>(src.raw_ptr),
             dptr = static_cast<Transpose4Byte*>(dst.raw_ptr);

        MEGDNN_DISPATCH_CPU_KERN_OPR(transpose_fallback::transpose<Transpose4Byte>(
                trans_param.batch, trans_param.m, trans_param.n, sptr, dptr,
                trans_param.stride_m));
        return;
    }

    exec_after_preprocess(src, dst, trans ? &trans_param : nullptr);
}

// vim: syntax=cpp.doxygen
