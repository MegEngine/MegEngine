/**
 * \file dnn/src/arm_common/convolution/int8x8x32/conv_backdata_stride2.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#if __ARM_FEATURE_DOTPROD
#include "src/arm_common/convolution/int8x8x32/conv_backdata_stride2.h"
#include "src/common/utils.h"

#include <cstring>
#include "src/arm_common/simd_macro/marm_neon.h"

using namespace megdnn;
using namespace arm_common;
using namespace deconv;

namespace {

bool need_dst_copy(const NCBKernSizeParam& param) {
    if (param.osz[1] % 4 != 0) {
        // If the size of output is not multiples of 4, we need to copy it.
        return true;
    }
    return false;
}

void get_rectified_size(size_t IH, size_t IW, size_t OH, size_t OW, size_t FH,
                        size_t FW, size_t PH, size_t PW, size_t& IH2,
                        size_t& IW2, size_t& OW2) {
    MEGDNN_MARK_USED_VAR(OH);
    MEGDNN_MARK_USED_VAR(IW);
    //! OW should be a multiple of 4
    OW2 = (OW + 3) & ~3;
    IH2 = 2 * IH - 1 + 2 * (FH - PH - 1);
    IW2 = (OW2 - FW + 2 * PW) / 2 + 1 + (FW - PW - 1) + 16;
}

WorkspaceBundle get_bundle(const NCBKernSizeParam& param) {
    UNPACK_CONV_F32_NCB_KERN_SIZES(param);
    MEGDNN_MARK_USED_VAR(N);
    MEGDNN_MARK_USED_VAR(OC);
    MEGDNN_MARK_USED_VAR(SH);
    MEGDNN_MARK_USED_VAR(SW);
    size_t src_size = 0, dst_size = 0;
    size_t IH2, IW2, OW2;
    get_rectified_size(IH, IW, OH, OW, FH, FW, PH, PW, IH2, IW2, OW2);
    src_size = sizeof(int8_t) * IH2 * IW2;
    if (need_dst_copy(param)) {
        dst_size = sizeof(int32_t) * IC * OH * OW2;
    }
    return WorkspaceBundle(nullptr, {src_size, dst_size});
}

inline int8x16_t vqtbl1q_s8_common(int8x16_t a, uint8x16_t index) {
    int8x8x2_t src;
    src.val[0] = vget_low_s8(a);
    src.val[1] = vget_high_s8(a);
    uint8x8_t index_low = vget_low_u8(index);
    uint8x8_t index_high = vget_high_u8(index);
    int8x8_t r00 = vtbl2_s8(src, vreinterpret_s8_u8(index_low));
    int8x8_t r01 = vtbl2_s8(src, vreinterpret_s8_u8(index_high));
    int8x16_t r = vcombine_s8(r00, r01);
    return r;
}

#define CALC_0(_k_idx, _c_idx)                     \
    _elem = vqtbl1q_s8_common(_tmp, _idx##_c_idx); \
    _sum0##_c_idx = vdotq_s32(_sum0##_c_idx, _k##_k_idx, _elem);

#define CALC_1(_k_idx, _c_idx)                     \
    _elem = vqtbl1q_s8_common(_tmp, _idx##_c_idx); \
    _sum1##_c_idx = vdotq_s32(_sum1##_c_idx, _k##_k_idx, _elem);

#define CALC_2(_k1_idx, _k2_idx, _c_idx)                          \
    _elem = vqtbl1q_s8_common(_tmp, _idx##_c_idx);                \
    _sum0##_c_idx = vdotq_s32(_sum0##_c_idx, _k##_k1_idx, _elem); \
    _sum1##_c_idx = vdotq_s32(_sum1##_c_idx, _k##_k2_idx, _elem);

template <bool even>
void deconv_direct_2x2(const int8_t* src, const int8_t* filter, int32_t* dst,
                       size_t IH, size_t IW, size_t OH, size_t OW, size_t IC) {
    MEGDNN_MARK_USED_VAR(IH);
    const size_t tail_step = IW - OW / 2;

    const uint8x16_t _idx0 = {0, 1, 16, 16, 1, 2, 16, 16,
                              2, 3, 16, 16, 3, 4, 16, 16};
    const uint8x16_t _idx1 = {4, 5, 16, 16, 5, 6, 16, 16,
                              6, 7, 16, 16, 7, 8, 16, 16};
    uint8x16_t _idx_r_0, _idx_r_1;
    if (even) {
        _idx_r_0 = {0, 16, 1, 16, 2, 16, 3, 16, 4, 16, 5, 16, 6, 16, 7, 16};
        _idx_r_1 = {16, 1, 16, 2, 16, 3, 16, 4, 16, 5, 16, 6, 16, 7, 16, 8};
    } else {
        _idx_r_0 = {16, 0, 16, 1, 16, 2, 16, 3, 16, 4, 16, 5, 16, 6, 16, 7};
        _idx_r_1 = {0, 16, 1, 16, 2, 16, 3, 16, 4, 16, 5, 16, 6, 16, 7, 16};
    }
    rep(ic, IC) {
        const int8_t* src_ptr = src;
        int32_t* dst_ptr = dst + OW * OH * ic;
        int32_t* outptr = dst_ptr;
        int32_t* outptr2 = dst_ptr + OW;

        const int8_t* r0 = src_ptr;
        const int8_t* r1 = src_ptr + IW;
        const int8_t* r2 = src_ptr + 2 * IW;

        const int8_t* k0 = filter;

        int8x16_t _k0 = vreinterpretq_s8_s32(
                vdupq_n_s32(*reinterpret_cast<const int32_t*>(k0)));
        uint8x16_t _idx_k = {3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0};
        int8x16_t _k = vqtbl1q_s8_common(_k0, _idx_k);
        uint8x16_t _idx = {0, 1, 16, 16, 0, 1, 16, 16,
                           0, 1, 16, 16, 0, 1, 16, 16};
        int8x16_t _k1 = vqtbl1q_s8_common(_k, _idx);
        _idx = {2, 3, 16, 16, 2, 3, 16, 16, 2, 3, 16, 16, 2, 3, 16, 16};
        int8x16_t _k23 = vqtbl1q_s8_common(_k, _idx);

        int8x16_t _tmp, _elem;
        const int width = OW >> 2;
        size_t h = 0;
        for (; h + 1 < OH; h += 2) {
            int w = 0;
            for (; w + 4 < width; w += 4) {
                int32x4x2_t _sum00, _sum01, _sum10, _sum11;
                _sum00 = vld2q_s32(outptr);
                _sum01 = vld2q_s32(outptr + 8);
                _sum10 = vld2q_s32(outptr2);
                _sum11 = vld2q_s32(outptr2 + 8);

                int8x16_t _r0_ori = vld1q_s8(r0);
                int8x16_t _r00 = vqtbl1q_s8_common(_r0_ori, _idx_r_0);
                int8x16_t _r01 = vqtbl1q_s8_common(_r0_ori, _idx_r_1);
                int8x16_t _r1_ori = vld1q_s8(r1);
                int8x16_t _r10 = vqtbl1q_s8_common(_r1_ori, _idx_r_0);
                int8x16_t _r11 = vqtbl1q_s8_common(_r1_ori, _idx_r_1);
                int8x16_t _r2_ori = vld1q_s8(r2);
                int8x16_t _r20 = vqtbl1q_s8_common(_r2_ori, _idx_r_0);
                int8x16_t _r21 = vqtbl1q_s8_common(_r2_ori, _idx_r_1);

                int16x8x2_t r_00 = vzipq_s16(vreinterpretq_s16_s8(_r00),
                                             vreinterpretq_s16_s8(_r10));
                int8x16_t _r0 = r_00.val[0];
                int8x16_t _r2 = r_00.val[1];

                int16x8x2_t r_11 = vzipq_s16(vreinterpretq_s16_s8(_r01),
                                             vreinterpretq_s16_s8(_r11));
                int8x16_t _r1 = r_11.val[0];
                int8x16_t _r3 = r_11.val[1];

                _sum00.val[0] = vdotq_s32(_sum00.val[0], _k, _r0);
                _sum00.val[1] = vdotq_s32(_sum00.val[1], _k, _r1);
                _sum01.val[0] = vdotq_s32(_sum01.val[0], _k, _r2);
                _sum01.val[1] = vdotq_s32(_sum01.val[1], _k, _r3);

                r_00 = vzipq_s16(vreinterpretq_s16_s8(_r10),
                                 vreinterpretq_s16_s8(_r20));
                _r0 = r_00.val[0];
                _r2 = r_00.val[1];

                r_11 = vzipq_s16(vreinterpretq_s16_s8(_r11),
                                 vreinterpretq_s16_s8(_r21));
                _r1 = r_11.val[0];
                _r3 = r_11.val[1];

                _sum10.val[0] = vdotq_s32(_sum10.val[0], _k, _r0);
                _sum10.val[1] = vdotq_s32(_sum10.val[1], _k, _r1);
                _sum11.val[0] = vdotq_s32(_sum11.val[0], _k, _r2);
                _sum11.val[1] = vdotq_s32(_sum11.val[1], _k, _r3);

                vst2q_s32(outptr, _sum00);
                vst2q_s32(outptr + 8, _sum01);
                vst2q_s32(outptr2, _sum10);
                vst2q_s32(outptr2 + 8, _sum11);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr += 16;
                outptr2 += 16;
            }
            for (; w + 2 < width; w += 2) {
                int32x4_t _sum00 = vld1q_s32(outptr);
                int32x4_t _sum01 = vld1q_s32(outptr + 4);
                int32x4_t _sum10 = vld1q_s32(outptr2);
                int32x4_t _sum11 = vld1q_s32(outptr2 + 4);

                int8x16_t _r_ori = vld1q_s8(r0);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(1, 0);
                CALC_0(1, 1);

                _r_ori = vld1q_s8(r1);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(23, 1, 0);
                CALC_2(23, 1, 1);

                _r_ori = vld1q_s8(r2);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_1(23, 0);
                CALC_1(23, 1);

                vst1q_s32(outptr, _sum00);
                vst1q_s32(outptr + 4, _sum01);
                vst1q_s32(outptr2, _sum10);
                vst1q_s32(outptr2 + 4, _sum11);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                outptr += 8;
                outptr2 += 8;
            }

            for (; w < width; w++) {
                int32x4_t _sum00 = vld1q_s32(outptr);
                int32x4_t _sum10 = vld1q_s32(outptr2);

                int8x16_t _r_ori = vtranslq_s8(vld1_s8(r0));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(1, 0);

                _r_ori = vtranslq_s8(vld1_s8(r1));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(23, 1, 0);

                _r_ori = vtranslq_s8(vld1_s8(r2));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_1(23, 0);

                vst1q_s32(outptr, _sum00);
                vst1q_s32(outptr2, _sum10);

                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr += 4;
                outptr2 += 4;
            }
            r0 += tail_step + IW;
            r1 += tail_step + IW;
            r2 += tail_step + IW;
            outptr += OW;
            outptr2 += OW;
        }

        for (; h < OH; h++) {
            int w = 0;
            for (; w + 4 < width; w += 4) {
                int32x4x2_t _sum0, _sum1;
                _sum0 = vld2q_s32(outptr);
                _sum1 = vld2q_s32(outptr + 8);

                int8x16_t _r0_ori = vld1q_s8(r0);
                int8x16_t _r00 = vqtbl1q_s8_common(_r0_ori, _idx_r_0);
                int8x16_t _r01 = vqtbl1q_s8_common(_r0_ori, _idx_r_1);
                int8x16_t _r1_ori = vld1q_s8(r1);
                int8x16_t _r10 = vqtbl1q_s8_common(_r1_ori, _idx_r_0);
                int8x16_t _r11 = vqtbl1q_s8_common(_r1_ori, _idx_r_1);

                int16x8x2_t r_00 = vzipq_s16(vreinterpretq_s16_s8(_r00),
                                             vreinterpretq_s16_s8(_r10));
                int8x16_t _r0 = r_00.val[0];
                int8x16_t _r2 = r_00.val[1];

                int16x8x2_t r_11 = vzipq_s16(vreinterpretq_s16_s8(_r01),
                                             vreinterpretq_s16_s8(_r11));
                int8x16_t _r1 = r_11.val[0];
                int8x16_t _r3 = r_11.val[1];

                _sum0.val[0] = vdotq_s32(_sum0.val[0], _k, _r0);
                _sum0.val[1] = vdotq_s32(_sum0.val[1], _k, _r1);
                _sum1.val[0] = vdotq_s32(_sum1.val[0], _k, _r2);
                _sum1.val[1] = vdotq_s32(_sum1.val[1], _k, _r3);

                vst2q_s32(outptr, _sum0);
                vst2q_s32(outptr + 8, _sum1);

                r0 += 8;
                r1 += 8;
                outptr += 16;
            }
            for (; w + 2 < width; w += 2) {
                int32x4_t _sum00 = vld1q_s32(outptr);
                int32x4_t _sum01 = vld1q_s32(outptr + 4);

                int8x16_t _r_ori = vld1q_s8(r0);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(1, 0);
                CALC_0(1, 1);

                _r_ori = vld1q_s8(r1);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(23, 0);
                CALC_0(23, 1);

                vst1q_s32(outptr, _sum00);
                vst1q_s32(outptr + 4, _sum01);

                r0 += 4;
                r1 += 4;
                outptr += 8;
            }

            for (; w < width; w++) {
                int32x4_t _sum00 = vld1q_s32(outptr);

                int8x16_t _r_ori = vtranslq_s8(vld1_s8(r0));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(1, 0);

                _r_ori = vtranslq_s8(vld1_s8(r1));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(23, 0);

                vst1q_s32(outptr, _sum00);

                r0 += 2;
                r1 += 2;
                outptr += 4;
            }
            r0 += tail_step;
            r1 += tail_step;
        }

        filter += 4;
    }
}

template <bool even>
void deconv_direct_3x3(const int8_t* src, const int8_t* filter, int32_t* dst,
                       size_t IH, size_t IW, size_t OH, size_t OW, size_t IC) {
    MEGDNN_MARK_USED_VAR(IH);
    const size_t tail_step = IW - OW / 2;

    const uint8x16_t _idx0 = {0, 1, 2, 16, 1, 2, 3, 16,
                              2, 3, 4, 16, 3, 4, 5, 16};
    const uint8x16_t _idx1 = {4, 5, 6, 16, 5, 6, 7, 16,
                              6, 7, 8, 16, 7, 8, 9, 16};
    const uint8x16_t _idx2 = {8,  9,  10, 16, 9,  10, 11, 16,
                              10, 11, 12, 16, 11, 12, 13, 16};

    uint8x16_t _idx_r_0;
    if (even) {
        _idx_r_0 = {0, 16, 1, 16, 2, 16, 3, 16, 4, 16, 5, 16, 6, 16, 7, 16};
    } else {
        _idx_r_0 = {16, 0, 16, 1, 16, 2, 16, 3, 16, 4, 16, 5, 16, 6, 16, 7};
    }
    rep(ic, IC) {
        const int8_t* src_ptr = src;
        int32_t* dst_ptr = dst + OW * OH * ic;
        int32_t* outptr = dst_ptr;
        int32_t* outptr2 = outptr + OW;

        const int8_t* r0 = src_ptr;
        const int8_t* r1 = src_ptr + IW;
        const int8_t* r2 = src_ptr + IW * 2;
        const int8_t* r3 = src_ptr + IW * 3;

        const int8_t* k0 = filter;

        int8x16_t _k_tmp = vcombine_s8(vld1_s8(k0), vdup_n_s8(k0[8]));
        uint8x16_t _idx = {8, 7, 6, 16, 8, 7, 6, 16, 8, 7, 6, 16, 8, 7, 6, 16};
        int8x16_t _k12 = vqtbl1q_s8_common(_k_tmp, _idx);
        _idx = {5, 4, 3, 16, 5, 4, 3, 16, 5, 4, 3, 16, 5, 4, 3, 16};
        int8x16_t _k345 = vqtbl1q_s8_common(_k_tmp, _idx);
        _idx = {2, 1, 0, 16, 2, 1, 0, 16, 2, 1, 0, 16, 2, 1, 0, 16};
        int8x16_t _k678 = vqtbl1q_s8_common(_k_tmp, _idx);

        int8x16_t _tmp, _elem;
        size_t h = 0;
        for (; h + 1 < OH; h += 2) {
            int width = OW >> 2;

            int w = 0;
            for (; w + 3 < width; w += 3) {
                //! As the inner kernel read 16 elements, and IW is times of 16
                int32x4_t _sum00 = vld1q_s32(outptr);
                int32x4_t _sum01 = vld1q_s32(outptr + 4);
                int32x4_t _sum02 = vld1q_s32(outptr + 8);
                int32x4_t _sum10 = vld1q_s32(outptr2);
                int32x4_t _sum11 = vld1q_s32(outptr2 + 4);
                int32x4_t _sum12 = vld1q_s32(outptr2 + 8);

                int8x16_t _r_ori = vld1q_s8(r0);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(12, 0);
                CALC_0(12, 1);
                CALC_0(12, 2);

                _r_ori = vld1q_s8(r1);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(345, 12, 0);
                CALC_2(345, 12, 1);
                CALC_2(345, 12, 2);

                _r_ori = vld1q_s8(r2);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(678, 345, 0);
                CALC_2(678, 345, 1);
                CALC_2(678, 345, 2);

                _r_ori = vld1q_s8(r3);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_1(678, 0);
                CALC_1(678, 1);
                CALC_1(678, 2);

                vst1q_s32(outptr, _sum00);
                vst1q_s32(outptr + 4, _sum01);
                vst1q_s32(outptr + 8, _sum02);
                vst1q_s32(outptr2, _sum10);
                vst1q_s32(outptr2 + 4, _sum11);
                vst1q_s32(outptr2 + 8, _sum12);

                r0 += 6;
                r1 += 6;
                r2 += 6;
                r3 += 6;
                outptr += 12;
                outptr2 += 12;
            }
            for (; w < width; w++) {
                int32x4_t _sum00 = vld1q_s32(outptr);
                int32x4_t _sum10 = vld1q_s32(outptr2);

                int8x16_t _r_ori = vtranslq_s8(vld1_s8(r0));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(12, 0);

                _r_ori = vtranslq_s8(vld1_s8(r1));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(345, 12, 0);

                _r_ori = vtranslq_s8(vld1_s8(r2));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(678, 345, 0);

                _r_ori = vtranslq_s8(vld1_s8(r3));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_1(678, 0);

                vst1q_s32(outptr, _sum00);
                vst1q_s32(outptr2, _sum10);

                r0 += 2;
                r1 += 2;
                r2 += 2;
                r3 += 2;
                outptr += 4;
                outptr2 += 4;
            }

            r0 += tail_step + IW;
            r1 += tail_step + IW;
            r2 += tail_step + IW;
            r3 += tail_step + IW;

            outptr += OW;
            outptr2 += OW;
        }

        for (; h < OH; h++) {
            int width = OW >> 2;

            int w = 0;
            for (; w + 3 < width; w += 3) {
                int32x4_t _sum00 = vld1q_s32(outptr);
                int32x4_t _sum01 = vld1q_s32(outptr + 4);
                int32x4_t _sum02 = vld1q_s32(outptr + 8);

                int8x16_t _r_ori = vld1q_s8(r0);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(12, 0);
                CALC_0(12, 1);
                CALC_0(12, 2);

                _r_ori = vld1q_s8(r1);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(345, 0);
                CALC_0(345, 1);
                CALC_0(345, 2);

                _r_ori = vld1q_s8(r2);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(678, 0);
                CALC_0(678, 1);
                CALC_0(678, 2);

                vst1q_s32(outptr, _sum00);
                vst1q_s32(outptr + 4, _sum01);
                vst1q_s32(outptr + 8, _sum02);

                r0 += 6;
                r1 += 6;
                r2 += 6;
                outptr += 12;
            }
            for (; w < width; w++) {
                int32x4_t _sum00 = vld1q_s32(outptr);

                int8x16_t _r_ori = vtranslq_s8(vld1_s8(r0));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(12, 0);

                _r_ori = vtranslq_s8(vld1_s8(r1));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(345, 0);

                _r_ori = vtranslq_s8(vld1_s8(r2));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(678, 0);

                vst1q_s32(outptr, _sum00);

                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr += 4;
            }
            r0 += tail_step;
            r1 += tail_step;
            r2 += tail_step;
        }

        filter += 9;
    }
}

#undef CALC_0
#undef CALC_1
#undef CALC_2

#define CALC_0(_k00_idx, _k01_idx, _c_idx)                         \
    _elem = vqtbl1q_s8_common(_tmp, _idx##_c_idx##0);              \
    _sum0##_c_idx = vdotq_s32(_sum0##_c_idx, _k##_k00_idx, _elem); \
    _elem = vqtbl1q_s8_common(_tmp, _idx##_c_idx##1);              \
    _sum0##_c_idx = vdotq_s32(_sum0##_c_idx, _k##_k01_idx, _elem);

#define CALC_1(_k00_idx, _k01_idx, _c_idx)                         \
    _elem = vqtbl1q_s8_common(_tmp, _idx##_c_idx##0);              \
    _sum1##_c_idx = vdotq_s32(_sum1##_c_idx, _k##_k00_idx, _elem); \
    _elem = vqtbl1q_s8_common(_tmp, _idx##_c_idx##1);              \
    _sum1##_c_idx = vdotq_s32(_sum1##_c_idx, _k##_k01_idx, _elem);

#define CALC_2(_k00_idx, _k01_idx, _k10_idx, _k11_idx, _c_idx)     \
    _elem = vqtbl1q_s8_common(_tmp, _idx##_c_idx##0);              \
    _sum0##_c_idx = vdotq_s32(_sum0##_c_idx, _k##_k00_idx, _elem); \
    _sum1##_c_idx = vdotq_s32(_sum1##_c_idx, _k##_k10_idx, _elem); \
    _elem = vqtbl1q_s8_common(_tmp, _idx##_c_idx##1);              \
    _sum0##_c_idx = vdotq_s32(_sum0##_c_idx, _k##_k01_idx, _elem); \
    _sum1##_c_idx = vdotq_s32(_sum1##_c_idx, _k##_k11_idx, _elem);

template <bool even>
void deconv_direct_5x5(const int8_t* src, const int8_t* filter, int32_t* dst,
                       size_t IH, size_t IW, size_t OH, size_t OW, size_t IC) {
    MEGDNN_MARK_USED_VAR(IH);
    const size_t tail_step = IW - OW / 2;

    const uint8x16_t _idx00 = {0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
    const uint8x16_t _idx01 = {4, 16, 16, 16, 5, 16, 16, 16,
                               6, 16, 16, 16, 7, 16, 16, 16};
    const uint8x16_t _idx10 = {4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10};
    const uint8x16_t _idx11 = {8,  16, 16, 16, 9,  16, 16, 16,
                               10, 16, 16, 16, 11, 16, 16, 16};
    const uint8x16_t _idx20 = {8,  9,  10, 11, 9,  10, 11, 12,
                               10, 11, 12, 13, 11, 12, 13, 14};
    const uint8x16_t _idx21 = {12, 16, 16, 16, 13, 16, 16, 16,
                               14, 16, 16, 16, 15, 16, 16, 16};

    uint8x16_t _idx_r_0;
    if (even) {
        _idx_r_0 = {0, 16, 1, 16, 2, 16, 3, 16, 4, 16, 5, 16, 6, 16, 7, 16};
    } else {
        _idx_r_0 = {16, 0, 16, 1, 16, 2, 16, 3, 16, 4, 16, 5, 16, 6, 16, 7};
    }
    int8x16_t _tmp, _elem;
    rep(ic, IC) {
        const int8_t* src_ptr = src;
        int32_t* dst_ptr = dst + OW * OH * ic;
        int32_t* outptr = dst_ptr;
        int32_t* outptr2 = outptr + OW;

        const int8_t* r0 = src_ptr;
        const int8_t* r1 = src_ptr + IW;
        const int8_t* r2 = src_ptr + IW * 2;
        const int8_t* r3 = src_ptr + IW * 3;
        const int8_t* r4 = src_ptr + IW * 4;
        const int8_t* r5 = src_ptr + IW * 5;

        const int8_t* k0 = filter;

        int8x16_t _k = vld1q_s8(k0 + 9);
        //! filter row 1
        uint8x16_t _idx = {15, 14, 13, 12, 15, 14, 13, 12,
                           15, 14, 13, 12, 15, 14, 13, 12};
        int8x16_t _k123 = vqtbl1q_s8_common(_k, _idx);
        _idx = {11, 16, 16, 16, 11, 16, 16, 16, 11, 16, 16, 16, 11, 16, 16, 16};
        int8x16_t _k4 = vqtbl1q_s8_common(_k, _idx);
        //! filter row 2
        _idx = {10, 9, 8, 7, 10, 9, 8, 7, 10, 9, 8, 7, 10, 9, 8, 7};
        int8x16_t _k5678 = vqtbl1q_s8_common(_k, _idx);
        _idx = {6, 16, 16, 16, 6, 16, 16, 16, 6, 16, 16, 16, 6, 16, 16, 16};
        int8x16_t _k9 = vqtbl1q_s8_common(_k, _idx);
        //! filter row 3
        _idx = {5, 4, 3, 2, 5, 4, 3, 2, 5, 4, 3, 2, 5, 4, 3, 2};
        int8x16_t _k10111213 = vqtbl1q_s8_common(_k, _idx);
        _idx = {1, 16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16, 1, 16, 16, 16};
        int8x16_t _k14 = vqtbl1q_s8_common(_k, _idx);
        //! 9 10 11 12 -> 13 14 15 16 -> 17 18 19 20 -> 21 22 23 24
        _k = vld1q_s8(k0);
        //! filter row 4
        _idx = {9, 8, 7, 6, 9, 8, 7, 6, 9, 8, 7, 6, 9, 8, 7, 6};
        int8x16_t _k15161718 = vqtbl1q_s8_common(_k, _idx);
        _idx = {5, 16, 16, 16, 5, 16, 16, 16, 5, 16, 16, 16, 5, 16, 16, 16};
        int8x16_t _k19 = vqtbl1q_s8_common(_k, _idx);
        //! filter row 5
        _idx = {4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1};
        int8x16_t _k20212223 = vqtbl1q_s8_common(_k, _idx);
        _idx = {0, 16, 16, 16, 0, 16, 16, 16, 0, 16, 16, 16, 0, 16, 16, 16};
        int8x16_t _k24 = vqtbl1q_s8_common(_k, _idx);

        const int width = OW >> 2;
        size_t h = 0;
        for (; h + 1 < OH; h += 2) {
            int w = 0;
            for (; w + 3 < width; w += 3) {
                //! As the inner kernel read 16 elements, and IW is times of 16
                int32x4_t _sum00 = vld1q_s32(outptr);
                int32x4_t _sum01 = vld1q_s32(outptr + 4);
                int32x4_t _sum02 = vld1q_s32(outptr + 8);
                int32x4_t _sum10 = vld1q_s32(outptr2);
                int32x4_t _sum11 = vld1q_s32(outptr2 + 4);
                int32x4_t _sum12 = vld1q_s32(outptr2 + 8);

                int8x16_t _r_ori = vld1q_s8(r0);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(123, 4, 0);
                CALC_0(123, 4, 1);
                CALC_0(123, 4, 2);

                _r_ori = vld1q_s8(r1);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(5678, 9, 123, 4, 0);
                CALC_2(5678, 9, 123, 4, 1);
                CALC_2(5678, 9, 123, 4, 2);

                _r_ori = vld1q_s8(r2);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(10111213, 14, 5678, 9, 0);
                CALC_2(10111213, 14, 5678, 9, 1);
                CALC_2(10111213, 14, 5678, 9, 2);

                _r_ori = vld1q_s8(r3);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(15161718, 19, 10111213, 14, 0);
                CALC_2(15161718, 19, 10111213, 14, 1);
                CALC_2(15161718, 19, 10111213, 14, 2);

                _r_ori = vld1q_s8(r4);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(20212223, 24, 15161718, 19, 0);
                CALC_2(20212223, 24, 15161718, 19, 1);
                CALC_2(20212223, 24, 15161718, 19, 2);

                _r_ori = vld1q_s8(r5);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_1(20212223, 24, 0);
                CALC_1(20212223, 24, 1);
                CALC_1(20212223, 24, 2);

                vst1q_s32(outptr, _sum00);
                vst1q_s32(outptr + 4, _sum01);
                vst1q_s32(outptr + 8, _sum02);
                vst1q_s32(outptr2, _sum10);
                vst1q_s32(outptr2 + 4, _sum11);
                vst1q_s32(outptr2 + 8, _sum12);

                r0 += 6;
                r1 += 6;
                r2 += 6;
                r3 += 6;
                r4 += 6;
                r5 += 6;
                outptr += 12;
                outptr2 += 12;
            }
            for (; w < width; w++) {
                int32x4_t _sum00 = vld1q_s32(outptr);
                int32x4_t _sum10 = vld1q_s32(outptr2);

                int8x16_t _r_ori = vtranslq_s8(vld1_s8(r0));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(123, 4, 0);

                _r_ori = vtranslq_s8(vld1_s8(r1));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(5678, 9, 123, 4, 0);

                _r_ori = vtranslq_s8(vld1_s8(r2));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(10111213, 14, 5678, 9, 0);

                _r_ori = vtranslq_s8(vld1_s8(r3));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(15161718, 19, 10111213, 14, 0);

                _r_ori = vtranslq_s8(vld1_s8(r4));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(20212223, 24, 15161718, 19, 0);

                _r_ori = vtranslq_s8(vld1_s8(r5));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_1(20212223, 24, 0);

                vst1q_s32(outptr, _sum00);
                vst1q_s32(outptr2, _sum10);

                r0 += 2;
                r1 += 2;
                r2 += 2;
                r3 += 2;
                r4 += 2;
                r5 += 2;
                outptr += 4;
                outptr2 += 4;
            }

            r0 += tail_step + IW;
            r1 += tail_step + IW;
            r2 += tail_step + IW;
            r3 += tail_step + IW;
            r4 += tail_step + IW;
            r5 += tail_step + IW;

            outptr += OW;
            outptr2 += OW;
        }

        for (; h < OH; h++) {
            int w = 0;
            for (; w + 3 < width; w += 3) {
                int32x4_t _sum00 = vld1q_s32(outptr);
                int32x4_t _sum01 = vld1q_s32(outptr + 4);
                int32x4_t _sum02 = vld1q_s32(outptr + 8);

                int8x16_t _r_ori = vld1q_s8(r0);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(123, 4, 0);
                CALC_0(123, 4, 1);
                CALC_0(123, 4, 2);

                _r_ori = vld1q_s8(r1);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(5678, 9, 0);
                CALC_0(5678, 9, 1);
                CALC_0(5678, 9, 2);

                _r_ori = vld1q_s8(r2);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(10111213, 14, 0);
                CALC_0(10111213, 14, 1);
                CALC_0(10111213, 14, 2);

                _r_ori = vld1q_s8(r3);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(15161718, 19, 0);
                CALC_0(15161718, 19, 1);
                CALC_0(15161718, 19, 2);

                _r_ori = vld1q_s8(r4);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(20212223, 24, 0);
                CALC_0(20212223, 24, 1);
                CALC_0(20212223, 24, 2);

                vst1q_s32(outptr, _sum00);
                vst1q_s32(outptr + 4, _sum01);
                vst1q_s32(outptr + 8, _sum02);

                r0 += 6;
                r1 += 6;
                r2 += 6;
                r3 += 6;
                r4 += 6;
                outptr += 12;
            }
            for (; w < width; w++) {
                int32x4_t _sum00 = vld1q_s32(outptr);

                int8x16_t _r_ori = vtranslq_s8(vld1_s8(r0));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(123, 4, 0);

                _r_ori = vtranslq_s8(vld1_s8(r1));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(5678, 9, 0);

                _r_ori = vtranslq_s8(vld1_s8(r2));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(10111213, 14, 0);

                _r_ori = vtranslq_s8(vld1_s8(r3));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(15161718, 19, 0);

                _r_ori = vtranslq_s8(vld1_s8(r4));
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(20212223, 24, 0);

                vst1q_s32(outptr, _sum00);

                r0 += 2;
                r1 += 2;
                r2 += 2;
                r3 += 2;
                r4 += 2;
                outptr += 4;
            }
            r0 += tail_step;
            r1 += tail_step;
            r2 += tail_step;
            r3 += tail_step;
            r4 += tail_step;
        }

        filter += 25;
    }
}

template <bool even>
void deconv_direct_7x7(const int8_t* src, const int8_t* filter, int32_t* dst,
                       size_t IH, size_t IW, size_t OH, size_t OW, size_t IC) {
    MEGDNN_MARK_USED_VAR(IH);
    const size_t tail_step = IW - OW / 2;

    const uint8x16_t _idx00 = {0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
    const uint8x16_t _idx01 = {4, 5, 6, 16, 5, 6, 7, 16,
                               6, 7, 8, 16, 7, 8, 9, 16};
    const uint8x16_t _idx10 = {4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10};
    const uint8x16_t _idx11 = {8,  9,  10, 16, 9,  10, 11, 16,
                               10, 11, 12, 16, 11, 12, 13, 16};

    uint8x16_t _idx_r_0;
    if (even) {
        _idx_r_0 = {0, 16, 1, 16, 2, 16, 3, 16, 4, 16, 5, 16, 6, 16, 7, 16};
    } else {
        _idx_r_0 = {16, 0, 16, 1, 16, 2, 16, 3, 16, 4, 16, 5, 16, 6, 16, 7};
    }
    int8x16_t _tmp, _elem;
    rep(ic, IC) {
        const int8_t* src_ptr = src;
        int32_t* dst_ptr = dst + OW * OH * ic;
        int32_t* outptr = dst_ptr;
        int32_t* outptr2 = outptr + OW;

        const int8_t* r0 = src_ptr;
        const int8_t* r1 = src_ptr + IW;
        const int8_t* r2 = src_ptr + IW * 2;
        const int8_t* r3 = src_ptr + IW * 3;
        const int8_t* r4 = src_ptr + IW * 4;
        const int8_t* r5 = src_ptr + IW * 5;
        const int8_t* r6 = src_ptr + IW * 6;
        const int8_t* r7 = src_ptr + IW * 7;

        const int8_t* k0 = filter;

        int8x16_t _k = vld1q_s8(k0 + 33);
        //! filter row 1
        uint8x16_t _idx = {15, 14, 13, 12, 15, 14, 13, 12,
                           15, 14, 13, 12, 15, 14, 13, 12};
        int8x16_t _k123 = vqtbl1q_s8_common(_k, _idx);
        _idx = {11, 10, 9, 16, 11, 10, 9, 16, 11, 10, 9, 16, 11, 10, 9, 16};
        int8x16_t _k456 = vqtbl1q_s8_common(_k, _idx);
        //! filter row 2
        _idx = {8, 7, 6, 5, 8, 7, 6, 5, 8, 7, 6, 5, 8, 7, 6, 5};
        int8x16_t _k78910 = vqtbl1q_s8_common(_k, _idx);
        _idx = {4, 3, 2, 16, 4, 3, 2, 16, 4, 3, 2, 16, 4, 3, 2, 16};
        int8x16_t _k111213 = vqtbl1q_s8_common(_k, _idx);

        //! 12 13 14 15 -> 16 17 18 19 -> 20 21 22 23 -> 24 25 26 27
        _k = vld1q_s8(k0 + 19);
        //! filter row 3
        _idx = {15, 14, 13, 12, 15, 14, 13, 12, 15, 14, 13, 12, 15, 14, 13, 12};
        int8x16_t _k14151617 = vqtbl1q_s8_common(_k, _idx);
        _idx = {11, 10, 9, 16, 11, 10, 9, 16, 11, 10, 9, 16, 11, 10, 9, 16};
        int8x16_t _k181920 = vqtbl1q_s8_common(_k, _idx);
        //! filter row 4
        _idx = {8, 7, 6, 5, 8, 7, 6, 5, 8, 7, 6, 5, 8, 7, 6, 5};
        int8x16_t _k21222324 = vqtbl1q_s8_common(_k, _idx);
        _idx = {4, 3, 2, 16, 4, 3, 2, 16, 4, 3, 2, 16, 4, 3, 2, 16};
        int8x16_t _k252627 = vqtbl1q_s8_common(_k, _idx);

        //! 24 25 26 27->28 29 30 31 -> 32 33 34 35 -> 36 37 38 39
        _k = vld1q_s8(k0 + 5);
        //! filter row 5
        _idx = {15, 14, 13, 12, 15, 14, 13, 12, 15, 14, 13, 12, 15, 14, 13, 12};
        int8x16_t _k28293031 = vqtbl1q_s8_common(_k, _idx);
        _idx = {11, 10, 9, 16, 11, 10, 9, 16, 11, 10, 9, 16, 11, 10, 9, 16};
        int8x16_t _k323334 = vqtbl1q_s8_common(_k, _idx);

        //! 33 34 35 36 -> 37 38 39 40 -> 41 42 43 44 -> 45 46 47 48
        _k = vld1q_s8(k0);
        //! filter row 6
        _idx = {13, 12, 11, 10, 13, 12, 11, 10, 13, 12, 11, 10, 13, 12, 11, 10};
        int8x16_t _k35363738 = vqtbl1q_s8_common(_k, _idx);
        _idx = {9, 8, 7, 16, 9, 8, 7, 16, 9, 8, 7, 16, 9, 8, 7, 16};
        int8x16_t _k394041 = vqtbl1q_s8_common(_k, _idx);

        //! filter row 7
        _idx = {6, 5, 4, 3, 6, 5, 4, 3, 6, 5, 4, 3, 6, 5, 4, 3};
        int8x16_t _k42434445 = vqtbl1q_s8_common(_k, _idx);
        _idx = {2, 1, 0, 16, 2, 1, 0, 16, 2, 1, 0, 16, 2, 1, 0, 16};
        int8x16_t _k464748 = vqtbl1q_s8_common(_k, _idx);

        const int width = OW >> 2;
        size_t h = 0;
        for (; h + 1 < OH; h += 2) {
            int w = 0;
            for (; w + 2 < width; w += 2) {
                //! As the inner kernel read 16 elements, and IW is times of 16
                int32x4_t _sum00 = vld1q_s32(outptr);
                int32x4_t _sum01 = vld1q_s32(outptr + 4);
                int32x4_t _sum10 = vld1q_s32(outptr2);
                int32x4_t _sum11 = vld1q_s32(outptr2 + 4);

                int8x16_t _r_ori = vld1q_s8(r0);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(123, 456, 0);
                CALC_0(123, 456, 1);

                _r_ori = vld1q_s8(r1);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(78910, 111213, 123, 456, 0);
                CALC_2(78910, 111213, 123, 456, 1);

                _r_ori = vld1q_s8(r2);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(14151617, 181920, 78910, 111213, 0);
                CALC_2(14151617, 181920, 78910, 111213, 1);

                _r_ori = vld1q_s8(r3);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(21222324, 252627, 14151617, 181920, 0);
                CALC_2(21222324, 252627, 14151617, 181920, 1);

                _r_ori = vld1q_s8(r4);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(28293031, 323334, 21222324, 252627, 0);
                CALC_2(28293031, 323334, 21222324, 252627, 1);

                _r_ori = vld1q_s8(r5);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(35363738, 394041, 28293031, 323334, 0);
                CALC_2(35363738, 394041, 28293031, 323334, 1);

                _r_ori = vld1q_s8(r6);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(42434445, 464748, 35363738, 394041, 0);
                CALC_2(42434445, 464748, 35363738, 394041, 1);

                _r_ori = vld1q_s8(r7);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_1(42434445, 464748, 0);
                CALC_1(42434445, 464748, 1);

                vst1q_s32(outptr, _sum00);
                vst1q_s32(outptr + 4, _sum01);
                vst1q_s32(outptr2, _sum10);
                vst1q_s32(outptr2 + 4, _sum11);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;
                r5 += 4;
                r6 += 4;
                r7 += 4;
                outptr += 8;
                outptr2 += 8;
            }
            for (; w < width; w++) {
                int32x4_t _sum00 = vld1q_s32(outptr);
                int32x4_t _sum10 = vld1q_s32(outptr2);

                int8x16_t _r_ori = vld1q_s8(r0);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(123, 456, 0);

                _r_ori = vld1q_s8(r1);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(78910, 111213, 123, 456, 0);

                _r_ori = vld1q_s8(r2);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(14151617, 181920, 78910, 111213, 0);

                _r_ori = vld1q_s8(r3);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(21222324, 252627, 14151617, 181920, 0);

                _r_ori = vld1q_s8(r4);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(28293031, 323334, 21222324, 252627, 0);

                _r_ori = vld1q_s8(r5);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(35363738, 394041, 28293031, 323334, 0);

                _r_ori = vld1q_s8(r6);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_2(42434445, 464748, 35363738, 394041, 0);

                _r_ori = vld1q_s8(r7);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_1(42434445, 464748, 0);

                vst1q_s32(outptr, _sum00);
                vst1q_s32(outptr2, _sum10);

                r0 += 2;
                r1 += 2;
                r2 += 2;
                r3 += 2;
                r4 += 2;
                r5 += 2;
                r6 += 2;
                r7 += 2;
                outptr += 4;
                outptr2 += 4;
            }

            r0 += tail_step + IW;
            r1 += tail_step + IW;
            r2 += tail_step + IW;
            r3 += tail_step + IW;
            r4 += tail_step + IW;
            r5 += tail_step + IW;
            r6 += tail_step + IW;
            r7 += tail_step + IW;

            outptr += OW;
            outptr2 += OW;
        }

        for (; h < OH; h++) {
            int w = 0;
            for (; w + 2 < width; w += 2) {
                int32x4_t _sum00 = vld1q_s32(outptr);
                int32x4_t _sum01 = vld1q_s32(outptr + 4);

                int8x16_t _r_ori = vld1q_s8(r0);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(123, 456, 0);
                CALC_0(123, 456, 1);

                _r_ori = vld1q_s8(r1);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(78910, 111213, 0);
                CALC_0(78910, 111213, 1);

                _r_ori = vld1q_s8(r2);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(14151617, 181920, 0);
                CALC_0(14151617, 181920, 1);

                _r_ori = vld1q_s8(r3);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(21222324, 252627, 0);
                CALC_0(21222324, 252627, 1);

                _r_ori = vld1q_s8(r4);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(28293031, 323334, 0);
                CALC_0(28293031, 323334, 1);

                _r_ori = vld1q_s8(r5);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(35363738, 394041, 0);
                CALC_0(35363738, 394041, 1);

                _r_ori = vld1q_s8(r6);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(42434445, 464748, 0);
                CALC_0(42434445, 464748, 1);

                vst1q_s32(outptr, _sum00);
                vst1q_s32(outptr + 4, _sum01);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;
                r5 += 4;
                r6 += 4;
                outptr += 8;
            }
            for (; w < width; w++) {
                int32x4_t _sum00 = vld1q_s32(outptr);

                int8x16_t _r_ori = vld1q_s8(r0);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(123, 456, 0);

                _r_ori = vld1q_s8(r1);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(78910, 111213, 0);

                _r_ori = vld1q_s8(r2);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(14151617, 181920, 0);

                _r_ori = vld1q_s8(r3);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(21222324, 252627, 0);

                _r_ori = vld1q_s8(r4);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(28293031, 323334, 0);

                _r_ori = vld1q_s8(r5);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(35363738, 394041, 0);

                _r_ori = vld1q_s8(r6);
                _tmp = vqtbl1q_s8_common(_r_ori, _idx_r_0);
                CALC_0(42434445, 464748, 0);

                vst1q_s32(outptr, _sum00);

                r0 += 2;
                r1 += 2;
                r2 += 2;
                r3 += 2;
                r4 += 2;
                r5 += 2;
                r6 += 2;
                outptr += 4;
            }
            r0 += tail_step;
            r1 += tail_step;
            r2 += tail_step;
            r3 += tail_step;
            r4 += tail_step;
            r5 += tail_step;
            r6 += tail_step;
        }

        filter += 49;
    }
}

#undef CALC_0
#undef CALC_1
#undef CALC_2

}  // anonymous namespace

size_t deconv::get_workspace_in_bytes_stride2_int8x8x32_dot(
        const NCBKernSizeParam& param) {
    return get_bundle(param).total_size_in_bytes();
}

bool deconv::can_stride2_int8x8x32_dot(const NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0], FW = fm.spatial[1], OC = fm.ocpg,
         PH = fm.padding[0], PW = fm.padding[1];
    bool avaiable = fm.format == param::Convolution::Format::NCHW &&
                    !fm.should_flip && fm.spatial_ndim == 2 &&
                    fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
                    fm.stride[0] == 2 && fm.stride[1] == 2 && FH == FW &&
                    (FH == 2 || FH == 3 || FH == 5 || FH == 7) &&
                    FH >= PH + 1 && FW >= PW + 1;

    avaiable &= (param.filter_type.enumv() == DTypeEnum::QuantizedS8 ||
                 param.filter_type.enumv() == DTypeEnum::Int8) &&
                (param.grad_type.enumv() == DTypeEnum::QuantizedS32 ||
                 param.grad_type.enumv() == DTypeEnum::Int32);
    return avaiable && ((FH == 2 && OC <= 4) || (FH == 3 && OC <= 8) ||
                        (FH == 5 && OC <= 16) || (FH == 7 && OC < 32));
}

void deconv::stride2_int8x8x32_dot(const NCBKernParam& param) {
    auto bundle = get_bundle(param);
    bundle.set(param.workspace_ptr);
    UNPACK_CONV_F32_NCB_KERN_SIZES(param);
    MEGDNN_MARK_USED_VAR(SH);
    MEGDNN_MARK_USED_VAR(SW);
    size_t IH2, IW2, OW2;
    int padding_h = FH - PH - 1, padding_w = FW - PW - 1;

    get_rectified_size(IH, IW, OH, OW, FH, FW, PH, PW, IH2, IW2, OW2);

    using Func = std::function<void(const int8_t*, const int8_t*, int32_t*,
                                    size_t, size_t, size_t, size_t, size_t)>;
    Func conv = nullptr;
    if (FH == 2) {
        if ((padding_w & 1) == 0)
            conv = deconv_direct_2x2<true>;
        else
            conv = deconv_direct_2x2<false>;
    } else if (FH == 3) {
        if ((padding_w & 1) == 0)
            conv = deconv_direct_3x3<true>;
        else
            conv = deconv_direct_3x3<false>;
    } else if (FH == 5) {
        if ((padding_w & 1) == 0)
            conv = deconv_direct_5x5<true>;
        else
            conv = deconv_direct_5x5<false>;
    } else if (FH == 7) {
        if ((padding_w & 1) == 0)
            conv = deconv_direct_7x7<true>;
        else
            conv = deconv_direct_7x7<false>;
    } else {
        megdnn_assert(0);
    }

    bool need_dst_copy_var = need_dst_copy(param);
    int8_t* base_src_ptr = const_cast<int8_t*>(param.diff<int8_t>());
    int32_t* base_dst_ptr = param.grad<int32_t>();
    const int8_t* fptr = param.filter<int8_t>();

    for (size_t n = 0; n < N; ++n) {
        int32_t* dptr_copied = static_cast<int32_t*>(bundle.get(1));
        int32_t* dptr_ori = base_dst_ptr + n * param.out_bs;
        int32_t* dptr = nullptr;
        size_t OW_real = OW;
        if (need_dst_copy_var) {
            dptr = dptr_copied;
            OW_real = OW2;
        } else {
            dptr = dptr_ori;
        }
        std::memset(dptr, 0, sizeof(int32_t) * IC * OH * OW_real);

        int8_t* sptr_ori = base_src_ptr + n * param.inp_bs;
        int8_t* sptr_copied = static_cast<int8_t*>(bundle.get(0));
        int8_t* sptr = nullptr;
        rep(oc, OC) {
            std::memset(sptr_copied, 0, sizeof(int8_t) * IH2 * IW2);
            copy_plane_in_bytes(sptr_copied + padding_h * IW2 + padding_w / 2,
                                sptr_ori + oc * IH * IW, IH,
                                IW * sizeof(int8_t), 2 * IW2 * sizeof(int8_t),
                                IW * sizeof(int8_t));
            sptr = sptr_copied;

            conv(sptr, fptr + oc * IC * FH * FW, dptr, IH2, IW2, OH, OW_real,
                 IC);
        }
        if (need_dst_copy_var) {
            for (size_t ic = 0; ic < IC; ++ic) {
                copy_plane_in_bytes(dptr_ori + ic * OH * OW,
                                    dptr + ic * OH * OW2, OH,
                                    OW * sizeof(int32_t), OW * sizeof(int32_t),
                                    OW2 * sizeof(int32_t));
            }
        }
    }
}

#endif
// vim: syntax=cpp.doxygen
