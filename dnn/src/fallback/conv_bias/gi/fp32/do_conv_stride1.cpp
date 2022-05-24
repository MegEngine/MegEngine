#include <algorithm>

#include "src/fallback/conv_bias/gi/fp32/do_conv_stride1.h"
#include "src/fallback/conv_bias/gi/postprocess_helper.h"
#include "src/fallback/conv_bias/opr_impl.h"
#include "src/fallback/general_intrinsic/gi_float.h"

#include "midout.h"

MIDOUT_DECL(megdnn_fallback_conv_bias_f32_convs1)

using namespace megdnn;
using namespace fallback;
using namespace fp32;
using namespace conv_stride1;

using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;
using NCBKernParam = fallback::ConvBiasImpl::NCBKernParam;

void conv_stride1::do_conv_2x2_stride1(
        const float* src, const float* filter, float* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC) {
    const size_t tail_step = IW - OW;
    //! unroll of 2
    size_t ic = 0;
    for (; ic + 1 < IC; ic += 2) {
        const float* src_ptr = src + IW * IH * ic;
        const float* src_ptr1 = src_ptr + IW * IH;
        float* outptr = dst;

        const float* r00 = src_ptr;
        const float* r01 = src_ptr + IW;
        const float* r10 = src_ptr1;
        const float* r11 = src_ptr1 + IW;

        const float* k0 = filter + ic * 4;
        const float* k1 = k0 + 4;

        GI_FLOAT32_t _k0 = GiLoadFloat32(k0);
        GI_FLOAT32_t _k1 = GiLoadFloat32(k1);
        rep(h, OH) {
            int width = OW >> 2;

            rep(i, width) {
                GI_FLOAT32_t _r000 = GiLoadFloat32(r00);
                GI_FLOAT32_t _r010 = GiLoadFloat32(r01);
                GI_FLOAT32_t _r001 = GiLoadFloat32(r00 + 1);
                GI_FLOAT32_t _r011 = GiLoadFloat32(r01 + 1);

                GI_FLOAT32_t _r100 = GiLoadFloat32(r10);
                GI_FLOAT32_t _r110 = GiLoadFloat32(r11);
                GI_FLOAT32_t _r101 = GiLoadFloat32(r10 + 1);
                GI_FLOAT32_t _r111 = GiLoadFloat32(r11 + 1);

                GI_FLOAT32_t _sum = GiLoadFloat32(outptr);

                _sum = GiVmlaqLaneFloat32LowHalf(_sum, _r000, _k0, 0);
                _sum = GiVmlaqLaneFloat32LowHalf(_sum, _r001, _k0, 1);
                _sum = GiMlaqLaneFloat32HighHalf(_sum, _r010, _k0, 0);
                _sum = GiMlaqLaneFloat32HighHalf(_sum, _r011, _k0, 1);

                _sum = GiVmlaqLaneFloat32LowHalf(_sum, _r100, _k1, 0);
                _sum = GiVmlaqLaneFloat32LowHalf(_sum, _r101, _k1, 1);
                _sum = GiMlaqLaneFloat32HighHalf(_sum, _r110, _k1, 0);
                _sum = GiMlaqLaneFloat32HighHalf(_sum, _r111, _k1, 1);

                GiStoreFloat32(outptr, _sum);

                r00 += 4;
                r01 += 4;
                r10 += 4;
                r11 += 4;
                outptr += 4;
            }

            r00 += tail_step;
            r01 += tail_step;
            r10 += tail_step;
            r11 += tail_step;
        }
    }
    for (; ic < IC; ic++) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;

        const float* k0 = filter + ic * 4;

        GI_FLOAT32_t _k0 = GiBroadcastFloat32(k0[0]);
        GI_FLOAT32_t _k1 = GiBroadcastFloat32(k0[1]);
        GI_FLOAT32_t _k2 = GiBroadcastFloat32(k0[2]);
        GI_FLOAT32_t _k3 = GiBroadcastFloat32(k0[3]);
        rep(h, OH) {
            int width = OW >> 2;

            rep(i, width) {
                GI_FLOAT32_t _r00 = GiLoadFloat32(r0);
                GI_FLOAT32_t _r10 = GiLoadFloat32(r1);
                GI_FLOAT32_t _r01 = GiLoadFloat32(r0 + 1);
                GI_FLOAT32_t _r11 = GiLoadFloat32(r1 + 1);

                GI_FLOAT32_t _sum = GiLoadFloat32(outptr);
                GI_FLOAT32_t _sum2;

                _sum = GiMlaqFloat32(_sum, _r00, _k0);
                _sum2 = GiMultiplyFloat32(_r01, _k1);
                _sum = GiMlaqFloat32(_sum, _r10, _k2);
                _sum2 = GiMlaqFloat32(_sum2, _r11, _k3);

                _sum = GiAddFloat32(_sum, _sum2);

                GiStoreFloat32(outptr, _sum);

                r0 += 4;
                r1 += 4;
                outptr += 4;
            }

            r0 += tail_step;
            r1 += tail_step;
        }
    }
}

void conv_stride1::do_conv_3x3_stride1(
        const float* src, const float* filter, float* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC) {
    const size_t tail_step = IW - OW;

    rep(ic, IC) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;
        float* outptr2 = outptr + OW;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;
        const float* r2 = src_ptr + IW * 2;
        const float* r3 = src_ptr + IW * 3;

        const float* k0 = filter;
        const float* k1 = filter + 3;
        const float* k2 = filter + 5;

        GI_FLOAT32_t _k0123 = GiLoadFloat32(k0);
        GI_FLOAT32_t _k3456 = GiLoadFloat32(k1);
        GI_FLOAT32_t _k5678 = GiLoadFloat32(k2);
        GI_FLOAT32_t _k6789 = GiExtqFloat32(_k5678, _k5678, 1);

        size_t h = 0;
        for (; h + 1 < OH; h += 2) {
            int width = OW >> 2;

            rep(i, width) {
                GI_FLOAT32_t _sum1 = GiLoadFloat32(outptr);
                GI_FLOAT32_t _sum2 = GiBroadcastFloat32(0.f);
                GI_FLOAT32_t _sum3 = GiLoadFloat32(outptr2);
                GI_FLOAT32_t _sum4 = GiBroadcastFloat32(0.f);

                GI_FLOAT32_t _r00 = GiLoadFloat32(r0);
                GI_FLOAT32_t _r00n = GiLoadFloat32(r0 + 4);
                GI_FLOAT32_t _r01 = GiExtqFloat32(_r00, _r00n, 1);
                GI_FLOAT32_t _r02 = GiExtqFloat32(_r00, _r00n, 2);

                GI_FLOAT32_t _r10 = GiLoadFloat32(r1);
                GI_FLOAT32_t _r10n = GiLoadFloat32(r1 + 4);
                GI_FLOAT32_t _r11 = GiExtqFloat32(_r10, _r10n, 1);
                GI_FLOAT32_t _r12 = GiExtqFloat32(_r10, _r10n, 2);

                GI_FLOAT32_t _r20 = GiLoadFloat32(r2);
                GI_FLOAT32_t _r20n = GiLoadFloat32(r2 + 4);
                GI_FLOAT32_t _r21 = GiExtqFloat32(_r20, _r20n, 1);
                GI_FLOAT32_t _r22 = GiExtqFloat32(_r20, _r20n, 2);

                GI_FLOAT32_t _r30 = GiLoadFloat32(r3);
                GI_FLOAT32_t _r30n = GiLoadFloat32LowHalf(r3 + 4);
                GI_FLOAT32_t _r31 = GiExtqFloat32(_r30, _r30n, 1);
                GI_FLOAT32_t _r32 = GiExtqFloat32(_r30, _r30n, 2);

                _sum1 = GiSimdFmaLane(_sum1, _r00, _k0123, 0);
                _sum2 = GiSimdFmaLane(_sum2, _r01, _k0123, 1);
                _sum1 = GiSimdFmaLane(_sum1, _r02, _k0123, 2);
                _sum2 = GiSimdFmaLane(_sum2, _r10, _k3456, 0);
                _sum1 = GiSimdFmaLane(_sum1, _r11, _k3456, 1);
                _sum2 = GiSimdFmaLane(_sum2, _r12, _k3456, 2);
                _sum1 = GiSimdFmaLane(_sum1, _r20, _k6789, 0);
                _sum2 = GiSimdFmaLane(_sum2, _r21, _k6789, 1);
                _sum1 = GiSimdFmaLane(_sum1, _r22, _k6789, 2);

                _sum3 = GiSimdFmaLane(_sum3, _r10, _k0123, 0);
                _sum4 = GiSimdFmaLane(_sum4, _r11, _k0123, 1);
                _sum3 = GiSimdFmaLane(_sum3, _r12, _k0123, 2);
                _sum4 = GiSimdFmaLane(_sum4, _r20, _k3456, 0);
                _sum3 = GiSimdFmaLane(_sum3, _r21, _k3456, 1);
                _sum4 = GiSimdFmaLane(_sum4, _r22, _k3456, 2);
                _sum3 = GiSimdFmaLane(_sum3, _r30, _k6789, 0);
                _sum4 = GiSimdFmaLane(_sum4, _r31, _k6789, 1);
                _sum3 = GiSimdFmaLane(_sum3, _r32, _k6789, 2);

                _sum1 = GiAddFloat32(_sum1, _sum2);
                _sum3 = GiAddFloat32(_sum3, _sum4);

                GiStoreFloat32(outptr, _sum1);
                GiStoreFloat32(outptr2, _sum3);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
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

            rep(i, width) {
                GI_FLOAT32_t _sum1 = GiLoadFloat32(outptr);
                GI_FLOAT32_t _sum2 = GiBroadcastFloat32(0.f);

                GI_FLOAT32_t _r00 = GiLoadFloat32(r0);
                GI_FLOAT32_t _r00n = GiLoadFloat32(r0 + 4);
                GI_FLOAT32_t _r01 = GiExtqFloat32(_r00, _r00n, 1);
                GI_FLOAT32_t _r02 = GiExtqFloat32(_r00, _r00n, 2);

                GI_FLOAT32_t _r10 = GiLoadFloat32(r1);
                GI_FLOAT32_t _r10n = GiLoadFloat32(r1 + 4);
                GI_FLOAT32_t _r11 = GiExtqFloat32(_r10, _r10n, 1);
                GI_FLOAT32_t _r12 = GiExtqFloat32(_r10, _r10n, 2);

                GI_FLOAT32_t _r20 = GiLoadFloat32(r2);
                GI_FLOAT32_t _r20n = GiLoadFloat32(r2 + 4);
                GI_FLOAT32_t _r21 = GiExtqFloat32(_r20, _r20n, 1);
                GI_FLOAT32_t _r22 = GiExtqFloat32(_r20, _r20n, 2);

                _sum1 = GiSimdFmaLane(_sum1, _r00, _k0123, 0);
                _sum2 = GiSimdFmaLane(_sum2, _r01, _k0123, 1);
                _sum1 = GiSimdFmaLane(_sum1, _r02, _k0123, 2);
                _sum2 = GiSimdFmaLane(_sum2, _r10, _k3456, 0);
                _sum1 = GiSimdFmaLane(_sum1, _r11, _k3456, 1);
                _sum2 = GiSimdFmaLane(_sum2, _r12, _k3456, 2);
                _sum1 = GiSimdFmaLane(_sum1, _r20, _k6789, 0);
                _sum2 = GiSimdFmaLane(_sum2, _r21, _k6789, 1);
                _sum1 = GiSimdFmaLane(_sum1, _r22, _k6789, 2);

                _sum1 = GiAddFloat32(_sum1, _sum2);

                GiStoreFloat32(outptr, _sum1);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                outptr += 4;
            }
            r0 += tail_step;
            r1 += tail_step;
            r2 += tail_step;
        }

        filter += 9;
    }
}

void conv_stride1::do_conv_5x5_stride1(
        const float* src, const float* filter, float* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC) {
    const size_t tail_step = IW - OW;

    rep(ic, IC) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;
        float* outptr2 = outptr + OW;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;
        const float* r2 = src_ptr + IW * 2;
        const float* r3 = src_ptr + IW * 3;
        const float* r4 = src_ptr + IW * 4;
        const float* r5 = src_ptr + IW * 5;

        GI_FLOAT32_t _k0123 = GiLoadFloat32(filter);
        GI_FLOAT32_t _k4567 = GiLoadFloat32(filter + 4);
        GI_FLOAT32_t _k891011 = GiLoadFloat32(filter + 8);
        GI_FLOAT32_t _k12131415 = GiLoadFloat32(filter + 12);
        GI_FLOAT32_t _k16171819 = GiLoadFloat32(filter + 16);
        GI_FLOAT32_t _k20212223 = GiLoadFloat32(filter + 20);
        GI_FLOAT32_t _k24242424 = GiBroadcastFloat32(filter[24]);

        size_t h = 0;
        for (; h + 1 < OH; h += 2) {
            int width = OW >> 2;

            rep(i, width) {
                GI_FLOAT32_t _sum = GiLoadFloat32(outptr);
                GI_FLOAT32_t _sum2 = GiLoadFloat32(outptr2);

                GI_FLOAT32_t _r00 = GiLoadFloat32(r0);
                GI_FLOAT32_t _r04 = GiLoadFloat32(r0 + 4);
                GI_FLOAT32_t _r01 = GiExtqFloat32(_r00, _r04, 1);
                GI_FLOAT32_t _r02 = GiExtqFloat32(_r00, _r04, 2);
                GI_FLOAT32_t _r03 = GiExtqFloat32(_r00, _r04, 3);

                GI_FLOAT32_t _r10 = GiLoadFloat32(r1);
                GI_FLOAT32_t _r14 = GiLoadFloat32(r1 + 4);
                GI_FLOAT32_t _r11 = GiExtqFloat32(_r10, _r14, 1);
                GI_FLOAT32_t _r12 = GiExtqFloat32(_r10, _r14, 2);
                GI_FLOAT32_t _r13 = GiExtqFloat32(_r10, _r14, 3);

                GI_FLOAT32_t _r20 = GiLoadFloat32(r2);
                GI_FLOAT32_t _r24 = GiLoadFloat32(r2 + 4);
                GI_FLOAT32_t _r21 = GiExtqFloat32(_r20, _r24, 1);
                GI_FLOAT32_t _r22 = GiExtqFloat32(_r20, _r24, 2);
                GI_FLOAT32_t _r23 = GiExtqFloat32(_r20, _r24, 3);

                GI_FLOAT32_t _r30 = GiLoadFloat32(r3);
                GI_FLOAT32_t _r34 = GiLoadFloat32(r3 + 4);
                GI_FLOAT32_t _r31 = GiExtqFloat32(_r30, _r34, 1);
                GI_FLOAT32_t _r32 = GiExtqFloat32(_r30, _r34, 2);
                GI_FLOAT32_t _r33 = GiExtqFloat32(_r30, _r34, 3);

                GI_FLOAT32_t _r40 = GiLoadFloat32(r4);
                GI_FLOAT32_t _r44 = GiLoadFloat32(r4 + 4);
                GI_FLOAT32_t _r41 = GiExtqFloat32(_r40, _r44, 1);
                GI_FLOAT32_t _r42 = GiExtqFloat32(_r40, _r44, 2);
                GI_FLOAT32_t _r43 = GiExtqFloat32(_r40, _r44, 3);

                GI_FLOAT32_t _r50 = GiLoadFloat32(r5);
                GI_FLOAT32_t _r54 = GiLoadFloat32(r5 + 4);
                GI_FLOAT32_t _r51 = GiExtqFloat32(_r50, _r54, 1);
                GI_FLOAT32_t _r52 = GiExtqFloat32(_r50, _r54, 2);
                GI_FLOAT32_t _r53 = GiExtqFloat32(_r50, _r54, 3);

                _sum = GiSimdFmaLane(_sum, _r00, _k0123, 0);
                _sum = GiSimdFmaLane(_sum, _r01, _k0123, 1);
                _sum = GiSimdFmaLane(_sum, _r02, _k0123, 2);
                _sum = GiSimdFmaLane(_sum, _r03, _k0123, 3);
                _sum = GiSimdFmaLane(_sum, _r04, _k4567, 0);

                _sum = GiSimdFmaLane(_sum, _r10, _k4567, 1);
                _sum = GiSimdFmaLane(_sum, _r11, _k4567, 2);
                _sum = GiSimdFmaLane(_sum, _r12, _k4567, 3);
                _sum = GiSimdFmaLane(_sum, _r13, _k891011, 0);
                _sum = GiSimdFmaLane(_sum, _r14, _k891011, 1);

                _sum = GiSimdFmaLane(_sum, _r20, _k891011, 2);
                _sum = GiSimdFmaLane(_sum, _r21, _k891011, 3);
                _sum = GiSimdFmaLane(_sum, _r22, _k12131415, 0);
                _sum = GiSimdFmaLane(_sum, _r23, _k12131415, 1);
                _sum = GiSimdFmaLane(_sum, _r24, _k12131415, 2);

                _sum = GiSimdFmaLane(_sum, _r30, _k12131415, 3);
                _sum = GiSimdFmaLane(_sum, _r31, _k16171819, 0);
                _sum = GiSimdFmaLane(_sum, _r32, _k16171819, 1);
                _sum = GiSimdFmaLane(_sum, _r33, _k16171819, 2);
                _sum = GiSimdFmaLane(_sum, _r34, _k16171819, 3);

                _sum = GiSimdFmaLane(_sum, _r40, _k20212223, 0);
                _sum = GiSimdFmaLane(_sum, _r41, _k20212223, 1);
                _sum = GiSimdFmaLane(_sum, _r42, _k20212223, 2);
                _sum = GiSimdFmaLane(_sum, _r43, _k20212223, 3);
                _sum = GiSimdFmaLane(_sum, _r44, _k24242424, 0);

                _sum2 = GiSimdFmaLane(_sum2, _r10, _k0123, 0);
                _sum2 = GiSimdFmaLane(_sum2, _r11, _k0123, 1);
                _sum2 = GiSimdFmaLane(_sum2, _r12, _k0123, 2);
                _sum2 = GiSimdFmaLane(_sum2, _r13, _k0123, 3);
                _sum2 = GiSimdFmaLane(_sum2, _r14, _k4567, 0);

                _sum2 = GiSimdFmaLane(_sum2, _r20, _k4567, 1);
                _sum2 = GiSimdFmaLane(_sum2, _r21, _k4567, 2);
                _sum2 = GiSimdFmaLane(_sum2, _r22, _k4567, 3);
                _sum2 = GiSimdFmaLane(_sum2, _r23, _k891011, 0);
                _sum2 = GiSimdFmaLane(_sum2, _r24, _k891011, 1);

                _sum2 = GiSimdFmaLane(_sum2, _r30, _k891011, 2);
                _sum2 = GiSimdFmaLane(_sum2, _r31, _k891011, 3);
                _sum2 = GiSimdFmaLane(_sum2, _r32, _k12131415, 0);
                _sum2 = GiSimdFmaLane(_sum2, _r33, _k12131415, 1);
                _sum2 = GiSimdFmaLane(_sum2, _r34, _k12131415, 2);

                _sum2 = GiSimdFmaLane(_sum2, _r40, _k12131415, 3);
                _sum2 = GiSimdFmaLane(_sum2, _r41, _k16171819, 0);
                _sum2 = GiSimdFmaLane(_sum2, _r42, _k16171819, 1);
                _sum2 = GiSimdFmaLane(_sum2, _r43, _k16171819, 2);
                _sum2 = GiSimdFmaLane(_sum2, _r44, _k16171819, 3);

                _sum2 = GiSimdFmaLane(_sum2, _r50, _k20212223, 0);
                _sum2 = GiSimdFmaLane(_sum2, _r51, _k20212223, 1);
                _sum2 = GiSimdFmaLane(_sum2, _r52, _k20212223, 2);
                _sum2 = GiSimdFmaLane(_sum2, _r53, _k20212223, 3);
                _sum2 = GiSimdFmaLane(_sum2, _r54, _k24242424, 0);

                GiStoreFloat32(outptr, _sum);
                GiStoreFloat32(outptr2, _sum2);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;
                r5 += 4;
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
            int width = OW >> 2;

            rep(i, width) {
                GI_FLOAT32_t _sum = GiLoadFloat32(outptr);

                GI_FLOAT32_t _r00 = GiLoadFloat32(r0);
                GI_FLOAT32_t _r04 = GiLoadFloat32(r0 + 4);
                GI_FLOAT32_t _r01 = GiExtqFloat32(_r00, _r04, 1);
                GI_FLOAT32_t _r02 = GiExtqFloat32(_r00, _r04, 2);
                GI_FLOAT32_t _r03 = GiExtqFloat32(_r00, _r04, 3);

                GI_FLOAT32_t _r10 = GiLoadFloat32(r1);
                GI_FLOAT32_t _r14 = GiLoadFloat32(r1 + 4);
                GI_FLOAT32_t _r11 = GiExtqFloat32(_r10, _r14, 1);
                GI_FLOAT32_t _r12 = GiExtqFloat32(_r10, _r14, 2);
                GI_FLOAT32_t _r13 = GiExtqFloat32(_r10, _r14, 3);

                GI_FLOAT32_t _r20 = GiLoadFloat32(r2);
                GI_FLOAT32_t _r24 = GiLoadFloat32(r2 + 4);
                GI_FLOAT32_t _r21 = GiExtqFloat32(_r20, _r24, 1);
                GI_FLOAT32_t _r22 = GiExtqFloat32(_r20, _r24, 2);
                GI_FLOAT32_t _r23 = GiExtqFloat32(_r20, _r24, 3);

                GI_FLOAT32_t _r30 = GiLoadFloat32(r3);
                GI_FLOAT32_t _r34 = GiLoadFloat32(r3 + 4);
                GI_FLOAT32_t _r31 = GiExtqFloat32(_r30, _r34, 1);
                GI_FLOAT32_t _r32 = GiExtqFloat32(_r30, _r34, 2);
                GI_FLOAT32_t _r33 = GiExtqFloat32(_r30, _r34, 3);

                GI_FLOAT32_t _r40 = GiLoadFloat32(r4);
                GI_FLOAT32_t _r44 = GiLoadFloat32(r4 + 4);
                GI_FLOAT32_t _r41 = GiExtqFloat32(_r40, _r44, 1);
                GI_FLOAT32_t _r42 = GiExtqFloat32(_r40, _r44, 2);
                GI_FLOAT32_t _r43 = GiExtqFloat32(_r40, _r44, 3);

                _sum = GiSimdFmaLane(_sum, _r00, _k0123, 0);
                _sum = GiSimdFmaLane(_sum, _r01, _k0123, 1);
                _sum = GiSimdFmaLane(_sum, _r02, _k0123, 2);
                _sum = GiSimdFmaLane(_sum, _r03, _k0123, 3);
                _sum = GiSimdFmaLane(_sum, _r04, _k4567, 0);

                _sum = GiSimdFmaLane(_sum, _r10, _k4567, 1);
                _sum = GiSimdFmaLane(_sum, _r11, _k4567, 2);
                _sum = GiSimdFmaLane(_sum, _r12, _k4567, 3);
                _sum = GiSimdFmaLane(_sum, _r13, _k891011, 0);
                _sum = GiSimdFmaLane(_sum, _r14, _k891011, 1);

                _sum = GiSimdFmaLane(_sum, _r20, _k891011, 2);
                _sum = GiSimdFmaLane(_sum, _r21, _k891011, 3);
                _sum = GiSimdFmaLane(_sum, _r22, _k12131415, 0);
                _sum = GiSimdFmaLane(_sum, _r23, _k12131415, 1);
                _sum = GiSimdFmaLane(_sum, _r24, _k12131415, 2);

                _sum = GiSimdFmaLane(_sum, _r30, _k12131415, 3);
                _sum = GiSimdFmaLane(_sum, _r31, _k16171819, 0);
                _sum = GiSimdFmaLane(_sum, _r32, _k16171819, 1);
                _sum = GiSimdFmaLane(_sum, _r33, _k16171819, 2);
                _sum = GiSimdFmaLane(_sum, _r34, _k16171819, 3);

                _sum = GiSimdFmaLane(_sum, _r40, _k20212223, 0);
                _sum = GiSimdFmaLane(_sum, _r41, _k20212223, 1);
                _sum = GiSimdFmaLane(_sum, _r42, _k20212223, 2);
                _sum = GiSimdFmaLane(_sum, _r43, _k20212223, 3);
                _sum = GiSimdFmaLane(_sum, _r44, _k24242424, 0);

                GiStoreFloat32(outptr, _sum);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;
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

void conv_stride1::do_conv_7x7_stride1(
        const float* src, const float* filter, float* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC) {
    const size_t tail_step = IW - OW;

    rep(ic, IC) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;
        const float* r2 = src_ptr + IW * 2;
        const float* r3 = src_ptr + IW * 3;
        const float* r4 = src_ptr + IW * 4;
        const float* r5 = src_ptr + IW * 5;
        const float* r6 = src_ptr + IW * 6;

        const float* k0 = filter;
        const float* k1 = filter + 7;
        const float* k2 = filter + 14;
        const float* k3 = filter + 21;
        const float* k4 = filter + 28;
        const float* k5 = filter + 35;
        const float* k6 = filter + 42;

        for (size_t i = 0; i < OH; i++) {
            int width = OW >> 2;

            rep(i, width) {
                GI_FLOAT32_t _sum = GiLoadFloat32(outptr);

                GI_FLOAT32_t _k0123 = GiLoadFloat32(k0);
                GI_FLOAT32_t _k4567 = GiLoadFloat32(k0 + 4);

                GI_FLOAT32_t _r00 = GiLoadFloat32(r0);              // 0 1 2 3
                GI_FLOAT32_t _r04 = GiLoadFloat32(r0 + 4);          // 4 5 6 7
                GI_FLOAT32_t _r00n = GiLoadFloat32(r0 + 8);         // 8 9 10 11
                GI_FLOAT32_t _r01 = GiExtqFloat32(_r00, _r04, 1);   // 1 2 3 4
                GI_FLOAT32_t _r02 = GiExtqFloat32(_r00, _r04, 2);   // 2 3 4 5
                GI_FLOAT32_t _r03 = GiExtqFloat32(_r00, _r04, 3);   // 3 4 5 6
                GI_FLOAT32_t _r05 = GiExtqFloat32(_r04, _r00n, 1);  // 5 6 7 8
                GI_FLOAT32_t _r06 = GiExtqFloat32(_r04, _r00n, 2);  // 6 7 8 9

                _sum = GiSimdFmaLane(_sum, _r00, _k0123, 0);
                _sum = GiSimdFmaLane(_sum, _r01, _k0123, 1);
                _sum = GiSimdFmaLane(_sum, _r02, _k0123, 2);
                _sum = GiSimdFmaLane(_sum, _r03, _k0123, 3);
                _sum = GiSimdFmaLane(_sum, _r04, _k4567, 0);
                _sum = GiSimdFmaLane(_sum, _r05, _k4567, 1);
                _sum = GiSimdFmaLane(_sum, _r06, _k4567, 2);

                GI_FLOAT32_t _k78910 = GiLoadFloat32(k1);
                GI_FLOAT32_t _k11121314 = GiLoadFloat32(k1 + 4);

                GI_FLOAT32_t _r10 = GiLoadFloat32(r1);
                GI_FLOAT32_t _r14 = GiLoadFloat32(r1 + 4);
                GI_FLOAT32_t _r10n = GiLoadFloat32(r1 + 8);
                GI_FLOAT32_t _r11 = GiExtqFloat32(_r10, _r14, 1);
                GI_FLOAT32_t _r12 = GiExtqFloat32(_r10, _r14, 2);
                GI_FLOAT32_t _r13 = GiExtqFloat32(_r10, _r14, 3);
                GI_FLOAT32_t _r15 = GiExtqFloat32(_r14, _r10n, 1);
                GI_FLOAT32_t _r16 = GiExtqFloat32(_r14, _r10n, 2);

                _sum = GiSimdFmaLane(_sum, _r10, _k78910, 0);
                _sum = GiSimdFmaLane(_sum, _r11, _k78910, 1);
                _sum = GiSimdFmaLane(_sum, _r12, _k78910, 2);
                _sum = GiSimdFmaLane(_sum, _r13, _k78910, 3);
                _sum = GiSimdFmaLane(_sum, _r14, _k11121314, 0);
                _sum = GiSimdFmaLane(_sum, _r15, _k11121314, 1);
                _sum = GiSimdFmaLane(_sum, _r16, _k11121314, 2);

                GI_FLOAT32_t _k14151617 = GiLoadFloat32(k2);
                GI_FLOAT32_t _k18192021 = GiLoadFloat32(k2 + 4);

                GI_FLOAT32_t _r20 = GiLoadFloat32(r2);
                GI_FLOAT32_t _r24 = GiLoadFloat32(r2 + 4);
                GI_FLOAT32_t _r20n = GiLoadFloat32(r2 + 8);
                GI_FLOAT32_t _r21 = GiExtqFloat32(_r20, _r24, 1);
                GI_FLOAT32_t _r22 = GiExtqFloat32(_r20, _r24, 2);
                GI_FLOAT32_t _r23 = GiExtqFloat32(_r20, _r24, 3);
                GI_FLOAT32_t _r25 = GiExtqFloat32(_r24, _r20n, 1);
                GI_FLOAT32_t _r26 = GiExtqFloat32(_r24, _r20n, 2);

                _sum = GiSimdFmaLane(_sum, _r20, _k14151617, 0);
                _sum = GiSimdFmaLane(_sum, _r21, _k14151617, 1);
                _sum = GiSimdFmaLane(_sum, _r22, _k14151617, 2);
                _sum = GiSimdFmaLane(_sum, _r23, _k14151617, 3);
                _sum = GiSimdFmaLane(_sum, _r24, _k18192021, 0);
                _sum = GiSimdFmaLane(_sum, _r25, _k18192021, 1);
                _sum = GiSimdFmaLane(_sum, _r26, _k18192021, 2);

                GI_FLOAT32_t _k21222324 = GiLoadFloat32(k3);
                GI_FLOAT32_t _k25262728 = GiLoadFloat32(k3 + 4);

                GI_FLOAT32_t _r30 = GiLoadFloat32(r3);
                GI_FLOAT32_t _r34 = GiLoadFloat32(r3 + 4);
                GI_FLOAT32_t _r30n = GiLoadFloat32(r3 + 8);
                GI_FLOAT32_t _r31 = GiExtqFloat32(_r30, _r34, 1);
                GI_FLOAT32_t _r32 = GiExtqFloat32(_r30, _r34, 2);
                GI_FLOAT32_t _r33 = GiExtqFloat32(_r30, _r34, 3);
                GI_FLOAT32_t _r35 = GiExtqFloat32(_r34, _r30n, 1);
                GI_FLOAT32_t _r36 = GiExtqFloat32(_r34, _r30n, 2);

                _sum = GiSimdFmaLane(_sum, _r30, _k21222324, 0);
                _sum = GiSimdFmaLane(_sum, _r31, _k21222324, 1);
                _sum = GiSimdFmaLane(_sum, _r32, _k21222324, 2);
                _sum = GiSimdFmaLane(_sum, _r33, _k21222324, 3);
                _sum = GiSimdFmaLane(_sum, _r34, _k25262728, 0);
                _sum = GiSimdFmaLane(_sum, _r35, _k25262728, 1);
                _sum = GiSimdFmaLane(_sum, _r36, _k25262728, 2);

                GI_FLOAT32_t _k28293031 = GiLoadFloat32(k4);
                GI_FLOAT32_t _k32333435 = GiLoadFloat32(k4 + 4);

                GI_FLOAT32_t _r40 = GiLoadFloat32(r4);
                GI_FLOAT32_t _r44 = GiLoadFloat32(r4 + 4);
                GI_FLOAT32_t _r40n = GiLoadFloat32(r4 + 8);
                GI_FLOAT32_t _r41 = GiExtqFloat32(_r40, _r44, 1);
                GI_FLOAT32_t _r42 = GiExtqFloat32(_r40, _r44, 2);
                GI_FLOAT32_t _r43 = GiExtqFloat32(_r40, _r44, 3);
                GI_FLOAT32_t _r45 = GiExtqFloat32(_r44, _r40n, 1);
                GI_FLOAT32_t _r46 = GiExtqFloat32(_r44, _r40n, 2);

                _sum = GiSimdFmaLane(_sum, _r40, _k28293031, 0);
                _sum = GiSimdFmaLane(_sum, _r41, _k28293031, 1);
                _sum = GiSimdFmaLane(_sum, _r42, _k28293031, 2);
                _sum = GiSimdFmaLane(_sum, _r43, _k28293031, 3);
                _sum = GiSimdFmaLane(_sum, _r44, _k32333435, 0);
                _sum = GiSimdFmaLane(_sum, _r45, _k32333435, 1);
                _sum = GiSimdFmaLane(_sum, _r46, _k32333435, 2);

                GI_FLOAT32_t _k35363738 = GiLoadFloat32(k5);
                GI_FLOAT32_t _k39404142 = GiLoadFloat32(k5 + 4);

                GI_FLOAT32_t _r50 = GiLoadFloat32(r5);
                GI_FLOAT32_t _r54 = GiLoadFloat32(r5 + 4);
                GI_FLOAT32_t _r50n = GiLoadFloat32(r5 + 8);
                GI_FLOAT32_t _r51 = GiExtqFloat32(_r50, _r54, 1);
                GI_FLOAT32_t _r52 = GiExtqFloat32(_r50, _r54, 2);
                GI_FLOAT32_t _r53 = GiExtqFloat32(_r50, _r54, 3);
                GI_FLOAT32_t _r55 = GiExtqFloat32(_r54, _r50n, 1);
                GI_FLOAT32_t _r56 = GiExtqFloat32(_r54, _r50n, 2);

                _sum = GiSimdFmaLane(_sum, _r50, _k35363738, 0);
                _sum = GiSimdFmaLane(_sum, _r51, _k35363738, 1);
                _sum = GiSimdFmaLane(_sum, _r52, _k35363738, 2);
                _sum = GiSimdFmaLane(_sum, _r53, _k35363738, 3);
                _sum = GiSimdFmaLane(_sum, _r54, _k39404142, 0);
                _sum = GiSimdFmaLane(_sum, _r55, _k39404142, 1);
                _sum = GiSimdFmaLane(_sum, _r56, _k39404142, 2);

                GI_FLOAT32_t _k42434445 = GiLoadFloat32(k6);
                GI_FLOAT32_t _k46474849 =
                        GiLd1qLaneFloat32(k6 + 4 + 2, GiLoadFloat32LowHalf(k6 + 4), 2);

                GI_FLOAT32_t _r60 = GiLoadFloat32(r6);
                GI_FLOAT32_t _r64 = GiLoadFloat32(r6 + 4);
                GI_FLOAT32_t _r60n = GiLoadFloat32(r6 + 8);
                GI_FLOAT32_t _r61 = GiExtqFloat32(_r60, _r64, 1);
                GI_FLOAT32_t _r62 = GiExtqFloat32(_r60, _r64, 2);
                GI_FLOAT32_t _r63 = GiExtqFloat32(_r60, _r64, 3);
                GI_FLOAT32_t _r65 = GiExtqFloat32(_r64, _r60n, 1);
                GI_FLOAT32_t _r66 = GiExtqFloat32(_r64, _r60n, 2);

                _sum = GiSimdFmaLane(_sum, _r60, _k42434445, 0);
                _sum = GiSimdFmaLane(_sum, _r61, _k42434445, 1);
                _sum = GiSimdFmaLane(_sum, _r62, _k42434445, 2);
                _sum = GiSimdFmaLane(_sum, _r63, _k42434445, 3);
                _sum = GiSimdFmaLane(_sum, _r64, _k46474849, 0);
                _sum = GiSimdFmaLane(_sum, _r65, _k46474849, 1);
                _sum = GiSimdFmaLane(_sum, _r66, _k46474849, 2);

                GiStoreFloat32(outptr, _sum);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;
                r5 += 4;
                r6 += 4;
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

#include "src/common/simd_macro/epilogue.h"
// vim: syntax=cpp.doxygen
