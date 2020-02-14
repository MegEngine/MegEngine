/**
 * \file dnn/src/fallback/convolution/do_conv_stride2_decl.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
// simd_macro/*_helper.h should be included before including this file.

#pragma once

#if defined(MEGDNN_YCM_COMPILE) && !defined(MEGDNN_SIMD_NAME)
#warning "include x86 simd macros for ycm to work"
#define MEGDNN_YCM_COMPILE_CLEANUP
#include "src/x86/simd_macro/sse_helper.h"
#endif

#include "src/common/macro_helper.h"
#include "megdnn/arch.h"

#include "./opr_impl.h"

namespace megdnn {
namespace conv_general_simd {

template <bool add_to_dst>
MEGDNN_SIMD_ATTRIBUTE_TARGET
void do_conv_2x2_stride2(const float *src, const float *filter, float *dst,
        size_t IH, size_t IW,
        size_t OH, size_t OW,
        size_t PH, size_t PW)
{
    size_t OH_start = div_ceil<size_t>(PH, 2),
           OH_stop = div_floor<size_t>(IH+PH-2, 2) + 1,
           OW_start = div_ceil<size_t>(PW, 2),
           OW_stop = div_floor<size_t>(IW+PW-2, 2) + 1;
    OH_start = std::min<size_t>(OH, OH_start);
    OH_stop = std::min<size_t>(OH, OH_stop);
    OW_start = std::min<size_t>(OW, OW_start);
    OW_stop = std::min<size_t>(OW, OW_stop);
    auto run_single = [&](size_t oh, size_t ow) {
        if (!add_to_dst) {
            dst[oh*OW + ow] = 0;
        }
        for (size_t fh = 0; fh < 2; ++fh)
        for (size_t fw = 0; fw < 2; ++fw)
        {
            size_t ih = oh*2+fh-PH;
            size_t iw = ow*2+fw-PW;
            if (ih < IH && iw < IW) {
                dst[oh*OW + ow] += src[ih*IW + iw] * filter[fh*2 + fw];
            }
        }
    };
    for (size_t oh = 0; oh < OH_start; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    for (size_t oh = OH_start; oh < OH_stop; ++oh) {
        for (size_t ow = 0; ow < OW_start; ++ow) run_single(oh, ow);
        for (size_t ow = OW_stop; ow < OW; ++ow) run_single(oh, ow);
    }
    for (size_t oh = OH_stop; oh < OH; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    // 4xMEGDNN_SIMD_WIDTH block
    size_t oh = OH_start;
    float cache_even[8*MEGDNN_SIMD_WIDTH];
    float cache_odd[8*MEGDNN_SIMD_WIDTH];
    const float* sptrs[2] = {
        cache_even + 0,
        cache_odd + 0
    };
    for (; oh+4 <= OH_stop; oh += 4) {
        size_t ih = oh*2-PH;
        size_t ow = OW_start;
        for (; ow + MEGDNN_SIMD_WIDTH <= OW_stop; ow += MEGDNN_SIMD_WIDTH) {
            size_t iw = ow*2-PW;
            float * __restrict dptr = dst + oh*OW + ow;
            const float * __restrict sptr = src + ih*IW + iw;
            const float * __restrict fptr = filter;
            //do prefetch for current line and the first two/three blocks of the next line
            const int prefetch_index_input = (ow + 4*MEGDNN_SIMD_WIDTH) < OW_stop?
                    ih*IW + iw + 4*MEGDNN_SIMD_WIDTH:
                    (((ow + 4*MEGDNN_SIMD_WIDTH - OW_stop)/MEGDNN_SIMD_WIDTH) * MEGDNN_SIMD_WIDTH + OW_start) * 2 - PW;
            const int prefetch_index_output = (ow + 4*MEGDNN_SIMD_WIDTH) < OW_stop?
                    oh*OW + ow + 4*MEGDNN_SIMD_WIDTH:
                    (((ow + 4*MEGDNN_SIMD_WIDTH - OW_stop)/MEGDNN_SIMD_WIDTH) * MEGDNN_SIMD_WIDTH + OW_start);
            const float* src_prefetch = src + prefetch_index_input;
            const float* dst_prefetch = dst + prefetch_index_output;
            for(int iw_id = 0;iw_id < 8;++iw_id){
                __builtin_prefetch(src_prefetch + iw_id * IW,0,3);
            }
            for(int ow_id = 0;ow_id < 4;++ow_id){
                __builtin_prefetch(dst_prefetch + ow_id * OW,1,3);
            }

            MEGDNN_SIMD_TYPE d0, d1, d2, d3;
            MEGDNN_SIMD_TYPE k0, k1, s;
            {
                // do transpose
                for (size_t i = 0; i < 8; ++i) {
                    MEGDNN_SIMD_TYPE s_low = MEGDNN_SIMD_LOADU(sptr + i*IW),
                                     s_high = MEGDNN_SIMD_LOADU(sptr + i*IW +
                                             MEGDNN_SIMD_WIDTH);
                    MEGDNN_SIMD_TYPE s_result0, s_result1;
                    MEGDNN_SIMD_UZP(s_low, s_high, s_result0, s_result1);
                    MEGDNN_SIMD_STOREU(cache_even + i*MEGDNN_SIMD_WIDTH, s_result0);
                    MEGDNN_SIMD_STOREU(cache_odd + i*MEGDNN_SIMD_WIDTH, s_result1);
                }
            }
            if (add_to_dst) {
                d0 = MEGDNN_SIMD_LOADU(dptr + 0*OW);
                d1 = MEGDNN_SIMD_LOADU(dptr + 1*OW);
                d2 = MEGDNN_SIMD_LOADU(dptr + 2*OW);
                d3 = MEGDNN_SIMD_LOADU(dptr + 3*OW);
            } else {
                d0 = MEGDNN_SIMD_SETZERO();
                d1 = MEGDNN_SIMD_SETZERO();
                d2 = MEGDNN_SIMD_SETZERO();
                d3 = MEGDNN_SIMD_SETZERO();
            }
            for (size_t fw = 0; fw < 2; ++fw) {
                k0 = MEGDNN_SIMD_SET1(fptr[0*2 + fw]);
                k1 = MEGDNN_SIMD_SET1(fptr[1*2 + fw]);

                // line 0
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 0*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k0, s, d0);

                // line 1
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 1*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k1, s, d0);

                // line 2
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 2*MEGDNN_SIMD_WIDTH);
                d1 = MEGDNN_SIMD_FMADD(k0, s, d1);

                // line 3
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 3*MEGDNN_SIMD_WIDTH);
                d1 = MEGDNN_SIMD_FMADD(k1, s, d1);

                // line 4
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 4*MEGDNN_SIMD_WIDTH);
                d2 = MEGDNN_SIMD_FMADD(k0, s, d2);

                // line 5
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 5*MEGDNN_SIMD_WIDTH);
                d2 = MEGDNN_SIMD_FMADD(k1, s, d2);

                // line 6
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 6*MEGDNN_SIMD_WIDTH);
                d3 = MEGDNN_SIMD_FMADD(k0, s, d3);

                // line 7
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 7*MEGDNN_SIMD_WIDTH);
                d3 = MEGDNN_SIMD_FMADD(k1, s, d3);
            }
            MEGDNN_SIMD_STOREU(dptr + 0*OW, d0);
            MEGDNN_SIMD_STOREU(dptr + 1*OW, d1);
            MEGDNN_SIMD_STOREU(dptr + 2*OW, d2);
            MEGDNN_SIMD_STOREU(dptr + 3*OW, d3);
        }
        //do prefetch for the 3th block in the next line
        const int prefetch_index_input = (ih + 8)* IW + 6*MEGDNN_SIMD_WIDTH + 2*OW_start - PW;
        const int prefetch_index_output = (oh + 4)* OW + 3*MEGDNN_SIMD_WIDTH + OW_start;
        const float* src_prefetch = src + prefetch_index_input;
        const float* dst_prefetch = dst + prefetch_index_output;
        for(int iw_id = 0;iw_id < 8;++iw_id){
            __builtin_prefetch(src_prefetch + iw_id * IW,0,3);
        }
        for(int ow_id = 0;ow_id < 4;++ow_id){
            __builtin_prefetch(dst_prefetch + ow_id * OW,1,3);
        }
        for (; ow < OW_stop; ++ow) {
            run_single(oh+0, ow);
            run_single(oh+1, ow);
            run_single(oh+2, ow);
            run_single(oh+3, ow);
        }
    }
    for (; oh < OH_stop; ++oh) {
        for (size_t ow = OW_start; ow < OW_stop; ++ow) {
            run_single(oh, ow);
        }
    }
}

template <bool add_to_dst>
MEGDNN_SIMD_ATTRIBUTE_TARGET
void do_conv_3x3_stride2(const float *src, const float *filter, float *dst,
        size_t IH, size_t IW,
        size_t OH, size_t OW,
        size_t PH, size_t PW)
{
    size_t OH_start = div_ceil<size_t>(PH, 2),
           OH_stop = div_floor<size_t>(IH+PH-3, 2) + 1,
           OW_start = div_ceil<size_t>(PW, 2),
           OW_stop = div_floor<size_t>(IW+PW-3, 2) + 1;
    OH_start = std::min<size_t>(OH, OH_start);
    OH_stop = std::min<size_t>(OH, OH_stop);
    OW_start = std::min<size_t>(OW, OW_start);
    OW_stop = std::min<size_t>(OW, OW_stop);
    auto run_single = [&](size_t oh, size_t ow) {
        if (!add_to_dst) {
            dst[oh*OW + ow] = 0;
        }
        for (size_t fh = 0; fh < 3; ++fh)
        for (size_t fw = 0; fw < 3; ++fw)
        {
            size_t ih = oh*2+fh-PH;
            size_t iw = ow*2+fw-PW;
            if (ih < IH && iw < IW) {
                dst[oh*OW + ow] += src[ih*IW + iw] * filter[fh*3 + fw];
            }
        }
    };
    for (size_t oh = 0; oh < OH_start; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    for (size_t oh = OH_start; oh < OH_stop; ++oh) {
        for (size_t ow = 0; ow < OW_start; ++ow) run_single(oh, ow);
        for (size_t ow = OW_stop; ow < OW; ++ow) run_single(oh, ow);
    }
    for (size_t oh = OH_stop; oh < OH; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    // 4xMEGDNN_SIMD_WIDTH block
    size_t oh = OH_start;
    float cache_even[9*2*MEGDNN_SIMD_WIDTH];
    float cache_odd[9*2*MEGDNN_SIMD_WIDTH];
    const float* sptrs[3] = {
        cache_even + 0,
        cache_odd + 0,
        cache_even + 1
    };
    for (; oh+4 <= OH_stop; oh += 4) {
        size_t ih = oh*2-PH;
        size_t ow = OW_start;
        for (; ow+ 4 * MEGDNN_SIMD_WIDTH < OW_stop; ow += MEGDNN_SIMD_WIDTH) {
            size_t iw = ow*2-PW;
            float * __restrict dptr = dst + oh*OW + ow;
            const float * __restrict sptr = src + ih*IW + iw;
            const float * __restrict fptr = filter;

            //do prefetch for current line
            const int prefetch_index_input = ih*IW + iw + 4*MEGDNN_SIMD_WIDTH;
            const int prefetch_index_output = oh*OW + ow + 4*MEGDNN_SIMD_WIDTH;
            const float* src_prefetch = src + prefetch_index_input;
            const float* dst_prefetch = dst + prefetch_index_output;
            for(int iw_id = 0;iw_id < 9;++iw_id){
                __builtin_prefetch(src_prefetch + iw_id * IW,0,3);
            }
            for(int ow_id = 0;ow_id < 4;++ow_id){
                __builtin_prefetch(dst_prefetch + ow_id * OW,1,3);
            }

            MEGDNN_SIMD_TYPE d0, d1, d2, d3;
            MEGDNN_SIMD_TYPE k0, k1, k2, s;
            {
                // do transpose
                for (size_t i = 0; i < 9; ++i) {
                    MEGDNN_SIMD_TYPE s_low = MEGDNN_SIMD_LOADU(sptr + i*IW);
                    MEGDNN_SIMD_TYPE s_high = MEGDNN_SIMD_LOADU(sptr + i*IW +
                            MEGDNN_SIMD_WIDTH);
                    MEGDNN_SIMD_TYPE s_result0, s_result1;
                    MEGDNN_SIMD_UZP(s_low, s_high, s_result0, s_result1);
                    MEGDNN_SIMD_STOREU(cache_even + i*2*MEGDNN_SIMD_WIDTH,
                            s_result0);
                    MEGDNN_SIMD_STOREU(cache_odd + i*2*MEGDNN_SIMD_WIDTH,
                            s_result1);
                    // last elements
                    cache_even[(i*2+1)*MEGDNN_SIMD_WIDTH] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH];
                }
            }
            if (add_to_dst) {
                d0 = MEGDNN_SIMD_LOADU(dptr + 0*OW);
                d1 = MEGDNN_SIMD_LOADU(dptr + 1*OW);
                d2 = MEGDNN_SIMD_LOADU(dptr + 2*OW);
                d3 = MEGDNN_SIMD_LOADU(dptr + 3*OW);
            } else {
                d0 = MEGDNN_SIMD_SETZERO();
                d1 = MEGDNN_SIMD_SETZERO();
                d2 = MEGDNN_SIMD_SETZERO();
                d3 = MEGDNN_SIMD_SETZERO();
            }
            for (size_t fw = 0; fw < 3; ++fw) {
                k0 = MEGDNN_SIMD_SET1(fptr[0*3 + fw]);
                k1 = MEGDNN_SIMD_SET1(fptr[1*3 + fw]);
                k2 = MEGDNN_SIMD_SET1(fptr[2*3 + fw]);

                // line 0
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 0*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k0, s, d0);

                // line 1
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 1*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k1, s, d0);

                // line 2
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 2*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k2, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k0, s, d1);

                // line 3
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 3*2*MEGDNN_SIMD_WIDTH);
                d1 = MEGDNN_SIMD_FMADD(k1, s, d1);

                // line 4
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 4*2*MEGDNN_SIMD_WIDTH);
                d1 = MEGDNN_SIMD_FMADD(k2, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k0, s, d2);

                // line 5
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 5*2*MEGDNN_SIMD_WIDTH);
                d2 = MEGDNN_SIMD_FMADD(k1, s, d2);

                // line 6
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 6*2*MEGDNN_SIMD_WIDTH);
                d2 = MEGDNN_SIMD_FMADD(k2, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k0, s, d3);

                // line 7
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 7*2*MEGDNN_SIMD_WIDTH);
                d3 = MEGDNN_SIMD_FMADD(k1, s, d3);

                // line 2*MEGDNN_SIMD_WIDTH
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 8*2*MEGDNN_SIMD_WIDTH);
                d3 = MEGDNN_SIMD_FMADD(k2, s, d3);
            }
            MEGDNN_SIMD_STOREU(dptr + 0*OW, d0);
            MEGDNN_SIMD_STOREU(dptr + 1*OW, d1);
            MEGDNN_SIMD_STOREU(dptr + 2*OW, d2);
            MEGDNN_SIMD_STOREU(dptr + 3*OW, d3);
        }
        for (; ow+MEGDNN_SIMD_WIDTH <= OW_stop; ow += MEGDNN_SIMD_WIDTH) {
            size_t iw = ow*2-PW;
            float * __restrict dptr = dst + oh*OW + ow;
            const float * __restrict sptr = src + ih*IW + iw;
            const float * __restrict fptr = filter;

            //do prefetch for the first two/three blocks of the next line
            const int prefetch_index_input = (ih + 8) * IW +
                (((ow + 4*MEGDNN_SIMD_WIDTH - OW_stop)/MEGDNN_SIMD_WIDTH) * MEGDNN_SIMD_WIDTH + OW_start) * 2 - PW;
            const int prefetch_index_output = (oh + 4) * OW +
                 (((ow + 4*MEGDNN_SIMD_WIDTH - OW_stop)/MEGDNN_SIMD_WIDTH) * MEGDNN_SIMD_WIDTH + OW_start);
            const float* src_prefetch = src + prefetch_index_input;
            const float* dst_prefetch = dst + prefetch_index_output;
            for(int iw_id = 0;iw_id < 9;++iw_id){
                __builtin_prefetch(src_prefetch + iw_id * IW,0,3);
            }
            for(int ow_id = 0;ow_id < 4;++ow_id){
                __builtin_prefetch(dst_prefetch + ow_id * OW,1,3);
            }

            MEGDNN_SIMD_TYPE d0, d1, d2, d3;
            MEGDNN_SIMD_TYPE k0, k1, k2, s;
            {
                // do transpose
                for (size_t i = 0; i < 9; ++i) {
                    MEGDNN_SIMD_TYPE s_low = MEGDNN_SIMD_LOADU(sptr + i*IW);
                    MEGDNN_SIMD_TYPE s_high = MEGDNN_SIMD_LOADU(sptr + i*IW +
                            MEGDNN_SIMD_WIDTH);
                    MEGDNN_SIMD_TYPE s_result0, s_result1;
                    MEGDNN_SIMD_UZP(s_low, s_high, s_result0, s_result1);
                    MEGDNN_SIMD_STOREU(cache_even + i*2*MEGDNN_SIMD_WIDTH,
                            s_result0);
                    MEGDNN_SIMD_STOREU(cache_odd + i*2*MEGDNN_SIMD_WIDTH,
                            s_result1);
                    // last elements
                    cache_even[(i*2+1)*MEGDNN_SIMD_WIDTH] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH];
                }
            }
            if (add_to_dst) {
                d0 = MEGDNN_SIMD_LOADU(dptr + 0*OW);
                d1 = MEGDNN_SIMD_LOADU(dptr + 1*OW);
                d2 = MEGDNN_SIMD_LOADU(dptr + 2*OW);
                d3 = MEGDNN_SIMD_LOADU(dptr + 3*OW);
            } else {
                d0 = MEGDNN_SIMD_SETZERO();
                d1 = MEGDNN_SIMD_SETZERO();
                d2 = MEGDNN_SIMD_SETZERO();
                d3 = MEGDNN_SIMD_SETZERO();
            }
            for (size_t fw = 0; fw < 3; ++fw) {
                k0 = MEGDNN_SIMD_SET1(fptr[0*3 + fw]);
                k1 = MEGDNN_SIMD_SET1(fptr[1*3 + fw]);
                k2 = MEGDNN_SIMD_SET1(fptr[2*3 + fw]);

                // line 0
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 0*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k0, s, d0);

                // line 1
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 1*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k1, s, d0);

                // line 2
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 2*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k2, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k0, s, d1);

                // line 3
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 3*2*MEGDNN_SIMD_WIDTH);
                d1 = MEGDNN_SIMD_FMADD(k1, s, d1);

                // line 4
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 4*2*MEGDNN_SIMD_WIDTH);
                d1 = MEGDNN_SIMD_FMADD(k2, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k0, s, d2);

                // line 5
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 5*2*MEGDNN_SIMD_WIDTH);
                d2 = MEGDNN_SIMD_FMADD(k1, s, d2);

                // line 6
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 6*2*MEGDNN_SIMD_WIDTH);
                d2 = MEGDNN_SIMD_FMADD(k2, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k0, s, d3);

                // line 7
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 7*2*MEGDNN_SIMD_WIDTH);
                d3 = MEGDNN_SIMD_FMADD(k1, s, d3);

                // line 2*MEGDNN_SIMD_WIDTH
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 8*2*MEGDNN_SIMD_WIDTH);
                d3 = MEGDNN_SIMD_FMADD(k2, s, d3);
            }
            MEGDNN_SIMD_STOREU(dptr + 0*OW, d0);
            MEGDNN_SIMD_STOREU(dptr + 1*OW, d1);
            MEGDNN_SIMD_STOREU(dptr + 2*OW, d2);
            MEGDNN_SIMD_STOREU(dptr + 3*OW, d3);
        }

        //do prefetch for the 3th block in the next line
        const int prefetch_index_input = (ih + 8)* IW + 6*MEGDNN_SIMD_WIDTH + 2*OW_start - PW;
        const int prefetch_index_output = (oh + 4)* OW + 3*MEGDNN_SIMD_WIDTH + OW_start;
        const float* src_prefetch = src + prefetch_index_input;
        const float* dst_prefetch = dst + prefetch_index_output;
        for(int iw_id = 0;iw_id < 9;++iw_id){
            __builtin_prefetch(src_prefetch + iw_id * IW,0,3);
        }
        for(int ow_id = 0;ow_id < 4;++ow_id){
            __builtin_prefetch(dst_prefetch + ow_id * OW,1,3);
        }

        for (; ow < OW_stop; ++ow) {
            run_single(oh+0, ow);
            run_single(oh+1, ow);
            run_single(oh+2, ow);
            run_single(oh+3, ow);
        }
    }
    for (; oh < OH_stop; ++oh) {
        for (size_t ow = OW_start; ow < OW_stop; ++ow) {
            run_single(oh, ow);
        }
    }
}

template <bool add_to_dst>
MEGDNN_SIMD_ATTRIBUTE_TARGET
void do_conv_5x5_stride2(const float *src, const float *filter, float *dst,
        size_t IH, size_t IW,
        size_t OH, size_t OW,
        size_t PH, size_t PW)
{
    size_t OH_start = div_ceil<size_t>(PH, 2),
           OH_stop = div_floor<size_t>(IH+PH-5, 2) + 1,
           OW_start = div_ceil<size_t>(PW, 2),
           OW_stop = div_floor<size_t>(IW+PW-5, 2) + 1;
    OH_start = std::min<size_t>(OH, OH_start);
    OH_stop = std::min<size_t>(OH, OH_stop);
    OW_start = std::min<size_t>(OW, OW_start);
    OW_stop = std::min<size_t>(OW, OW_stop);
    auto run_single = [&](size_t oh, size_t ow) {
        if (!add_to_dst) {
            dst[oh*OW + ow] = 0;
        }
        for (size_t fh = 0; fh < 5; ++fh)
        for (size_t fw = 0; fw < 5; ++fw)
        {
            size_t ih = oh*2+fh-PH;
            size_t iw = ow*2+fw-PW;
            if (ih < IH && iw < IW) {
                dst[oh*OW + ow] += src[ih*IW + iw] * filter[fh*5 + fw];
            }
        }
    };
    for (size_t oh = 0; oh < OH_start; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    for (size_t oh = OH_start; oh < OH_stop; ++oh) {
        for (size_t ow = 0; ow < OW_start; ++ow) run_single(oh, ow);
        for (size_t ow = OW_stop; ow < OW; ++ow) run_single(oh, ow);
    }
    for (size_t oh = OH_stop; oh < OH; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    // 4x4 block
    size_t oh = OH_start;
    float cache_even[11*2*MEGDNN_SIMD_WIDTH];
    float cache_odd[11*2*MEGDNN_SIMD_WIDTH];
    const float* sptrs[5] = {
        cache_even + 0,
        cache_odd + 0,
        cache_even + 1,
        cache_odd + 1,
        cache_even + 2,
    };
    for (; oh+4 <= OH_stop; oh += 4) {
        size_t ih = oh*2-PH;
        size_t ow = OW_start;
        for (; ow+4*MEGDNN_SIMD_WIDTH < OW_stop; ow += MEGDNN_SIMD_WIDTH) {
            size_t iw = ow*2-PW;
            float * __restrict dptr = dst + oh*OW + ow;
            const float * __restrict sptr = src + ih*IW + iw;
            const float * __restrict fptr = filter;

            //do prefetch for current line
            const int prefetch_index_input = ih*IW + iw + 4*MEGDNN_SIMD_WIDTH;
            const int prefetch_index_output = oh*OW + ow + 4*MEGDNN_SIMD_WIDTH;
            const float* src_prefetch = src + prefetch_index_input;
            const float* dst_prefetch = dst + prefetch_index_output;
            for(int iw_id = 0;iw_id < 11;++iw_id){
                __builtin_prefetch(src_prefetch + iw_id * IW,0,3);
            }
            for(int ow_id = 0;ow_id < 4;++ow_id){
                __builtin_prefetch(dst_prefetch + ow_id * OW,1,3);
            }

            MEGDNN_SIMD_TYPE d0, d1, d2, d3;
            MEGDNN_SIMD_TYPE k0, k1, k2, k3, k4, s;
            {
                // do transpose
                for (size_t i = 0; i < 11; ++i) {
                    MEGDNN_SIMD_TYPE s_low = MEGDNN_SIMD_LOADU(sptr + i*IW);
                    MEGDNN_SIMD_TYPE s_high = MEGDNN_SIMD_LOADU(sptr + i*IW +
                            MEGDNN_SIMD_WIDTH);
                    MEGDNN_SIMD_TYPE s_result0, s_result1;
                    MEGDNN_SIMD_UZP(s_low, s_high, s_result0, s_result1);
                    MEGDNN_SIMD_STOREU(cache_even + i*2*MEGDNN_SIMD_WIDTH,
                            s_result0);
                    MEGDNN_SIMD_STOREU(cache_odd + i*2*MEGDNN_SIMD_WIDTH,
                            s_result1);
                    // last elements
                    cache_even[(i*2+1)*MEGDNN_SIMD_WIDTH] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH + 0];
                    cache_odd[(i*2+1)*MEGDNN_SIMD_WIDTH] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH + 1];
                    cache_even[(i*2+1)*MEGDNN_SIMD_WIDTH+1] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH + 2];
                }
            }
            if (add_to_dst) {
                d0 = MEGDNN_SIMD_LOADU(dptr + 0*OW);
                d1 = MEGDNN_SIMD_LOADU(dptr + 1*OW);
                d2 = MEGDNN_SIMD_LOADU(dptr + 2*OW);
                d3 = MEGDNN_SIMD_LOADU(dptr + 3*OW);
            } else {
                d0 = MEGDNN_SIMD_SETZERO();
                d1 = MEGDNN_SIMD_SETZERO();
                d2 = MEGDNN_SIMD_SETZERO();
                d3 = MEGDNN_SIMD_SETZERO();
            }
            for (size_t fw = 0; fw < 5; ++fw) {
                k0 = MEGDNN_SIMD_SET1(fptr[0*5 + fw]);
                k1 = MEGDNN_SIMD_SET1(fptr[1*5 + fw]);
                k2 = MEGDNN_SIMD_SET1(fptr[2*5 + fw]);
                k3 = MEGDNN_SIMD_SET1(fptr[3*5 + fw]);
                k4 = MEGDNN_SIMD_SET1(fptr[4*5 + fw]);

                // line 0
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 0*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k0, s, d0);

                // line 1
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 1*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k1, s, d0);

                // line 2
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 2*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k2, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k0, s, d1);

                // line 3
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 3*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k3, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k1, s, d1);

                // line 4
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 4*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k4, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k2, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k0, s, d2);

                // line 5
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 5*2*MEGDNN_SIMD_WIDTH);
                d1 = MEGDNN_SIMD_FMADD(k3, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k1, s, d2);

                // line 6
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 6*2*MEGDNN_SIMD_WIDTH);
                d1 = MEGDNN_SIMD_FMADD(k4, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k2, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k0, s, d3);

                // line 7
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 7*2*MEGDNN_SIMD_WIDTH);
                d2 = MEGDNN_SIMD_FMADD(k3, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k1, s, d3);

                // line 2*MEGDNN_SIMD_WIDTH
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 8*2*MEGDNN_SIMD_WIDTH);
                d2 = MEGDNN_SIMD_FMADD(k4, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k2, s, d3);

                // line 9
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 9*2*MEGDNN_SIMD_WIDTH);
                d3 = MEGDNN_SIMD_FMADD(k3, s, d3);

                // line 9
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 10*2*MEGDNN_SIMD_WIDTH);
                d3 = MEGDNN_SIMD_FMADD(k4, s, d3);
            }
            MEGDNN_SIMD_STOREU(dptr + 0*OW, d0);
            MEGDNN_SIMD_STOREU(dptr + 1*OW, d1);
            MEGDNN_SIMD_STOREU(dptr + 2*OW, d2);
            MEGDNN_SIMD_STOREU(dptr + 3*OW, d3);
        }
        for (; ow+MEGDNN_SIMD_WIDTH <= OW_stop; ow += MEGDNN_SIMD_WIDTH) {
            size_t iw = ow*2-PW;
            float * __restrict dptr = dst + oh*OW + ow;
            const float * __restrict sptr = src + ih*IW + iw;
            const float * __restrict fptr = filter;

            //do prefetch for the first two/three blocks of the next line
            const int prefetch_index_input = (ih + 8) * IW +
                (((ow + 4*MEGDNN_SIMD_WIDTH - OW_stop)/MEGDNN_SIMD_WIDTH) * MEGDNN_SIMD_WIDTH + OW_start) * 2 - PW;
            const int prefetch_index_output = (oh + 4) * OW +
                 (((ow + 4*MEGDNN_SIMD_WIDTH - OW_stop)/MEGDNN_SIMD_WIDTH) * MEGDNN_SIMD_WIDTH + OW_start);
            const float* src_prefetch = src + prefetch_index_input;
            const float* dst_prefetch = dst + prefetch_index_output;
            for(int iw_id = 0;iw_id < 11;++iw_id){
                __builtin_prefetch(src_prefetch + iw_id * IW,0,3);
            }
            for(int ow_id = 0;ow_id < 4;++ow_id){
                __builtin_prefetch(dst_prefetch + ow_id * OW,1,3);
            }

            MEGDNN_SIMD_TYPE d0, d1, d2, d3;
            MEGDNN_SIMD_TYPE k0, k1, k2, k3, k4, s;
            {
                // do transpose
                for (size_t i = 0; i < 11; ++i) {
                    MEGDNN_SIMD_TYPE s_low = MEGDNN_SIMD_LOADU(sptr + i*IW);
                    MEGDNN_SIMD_TYPE s_high = MEGDNN_SIMD_LOADU(sptr + i*IW +
                            MEGDNN_SIMD_WIDTH);
                    MEGDNN_SIMD_TYPE s_result0, s_result1;
                    MEGDNN_SIMD_UZP(s_low, s_high, s_result0, s_result1);
                    MEGDNN_SIMD_STOREU(cache_even + i*2*MEGDNN_SIMD_WIDTH,
                            s_result0);
                    MEGDNN_SIMD_STOREU(cache_odd + i*2*MEGDNN_SIMD_WIDTH,
                            s_result1);
                    // last elements
                    cache_even[(i*2+1)*MEGDNN_SIMD_WIDTH] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH + 0];
                    cache_odd[(i*2+1)*MEGDNN_SIMD_WIDTH] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH + 1];
                    cache_even[(i*2+1)*MEGDNN_SIMD_WIDTH+1] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH + 2];
                }
            }
            if (add_to_dst) {
                d0 = MEGDNN_SIMD_LOADU(dptr + 0*OW);
                d1 = MEGDNN_SIMD_LOADU(dptr + 1*OW);
                d2 = MEGDNN_SIMD_LOADU(dptr + 2*OW);
                d3 = MEGDNN_SIMD_LOADU(dptr + 3*OW);
            } else {
                d0 = MEGDNN_SIMD_SETZERO();
                d1 = MEGDNN_SIMD_SETZERO();
                d2 = MEGDNN_SIMD_SETZERO();
                d3 = MEGDNN_SIMD_SETZERO();
            }
            for (size_t fw = 0; fw < 5; ++fw) {
                k0 = MEGDNN_SIMD_SET1(fptr[0*5 + fw]);
                k1 = MEGDNN_SIMD_SET1(fptr[1*5 + fw]);
                k2 = MEGDNN_SIMD_SET1(fptr[2*5 + fw]);
                k3 = MEGDNN_SIMD_SET1(fptr[3*5 + fw]);
                k4 = MEGDNN_SIMD_SET1(fptr[4*5 + fw]);

                // line 0
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 0*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k0, s, d0);

                // line 1
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 1*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k1, s, d0);

                // line 2
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 2*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k2, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k0, s, d1);

                // line 3
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 3*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k3, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k1, s, d1);

                // line 4
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 4*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k4, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k2, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k0, s, d2);

                // line 5
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 5*2*MEGDNN_SIMD_WIDTH);
                d1 = MEGDNN_SIMD_FMADD(k3, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k1, s, d2);

                // line 6
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 6*2*MEGDNN_SIMD_WIDTH);
                d1 = MEGDNN_SIMD_FMADD(k4, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k2, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k0, s, d3);

                // line 7
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 7*2*MEGDNN_SIMD_WIDTH);
                d2 = MEGDNN_SIMD_FMADD(k3, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k1, s, d3);

                // line 2*MEGDNN_SIMD_WIDTH
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 8*2*MEGDNN_SIMD_WIDTH);
                d2 = MEGDNN_SIMD_FMADD(k4, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k2, s, d3);

                // line 9
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 9*2*MEGDNN_SIMD_WIDTH);
                d3 = MEGDNN_SIMD_FMADD(k3, s, d3);

                // line 9
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 10*2*MEGDNN_SIMD_WIDTH);
                d3 = MEGDNN_SIMD_FMADD(k4, s, d3);
            }
            MEGDNN_SIMD_STOREU(dptr + 0*OW, d0);
            MEGDNN_SIMD_STOREU(dptr + 1*OW, d1);
            MEGDNN_SIMD_STOREU(dptr + 2*OW, d2);
            MEGDNN_SIMD_STOREU(dptr + 3*OW, d3);
        }
        //do prefetch for the 3th block in the next line
        const int prefetch_index_input = (ih + 8)* IW + 6*MEGDNN_SIMD_WIDTH + 2*OW_start - PW;
        const int prefetch_index_output = (oh + 4)* OW + 3*MEGDNN_SIMD_WIDTH + OW_start;
        const float* src_prefetch = src + prefetch_index_input;
        const float* dst_prefetch = dst + prefetch_index_output;
        for(int iw_id = 0;iw_id < 11;++iw_id){
            __builtin_prefetch(src_prefetch + iw_id * IW,0,3);
        }
        for(int ow_id = 0;ow_id < 4;++ow_id){
            __builtin_prefetch(dst_prefetch + ow_id * OW,1,3);
        }
        for (; ow < OW_stop; ++ow) {
            run_single(oh+0, ow);
            run_single(oh+1, ow);
            run_single(oh+2, ow);
            run_single(oh+3, ow);
        }
    }
    for (; oh < OH_stop; ++oh) {
        for (size_t ow = OW_start; ow < OW_stop; ++ow) {
            run_single(oh, ow);
        }
    }
}

template <bool add_to_dst>
MEGDNN_SIMD_ATTRIBUTE_TARGET
void do_conv_7x7_stride2(const float *src, const float *filter, float *dst,
        size_t IH, size_t IW,
        size_t OH, size_t OW,
        size_t PH, size_t PW)
{
    size_t OH_start = div_ceil<size_t>(PH, 2),
           OH_stop = div_floor<size_t>(IH+PH-7, 2) + 1,
           OW_start = div_ceil<size_t>(PW, 2),
           OW_stop = div_floor<size_t>(IW+PW-7, 2) + 1;
    OH_start = std::min<size_t>(OH, OH_start);
    OH_stop = std::min<size_t>(OH, OH_stop);
    OW_start = std::min<size_t>(OW, OW_start);
    OW_stop = std::min<size_t>(OW, OW_stop);
    auto run_single = [&](size_t oh, size_t ow) {
        if (!add_to_dst) {
            dst[oh*OW + ow] = 0;
        }
        for (size_t fh = 0; fh < 7; ++fh)
        for (size_t fw = 0; fw < 7; ++fw)
        {
            size_t ih = oh*2+fh-PH;
            size_t iw = ow*2+fw-PW;
            if (ih < IH && iw < IW) {
                dst[oh*OW + ow] += src[ih*IW + iw] * filter[fh*7 + fw];
            }
        }
    };
    for (size_t oh = 0; oh < OH_start; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    for (size_t oh = OH_start; oh < OH_stop; ++oh) {
        for (size_t ow = 0; ow < OW_start; ++ow) run_single(oh, ow);
        for (size_t ow = OW_stop; ow < OW; ++ow) run_single(oh, ow);
    }
    for (size_t oh = OH_stop; oh < OH; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    // 4x4 block
    size_t oh = OH_start;
    float cache_even[13*2*MEGDNN_SIMD_WIDTH];
    float cache_odd[13*2*MEGDNN_SIMD_WIDTH];
    const float* sptrs[7] = {
        cache_even + 0,
        cache_odd + 0,
        cache_even + 1,
        cache_odd + 1,
        cache_even + 2,
        cache_odd + 2,
        cache_even + 3,
    };
    for (; oh+4 <= OH_stop; oh += 4) {
        size_t ih = oh*2-PH;
        size_t ow = OW_start;
        for (; ow+4*MEGDNN_SIMD_WIDTH < OW_stop; ow += MEGDNN_SIMD_WIDTH) {
            size_t iw = ow*2-PW;
            float * __restrict dptr = dst + oh*OW + ow;
            const float * __restrict sptr = src + ih*IW + iw;
            const float * __restrict fptr = filter;

            //do prefetch for current line
            const int prefetch_index_input = ih*IW + iw + 4*MEGDNN_SIMD_WIDTH;
            const int prefetch_index_output = oh*OW + ow + 4*MEGDNN_SIMD_WIDTH;
            const float* src_prefetch = src + prefetch_index_input;
            const float* dst_prefetch = dst + prefetch_index_output;
            for(int iw_id = 0;iw_id < 13;++iw_id){
                __builtin_prefetch(src_prefetch + iw_id * IW,0,3);
            }
            for(int ow_id = 0;ow_id < 4;++ow_id){
                __builtin_prefetch(dst_prefetch + ow_id * OW,1,3);
            }

            MEGDNN_SIMD_TYPE d0, d1, d2, d3;
            MEGDNN_SIMD_TYPE k0, k1, k2, k3, k4, k5, k6, s;
            {
                // do transpose
                for (size_t i = 0; i < 13; ++i) {
                    MEGDNN_SIMD_TYPE s_low = MEGDNN_SIMD_LOADU(sptr + i*IW);
                    MEGDNN_SIMD_TYPE s_high = MEGDNN_SIMD_LOADU(sptr + i*IW +
                            MEGDNN_SIMD_WIDTH);
                    MEGDNN_SIMD_TYPE s_result0, s_result1;
                    MEGDNN_SIMD_UZP(s_low, s_high, s_result0, s_result1);
                    MEGDNN_SIMD_STOREU(cache_even + i*2*MEGDNN_SIMD_WIDTH,
                            s_result0);
                    MEGDNN_SIMD_STOREU(cache_odd + i*2*MEGDNN_SIMD_WIDTH,
                            s_result1);
                    // last elements
                    cache_even[(i*2+1)*MEGDNN_SIMD_WIDTH] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH + 0];
                    cache_odd[(i*2+1)*MEGDNN_SIMD_WIDTH] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH + 1];
                    cache_even[(i*2+1)*MEGDNN_SIMD_WIDTH+1] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH + 2];
                    cache_odd[(i*2+1)*MEGDNN_SIMD_WIDTH+1] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH + 3];
                    cache_even[(i*2+1)*MEGDNN_SIMD_WIDTH+2] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH + 4];
                }
            }
            if (add_to_dst) {
                d0 = MEGDNN_SIMD_LOADU(dptr + 0*OW);
                d1 = MEGDNN_SIMD_LOADU(dptr + 1*OW);
                d2 = MEGDNN_SIMD_LOADU(dptr + 2*OW);
                d3 = MEGDNN_SIMD_LOADU(dptr + 3*OW);
            } else {
                d0 = MEGDNN_SIMD_SETZERO();
                d1 = MEGDNN_SIMD_SETZERO();
                d2 = MEGDNN_SIMD_SETZERO();
                d3 = MEGDNN_SIMD_SETZERO();
            }
            for (size_t fw = 0; fw < 7; ++fw) {
                k0 = MEGDNN_SIMD_SET1(fptr[0*7 + fw]);
                k1 = MEGDNN_SIMD_SET1(fptr[1*7 + fw]);
                k2 = MEGDNN_SIMD_SET1(fptr[2*7 + fw]);
                k3 = MEGDNN_SIMD_SET1(fptr[3*7 + fw]);
                k4 = MEGDNN_SIMD_SET1(fptr[4*7 + fw]);
                k5 = MEGDNN_SIMD_SET1(fptr[5*7 + fw]);
                k6 = MEGDNN_SIMD_SET1(fptr[6*7 + fw]);

                // line 0
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 0*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k0, s, d0);

                // line 1
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 1*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k1, s, d0);

                // line 2
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 2*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k2, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k0, s, d1);

                // line 3
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 3*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k3, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k1, s, d1);

                // line 4
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 4*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k4, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k2, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k0, s, d2);

                // line 5
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 5*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k5, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k3, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k1, s, d2);

                // line 6
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 6*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k6, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k4, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k2, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k0, s, d3);

                // line 7
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 7*2*MEGDNN_SIMD_WIDTH);
                d1 = MEGDNN_SIMD_FMADD(k5, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k3, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k1, s, d3);

                // line 2*MEGDNN_SIMD_WIDTH
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 8*2*MEGDNN_SIMD_WIDTH);
                d1 = MEGDNN_SIMD_FMADD(k6, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k4, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k2, s, d3);

                // line 9
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 9*2*MEGDNN_SIMD_WIDTH);
                d2 = MEGDNN_SIMD_FMADD(k5, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k3, s, d3);

                // line 10
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 10*2*MEGDNN_SIMD_WIDTH);
                d2 = MEGDNN_SIMD_FMADD(k6, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k4, s, d3);

                // line 11
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 11*2*MEGDNN_SIMD_WIDTH);
                d3 = MEGDNN_SIMD_FMADD(k5, s, d3);

                // line 12
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 12*2*MEGDNN_SIMD_WIDTH);
                d3 = MEGDNN_SIMD_FMADD(k6, s, d3);
            }
            MEGDNN_SIMD_STOREU(dptr + 0*OW, d0);
            MEGDNN_SIMD_STOREU(dptr + 1*OW, d1);
            MEGDNN_SIMD_STOREU(dptr + 2*OW, d2);
            MEGDNN_SIMD_STOREU(dptr + 3*OW, d3);
        }
        for (; ow+MEGDNN_SIMD_WIDTH <= OW_stop; ow += MEGDNN_SIMD_WIDTH) {
            size_t iw = ow*2-PW;
            float * __restrict dptr = dst + oh*OW + ow;
            const float * __restrict sptr = src + ih*IW + iw;
            const float * __restrict fptr = filter;

            //do prefetch for the first two/three blocks of the next line
            const int prefetch_index_input = (ih + 8) * IW +
                (((ow + 4*MEGDNN_SIMD_WIDTH - OW_stop)/MEGDNN_SIMD_WIDTH) * MEGDNN_SIMD_WIDTH + OW_start) * 2 - PW;
            const int prefetch_index_output = (oh + 4) * OW +
                 (((ow + 4*MEGDNN_SIMD_WIDTH - OW_stop)/MEGDNN_SIMD_WIDTH) * MEGDNN_SIMD_WIDTH + OW_start);
            const float* src_prefetch = src + prefetch_index_input;
            const float* dst_prefetch = dst + prefetch_index_output;
            for(int iw_id = 0;iw_id < 13;++iw_id){
                __builtin_prefetch(src_prefetch + iw_id * IW,0,3);
            }
            for(int ow_id = 0;ow_id < 4;++ow_id){
                __builtin_prefetch(dst_prefetch + ow_id * OW,1,3);
            }

            MEGDNN_SIMD_TYPE d0, d1, d2, d3;
            MEGDNN_SIMD_TYPE k0, k1, k2, k3, k4, k5, k6, s;
            {
                // do transpose
                for (size_t i = 0; i < 13; ++i) {
                    MEGDNN_SIMD_TYPE s_low = MEGDNN_SIMD_LOADU(sptr + i*IW);
                    MEGDNN_SIMD_TYPE s_high = MEGDNN_SIMD_LOADU(sptr + i*IW +
                            MEGDNN_SIMD_WIDTH);
                    MEGDNN_SIMD_TYPE s_result0, s_result1;
                    MEGDNN_SIMD_UZP(s_low, s_high, s_result0, s_result1);
                    MEGDNN_SIMD_STOREU(cache_even + i*2*MEGDNN_SIMD_WIDTH,
                            s_result0);
                    MEGDNN_SIMD_STOREU(cache_odd + i*2*MEGDNN_SIMD_WIDTH,
                            s_result1);
                    // last elements
                    cache_even[(i*2+1)*MEGDNN_SIMD_WIDTH] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH + 0];
                    cache_odd[(i*2+1)*MEGDNN_SIMD_WIDTH] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH + 1];
                    cache_even[(i*2+1)*MEGDNN_SIMD_WIDTH+1] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH + 2];
                    cache_odd[(i*2+1)*MEGDNN_SIMD_WIDTH+1] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH + 3];
                    cache_even[(i*2+1)*MEGDNN_SIMD_WIDTH+2] = sptr[i*IW +
                        2*MEGDNN_SIMD_WIDTH + 4];
                }
            }
            if (add_to_dst) {
                d0 = MEGDNN_SIMD_LOADU(dptr + 0*OW);
                d1 = MEGDNN_SIMD_LOADU(dptr + 1*OW);
                d2 = MEGDNN_SIMD_LOADU(dptr + 2*OW);
                d3 = MEGDNN_SIMD_LOADU(dptr + 3*OW);
            } else {
                d0 = MEGDNN_SIMD_SETZERO();
                d1 = MEGDNN_SIMD_SETZERO();
                d2 = MEGDNN_SIMD_SETZERO();
                d3 = MEGDNN_SIMD_SETZERO();
            }
            for (size_t fw = 0; fw < 7; ++fw) {
                k0 = MEGDNN_SIMD_SET1(fptr[0*7 + fw]);
                k1 = MEGDNN_SIMD_SET1(fptr[1*7 + fw]);
                k2 = MEGDNN_SIMD_SET1(fptr[2*7 + fw]);
                k3 = MEGDNN_SIMD_SET1(fptr[3*7 + fw]);
                k4 = MEGDNN_SIMD_SET1(fptr[4*7 + fw]);
                k5 = MEGDNN_SIMD_SET1(fptr[5*7 + fw]);
                k6 = MEGDNN_SIMD_SET1(fptr[6*7 + fw]);

                // line 0
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 0*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k0, s, d0);

                // line 1
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 1*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k1, s, d0);

                // line 2
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 2*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k2, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k0, s, d1);

                // line 3
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 3*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k3, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k1, s, d1);

                // line 4
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 4*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k4, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k2, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k0, s, d2);

                // line 5
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 5*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k5, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k3, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k1, s, d2);

                // line 6
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 6*2*MEGDNN_SIMD_WIDTH);
                d0 = MEGDNN_SIMD_FMADD(k6, s, d0);
                d1 = MEGDNN_SIMD_FMADD(k4, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k2, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k0, s, d3);

                // line 7
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 7*2*MEGDNN_SIMD_WIDTH);
                d1 = MEGDNN_SIMD_FMADD(k5, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k3, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k1, s, d3);

                // line 2*MEGDNN_SIMD_WIDTH
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 8*2*MEGDNN_SIMD_WIDTH);
                d1 = MEGDNN_SIMD_FMADD(k6, s, d1);
                d2 = MEGDNN_SIMD_FMADD(k4, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k2, s, d3);

                // line 9
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 9*2*MEGDNN_SIMD_WIDTH);
                d2 = MEGDNN_SIMD_FMADD(k5, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k3, s, d3);

                // line 10
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 10*2*MEGDNN_SIMD_WIDTH);
                d2 = MEGDNN_SIMD_FMADD(k6, s, d2);
                d3 = MEGDNN_SIMD_FMADD(k4, s, d3);

                // line 11
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 11*2*MEGDNN_SIMD_WIDTH);
                d3 = MEGDNN_SIMD_FMADD(k5, s, d3);

                // line 12
                s = MEGDNN_SIMD_LOADU(sptrs[fw] + 12*2*MEGDNN_SIMD_WIDTH);
                d3 = MEGDNN_SIMD_FMADD(k6, s, d3);
            }
            MEGDNN_SIMD_STOREU(dptr + 0*OW, d0);
            MEGDNN_SIMD_STOREU(dptr + 1*OW, d1);
            MEGDNN_SIMD_STOREU(dptr + 2*OW, d2);
            MEGDNN_SIMD_STOREU(dptr + 3*OW, d3);
        }
        //do prefetch for the 3th block in the next line
        const int prefetch_index_input = (ih + 8)* IW + 6*MEGDNN_SIMD_WIDTH + 2*OW_start - PW;
        const int prefetch_index_output = (oh + 4)* OW + 3*MEGDNN_SIMD_WIDTH + OW_start;
        const float* src_prefetch = src + prefetch_index_input;
        const float* dst_prefetch = dst + prefetch_index_output;
        for(int iw_id = 0;iw_id < 13;++iw_id){
            __builtin_prefetch(src_prefetch + iw_id * IW,0,3);
        }
        for(int ow_id = 0;ow_id < 4;++ow_id){
            __builtin_prefetch(dst_prefetch + ow_id * OW,1,3);
        }
        for (; ow < OW_stop; ++ow) {
            run_single(oh+0, ow);
            run_single(oh+1, ow);
            run_single(oh+2, ow);
            run_single(oh+3, ow);
        }
    }
    for (; oh < OH_stop; ++oh) {
        for (size_t ow = OW_start; ow < OW_stop; ++ow) {
            run_single(oh, ow);
        }
    }
}using NCBKernSizeParam = fallback::ConvolutionImpl::NCBKernSizeParam;
using NCBKernParam = fallback::ConvolutionImpl::NCBKernParam;

void WITH_SIMD_SUFFIX(do_conv_stride2)(const NCBKernParam &param)
    MEGDNN_SIMD_ATTRIBUTE_TARGET;

size_t WITH_SIMD_SUFFIX(get_workspace_in_bytes_do_conv_stride2)(
        const NCBKernSizeParam &param);

static inline bool can_do_conv_stride2(
        const fallback::ConvolutionImpl::NCBKernSizeParam &param) {
    auto &&fm = param.filter_meta;
    auto FH = fm.spatial[0];
    return
        param.filter_meta.format == param::Convolution::Format::NCHW &&
        param.src_type.enumv() == DTypeEnum::Float32 &&
        param.filter_type.enumv() == DTypeEnum::Float32 &&
        param.dst_type.enumv() == DTypeEnum::Float32 &&
        !fm.should_flip &&
        fm.group == 1 &&
        fm.spatial_ndim == 2 &&
        fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
        fm.stride[0] == 2 && fm.stride[1] == 2 &&
        FH == fm.spatial[1] &&
        (FH == 2 || FH == 3 || FH == 5 || FH == 7);
}

} // namespace conv_general_simd
} // namespace megdnn

#include "src/common/macro_helper_epilogue.h"


#ifdef MEGDNN_YCM_COMPILE_CLEANUP
#undef MEGDNN_YCM_COMPILE_CLEANUP
#include "src/x86/simd_macro/sse_helper_epilogue.h"
#endif
