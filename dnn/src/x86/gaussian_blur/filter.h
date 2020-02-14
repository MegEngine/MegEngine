/**
 * By downloading, copying, installing or using the software you agree to this license.
 * If you do not agree to this license, do not download, install,
 * copy or use the software.
 *
 *
 *                           License Agreement
 *                For Open Source Computer Vision Library
 *                        (3-clause BSD License)
 *
 * Copyright (C) 2000-2020, Intel Corporation, all rights reserved.
 * Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
 * Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
 * Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
 * Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
 * Copyright (C) 2019-2020, Xperience AI, all rights reserved.
 * Third party copyrights are property of their respective owners.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   * Neither the names of the copyright holders nor the names of the contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 * This software is provided by the copyright holders and contributors "as is" and
 * any express or implied warranties, including, but not limited to, the implied
 * warranties of merchantability and fitness for a particular purpose are disclaimed.
 * In no event shall copyright holders or contributors be liable for any direct,
 * indirect, incidental, special, exemplary, or consequential damages
 * (including, but not limited to, procurement of substitute goods or services;
 * loss of use, data, or profits; or business interruption) however caused
 * and on any theory of liability, whether in contract, strict liability,
 * or tort (including negligence or otherwise) arising in any way out of
 * the use of this software, even if advised of the possibility of such damage.
 *
 * ---------------------------------------------------------------------------
 * \file dnn/src/x86/gaussian_blur/filter.h
 *
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
 *
 * ---------------------------------------------------------------------------
 */



#include "src/common/cv/filter.h"
#include <cfloat>
#include <cmath>
#include <pmmintrin.h>
#include <smmintrin.h>

namespace megdnn {
namespace megcv {
namespace gaussian_blur {

using namespace filter_common;

struct RowVec_8u32s
{
    RowVec_8u32s() {}
    RowVec_8u32s( const uchar * _kernel, int _len)
    {
        ksize = _len;
        kernel = (int*)_kernel;
    }

    MEGDNN_ATTRIBUTE_TARGET("sse2")
    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
        int i = 0, k, _ksize = ksize;
        int* dst = (int*)_dst;
        const int * _kx = kernel;
        width *= cn;

        for( ; i <= width - 16; i += 16 )
        {
            const uchar* src = _src + i;
            __m128i f, z = _mm_setzero_si128(), s0 = z, s1 = z, s2 = z, s3 = z;
            __m128i x0, x1, x2, x3;

            for( k = 0; k < _ksize; k++, src += cn )
            {
                f = _mm_cvtsi32_si128(_kx[k]);
                f = _mm_shuffle_epi32(f, 0);
                f = _mm_packs_epi32(f, f);

                x0 = _mm_loadu_si128((const __m128i*)src);
                x2 = _mm_unpackhi_epi8(x0, z);
                x0 = _mm_unpacklo_epi8(x0, z);
                x1 = _mm_mulhi_epi16(x0, f);
                x3 = _mm_mulhi_epi16(x2, f);
                x0 = _mm_mullo_epi16(x0, f);
                x2 = _mm_mullo_epi16(x2, f);

                s0 = _mm_add_epi32(s0, _mm_unpacklo_epi16(x0, x1));
                s1 = _mm_add_epi32(s1, _mm_unpackhi_epi16(x0, x1));
                s2 = _mm_add_epi32(s2, _mm_unpacklo_epi16(x2, x3));
                s3 = _mm_add_epi32(s3, _mm_unpackhi_epi16(x2, x3));
            }

            _mm_store_si128((__m128i*)(dst + i), s0);
            _mm_store_si128((__m128i*)(dst + i + 4), s1);
            _mm_store_si128((__m128i*)(dst + i + 8), s2);
            _mm_store_si128((__m128i*)(dst + i + 12), s3);
        }

        for( ; i <= width - 4; i += 4 )
        {
            const uchar* src = _src + i;
            __m128i f, z = _mm_setzero_si128(), s0 = z, x0, x1;

            for( k = 0; k < _ksize; k++, src += cn )
            {
                f = _mm_cvtsi32_si128(_kx[k]);
                f = _mm_shuffle_epi32(f, 0);
                f = _mm_packs_epi32(f, f);

                x0 = _mm_cvtsi32_si128(*(const int*)src);
                x0 = _mm_unpacklo_epi8(x0, z);
                x1 = _mm_mulhi_epi16(x0, f);
                x0 = _mm_mullo_epi16(x0, f);
                s0 = _mm_add_epi32(s0, _mm_unpacklo_epi16(x0, x1));
            }
            _mm_store_si128((__m128i*)(dst + i), s0);
        }
        return i;
    }

    int * kernel;
    size_t ksize;
};

struct SymmRowSmallVec_8u32s
{
    SymmRowSmallVec_8u32s() {}
    SymmRowSmallVec_8u32s( const uchar * _kernel, int _len)
    {
        kernel = (int *)_kernel;
        ksize = _len;
    }

    MEGDNN_ATTRIBUTE_TARGET("sse2")
    int operator()(const uchar* src, uchar* _dst, int width, int cn) const
    {
        int i = 0, j, k, _ksize = ksize;
        int* dst = (int*)_dst;
        const int * kx = kernel + _ksize/2;

        src += (_ksize/2)*cn;
        width *= cn;

        __m128i z = _mm_setzero_si128();
        {
            if( _ksize == 1 )
                return 0;
            if( _ksize == 3 )
            {
                __m128i k0 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[0]), 0),
                        k1 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[1]), 0);
                k0 = _mm_packs_epi32(k0, k0);
                k1 = _mm_packs_epi32(k1, k1);

                for( ; i <= width - 16; i += 16, src += 16 )
                {
                    __m128i x0, x1, x2, y0, y1, t0, t1, z0, z1, z2, z3;
                    x0 = _mm_loadu_si128((__m128i*)(src - cn));
                    x1 = _mm_loadu_si128((__m128i*)src);
                    x2 = _mm_loadu_si128((__m128i*)(src + cn));

                    y0 = _mm_add_epi16(_mm_unpackhi_epi8(x0, z), _mm_unpackhi_epi8(x2, z));
                    x0 = _mm_add_epi16(_mm_unpacklo_epi8(x0, z), _mm_unpacklo_epi8(x2, z));
                    y1 = _mm_unpackhi_epi8(x1, z);
                    x1 = _mm_unpacklo_epi8(x1, z);

                    t1 = _mm_mulhi_epi16(x1, k0);
                    t0 = _mm_mullo_epi16(x1, k0);
                    x2 = _mm_mulhi_epi16(x0, k1);
                    x0 = _mm_mullo_epi16(x0, k1);
                    z0 = _mm_unpacklo_epi16(t0, t1);
                    z1 = _mm_unpackhi_epi16(t0, t1);
                    z0 = _mm_add_epi32(z0, _mm_unpacklo_epi16(x0, x2));
                    z1 = _mm_add_epi32(z1, _mm_unpackhi_epi16(x0, x2));

                    t1 = _mm_mulhi_epi16(y1, k0);
                    t0 = _mm_mullo_epi16(y1, k0);
                    y1 = _mm_mulhi_epi16(y0, k1);
                    y0 = _mm_mullo_epi16(y0, k1);
                    z2 = _mm_unpacklo_epi16(t0, t1);
                    z3 = _mm_unpackhi_epi16(t0, t1);
                    z2 = _mm_add_epi32(z2, _mm_unpacklo_epi16(y0, y1));
                    z3 = _mm_add_epi32(z3, _mm_unpackhi_epi16(y0, y1));
                    _mm_store_si128((__m128i*)(dst + i), z0);
                    _mm_store_si128((__m128i*)(dst + i + 4), z1);
                    _mm_store_si128((__m128i*)(dst + i + 8), z2);
                    _mm_store_si128((__m128i*)(dst + i + 12), z3);
                }
            }
            else if( _ksize == 5 )
            {
                __m128i k0 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[0]), 0),
                        k1 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[1]), 0),
                        k2 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[2]), 0);
                k0 = _mm_packs_epi32(k0, k0);
                k1 = _mm_packs_epi32(k1, k1);
                k2 = _mm_packs_epi32(k2, k2);

                for( ; i <= width - 16; i += 16, src += 16 )
                {
                    __m128i x0, x1, x2, y0, y1, t0, t1, z0, z1, z2, z3;
                    x0 = _mm_loadu_si128((__m128i*)(src - cn));
                    x1 = _mm_loadu_si128((__m128i*)src);
                    x2 = _mm_loadu_si128((__m128i*)(src + cn));
                    y0 = _mm_add_epi16(_mm_unpackhi_epi8(x0, z), _mm_unpackhi_epi8(x2, z));
                    x0 = _mm_add_epi16(_mm_unpacklo_epi8(x0, z), _mm_unpacklo_epi8(x2, z));
                    y1 = _mm_unpackhi_epi8(x1, z);
                    x1 = _mm_unpacklo_epi8(x1, z);

                    t1 = _mm_mulhi_epi16(x1, k0);
                    t0 = _mm_mullo_epi16(x1, k0);
                    x2 = _mm_mulhi_epi16(x0, k1);
                    x0 = _mm_mullo_epi16(x0, k1);
                    z0 = _mm_unpacklo_epi16(t0, t1);
                    z1 = _mm_unpackhi_epi16(t0, t1);
                    z0 = _mm_add_epi32(z0, _mm_unpacklo_epi16(x0, x2));
                    z1 = _mm_add_epi32(z1, _mm_unpackhi_epi16(x0, x2));

                    t1 = _mm_mulhi_epi16(y1, k0);
                    t0 = _mm_mullo_epi16(y1, k0);
                    y1 = _mm_mulhi_epi16(y0, k1);
                    y0 = _mm_mullo_epi16(y0, k1);
                    z2 = _mm_unpacklo_epi16(t0, t1);
                    z3 = _mm_unpackhi_epi16(t0, t1);
                    z2 = _mm_add_epi32(z2, _mm_unpacklo_epi16(y0, y1));
                    z3 = _mm_add_epi32(z3, _mm_unpackhi_epi16(y0, y1));

                    x0 = _mm_loadu_si128((__m128i*)(src - cn*2));
                    x1 = _mm_loadu_si128((__m128i*)(src + cn*2));
                    y1 = _mm_add_epi16(_mm_unpackhi_epi8(x0, z), _mm_unpackhi_epi8(x1, z));
                    y0 = _mm_add_epi16(_mm_unpacklo_epi8(x0, z), _mm_unpacklo_epi8(x1, z));

                    t1 = _mm_mulhi_epi16(y0, k2);
                    t0 = _mm_mullo_epi16(y0, k2);
                    y0 = _mm_mullo_epi16(y1, k2);
                    y1 = _mm_mulhi_epi16(y1, k2);
                    z0 = _mm_add_epi32(z0, _mm_unpacklo_epi16(t0, t1));
                    z1 = _mm_add_epi32(z1, _mm_unpackhi_epi16(t0, t1));
                    z2 = _mm_add_epi32(z2, _mm_unpacklo_epi16(y0, y1));
                    z3 = _mm_add_epi32(z3, _mm_unpackhi_epi16(y0, y1));

                    _mm_store_si128((__m128i*)(dst + i), z0);
                    _mm_store_si128((__m128i*)(dst + i + 4), z1);
                    _mm_store_si128((__m128i*)(dst + i + 8), z2);
                    _mm_store_si128((__m128i*)(dst + i + 12), z3);
                }
            }
        }

        src -= (_ksize/2)*cn;
        kx -= _ksize/2;
        for( ; i <= width - 4; i += 4, src += 4 )
        {
            __m128i f, s0 = z, x0, x1;

            for( k = j = 0; k < _ksize; k++, j += cn )
            {
                f = _mm_cvtsi32_si128(kx[k]);
                f = _mm_shuffle_epi32(f, 0);
                f = _mm_packs_epi32(f, f);

                x0 = _mm_cvtsi32_si128(*(const int*)(src + j));
                x0 = _mm_unpacklo_epi8(x0, z);
                x1 = _mm_mulhi_epi16(x0, f);
                x0 = _mm_mullo_epi16(x0, f);
                s0 = _mm_add_epi32(s0, _mm_unpacklo_epi16(x0, x1));
            }
            _mm_store_si128((__m128i*)(dst + i), s0);
        }

        return i;
    }

    int * kernel;
    size_t ksize;
};


struct SymmColumnSmallVec_32s8u
{
    SymmColumnSmallVec_32s8u() {}
    SymmColumnSmallVec_32s8u(const uchar * _kernel, int _len, int _bits)
    {
        ksize = _len;
        kernel = (float *)malloc(sizeof(float)*ksize);
        for(size_t i=0; i<ksize; i++)
            kernel[i] = (float)(((int *)_kernel)[i]) * (1./(1<<_bits));
    }

    SymmColumnSmallVec_32s8u(const SymmColumnSmallVec_32s8u& rhs) {
        ksize = rhs.ksize;
        kernel = (float*)malloc(sizeof(float)*ksize);
        memcpy(kernel, rhs.kernel, sizeof(float)*ksize);
    }

    SymmColumnSmallVec_32s8u& operator=(const SymmColumnSmallVec_32s8u& rhs) {
        ksize = rhs.ksize;
        kernel = (float*)malloc(sizeof(float)*ksize);
        memcpy(kernel, rhs.kernel, sizeof(float)*ksize);
        return *this;
    }

    ~SymmColumnSmallVec_32s8u() {
        free(kernel);
    }

    MEGDNN_ATTRIBUTE_TARGET("sse2")
    int operator()(const uchar** _src, uchar* dst, int & count, int width) const
    {
        int ksize2 = (ksize)/2;
        const float * ky = kernel + ksize2;
        int i = 0, k;
        const int ** src = (const int**)_src;
        const __m128i *S, *S0, *S1, *S2;

        if(ksize == 3 && count == 4)
        {
            __m128 f0 = _mm_load_ss(ky);
            f0 = _mm_shuffle_ps(f0, f0, 0);

            __m128 f1 = _mm_load_ss(ky+1);
            f1 = _mm_shuffle_ps(f1, f1, 0);

            for( ; i <= width - 16; i += 16 )
            {
                __m128 s00, s01, s02, s03;
                __m128 s10, s11, s12, s13;
                __m128 s20, s21, s22, s23;

                __m128 d00, d01, d02, d03;

                __m128i d_0, d_1;

                S0 = (const __m128i*)(src[-1] + i);
                S1 = (const __m128i*)(src[0] + i);
                S2 = (const __m128i*)(src[1] + i);

                s20 = _mm_cvtepi32_ps(_mm_load_si128(S2));
                s21 = _mm_cvtepi32_ps(_mm_load_si128(S2+1));
                s22 = _mm_cvtepi32_ps(_mm_load_si128(S2+2));
                s23 = _mm_cvtepi32_ps(_mm_load_si128(S2+3));

                s10 = _mm_cvtepi32_ps(_mm_load_si128(S1));
                d00 = _mm_mul_ps(s10, f0);
                s11 = _mm_cvtepi32_ps(_mm_load_si128(S1+1));
                d01 = _mm_mul_ps(s11, f0);
                s12 = _mm_cvtepi32_ps(_mm_load_si128(S1+2));
                d02 = _mm_mul_ps(s12, f0);
                s13 = _mm_cvtepi32_ps(_mm_load_si128(S1+3));
                d03 = _mm_mul_ps(s13, f0);

                s00 = _mm_cvtepi32_ps(_mm_load_si128(S0));
                d00 = _mm_add_ps(d00, _mm_mul_ps(_mm_add_ps(s00, s20), f1));
                s01 = _mm_cvtepi32_ps(_mm_load_si128(S0+1));
                d01 = _mm_add_ps(d01, _mm_mul_ps(_mm_add_ps(s01, s21), f1));
                d_0 = _mm_packs_epi32(_mm_cvtps_epi32(d00), _mm_cvtps_epi32(d01));
                s02 = _mm_cvtepi32_ps(_mm_load_si128(S0+2));
                d02 = _mm_add_ps(d02, _mm_mul_ps(_mm_add_ps(s02, s22), f1));
                s03 = _mm_cvtepi32_ps(_mm_load_si128(S0+3));
                d03 = _mm_add_ps(d03, _mm_mul_ps(_mm_add_ps(s03, s23), f1));

                d_1 = _mm_packs_epi32(_mm_cvtps_epi32(d02), _mm_cvtps_epi32(d03));
                d_0 = _mm_packus_epi16(d_0, d_1);

                _mm_storeu_si128((__m128i*)(dst + i), d_0);

                S2 = (const __m128i*)(src[2] + i);
                s00 = _mm_cvtepi32_ps(_mm_load_si128(S2));
                d00 = _mm_mul_ps(s20, f0);
                d00 = _mm_add_ps(d00, _mm_mul_ps(_mm_add_ps(s00, s10), f1));
                s01 = _mm_cvtepi32_ps(_mm_load_si128(S2+1));
                d01 = _mm_mul_ps(s21, f0);
                d01 = _mm_add_ps(d01, _mm_mul_ps(_mm_add_ps(s01, s11), f1));
                d_0 = _mm_packs_epi32(_mm_cvtps_epi32(d00), _mm_cvtps_epi32(d01));
                s02 = _mm_cvtepi32_ps(_mm_load_si128(S2+2));
                d02 = _mm_mul_ps(s22, f0);
                d02 = _mm_add_ps(d02, _mm_mul_ps(_mm_add_ps(s02, s12), f1));
                s03 = _mm_cvtepi32_ps(_mm_load_si128(S2+3));
                d03 = _mm_mul_ps(s23, f0);
                d03 = _mm_add_ps(d03, _mm_mul_ps(_mm_add_ps(s03, s13), f1));

                d_1 = _mm_packs_epi32(_mm_cvtps_epi32(d02), _mm_cvtps_epi32(d03));
                d_0 = _mm_packus_epi16(d_0, d_1);

                _mm_storeu_si128((__m128i*)(dst + width + i), d_0);

                S2 = (const __m128i*)(src[3] + i);
                s10 = _mm_cvtepi32_ps(_mm_load_si128(S2));
                d00 = _mm_mul_ps(s00, f0);
                d00 = _mm_add_ps(d00, _mm_mul_ps(_mm_add_ps(s20, s10), f1));
                s11 = _mm_cvtepi32_ps(_mm_load_si128(S2+1));
                d01 = _mm_mul_ps(s01, f0);
                d01 = _mm_add_ps(d01, _mm_mul_ps(_mm_add_ps(s21, s11), f1));
                d_0 = _mm_packs_epi32(_mm_cvtps_epi32(d00), _mm_cvtps_epi32(d01));
                s12 = _mm_cvtepi32_ps(_mm_load_si128(S2+2));
                d02 = _mm_mul_ps(s02, f0);
                d02 = _mm_add_ps(d02, _mm_mul_ps(_mm_add_ps(s22, s12), f1));
                s13 = _mm_cvtepi32_ps(_mm_load_si128(S2+3));
                d03 = _mm_mul_ps(s03, f0);
                d03 = _mm_add_ps(d03, _mm_mul_ps(_mm_add_ps(s23, s13), f1));

                d_1 = _mm_packs_epi32(_mm_cvtps_epi32(d02), _mm_cvtps_epi32(d03));
                d_0 = _mm_packus_epi16(d_0, d_1);

                _mm_storeu_si128((__m128i*)(dst + width*2 + i), d_0);

                S2 = (const __m128i*)(src[4] + i);
                s20 = _mm_cvtepi32_ps(_mm_load_si128(S2));
                d00 = _mm_mul_ps(s10, f0);
                d00 = _mm_add_ps(d00, _mm_mul_ps(_mm_add_ps(s00, s20), f1));
                s21 = _mm_cvtepi32_ps(_mm_load_si128(S2+1));
                d01 = _mm_mul_ps(s11, f0);
                d01 = _mm_add_ps(d01, _mm_mul_ps(_mm_add_ps(s01, s21), f1));
                d_0 = _mm_packs_epi32(_mm_cvtps_epi32(d00), _mm_cvtps_epi32(d01));
                s22 = _mm_cvtepi32_ps(_mm_load_si128(S2+2));
                d02 = _mm_mul_ps(s12, f0);
                d02 = _mm_add_ps(d02, _mm_mul_ps(_mm_add_ps(s02, s22), f1));
                s23 = _mm_cvtepi32_ps(_mm_load_si128(S2+3));
                d03 = _mm_mul_ps(s13, f0);
                d03 = _mm_add_ps(d03, _mm_mul_ps(_mm_add_ps(s03, s23), f1));

                d_1 = _mm_packs_epi32(_mm_cvtps_epi32(d02), _mm_cvtps_epi32(d03));
                d_0 = _mm_packus_epi16(d_0, d_1);

                _mm_storeu_si128((__m128i*)(dst + width*3 + i), d_0);
            }

            for( ; i <= width - 4; i += 4 )
            {
                __m128i x0;
                __m128 s0, s1, s2;
                __m128 d0, d1;

                s2 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(src[1] + i)));
                d1 = _mm_mul_ps(s2, f0);

                s1 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(src[0] + i)));
                d0 = _mm_mul_ps(s1, f0);

                s0 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(src[-1] + i)));
                d0 = _mm_add_ps(d0, _mm_mul_ps(_mm_add_ps(s0, s2), f1));

                x0 = _mm_cvtps_epi32(d0);
                x0 = _mm_packs_epi32(x0, x0);
                x0 = _mm_packus_epi16(x0, x0);
                *(int*)(dst + i) = _mm_cvtsi128_si32(x0);

                s0 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(src[2] + i)));
                d0 = _mm_mul_ps(s0, f0);
                d1 = _mm_add_ps(d1, _mm_mul_ps(_mm_add_ps(s0, s1), f1));
                x0 = _mm_cvtps_epi32(d1);
                x0 = _mm_packs_epi32(x0, x0);
                x0 = _mm_packus_epi16(x0, x0);
                *(int*)(dst + width + i) = _mm_cvtsi128_si32(x0);

                s1 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(src[3] + i)));
                d1 = _mm_mul_ps(s1, f0);
                d0 = _mm_add_ps(d0, _mm_mul_ps(_mm_add_ps(s2, s1), f1));
                x0 = _mm_cvtps_epi32(d0);
                x0 = _mm_packs_epi32(x0, x0);
                x0 = _mm_packus_epi16(x0, x0);
                *(int*)(dst + width*2 + i) = _mm_cvtsi128_si32(x0);

                s2 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(src[4] + i)));
                d1 = _mm_add_ps(d1, _mm_mul_ps(_mm_add_ps(s0, s2), f1));
                x0 = _mm_cvtps_epi32(d1);
                x0 = _mm_packs_epi32(x0, x0);
                x0 = _mm_packus_epi16(x0, x0);
                *(int*)(dst + width*3 + i) = _mm_cvtsi128_si32(x0);
            }

            float f_0 = *ky;
            float f_1 = *(ky+1);
            for( ; i < width; i ++ )
            {
                float s0, s1, s2;
                float d0, d1;

                s2 = (float)(*(src[1] + i));
                d1 = s2 * f_0;

                s1 = (float)(*(src[0] + i));
                d0 = s1 * f_0;

                s0 = (float)(*(src[-1] + i));
                d0 += (s0 + s2) * f_1;

                *(dst + i) = (uchar)((int)d0);

                s0 = (float)(*(src[2] + i));
                d0 = s0 * f_0;
                d1 += (s0 + s1) * f_1;

                *(dst + width + i) = (uchar)((int)d1);

                s1 = (float)(*(src[3] + i));
                d1 = s1 * f_0;
                d0 += (s2 + s1) * f_1;

                *(dst + width*2 + i) = (uchar)((int)d0);

                s2 = (float)(*(src[4] + i));
                d1 += (s0 + s2) * f_1;

                *(dst + width*3 + i) = (uchar)((int)d1);
            }
            count -= 4;
        }
        else
        {
            for(; count >0 ; count --, src ++, dst += width)
            {
                i = 0;
                __m128 f0 = _mm_load_ss(ky);
                f0 = _mm_shuffle_ps(f0, f0, 0);
                for( ; i <= width - 16; i += 16 )
                {
                    __m128 s0, s1, s2, s3;
                    __m128i x0, x1;
                    S = (const __m128i*)(src[0] + i);
                    s0 = _mm_cvtepi32_ps(_mm_load_si128(S));
                    s0 = _mm_mul_ps(s0, f0);
                    s1 = _mm_cvtepi32_ps(_mm_load_si128(S+1));
                    s1 = _mm_mul_ps(s1, f0);
                    s2 = _mm_cvtepi32_ps(_mm_load_si128(S+2));
                    s2 = _mm_mul_ps(s2, f0);
                    s3 = _mm_cvtepi32_ps(_mm_load_si128(S+3));
                    s3 = _mm_mul_ps(s3, f0);

                    for( k = 1; k <= ksize2; k++ )
                    {
                        S = (const __m128i*)(src[k] + i);
                        S2 = (const __m128i*)(src[-k] + i);
                        __m128 f = _mm_load_ss(ky+k);
                        f = _mm_shuffle_ps(f, f, 0);
                        x0 = _mm_add_epi32(_mm_load_si128(S), _mm_load_si128(S2));
                        s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                        x1 = _mm_add_epi32(_mm_load_si128(S+1), _mm_load_si128(S2+1));
                        s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_cvtepi32_ps(x1), f));
                        x0 = _mm_add_epi32(_mm_load_si128(S+2), _mm_load_si128(S2+2));
                        s2 = _mm_add_ps(s2, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                        x1 = _mm_add_epi32(_mm_load_si128(S+3), _mm_load_si128(S2+3));
                        s3 = _mm_add_ps(s3, _mm_mul_ps(_mm_cvtepi32_ps(x1), f));
                    }

                    x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), _mm_cvtps_epi32(s1));
                    x1 = _mm_packs_epi32(_mm_cvtps_epi32(s2), _mm_cvtps_epi32(s3));
                    x0 = _mm_packus_epi16(x0, x1);
                    _mm_storeu_si128((__m128i*)(dst + i), x0);
                }

                for( ; i <= width - 4; i += 4 )
                {
                    __m128 f = _mm_load_ss(ky);
                    f = _mm_shuffle_ps(f, f, 0);
                    __m128i x0;
                    __m128 s0 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(src[0] + i)));
                    s0 = _mm_mul_ps(s0, f);

                    for( k = 1; k <= ksize2; k++ )
                    {
                        S = (const __m128i*)(src[k] + i);
                        S2 = (const __m128i*)(src[-k] + i);
                        f = _mm_load_ss(ky+k);
                        f = _mm_shuffle_ps(f, f, 0);
                        x0 = _mm_add_epi32(_mm_load_si128(S), _mm_load_si128(S2));
                        s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                    }

                    x0 = _mm_cvtps_epi32(s0);
                    x0 = _mm_packs_epi32(x0, x0);
                    x0 = _mm_packus_epi16(x0, x0);
                    *(int*)(dst + i) = _mm_cvtsi128_si32(x0);
                }
                float f_0 = *ky;
                for( ; i < width; i ++ )
                {
                    float d0;
                    d0 = (float)(*(src[0] + i)) * f_0;

                    for( k = 1; k <= ksize2; k++ )
                        d0 += ((float)(*(src[-k] + i)) + (float)(*(src[k] + i))) * (*(ky + k));

                    *(dst + i) = (uchar)((int)d0);
                }
            }
        }

        return i;
    }

    float * kernel;
    size_t ksize;
};

struct SymmColumnVec_32s8u
{
    SymmColumnVec_32s8u() {}
    SymmColumnVec_32s8u(const uchar * _kernel, int _len, int _bits)
    {
        ksize = _len;
        kernel = (float *)malloc(sizeof(float)*ksize);

        for(size_t i=0; i<ksize; i++)
            kernel[i] = (float)(((int *)_kernel)[i]) * (1./(1<<_bits));
    }

    SymmColumnVec_32s8u(const SymmColumnVec_32s8u &rhs) {
        ksize = rhs.ksize;
        kernel = (float*)malloc(sizeof(float)*ksize);
        memcpy(kernel, rhs.kernel, sizeof(float)*ksize);
    }

    SymmColumnVec_32s8u& operator=(const SymmColumnVec_32s8u& rhs) {
        ksize = rhs.ksize;
        kernel = (float*)malloc(sizeof(float)*ksize);
        memcpy(kernel, rhs.kernel, sizeof(float)*ksize);
        return *this;
    }

    ~SymmColumnVec_32s8u() {
        free(kernel);
    }

    MEGDNN_ATTRIBUTE_TARGET("sse2")
    int operator()(const uchar** _src, uchar* dst, int & count, int width) const
    {
        (void)count;
        int ksize2 = (ksize)/2;
        const float * ky = kernel + ksize2;
        int i = 0, k;
        const int** src = (const int**)_src;
        const __m128i *S, *S2;
        __m128 f0 = _mm_load_ss(ky);
        f0 = _mm_shuffle_ps(f0, f0, 0);
        __m128 f;
        i = 0;
        for (; i <= width - 16; i += 16) {
            __m128 s0, s1, s2, s3;
            __m128i x0, x1;
            S = (const __m128i*)(src[0] + i);
            s0 = _mm_cvtepi32_ps(_mm_load_si128(S));
            s0 = _mm_mul_ps(s0, f0);
            s1 = _mm_cvtepi32_ps(_mm_load_si128(S + 1));
            s1 = _mm_mul_ps(s1, f0);
            s2 = _mm_cvtepi32_ps(_mm_load_si128(S + 2));
            s2 = _mm_mul_ps(s2, f0);
            s3 = _mm_cvtepi32_ps(_mm_load_si128(S + 3));
            s3 = _mm_mul_ps(s3, f0);

            for (k = 1; k <= ksize2; k++) {
                S = (const __m128i*)(src[k] + i);
                S2 = (const __m128i*)(src[-k] + i);
                f = _mm_load_ss(ky + k);
                f = _mm_shuffle_ps(f, f, 0);
                x0 = _mm_add_epi32(_mm_load_si128(S), _mm_load_si128(S2));
                s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                x1 = _mm_add_epi32(_mm_load_si128(S + 1),
                                   _mm_load_si128(S2 + 1));
                s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_cvtepi32_ps(x1), f));
                x0 = _mm_add_epi32(_mm_load_si128(S + 2),
                                   _mm_load_si128(S2 + 2));
                s2 = _mm_add_ps(s2, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                x1 = _mm_add_epi32(_mm_load_si128(S + 3),
                                   _mm_load_si128(S2 + 3));
                s3 = _mm_add_ps(s3, _mm_mul_ps(_mm_cvtepi32_ps(x1), f));
            }

            x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), _mm_cvtps_epi32(s1));
            x1 = _mm_packs_epi32(_mm_cvtps_epi32(s2), _mm_cvtps_epi32(s3));
            x0 = _mm_packus_epi16(x0, x1);
            _mm_storeu_si128((__m128i*)(dst + i), x0);
        }

        for (; i <= width - 4; i += 4) {
            __m128i x0;
            __m128 s0 = _mm_cvtepi32_ps(
                    _mm_load_si128((const __m128i*)(src[0] + i)));
            s0 = _mm_mul_ps(s0, f0);

            for (k = 1; k <= ksize2; k++) {
                S = (const __m128i*)(src[k] + i);
                S2 = (const __m128i*)(src[-k] + i);
                f = _mm_load_ss(ky + k);
                f = _mm_shuffle_ps(f, f, 0);
                x0 = _mm_add_epi32(_mm_load_si128(S), _mm_load_si128(S2));
                s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
            }

            x0 = _mm_cvtps_epi32(s0);
            x0 = _mm_packs_epi32(x0, x0);
            x0 = _mm_packus_epi16(x0, x0);
            *(int*)(dst + i) = _mm_cvtsi128_si32(x0);
        }

        return i;
    }

    float * kernel;
    size_t ksize;
};

/////////////////////////////////////// 32f //////////////////////////////////

struct RowVec_32f
{
    RowVec_32f()
    {}

    RowVec_32f( const uchar * _kernel, int _len)
    {
        ksize = _len;
        kernel = (float*)_kernel;
    }

    MEGDNN_ATTRIBUTE_TARGET("sse")
    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
        int _ksize = ksize;
        const float* src0 = (const float*)_src;
        float* dst = (float*)_dst;
        const float* _kx = kernel;

        int i = 0, k;
        width *= cn;

        for( ; i <= width - 8; i += 8 )
        {
            const float* src = src0 + i;
            __m128 f, s0 = _mm_setzero_ps(), s1 = s0, x0, x1;
            for( k = 0; k < _ksize; k++, src += cn )
            {
                f = _mm_load_ss(_kx+k);
                f = _mm_shuffle_ps(f, f, 0);

                x0 = _mm_loadu_ps(src);
                x1 = _mm_loadu_ps(src + 4);
                s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
            }
            _mm_store_ps(dst + i, s0);
            _mm_store_ps(dst + i + 4, s1);
        }
        return i;
    }

    float * kernel;
    int ksize;
};

struct SymmRowSmallVec_32f
{
    SymmRowSmallVec_32f() {}
    SymmRowSmallVec_32f( const uchar * _kernel, int _len)
    {
        ksize = _len;
        kernel = (float*)_kernel;
    }

    MEGDNN_ATTRIBUTE_TARGET("sse2")
    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
        int i = 0, _ksize = ksize;
        float* dst = (float*)_dst;
        const float* src = (const float*)_src + (_ksize/2)*cn;
        const float* kx = kernel + _ksize/2;
        width *= cn;

        {
            if( _ksize == 1 )
                return 0;
            if( _ksize == 3 )
            {
                __m128 k0 = _mm_set1_ps(kx[0]), k1 = _mm_set1_ps(kx[1]);
                for( ; i <= width - 8; i += 8, src += 8 )
                {
                    __m128 x0, x1, x2, y0, y1, y2;
                    x0 = _mm_loadu_ps(src - cn);
                    x1 = _mm_loadu_ps(src);
                    x2 = _mm_loadu_ps(src + cn);
                    y0 = _mm_loadu_ps(src - cn + 4);
                    y1 = _mm_loadu_ps(src + 4);
                    y2 = _mm_loadu_ps(src + cn + 4);

                    x0 = _mm_mul_ps(_mm_add_ps(x0, x2), k1);
                    y0 = _mm_mul_ps(_mm_add_ps(y0, y2), k1);
                    x0 = _mm_add_ps(x0, _mm_mul_ps(x1, k0));
                    y0 = _mm_add_ps(y0, _mm_mul_ps(y1, k0));
                    _mm_store_ps(dst + i, x0);
                    _mm_store_ps(dst + i + 4, y0);
                }
            }
            else if( _ksize == 5 )
            {
                __m128 k0 = _mm_set1_ps(kx[0]), k1 = _mm_set1_ps(kx[1]), k2 = _mm_set1_ps(kx[2]);
                for( ; i <= width - 8; i += 8, src += 8 )
                {
                    __m128 x0, x1, x2, y0, y1, y2;
                    x0 = _mm_loadu_ps(src - cn);
                    x1 = _mm_loadu_ps(src);
                    x2 = _mm_loadu_ps(src + cn);
                    y0 = _mm_loadu_ps(src - cn + 4);
                    y1 = _mm_loadu_ps(src + 4);
                    y2 = _mm_loadu_ps(src + cn + 4);

                    x0 = _mm_mul_ps(_mm_add_ps(x0, x2), k1);
                    y0 = _mm_mul_ps(_mm_add_ps(y0, y2), k1);
                    x0 = _mm_add_ps(x0, _mm_mul_ps(x1, k0));
                    y0 = _mm_add_ps(y0, _mm_mul_ps(y1, k0));

                    x2 = _mm_add_ps(_mm_loadu_ps(src + cn*2), _mm_loadu_ps(src - cn*2));
                    y2 = _mm_add_ps(_mm_loadu_ps(src + cn*2 + 4), _mm_loadu_ps(src - cn*2 + 4));
                    x0 = _mm_add_ps(x0, _mm_mul_ps(x2, k2));
                    y0 = _mm_add_ps(y0, _mm_mul_ps(y2, k2));

                    _mm_store_ps(dst + i, x0);
                    _mm_store_ps(dst + i + 4, y0);
                }
            }
        }
        return i;
    }

    float * kernel;
    int ksize;
};

struct SymmColumnVec_32f
{
    SymmColumnVec_32f() { }
    SymmColumnVec_32f(const uchar * _kernel, int _len, int)
    {
        ksize = _len;
        kernel = (float*)_kernel;
    }

    MEGDNN_ATTRIBUTE_TARGET("sse2")
    int operator()(const uchar** _src, uchar* _dst, int &, int width) const
    {
        int ksize2 = (ksize)/2;
        const float* ky = kernel + ksize2;
        int i = 0, k;
        const float** src = (const float**)_src;
        const float *S, *S2;
        float* dst = (float*)_dst;

        {
            for( ; i <= width - 16; i += 16 )
            {
                __m128 f = _mm_load_ss(ky);
                f = _mm_shuffle_ps(f, f, 0);
                __m128 s0, s1, s2, s3;
                __m128 x0, x1;
                S = src[0] + i;
                s0 = _mm_load_ps(S);
                s1 = _mm_load_ps(S+4);
                s0 = _mm_mul_ps(s0, f);
                s1 = _mm_mul_ps(s1, f);
                s2 = _mm_load_ps(S+8);
                s3 = _mm_load_ps(S+12);
                s2 = _mm_mul_ps(s2, f);
                s3 = _mm_mul_ps(s3, f);

                for( k = 1; k <= ksize2; k++ )
                {
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_add_ps(_mm_load_ps(S), _mm_load_ps(S2));
                    x1 = _mm_add_ps(_mm_load_ps(S+4), _mm_load_ps(S2+4));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
                    x0 = _mm_add_ps(_mm_load_ps(S+8), _mm_load_ps(S2+8));
                    x1 = _mm_add_ps(_mm_load_ps(S+12), _mm_load_ps(S2+12));
                    s2 = _mm_add_ps(s2, _mm_mul_ps(x0, f));
                    s3 = _mm_add_ps(s3, _mm_mul_ps(x1, f));

                }
                _mm_storeu_ps(dst + i, s0);
                _mm_storeu_ps(dst + i + 4, s1);
                _mm_storeu_ps(dst + i + 8, s2);
                _mm_storeu_ps(dst + i + 12, s3);
            }

            for( ; i <= width - 4; i += 4 )
            {
                __m128 f = _mm_load_ss(ky);
                f = _mm_shuffle_ps(f, f, 0);
                __m128 x0, s0 = _mm_load_ps(src[0] + i);
                s0 = _mm_mul_ps(s0, f);

                for( k = 1; k <= ksize2; k++ )
                {
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    x0 = _mm_add_ps(_mm_load_ps(src[k]+i), _mm_load_ps(src[-k] + i));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));

                    // for test
                    //s0 += _mm_add_ps(s0, _mm_mul_ps(_mm_load_ps(src[k]+i), f));
                    //s0 += _mm_add_ps(s0, _mm_mul_ps(_mm_load_ps(src[-k]+i), f));
                }

                _mm_storeu_ps(dst + i, s0);
            }
        }

        return i;
    }

    float * kernel;
    int ksize;
};

struct SymmColumnSmallVec_32f
{
    SymmColumnSmallVec_32f() { }
    SymmColumnSmallVec_32f(const uchar * _kernel, int _len, int)
    {
        ksize = _len;
        kernel = (float*)_kernel;
    }

    MEGDNN_ATTRIBUTE_TARGET("sse2")
    int operator()(const uchar** _src, uchar* _dst, int & count, int width) const
    {
        (void)count;

        int ksize2 = (ksize)/2;
        const float* ky = kernel + ksize2;
        int i = 0;
        const float** src = (const float**)_src;
        const float *S0 = src[-1], *S1 = src[0], *S2 = src[1];
        float* dst = (float*)_dst;

        if (ky[0] == 2 && ky[1] == 1) {
            for (; i <= width - 8; i += 8) {
                __m128 s0, s1, s2, s3, s4, s5;
                s0 = _mm_load_ps(S0 + i);
                s1 = _mm_load_ps(S0 + i + 4);
                s2 = _mm_load_ps(S1 + i);
                s3 = _mm_load_ps(S1 + i + 4);
                s4 = _mm_load_ps(S2 + i);
                s5 = _mm_load_ps(S2 + i + 4);
                s0 = _mm_add_ps(s0, _mm_add_ps(s4, _mm_add_ps(s2, s2)));
                s1 = _mm_add_ps(s1, _mm_add_ps(s5, _mm_add_ps(s3, s3)));
                _mm_storeu_ps(dst + i, s0);
                _mm_storeu_ps(dst + i + 4, s1);
            }
        } else if (ky[0] == -2 && ky[1] == 1) {
            for (; i <= width - 8; i += 8) {
                __m128 s0, s1, s2, s3, s4, s5;
                s0 = _mm_load_ps(S0 + i);
                s1 = _mm_load_ps(S0 + i + 4);
                s2 = _mm_load_ps(S1 + i);
                s3 = _mm_load_ps(S1 + i + 4);
                s4 = _mm_load_ps(S2 + i);
                s5 = _mm_load_ps(S2 + i + 4);
                s0 = _mm_add_ps(s0, _mm_sub_ps(s4, _mm_add_ps(s2, s2)));
                s1 = _mm_add_ps(s1, _mm_sub_ps(s5, _mm_add_ps(s3, s3)));
                _mm_storeu_ps(dst + i, s0);
                _mm_storeu_ps(dst + i + 4, s1);
            }
        } else {
            __m128 k0 = _mm_set1_ps(ky[0]), k1 = _mm_set1_ps(ky[1]);
            for (; i <= width - 8; i += 8) {
                __m128 s0, s1, x0, x1;
                s0 = _mm_load_ps(S1 + i);
                s1 = _mm_load_ps(S1 + i + 4);
                s0 = _mm_mul_ps(s0, k0);
                s1 = _mm_mul_ps(s1, k0);
                x0 = _mm_add_ps(_mm_load_ps(S0 + i), _mm_load_ps(S2 + i));
                x1 = _mm_add_ps(_mm_load_ps(S0 + i + 4),
                                _mm_load_ps(S2 + i + 4));
                s0 = _mm_add_ps(s0, _mm_mul_ps(x0, k1));
                s1 = _mm_add_ps(s1, _mm_mul_ps(x1, k1));
                _mm_storeu_ps(dst + i, s0);
                _mm_storeu_ps(dst + i + 4, s1);
            }
        }

        return i;
    }

    float * kernel;
    int ksize;
};

/*!
 * \brief get the column filter.
 * \tparam FT The inner buffer type, used to store the product of src and
 * filter.
 * \tparam DT The dst image type.
 */
template <typename FT, typename DT>
static BaseColumnFilter* getLinearColumnFilter(Mat<FT>& kernel, int bits) {
    int ksize = kernel.cols();
    int anchor = ksize / 2;
    uchar* kernel_str = static_cast<uchar*>(kernel.raw_ptr());

    {
        if (ksize == 3) {
            if (std::is_same<DT, uchar>::value && std::is_same<FT, int>::value)
                return new SymmColumnSmallFilter<FixedPtCastEx<FT, DT>,
                                                 SymmColumnSmallVec_32s8u>(
                        kernel, anchor, FixedPtCastEx<FT, DT>(bits),
                        SymmColumnSmallVec_32s8u(kernel_str, ksize, bits));

            if (std::is_same<DT, float>::value && std::is_same<FT, float>::value)
                return new SymmColumnSmallFilter<FixedPtCastEx<FT, DT>,
                                                 SymmColumnSmallVec_32f>(
                        kernel, anchor, FixedPtCastEx<FT, DT>(0),
                        SymmColumnSmallVec_32f(kernel_str, ksize, 0));
        }
        if (std::is_same<DT, uchar>::value && std::is_same<FT, int>::value)
            return new SymmColumnFilter<FixedPtCastEx<FT, DT>,
                                        SymmColumnVec_32s8u>(
                    kernel, anchor, FixedPtCastEx<FT, DT>(bits),
                    SymmColumnVec_32s8u(kernel_str, ksize, bits));

        if (std::is_same<DT, float>::value && std::is_same<FT, float>::value)
            return new SymmColumnFilter<FixedPtCastEx<FT, DT>,
                                        SymmColumnVec_32f>(
                    kernel, anchor, FixedPtCastEx<FT, DT>(),
                    SymmColumnVec_32f(kernel_str, ksize, 0));
    }

    MegCVException(
            "Unsupported combination of source format and buffer format\n");
}

/*!
 * \brief get the row filter.
 * \tparam ST The src image type.
 * \tparam FT The inner buffer type, used to store the product of src and
 * filter.
 */
template <typename ST, typename FT>
static BaseRowFilter* getLinearRowFilter(Mat<FT>& kernel) {
    int ksize = kernel.cols();
    int anchor = ksize / 2;

    uchar* kernel_str = static_cast<uchar*>(kernel.raw_ptr());

    if (ksize <= 5) {
        if (std::is_same<ST, uchar>::value && std::is_same<FT, int>::value)
            return new SymmRowSmallFilter<ST, FT, SymmRowSmallVec_8u32s>(
                    kernel, anchor, SymmRowSmallVec_8u32s(kernel_str, ksize));

        if (std::is_same<ST, float>::value && std::is_same<FT, float>::value)
            return new SymmRowSmallFilter<ST, FT, SymmRowSmallVec_32f>(
                    kernel, anchor, SymmRowSmallVec_32f(kernel_str, ksize));
    }

    if (std::is_same<ST, uchar>::value && std::is_same<FT, int>::value)
        return new RowFilter<ST, FT, RowVec_8u32s>(
                kernel, anchor, RowVec_8u32s(kernel_str, ksize));

    if (std::is_same<ST, float>::value && std::is_same<FT, float>::value)
        return new RowFilter<ST, FT, RowVec_32f>(kernel, anchor,
                                                RowVec_32f(kernel_str, ksize));

    MegCVException(
            "Unsupported combination of source format and buffer format\n");
}

}  // namespace gaussian_blur
}  // namespace megcv
}  // namespace megdnn

// vim: syntax=cpp.doxygen
