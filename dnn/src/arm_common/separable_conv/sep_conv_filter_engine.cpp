/**
 * \file dnn/src/arm_common/separable_conv/sep_conv_filter_engine.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./sep_conv_filter.h"

#include <cfloat>
#include <cmath>
#include "src/arm_common/simd_macro/marm_neon.h"
#include <cstring>

namespace megdnn {
namespace arm_common {
namespace sep_conv {
using BorderMode = SeparableConv::Param::BorderMode;
using uchar = unsigned char;
using ushort = unsigned short;

//////////////////////////////////////////////
//vecOp
/////////////////////////////////////////////

struct RowVec_32f
{
    RowVec_32f()
    {}

    RowVec_32f(int _len)
    {
        ksize = _len;
    }

    int operator()(const uchar* _src, uchar* _dst, uchar * kernel, int width, int cn) const
    {

        int _ksize = ksize;
        const float* src0 = (const float*)_src;
        float* dst = (float*)_dst;
        const float* _kx = (float*)kernel;

        int i = 0, k;
        width *= cn;

        for( ; i <= width - 8; i += 8 )
        {
            const float* src = src0 + i;
            float32x4_t f, s0 = vdupq_n_f32(0), s1 = s0, x0, x1;
            for( k = 0; k < _ksize; k++, src += cn )
            {
                f = vdupq_n_f32(_kx[k]);

                x0 = vld1q_f32(src);
                x1 = vld1q_f32(src + 4);
                s0 = vmlaq_f32(s0, x0, f);
                s1 = vmlaq_f32(s1, x1, f);
            }
            vst1q_f32(dst + i, s0);
            vst1q_f32(dst + i + 4, s1);
        }
        for( ; i <= width - 4; i += 4 )
        {
            const float* src = src0 + i;
            float32x4_t f, s0 = vdupq_n_f32(0), x0;
            for( k = 0; k < _ksize; k++, src += cn )
            {
                f = vdupq_n_f32(_kx[k]);

                x0 = vld1q_f32(src);
                s0 = vmlaq_f32(s0, x0, f);
            }
            vst1q_f32(dst + i, s0);
        }
        return i;
    }
    int ksize;
};

struct SymmRowSmallVec_32f
{
    SymmRowSmallVec_32f() {}
    SymmRowSmallVec_32f(int _len)
    {
        ksize = _len;
    }

    int operator()(const uchar* _src, uchar* _dst, uchar * kernel, int width, int cn) const
    {
        int i = 0, _ksize = ksize;
        float* dst = (float*)_dst;
        const float* src = (const float*)_src + (_ksize/2)*cn;
        const float* kx = (float*)kernel + _ksize/2;
        width *= cn;

        {
            if( _ksize == 1 )
                return 0;
            if( _ksize == 3 )
            {
                float32x4_t k0 = vdupq_n_f32(kx[0]), k1 = vdupq_n_f32(kx[1]);
                for( ; i <= width - 8; i += 8, src += 8 )
                {
                    float32x4_t x0, x1, x2, y0, y1, y2;
                    x0 = vld1q_f32(src - cn);
                    x1 = vld1q_f32(src);
                    x2 = vld1q_f32(src + cn);
                    y0 = vld1q_f32(src - cn + 4);
                    y1 = vld1q_f32(src + 4);
                    y2 = vld1q_f32(src + cn + 4);

                    x0 = vmulq_f32(vaddq_f32(x0, x2), k1);
                    y0 = vmulq_f32(vaddq_f32(y0, y2), k1);
                    x0 = vmlaq_f32(x0, x1, k0);
                    y0 = vmlaq_f32(y0, y1, k0);
                    vst1q_f32(dst + i, x0);
                    vst1q_f32(dst + i + 4, y0);
                }
            }
            else if( _ksize == 5 )
            {
                float32x4_t k0 = vdupq_n_f32(kx[0]), k1 = vdupq_n_f32(kx[1]), k2 = vdupq_n_f32(kx[2]);
                for( ; i <= width - 8; i += 8, src += 8 )
                {
                    float32x4_t x0, x1, x2, y0, y1, y2;
                    x0 = vld1q_f32(src - cn);
                    x1 = vld1q_f32(src);
                    x2 = vld1q_f32(src + cn);
                    y0 = vld1q_f32(src - cn + 4);
                    y1 = vld1q_f32(src + 4);
                    y2 = vld1q_f32(src + cn + 4);

                    x0 = vmulq_f32(vaddq_f32(x0, x2), k1);
                    y0 = vmulq_f32(vaddq_f32(y0, y2), k1);
                    x0 = vmlaq_f32(x0, x1, k0);
                    y0 = vmlaq_f32(y0, y1, k0);

                    x2 = vaddq_f32(vld1q_f32(src + cn*2), vld1q_f32(src - cn*2));
                    y2 = vaddq_f32(vld1q_f32(src + cn*2 + 4), vld1q_f32(src - cn*2 + 4));
                    x0 = vmlaq_f32(x0, x2, k2);
                    y0 = vmlaq_f32(y0, y2, k2);

                    vst1q_f32(dst + i, x0);
                    vst1q_f32(dst + i + 4, y0);
                }
            }

        }
        return i;
    }
    int ksize;
};

struct ColumnVec_32f
{
    ColumnVec_32f() {}
    ColumnVec_32f(int _len, int)
    {
        ksize = _len;
    }

    int operator()(const uchar** _src, uchar* _dst, uchar * kernel, int &, int width) const
    {
        const float* ky = (const float*)kernel;
        int i = 0, k;
        const float** src = (const float**)_src;
        const float *S;
        float* dst = (float*)_dst;

        {
            for( ; i <= width - 16; i += 16 )
            {
                float32x4_t f = vdupq_n_f32(ky[0]);

                float32x4_t s0, s1, s2, s3;
                float32x4_t x0, x1;
                S = src[0] + i;
                s0 = vld1q_f32(S);
                s1 = vld1q_f32(S+4);
                s0 = vmulq_f32(s0, f);
                s1 = vmulq_f32(s1, f);
                s2 = vld1q_f32(S+8);
                s3 = vld1q_f32(S+12);
                s2 = vmulq_f32(s2, f);
                s3 = vmulq_f32(s3, f);

                for( k = 1; k < ksize; k++ )
                {
                    S = src[k] + i;
                    float32x4_t f = vdupq_n_f32(ky[k]);
                    x0 = vld1q_f32(S);
                    x1 = vld1q_f32(S+4);
                    s0 = vmlaq_f32(s0, f, x0);
                    s1 = vmlaq_f32(s1, f, x1);

                    x0 = vld1q_f32(S+8);
                    x1 = vld1q_f32(S+12);
                    s2 = vmlaq_f32(s2, f, x0);
                    s3 = vmlaq_f32(s3, f, x1);
                }
                s0 = vaddq_f32(s0, vld1q_f32(dst+i));
                s1 = vaddq_f32(s1, vld1q_f32(dst+i+4));
                s2 = vaddq_f32(s2, vld1q_f32(dst+i+8));
                s3 = vaddq_f32(s3, vld1q_f32(dst+i+12));

                vst1q_f32(dst + i, s0);
                vst1q_f32(dst + i + 4, s1);
                vst1q_f32(dst + i + 8, s2);
                vst1q_f32(dst + i + 12, s3);
            }

            for( ; i <= width - 4; i += 4 )
            {
                float32x4_t f = vdupq_n_f32(ky[0]);

                float32x4_t x0, s0 = vld1q_f32(src[0] + i);
                s0 = vmulq_f32(s0, f);

                for( k = 1; k < ksize; k++ )
                {
                    float32x4_t f = vdupq_n_f32(ky[k]);
                    S = src[k] + i;
                    x0 = vld1q_f32(S);
                    s0 = vmlaq_f32(s0, f, x0);
                }
                s0 = vaddq_f32(s0, vld1q_f32(dst + i));
                vst1q_f32(dst + i, s0);
            }
        }

        return i;
    }
    int ksize;
};

struct SymmColumnVec_32f
{
    SymmColumnVec_32f() {}
    SymmColumnVec_32f(int _len, int)
    {
        ksize = _len;
    }

    int operator()(const uchar** _src, uchar* _dst, uchar * kernel, int &, int width) const
    {
        int ksize2 = (ksize)/2;
        const float* ky = (const float*)kernel + ksize2;
        int i = 0, k;
        const float** src = (const float**)_src;
        const float *S, *S2;
        float* dst = (float*)_dst;

        {
            for( ; i <= width - 16; i += 16 )
            {
                float32x4_t f = vdupq_n_f32(ky[0]);

                float32x4_t s0, s1, s2, s3;
                float32x4_t x0, x1;
                S = src[0] + i;
                s0 = vld1q_f32(S);
                s1 = vld1q_f32(S+4);
                s0 = vmulq_f32(s0, f);
                s1 = vmulq_f32(s1, f);
                s2 = vld1q_f32(S+8);
                s3 = vld1q_f32(S+12);
                s2 = vmulq_f32(s2, f);
                s3 = vmulq_f32(s3, f);

                for( k = 1; k <= ksize2; k++ )
                {
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    float32x4_t f = vdupq_n_f32(ky[k]);

                    x0 = vaddq_f32(vld1q_f32(S), vld1q_f32(S2));
                    x1 = vaddq_f32(vld1q_f32(S+4), vld1q_f32(S2+4));
                    s0 = vmlaq_f32(s0, x0, f);
                    s1 = vmlaq_f32(s1, x1, f);
                    x0 = vaddq_f32(vld1q_f32(S+8), vld1q_f32(S2+8));
                    x1 = vaddq_f32(vld1q_f32(S+12), vld1q_f32(S2+12));
                    s2 = vmlaq_f32(s2, x0, f);
                    s3 = vmlaq_f32(s3, x1, f);

                }
                s0 = vaddq_f32(s0, vld1q_f32(dst+i));
                s1 = vaddq_f32(s1, vld1q_f32(dst+i+4));
                s2 = vaddq_f32(s2, vld1q_f32(dst+i+8));
                s3 = vaddq_f32(s3, vld1q_f32(dst+i+12));

                vst1q_f32(dst + i, s0);
                vst1q_f32(dst + i + 4, s1);
                vst1q_f32(dst + i + 8, s2);
                vst1q_f32(dst + i + 12, s3);
            }

            for( ; i <= width - 4; i += 4 )
            {
                float32x4_t f = vdupq_n_f32(ky[0]);
                float32x4_t x0, s0 = vld1q_f32(src[0] + i);
                s0 = vmulq_f32(s0, f);

                for( k = 1; k <= ksize2; k++ )
                {
                    float32x4_t f = vdupq_n_f32(ky[k]);
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    x0 = vaddq_f32(vld1q_f32(S), vld1q_f32(S2));
                    s0 = vmlaq_f32(s0, x0, f);
                }
                s0 = vaddq_f32(s0, vld1q_f32(dst + i));
                vst1q_f32(dst + i, s0);
            }
        }

        return i;

    }
    int ksize;
};


struct SymmColumnSmallVec_32f
{
    SymmColumnSmallVec_32f() { }
    SymmColumnSmallVec_32f(int _len, int)
    {
        ksize = _len;
    }

    int operator()(const uchar** _src, uchar* _dst, uchar * kernel, int & count, int width) const
    {
        (void)count;

        int ksize2 = (ksize)/2;
        const float* ky = (float*)kernel + ksize2;
        int i = 0;
        const float** src = (const float**)_src;
        const float *S0 = src[-1], *S1 = src[0], *S2 = src[1];
        float* dst = (float*)_dst;
        {
            float32x4_t k0 = vdupq_n_f32(ky[0]), k1 = vdupq_n_f32(ky[1]);
            for( ; i <= width - 8; i += 8 )
            {
                float32x4_t s0, s1, x0, x1;
                s0 = vld1q_f32(S1 + i);
                s1 = vld1q_f32(S1 + i + 4);
                s0 = vmulq_f32(s0, k0);
                s1 = vmulq_f32(s1, k0);
                x0 = vaddq_f32(vld1q_f32(S0 + i), vld1q_f32(S2 + i));
                x1 = vaddq_f32(vld1q_f32(S0 + i + 4), vld1q_f32(S2 + i + 4));
                s0 = vmlaq_f32(s0, x0, k1);
                s1 = vmlaq_f32(s1, x1, k1);
                s0 = vaddq_f32(s0, vld1q_f32(dst + i));
                s1 = vaddq_f32(s1, vld1q_f32(dst + i + 4));
                vst1q_f32(dst + i, s0);
                vst1q_f32(dst + i + 4, s1);
            }
        }

        return i;
    }
    int ksize;
};

//////////////////////////////////////////////////////////////////////////////////////
//%RowFilter%
//////////////////////////////////////////////////////////////////////////////////////

BaseRowFilter::BaseRowFilter() { ksize = anchor = -1; }
BaseRowFilter::~BaseRowFilter() {}

template<typename ST, typename DT, class VecOp> struct RowFilter : public BaseRowFilter
{
    RowFilter(int _ksize, int _anchor, const VecOp& _vecOp=VecOp() )
    {
        anchor = _anchor;
        ksize = _ksize;
        vecOp = _vecOp;
    }

    void operator()(const uchar* src, uchar* dst, uchar* kernel, int width, int cn)
    {
        int _ksize = ksize;
        const DT* kx = (DT* )kernel;
        const ST* S;
        DT* D = (DT*)dst;
        int i, k;

        i = vecOp(src, dst, kernel, width, cn);
        width *= cn;
#if MEGCV_ENABLE_UNROLLED
        for( ; i <= width - 4; i += 4 )
        {
            S = (const ST*)src + i;
            DT f = kx[0];
            DT s0 = f*S[0], s1 = f*S[1], s2 = f*S[2], s3 = f*S[3];

            for( k = 1; k < _ksize; k++ )
            {
                S += cn;
                f = kx[k];
                s0 += f*S[0]; s1 += f*S[1];
                s2 += f*S[2]; s3 += f*S[3];
            }

            D[i] = s0; D[i+1] = s1;
            D[i+2] = s2; D[i+3] = s3;
        }
#endif
        for( ; i < width; i++ )
        {
            S = (const ST*)src + i;
            DT s0 = kx[0]*S[0];
            for( k = 1; k < _ksize; k++ )
            {
                S += cn;
                s0 += kx[k]*S[0];
            }
            D[i] = s0;
        }
    }
    VecOp vecOp;
};


template<typename ST, typename DT, class VecOp> struct SymmRowSmallFilter :
    public RowFilter<ST, DT, VecOp>
{
    SymmRowSmallFilter(int _ksize, int _anchor,
            const VecOp& _vecOp = VecOp() )
        : RowFilter<ST, DT, VecOp>( _ksize, _anchor, _vecOp )
    {}

    void operator()(const uchar* src, uchar* dst, uchar* kernel, int width, int cn)
    {
        int ksize2 = this->ksize/2, ksize2n = ksize2*cn;
        const DT* kx = (DT*)kernel + ksize2;
        DT* D = (DT*)dst;
        int i = this->vecOp(src, dst, kernel, width, cn), j, k;
        const ST* S = (const ST*)src + i + ksize2n;
        width *= cn;

        {
            if( this->ksize == 1 && kx[0] == 1 )
            {
                for( ; i <= width - 2; i += 2 )
                {
                    DT s0 = S[i], s1 = S[i+1];
                    D[i] = s0; D[i+1] = s1;
                }
                S += i;
            }
            else if( this->ksize == 3 )
            {
                DT k0 = kx[0], k1 = kx[1];
                for( ; i <= width - 2; i += 2, S += 2 )
                {
                    DT s0 = S[0]*k0 + (S[-cn] + S[cn])*k1, s1 = S[1]*k0 + (S[1-cn] + S[1+cn])*k1;
                    D[i] = s0; D[i+1] = s1;
                }
            }
            else if( this->ksize == 5 )
            {
                DT k0 = kx[0], k1 = kx[1], k2 = kx[2];
                for( ; i <= width - 2; i += 2, S += 2 )
                {
                    DT s0 = S[0]*k0 + (S[-cn] + S[cn])*k1 + (S[-cn*2] + S[cn*2])*k2;
                    DT s1 = S[1]*k0 + (S[1-cn] + S[1+cn])*k1 + (S[1-cn*2] + S[1+cn*2])*k2;
                    D[i] = s0; D[i+1] = s1;
                }
            }

            for( ; i < width; i++, S++ )
            {
                DT s0 = kx[0]*S[0];
                for( k = 1, j = cn; k <= ksize2; k++, j += cn )
                    s0 += kx[k]*(S[j] + S[-j]);
                D[i] = s0;
            }
        }
    }
};

template <typename T, typename T1>
    BaseRowFilter * getLinearRowFilter(int ksize, bool is_symm_kernel)
    {
        // TODO: calculate anchor
        int anchor = ksize/2;
        if(is_symm_kernel) {
            if( ksize <= 5 )
            {
                //if( typeid(T) == typeid(float) && typeid(T1) == typeid(float))
                    return new SymmRowSmallFilter<T, T1, SymmRowSmallVec_32f>
                        (ksize, anchor, SymmRowSmallVec_32f(ksize));
            }

            //if( typeid(T) == typeid(float) && typeid(T1) == typeid(float))
                return new RowFilter<T, T1, RowVec_32f>
                    (ksize, anchor, RowVec_32f(ksize));
        } else {
            //if( typeid(T) == typeid(float) && typeid(T1) == typeid(float))
                return new RowFilter<T, T1, RowVec_32f>
                    (ksize, anchor, RowVec_32f(ksize));
        }

        //printf("Unsupported combination of source format (=%s), and buffer format (=%s)",
        //        typeid(T).name(), typeid(T1).name());
        //exit(1);
    }
//////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////
//%BaseColFilter%
//////////////////////////////////////////////////////////////////////////////////////

BaseColumnFilter::BaseColumnFilter() { ksize = anchor = -1; }
BaseColumnFilter::~BaseColumnFilter() {}
void BaseColumnFilter::reset() {}

template<class CastOp, class VecOp> struct ColumnFilter : public BaseColumnFilter
{
    typedef typename CastOp::type1 ST;
    typedef typename CastOp::rtype DT;

    ColumnFilter(int _ksize, int _anchor,
            const CastOp& _castOp=CastOp(),
            const VecOp& _vecOp=VecOp())
    {
        this->anchor = _anchor;
        this->ksize = _ksize;
        this->castOp0 = _castOp;
        this->vecOp = _vecOp;
    }

    void operator()(const uchar** src, uchar* dst, uchar* kernel, int dststep, int count, int width)
    {
        const ST* ky = (ST*)kernel;
        int i = 0, k;
        CastOp castOp = this->castOp0;

        {
            for( ; count > 0; count--, dst += dststep, src++ )
            {
                DT* D = (DT*)dst;
                i = (this->vecOp)(src, dst, kernel, count, width);
#if MEGCV_ENABLE_UNROLLED
                for( ; i <= width - 4; i += 4 )
                {
                    ST f = ky[0];
                    const ST* S = (const ST*)src[0] + i;
                    ST s0 = f*S[0], s1 = f*S[1],
                       s2 = f*S[2], s3 = f*S[3];

                    for( k = 1; k < ksize; k++ )
                    {
                        S = (const ST*)src[k] + i;
                        f = ky[k];
                        s0 += f*S[0];
                        s1 += f*S[1];
                        s2 += f*S[2];
                        s3 += f*S[3];
                    }

                    D[i] += castOp(s0); D[i+1] += castOp(s1);
                    D[i+2] += castOp(s2); D[i+3] += castOp(s3);
                }
#endif
                for( ; i < width; i++ )
                {
                    ST s0 = D[i];
                    //ST s0 = ky[0]*((const ST*)src[0])[i];
                    for( k = 0; k < ksize; k++ ) {
                        s0 += ky[k]* ((const ST*)src[k])[i];
                    }
                    D[i] = castOp(s0);
                    //D[i] += castOp(s0);
                }
            }
        }
    }
    CastOp castOp0;
    VecOp vecOp;
};

template<class CastOp, class VecOp> struct SymmColumnFilter : public BaseColumnFilter
{
    typedef typename CastOp::type1 ST;
    typedef typename CastOp::rtype DT;

    SymmColumnFilter(int _ksize, int _anchor,
            const CastOp& _castOp=CastOp(),
            const VecOp& _vecOp=VecOp())
    {
        this->anchor = _anchor;
        this->ksize = _ksize;
        this->castOp0 = _castOp;
        this->vecOp = _vecOp;
    }

    void operator()(const uchar** src, uchar* dst, uchar* kernel, int dststep, int count, int width)
    {
        int ksize2 = this->ksize/2;
        const ST* ky = (ST*)kernel + ksize2;
        int i, k;
        CastOp castOp = this->castOp0;
        src += ksize2;

        {
            for( ; count > 0; count--, dst += dststep, src++ )
            {
                DT* D = (DT*)dst;
                i = (this->vecOp)(src, dst, kernel, count, width);
#if MEGCV_ENABLE_UNROLLED
                for( ; i <= width - 4; i += 4 )
                {
                    ST f = ky[0];
                    const ST* S = (const ST*)src[0] + i, *S2;
                    ST s0 = f*S[0], s1 = f*S[1],
                       s2 = f*S[2], s3 = f*S[3];

                    for( k = 1; k <= ksize2; k++ )
                    {
                        S = (const ST*)src[k] + i;
                        S2 = (const ST*)src[-k] + i;
                        f = ky[k];
                        s0 += f*(S[0] + S2[0]);
                        s1 += f*(S[1] + S2[1]);
                        s2 += f*(S[2] + S2[2]);
                        s3 += f*(S[3] + S2[3]);
                    }

                    D[i] += castOp(s0); D[i+1] += castOp(s1);
                    D[i+2] += castOp(s2); D[i+3] += castOp(s3);
                }
#endif
                for( ; i < width; i++ )
                {
                    ST s0 = ky[0]*((const ST*)src[0])[i];
                    for( k = 1; k <= ksize2; k++ ) {
                        s0 += ky[k]*(((const ST*)src[k])[i] + ((const ST*)src[-k])[i]);
                        //s0 += ky[k]*((const ST*)src[k])[i];
                        //s0 += ky[k]*((const ST*)src[-k])[i];
                    }
                    D[i] += castOp(s0);
                }
            }
        }
    }
    CastOp castOp0;
    VecOp vecOp;
};


template<class CastOp, class VecOp>
    struct SymmColumnSmallFilter : public SymmColumnFilter<CastOp, VecOp>
{
    typedef typename CastOp::type1 ST;
    typedef typename CastOp::rtype DT;

    SymmColumnSmallFilter( int _ksize, int _anchor,
            const CastOp & _castOp=CastOp(),
            const VecOp & _vecOp=VecOp())
        : SymmColumnFilter<CastOp, VecOp>(_ksize, _anchor, _castOp, _vecOp )
    {
        megdnn_assert(this->ksize == 3 );
    }

    void operator()(const uchar** src, uchar* dst, uchar* kernel, int dststep, int count, int width)
    {
        int ksize2 = this->ksize/2;
        const ST* ky = (ST*)kernel + ksize2;
        int i = 0;
        ST f0 = ky[0], f1 = ky[1];
        CastOp castOp = this->castOp0;
        src += ksize2;

        /*
        if((typeid(ST) == typeid(int) && typeid(DT) == typeid(uchar)))
        {
            (this->vecOp)(src, dst, kernel, count, width);
        }
        */
        for( ; count > 0; count--, dst += dststep, src++ )
        {
            DT* D = (DT*)dst;

            i = (this->vecOp)(src, dst, kernel, count, width);
            if(count == 0)
                break;
            const ST* S0 = (const ST*)src[-1];
            const ST* S1 = (const ST*)src[0];
            const ST* S2 = (const ST*)src[1];
            {
#if MEGCV_ENABLE_UNROLLED
                for( ; i <= width - 4; i += 4 )
                {
                    ST s0 = (S0[i] + S2[i])*f1 + S1[i]*f0;
                    ST s1 = (S0[i+1] + S2[i+1])*f1 + S1[i+1]*f0;
                    D[i] += castOp(s0);
                    D[i+1] += castOp(s1);

                    s0 = (S0[i+2] + S2[i+2])*f1 + S1[i+2]*f0;
                    s1 = (S0[i+3] + S2[i+3])*f1 + S1[i+3]*f0;
                    D[i+2] += castOp(s0);
                    D[i+3] += castOp(s1);
                }
#endif
                for( ; i < width; i ++ )
                {
                    ST s0 = (S0[i] + S2[i])*f1 + S1[i]*f0;
                    D[i] += castOp(s0);
                }
            }
        }

    }
};


template<typename T1, typename T>
    BaseColumnFilter * getLinearColumnFilter(int ksize, int /*bits*/, bool is_symm_kernel)
    {
        int anchor = ksize/2;
        {
            if(is_symm_kernel) {
                if( ksize == 3 )
                {

                    //if( typeid(T1) == typeid(float) && typeid(T) == typeid(float) )
                        return new SymmColumnSmallFilter<FixedPtCastEx<T1, T>,SymmColumnSmallVec_32f>
                            (ksize, anchor, FixedPtCastEx<T1, T>(0),
                             SymmColumnSmallVec_32f(ksize, 0));
                }
                //if( typeid(T1) == typeid(float) && typeid(T) == typeid(float) )
                    return new SymmColumnFilter<FixedPtCastEx<T1, T>, SymmColumnVec_32f>
                        (ksize, anchor, FixedPtCastEx<T1, T>(),
                     SymmColumnVec_32f(ksize, 0));
            } else {
                //if( typeid(T1) == typeid(float) && typeid(T) == typeid(float) )
                    return new ColumnFilter<FixedPtCastEx<T1, T>, ColumnVec_32f>
                        (ksize, anchor, FixedPtCastEx<T1, T>(),
                     ColumnVec_32f(ksize, 0));
            }
        }
        //printf("Unsupported combination of buffer format (=%s), and destination format (=%s)",
        //        typeid(T1).name(), typeid(T).name());
        //exit(1);
    }

//////////////////////////////////////////////////////////////////////////////////////
////%FilterEngine%
//////////////////////////////////////////////////////////////////////////////////////

    FilterEngine::FilterEngine(const int &ih, const int &iw,
                    const int &oh, const int &ow,
                    const int &kh, const int &kw,
                    const int &anchor_h, const int &anchor_w,
                    BorderMode borderType,
                    bool is_symm_kernel) {
        init(ih, iw, oh, ow, kh, kw, anchor_h, anchor_w, borderType, is_symm_kernel);
    }


    FilterEngine::~FilterEngine()
    {
        delete rowFilter_;
        delete colFilter_;
    }

    void FilterEngine::init(const int &ih, const int &iw,
                    const int &oh, const int &ow,
                    const int &kh, const int &kw,
                    const int &anchor_h, const int &anchor_w,
                    BorderMode borderType,
                    bool is_symm_kernel) {
        // reduce warning
        int wrn = ih + iw + oh; ++wrn;

        ksize_x_ = kw;
        ksize_y_ = kh;
        anchor_x_ =  anchor_w;
        anchor_y_ =  anchor_h;
        borderType_ = borderType;
        is_symm_kernel_ = is_symm_kernel;

        rowFilter_ = getLinearRowFilter<float, float>(kw, is_symm_kernel_);
        colFilter_ = getLinearColumnFilter<float, float>(kh, 0, is_symm_kernel_);

        rowBufferOutputRow_ = 1;
        maxBufferRow_ = ksize_y_ + rowBufferOutputRow_ - 1;
        //int rowBuffStride_ = sizeof(float)*(int)align_size(maxWidth + (ksize_y_ - 1),VEC_ALIGN);
        rowBuffStride_ = sizeof(float) * (int)align_size(ow, VEC_ALIGN);
        row_ptr_.resize(maxBufferRow_);
        ringBuf_.resize(rowBuffStride_ * maxBufferRow_ + VEC_ALIGN);

        // There is no need to use constBorder when padding == 0.
        //if (borderType_ = BORDER_CONSTANT) {
        //    constBorderRow.resize(sizeof(int) * (maxWidth + ksize.cols() - 1) + VEC_ALIGN);
        //}


    }

    void FilterEngine::exec( const TensorND & src,
                const TensorND & kernel_x,
                const TensorND & kernel_y,
                const TensorND & dst) {

        //int stride_src = src.layout.stride[1];
        //int stride_dst = dst.layout.stride[1];
        //float *src0 = src.ptr();
        //float *dst0 = dst.ptr();
        float * src_cur_row = src.ptr<float>();
        float * src_cur_step = src.ptr<float>();
        float * dst_cur_chan = dst.ptr<float>();
        int width_src = (int)src.layout.shape[3];
        int width_dst = (int)dst.layout.shape[3];
        int height_src = (int)src.layout.shape[2];
        //int height_dst =  dst.layout.shape[2];
        int kernel_chan_stride = (int)kernel_x.layout.stride[1];
        memset(dst.ptr<float>(), 0, sizeof(float) * dst.layout.total_nr_elems());

        for(int step  = 0; step < (int)src.layout.shape[0]; ++step) {
            for(int chan_out = 0; chan_out < (int)dst.layout.shape[1];
                ++ chan_out, dst_cur_chan += dst.layout.stride[1]) {
                float* kx = kernel_x.ptr<float>();
                float* ky = kernel_y.ptr<float>();
                src_cur_row = src_cur_step;
                // handle a channel of input
                for(int chan_in = 0; chan_in < (int)src.layout.shape[1]; ++ chan_in) {
                    // 1. init row buffer borden
                    // No need to init row border when padding == 0.

                    // 2. fill ring buffer & calculate
                    int row_count = 0;
                    int row_ptr_pos = 0;
                    int dststep = dst.layout.stride[2];
                    int bufRows = (int)row_ptr_.size();
                    int bi = 0;
                    float* dst_cur_row = dst_cur_chan;
                    for(row_count = 0; row_count < height_src;
                        ++row_count, src_cur_row += width_src) {

                        //2.1 Get tab row. No need to do this when padding == 0.

                        //2.2 Calculate a row.
                        bi = row_count % bufRows;
                        uchar* brow = align_ptr(&ringBuf_[0], VEC_ALIGN) + bi * rowBuffStride_;
                        if(row_count < bufRows - 1) {
                            row_ptr_[bi] = (float*)brow;
                        } else {
                            row_ptr_[bufRows - 1] = (float*)brow;
                        }

                        // Get a row & make border
                        //uchar* row = &srcRow[0];
                        //memcpy( row + _dx1*esz, src, (width1 - _dx2 - _dx1)*esz );
                        uchar* row = (uchar*)src_cur_row;
                        (*rowFilter_)(row, brow, (uchar*)kx, width_dst, 1);
                        // operator()(const uchar* src, uchar* dst, uchar* kernel, int width, int cn)

                        // Keeping fill the ring_buff until its length is ky
                        if(row_count < bufRows - 1) {
                            ++ row_ptr_pos;
                            continue;
                        }

                        // 2.3 Calculate column
                        // operator()(const uchar** src, uchar* dst, ST* kernel, int dststep, int count, int width)
                        (*colFilter_)((const uchar**)(&row_ptr_[0]), (uchar*)dst_cur_row,
                                    (uchar*)ky, dststep, rowBufferOutputRow_, width_dst);

                        // Update row_ptr
                        for(int i = 0; i< bufRows - 1; ++i) {
                            row_ptr_[i] = row_ptr_[i+1];
                        }
                        dst_cur_row += width_dst; //dst.layout.stride[2];
                    }
                    kx += kernel_chan_stride;
                    ky += kernel_chan_stride;
                } // chan_in
            } // chan_out
            src_cur_step += src.layout.shape[0];
        } //step_in
    }

} // namespace sep_conv
} // namespace arm_common
} // namespace megdnn
