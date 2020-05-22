/**
 * \file dnn/src/arm_common/separable_conv/sep_conv_common.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/common/utils.h"
#include "megdnn/oprs.h"

namespace megdnn {
namespace arm_common {
namespace sep_conv {

#define VEC_ALIGN 16

using BorderMode = SeparableConv::Param::BorderMode;
using uchar = unsigned char;
using ushort = unsigned short;

///////////  helper  ///////////

static inline size_t align_size(size_t sz, int n)
{
    megdnn_assert((n & (n - 1)) == 0);
    return (sz + n-1) & -n;
}

static inline int clip(int x, int a, int b)
{
    return x >= a ? (x < b ? x : b-1) : a;
}

template<typename _Tp> static inline _Tp* align_ptr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

template <typename T>
T saturate_cast(T x)
{ return x; }

template <typename T>
T saturate_cast(int x)
{
    return static_cast<T>(x);
}
template <typename T>
T saturate_cast(float x)
{
    return static_cast<T>(x);
}
template <typename T>
T saturate_cast(double x)
{
    return static_cast<T>(x);
}

// int -> uchar
template<> unsigned char saturate_cast<unsigned char>(int x);
// int -> short
template<> short saturate_cast<short>(int x);
// float -> int
template<> int saturate_cast<int>(float x);
// float -> short
template<> short saturate_cast<short>(float x);
// double -> int
template<> int saturate_cast<int>(double x);


template<typename ST, typename DT, int bits> struct FixedPtCast
{
    typedef ST type1;
    typedef DT rtype;
    enum { SHIFT = bits, DELTA = 1 << (bits-1) };

    DT operator()(ST val) const
    { return saturate_cast<DT>((val + DELTA)>>SHIFT); }
};

template<typename ST, typename DT> struct FixedPtCastEx
{
    typedef ST type1;
    typedef DT rtype;

    FixedPtCastEx() : SHIFT(0), DELTA(0) {}
    FixedPtCastEx(int bits) : SHIFT(bits), DELTA(bits ? 1 << (bits-1) : 0) {}
    DT operator()(ST val) const { return saturate_cast<DT>(val + DELTA); }
    int SHIFT, DELTA;
};

template<> struct FixedPtCastEx <int, uchar>
{
    typedef int type1;
    typedef uchar rtype;

    FixedPtCastEx() : SHIFT(0), DELTA(0) {}
    FixedPtCastEx(int bits) : SHIFT(bits), DELTA(bits ? 1 << (bits-1) : 0) {}
    uchar operator()(int val) const { return saturate_cast<uchar>((val + DELTA)>>SHIFT); }
    int SHIFT, DELTA;
};


template<typename ST, typename DT> struct Cast
{
    typedef ST type1;
    typedef DT rtype;

    DT operator()(ST val) const { return saturate_cast<DT>(val); }
};

static inline int border_interpolate(int p, int len, BorderMode bmode)
{
    if( (unsigned)p < (unsigned)len )
        ;
    else if( bmode == BorderMode::BORDER_REPLICATE )
        p = p < 0 ? 0 : len - 1;
    else if( bmode == BorderMode::BORDER_REFLECT || bmode == BorderMode::BORDER_REFLECT_101 )
    {
        int delta = (bmode == BorderMode::BORDER_REFLECT_101);
        if( len == 1 )
            return 0;
        do
        {
            if( p < 0 )
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        }
        while( (unsigned)p >= (unsigned)len );
    }
    else if( bmode == BorderMode::BORDER_WRAP )
    {
        megdnn_assert(len > 0);
        if( p < 0 )
            p -= ((p-len+1)/len)*len;
        while (p >= len) {
            p -= len;
        }
    }
    else if( bmode == BorderMode::BORDER_CONSTANT )
        p = -1;
    else
        megdnn_throw("Unknown/unsupported border type");
    return p;
}
///////////  helper  ///////////

} // namespace sep_conv
} // namespace arm_common
} // namespace megdnn

// vim: syntax=cpp.doxygen
