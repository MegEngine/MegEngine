# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import itertools

def gen(mode, simd, fsize):
    funcname = "convolution_{mode}_fh{fsize}_{simd}".format(**vars())
    filename = funcname + ".cpp"
    if simd == 'fma':
        MAX_H = 15 - fsize
    elif simd == 'avx' or simd == 'sse':
        MAX_H = 14 - fsize
    else:
        assert False
    if simd == "sse":
        width = 4
        mm_type = "__m128"
        mm_load = "_mm_loadu_ps"
        mm_store = "_mm_storeu_ps"
        mm_mul = "_mm_mul_ps"
        mm_add = "_mm_add_ps"
        mm_set1 = "_mm_set1_ps"
        mm_set0 = "_mm_setzero_ps"
        mm_max = "_mm_max_ps"
        mm_set1_sign = ""
        header = ["xmmintrin.h"]
    elif simd == "avx":
        width = 8
        mm_type = "__m256"
        mm_load = "_mm256_loadu_ps"
        mm_store = "_mm256_storeu_ps"
        mm_mul = "_mm256_mul_ps"
        mm_add = "_mm256_add_ps"
        mm_set1 = "_mm256_broadcast_ss"
        mm_set0 = "_mm256_setzero_ps"
        mm_max = "_mm256_max_ps"
        mm_set1_sign = "&"
        header = ["immintrin.h", "avxintrin.h"]
    elif simd == "fma":
        width = 8
        mm_type = "__m256"
        mm_load = "_mm256_loadu_ps"
        mm_store = "_mm256_storeu_ps"
        mm_set1 = "_mm256_broadcast_ss"
        mm_set0 = "_mm256_setzero_ps"
        mm_max = "_mm256_max_ps"
        mm_set1_sign = "&"
        header = ["immintrin.h", "avxintrin.h", "fmaintrin.h"]
    with open(filename, 'w') as f:
        for H in range(1, MAX_H+1):
            f.write("""#define SIMD_H{H} do {{ \\
const size_t sh = dh; \\
const float *src_d = src + sh*src_w; \\
float *dst_d = dst + dh*dst_w; \\
size_t dw = dst_w_beg; \\
for (; dw < dst_w_end; dw += {width}) {{ \\
    const size_t sw = dw; \\
    float *dst_dd = dst_d + dw; \\
    {mm_type} tmp0; \\
""".format(**vars()))
            if simd != "fma":
                f.write("    {mm_type} tmp1; \\\n".format(**vars()))
            for h in range(H):
                f.write("""    {mm_type} res{h}; \\
    res{h} = {mm_load}(dst_dd + {h}*dst_w); \\
""".format(**vars()))
            f.write("""    for (size_t fw = 0; fw < flt_w; ++fw) {{ \\
        const float *src_dd = src_d + sw + fw; \\
""".format(**vars()))
            for fh in range(fsize):
                if mode == 'xcorr':
                    f.write("""        {mm_type} vf{fh} = {mm_set1}({mm_set1_sign}filter[{fh}*flt_w+fw]); \\
""".format(**vars()))
                elif mode == 'conv':
                    f.write("""        {mm_type} vf{fh} = {mm_set1}({mm_set1_sign}filter[{fh}*flt_w+flt_w-fw-1]); \\
""".format(**vars()))
                else:
                    assert False
            for ih in range(H+fsize-1):
                f.write("""        tmp0 = {mm_load}(src_dd + {ih}*src_w); \\
""".format(**vars()))
                for fh in range(fsize):
                    if mode == 'xcorr':
                        oh = ih - fh
                    elif mode == 'conv':
                        oh = ih - (fsize-fh-1)
                    else:
                        assert False
                    if oh >= 0 and oh < H:
                        if simd == "fma":
                            f.write("""        res{oh} = _mm256_fmadd_ps(tmp0, vf{fh}, res{oh}); \\
""".format(**vars()))
                        else:
                            f.write("""        tmp1 = {mm_mul}(tmp0, vf{fh}); \\
""".format(**vars()))
                            f.write("""        res{oh} = {mm_add}(res{oh}, tmp1); \\
""".format(**vars()))
            f.write("""    }} \\
""".format(**vars()))
            for h in range(H):
                f.write("""    {mm_store}(dst_dd + {h}*dst_w, res{h}); \\
""".format(**vars()))
            f.write("""}} \\
}} while (0)
""".format(**vars()))
            f.write("\n")


        for i in header:
            f.write('#include <{}>\n'.format(i))
        f.write("""#include <algorithm>

#include "../convolution_direct_special_cases.h"

namespace megdnn {{
namespace x86 {{
namespace detail {{

void {funcname}(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w)
{{
    (void)src_h;
    const size_t dst_h_beg = 0;
    const size_t dst_h_end = dst_h;
    const size_t dst_w_beg = 0;
    const size_t dst_w_end = dst_w;
""".format(**vars()))

        f.write("""
    size_t dh = dst_h_beg;
    for (; dh + {MAX_H} <= dst_h_end; dh += {MAX_H}) {{
        SIMD_H{MAX_H};
    }}
    switch (dst_h_end - dh) {{
""".format(**vars()))
        for H in range(1, MAX_H):
            f.write("""        case {H}:
            SIMD_H{H};
            break;
""".format(**vars()))
        f.write("""    }}
}}

}} // namespace detail
}} // namespace x86
}} // namespace megdnn
""".format(**vars()))
        for H in range(1, MAX_H+1):
            f.write("""#undef SIMD_H{H}
""".format(**vars()))

def gen_header(modes, simds, fsizes):
    with open('convolution_direct_special_cases.h', 'w') as f:
        f.write("""#pragma once

#include <cstddef>
#include "megdnn/arch.h"

namespace megdnn {
namespace x86 {
namespace detail {
""")
        for mode, simd, fsize in itertools.product(modes, simds, fsizes):
            funcname = "convolution_{mode}_fh{fsize}_{simd}".format(**vars())
            f.write("""
void {funcname}(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("{simd}");
""".format(**vars()))

        f.write("""} // namespace detail
} // namespace x86
} // namespace megdnn
""")

if __name__ == '__main__':
    for mode in ['xcorr', 'conv']:
        for fsize in range(1, 8):
            for simd in ['sse', 'avx', 'fma']:
                gen(mode, simd, fsize)
    gen_header(['xcorr', 'conv'], ['sse', 'avx', 'fma'], range(1, 8))
