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
 * \file dnn/src/common/cv/interp_helper.cpp
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

#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
// TableHolderBase has no problem; ignore the warning for old clang versions

#include "./helper.h"
#include "./interp_helper.h"

#include "src/common/utils.h"

using namespace megdnn;
using namespace megdnn::megcv;

static constexpr double MEGCV_PI_4 = 0.78539816339744830962; /* pi/4 */

#define DEF_FUN(_ret)                                                      \
    template <int INTER_BITS_, int INTER_MAX_, int INTER_REMAP_COEF_BITS_> \
    _ret InterpolationTable<INTER_BITS_, INTER_MAX_, INTER_REMAP_COEF_BITS_>::

#define DEF_TABLE_HOLDER(_name, _ksize)                                    \
    template <int INTER_BITS_, int INTER_MAX_, int INTER_REMAP_COEF_BITS_> \
    typename InterpolationTable<                                           \
            INTER_BITS_, INTER_MAX_,                                       \
            INTER_REMAP_COEF_BITS_>::template TableHolder<_ksize>          \
            InterpolationTable<INTER_BITS_, INTER_MAX_,                    \
                               INTER_REMAP_COEF_BITS_>::_name

DEF_TABLE_HOLDER(sm_tab_linear, 2);
DEF_TABLE_HOLDER(sm_tab_cubic, 4);
DEF_TABLE_HOLDER(sm_tab_lanczos4, 8);

DEF_FUN(void) interpolate_linear(float x, float* coeffs) {
    coeffs[0] = 1.f - x;
    coeffs[1] = x;
}

DEF_FUN(void) interpolate_cubic(float x, float* coeffs) {
    const float A = -0.75f;
    coeffs[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    coeffs[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    coeffs[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

DEF_FUN(void) interpolate_lanczos4(float x, float* coeffs) {
    static const double s45 = 0.70710678118654752440084436210485;
    static const double cs[][2] = {{1, 0},  {-s45, -s45}, {0, 1},  {s45, -s45},
                                   {-1, 0}, {s45, s45},   {0, -1}, {-s45, s45}};
    if (x < FLT_EPSILON) {
        for (int i = 0; i < 8; i++)
            coeffs[i] = 0;
        coeffs[3] = 1;
        return;
    }
    float sum = 0;
    double y0 = -(x + 3) * MEGCV_PI_4, s0 = sin(y0), c0 = cos(y0);
    for (int i = 0; i < 8; i++) {
        double y = -(x + 3 - i) * MEGCV_PI_4;
        coeffs[i] = (float)((cs[i][0] * s0 + cs[i][1] * c0) / (y * y));
        sum += coeffs[i];
    }
    sum = 1.f / sum;
    for (int i = 0; i < 8; i++)
        coeffs[i] *= sum;
}

DEF_FUN(void)
init_inter_tab_1d(InterpolationMode imode, float* tab, int tabsz) {
    float scale = 1.f / tabsz;
    switch (imode) {
        case IMode::INTER_LINEAR:
            for (int i = 0; i < tabsz; ++i, tab += 2)
                interpolate_linear(i * scale, tab);
            break;
        case IMode::INTER_CUBIC:
            for (int i = 0; i < tabsz; ++i, tab += 4)
                interpolate_cubic(i * scale, tab);
            break;
        case IMode::INTER_LANCZOS4:
            for (int i = 0; i < tabsz; ++i, tab += 8)
                interpolate_lanczos4(i * scale, tab);
            break;
        default:
            megdnn_throw("unsupported interpolation mode");
    }
}

#if MEGDNN_X86
DEF_FUN(const int16_t*) get_linear_ic4_table() {
    auto table_holder = &sm_tab_linear;
    std::lock_guard<std::mutex> lg{table_holder->mtx};
    float* tab = nullptr;
    short* itab = nullptr;
    MEGDNN_MARK_USED_VAR(tab);
    MEGDNN_MARK_USED_VAR(itab);
    megdnn_assert(table_holder->get(&tab, &itab),
                  "invoke get_table before get_linear_ic4_table");
    return table_holder->table->bilineartab_ic4_buf;
}
#endif

DEF_FUN(const void*) get_table(InterpolationMode imode, bool fixpt) {
    TableHolderBase* table_holder = nullptr;
    int ksize = 0;
    switch (imode) {
        case IMode::INTER_LINEAR:
            table_holder = &sm_tab_linear;
            ksize = 2;
            break;
        case IMode::INTER_CUBIC:
            table_holder = &sm_tab_cubic;
            ksize = 4;
            break;
        case IMode::INTER_LANCZOS4:
            table_holder = &sm_tab_lanczos4;
            ksize = 8;
            break;
        default:
            megdnn_throw(("unsupported interpolation mode"));
    }
    std::lock_guard<std::mutex> lg{table_holder->mtx};

    float* tab = nullptr;
    short* itab = nullptr;
    if (!table_holder->get(&tab, &itab)) {
        float _tab[8 * INTER_TAB_SIZE];
        int i, j, k1, k2;
        init_inter_tab_1d(imode, _tab, INTER_TAB_SIZE);
        for (i = 0; i < INTER_TAB_SIZE; ++i) {
            for (j = 0; j < INTER_TAB_SIZE;
                 ++j, tab += ksize * ksize, itab += ksize * ksize) {
                int isum = 0;
                for (k1 = 0; k1 < ksize; ++k1) {
                    float vy = _tab[i * ksize + k1];
                    for (k2 = 0; k2 < ksize; ++k2) {
                        float v = vy * _tab[j * ksize + k2];
                        tab[k1 * ksize + k2] = v;
                        isum += itab[k1 * ksize + k2] = saturate_cast<short>(
                                v * INTER_REMAP_COEF_SCALE);
                    }
                }
                if (isum != INTER_REMAP_COEF_SCALE) {
                    int diff = isum - INTER_REMAP_COEF_SCALE;
                    int ksize2 = ksize / 2, Mk1 = ksize2, Mk2 = ksize2;
                    int mk1 = ksize2, mk2 = ksize2;
                    for (k1 = ksize2; k1 < ksize2 + 2; ++k1)
                        for (k2 = ksize2; k2 < ksize2 + 2; ++k2) {
                            if (itab[k1 * ksize + k2] <
                                itab[mk1 * ksize + mk2]) {
                                mk1 = k1;
                                mk2 = k2;
                            } else if (itab[k1 * ksize + k2] >
                                       itab[Mk1 * ksize + Mk2]) {
                                Mk1 = k1;
                                Mk2 = k2;
                            }
                        }
                    if (diff < 0)
                        itab[Mk1 * ksize + Mk2] =
                                (short)(itab[Mk1 * ksize + Mk2] - diff);
                    else
                        itab[mk1 * ksize + mk2] =
                                (short)(itab[mk1 * ksize + mk2] - diff);
                }
            }
        }
        tab -= INTER_TAB_SIZE2 * ksize * ksize;
        itab -= INTER_TAB_SIZE2 * ksize * ksize;

#if MEGDNN_X86
        if (imode == IMode::INTER_LINEAR) {
            int16_t* bilineartab_ic4_buf =
                    sm_tab_linear.table->bilineartab_ic4_buf;
            for (i = 0; i < INTER_TAB_SIZE2; i++)
                for (j = 0; j < 4; j++) {
                    bilineartab_ic4_buf[i * 2 * 8 + 0 * 8 + j * 2] =
                            itab[i * ksize * ksize + 0 * ksize + 0];
                    bilineartab_ic4_buf[i * 2 * 8 + 0 * 8 + j * 2 + 1] =
                            itab[i * ksize * ksize + 0 * ksize + 1];
                    bilineartab_ic4_buf[i * 2 * 8 + 1 * 8 + j * 2] =
                            itab[i * ksize * ksize + 1 * ksize + 0];
                    bilineartab_ic4_buf[i * 2 * 8 + 1 * 8 + j * 2 + 1] =
                            itab[i * ksize * ksize + 1 * ksize + 1];
                }
        }
#endif
    }
    return fixpt ? static_cast<void*>(itab) : static_cast<void*>(tab);
}

namespace megdnn {
namespace megcv {

// explicit inst
template class InterpolationTable<5, 7, 15>;

}  // namespace megcv
}  // namespace megdnn

// vim: syntax=cpp.doxygen
