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
 * \file dnn/src/common/cv/interp_helper.h
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

#pragma once

#include "src/common/cv/aligned_allocator.h"

#include "megdnn/opr_param_defs.h"

#include <cstdint>
#include <memory>
#include <mutex>

namespace megdnn {
namespace megcv {

using InterpolationMode = megdnn::param::WarpPerspective::InterpolationMode;
using BorderMode = megdnn::param::WarpPerspective::BorderMode;

/*!
 * \brief helper for generating interpolation tables for different interpolation
 *        modes
 */
template <int INTER_BITS_ = 5, int INTER_MAX_ = 7,
          int INTER_REMAP_COEF_BITS_ = 15>
class InterpolationTable {
public:
    using IMode = InterpolationMode;

    static constexpr int INTER_BITS = INTER_BITS_;
    static constexpr int INTER_MAX = INTER_MAX_;
    static constexpr int INTER_REMAP_COEF_BITS = INTER_REMAP_COEF_BITS_;
    static constexpr int INTER_TAB_SIZE = (1 << INTER_BITS);
    static constexpr int INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE;
    static constexpr int INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS;

    /*!
     * \brief get interpolation table
     *
     * The table dimension is [INTER_TAB_SIZE][INTER_TAB_SIZE][ksize][ksize]
     *
     * \param imode interpolation mode
     * \param fixpt if this is true, return a table for int16_t; else return a
     *              table for float
     * \return table for int16 or float according to fixpt
     */
    static const void* get_table(InterpolationMode imode, bool fixpt);
#if MEGDNN_X86
    /**
     * \brief get interpolation table for linear mode.
     *
     * This current only avaiable in \warning X86.
     *
     * \return bilineartab_ic4_buf
     */
    static const int16_t* get_linear_ic4_table();
#endif

private:
    template <int ksize>
    struct Table {
        float ftab[INTER_TAB_SIZE2 * ksize * ksize];
        int16_t itab[INTER_TAB_SIZE2 * ksize * ksize];
#if MEGDNN_X86
        alignas(128) int16_t bilineartab_ic4_buf[INTER_TAB_SIZE2 * 2 * 8];

        static void* operator new(std::size_t sz) {
            return ah::aligned_allocator<Table, 128>().allocate(sz /
                                                                sizeof(Table));
        }
        void operator delete(void* ptr) noexcept {
            ah::aligned_allocator<Table, 128>().deallocate(
                    reinterpret_cast<Table*>(ptr), 0);
        }
#endif
    };

    struct TableHolderBase {
        std::mutex mtx;

        //! get table pointer; return whether already init
        virtual bool get(float**, int16_t**) = 0;

    protected:
        ~TableHolderBase() = default;
    };

    template <int ksize>
    struct TableHolder final : public TableHolderBase {
        std::unique_ptr<Table<ksize>> table;

        bool get(float** ftab, int16_t** itab) override {
            bool ret = true;
            if (!table) {
                ret = false;
                table.reset(new Table<ksize>);
            }
            *ftab = table->ftab;
            *itab = table->itab;
            return ret;
        }
    };

    static void init_inter_tab_1d(InterpolationMode imode, float* tab,
                                  int tabsz);

    static inline void interpolate_linear(float x, float* coeffs);
    static inline void interpolate_cubic(float x, float* coeffs);
    static inline void interpolate_lanczos4(float x, float* coeffs);

    static TableHolder<2> sm_tab_linear;
    static TableHolder<4> sm_tab_cubic;
    static TableHolder<8> sm_tab_lanczos4;
};

}  // namespace megcv
}  // namespace megdnn

// vim: syntax=cpp.doxygen
