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
 * \file dnn/src/naive/separable_filter/opr_impl.cpp
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
#include "src/naive/separable_filter/opr_impl.h"
#include "src/common/cv/helper.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"
#include <cstring>

namespace megdnn {
namespace naive {
using namespace megcv;

template <typename T, param::WarpPerspective::BorderMode bmode>
struct remap_func_holder {
    static void border_interpolate_exec(_megdnn_tensor_in src,
                                        _megdnn_tensor_in kx,
                                        _megdnn_tensor_in ky,
                                        _megdnn_tensor_out dst) {
        auto N = src.layout.shape[0], IH = src.layout.shape[1],
             IW = src.layout.shape[2], IC = src.layout.shape[3];

        uint32_t kernel_height = ky.layout.shape[3];
        uint32_t kernel_width = kx.layout.shape[3];
        uint32_t half_h = kernel_height / 2;
        uint32_t half_w = kernel_width / 2;

        rep(n, N) rep(h, IH) rep(w, IW) rep(c, IC) {
            double val = 0;
            rep(iy, kernel_height) {
                int y = border_interpolate<bmode>(h + iy - half_h, IH);
                rep(ix, kernel_width) {
                    int x = border_interpolate<bmode>(w + ix - half_w, IW);
                    if (x != -1 && y != -1) {
                        val += kx.ptr<T>()[ix] * ky.ptr<T>()[iy] *
                               src.ptr<T>()[n * src.layout.stride[0] +
                                            y * src.layout.stride[1] +
                                            x * src.layout.stride[2] +
                                            c * src.layout.stride[3]];
                    }
                }
            }
            dst.ptr<T>()[n * dst.layout.stride[0] + h * dst.layout.stride[1] +
                         w * dst.layout.stride[2] + c * dst.layout.stride[3]] =
                    static_cast<T>(val);
        }
    }
};

template <param::WarpPerspective::BorderMode bmode>
struct remap_func_holder<uint8_t, bmode> {
    static void border_interpolate_exec(_megdnn_tensor_in src,
                                        _megdnn_tensor_in filter_x,
                                        _megdnn_tensor_in filter_y,
                                        _megdnn_tensor_out dst) {
        auto N = src.layout.shape[0], IH = src.layout.shape[1],
             IW = src.layout.shape[2], IC = src.layout.shape[3];

        using namespace megcv;

        Mat<float> kx_(1, filter_x.layout.shape[3], 1,
                         static_cast<float*>(filter_x.raw_ptr));
        Mat<float> ky_(1, filter_y.layout.shape[3], 1,
                         static_cast<float*>(filter_y.raw_ptr));

        uint32_t kernel_height = ky_.width();
        uint32_t kernel_width = kx_.width();
        Mat<int> kx(1, kernel_width, 1);
        Mat<int> ky(1, kernel_height, 1);
        const uint8_t bits = 8;
        for (size_t i = 0; i < kernel_height; i++) {
            ky.at(0, i, 0) = static_cast<int>(ky_.at(0, i, 0) * (1 << bits));
        }
        for (size_t i = 0; i < kernel_width; i++) {
            kx.at(0, i, 0) = static_cast<int>(kx_.at(0, i, 0) * (1 << bits));
        }

        FixedPtCastEx<int, uint8_t> cast_op(2 * bits);
        rep(n, N) rep(h, IH) rep(w, IW) rep(c, IC) {
            int val = 0;
            rep(iy, kernel_height) {
                int y = border_interpolate<bmode>(h + iy - kernel_height / 2,
                                                  IH);
                rep(ix, kernel_width) {
                    int x = border_interpolate<bmode>(w + ix - kernel_width / 2,
                                                      IW);

                    //! BORDER_CONSTANT or BORDER_TRANSPARENT
                    if (x != -1 && y != -1) {
                        val += kx.at(0, ix, 0) * ky.at(0, iy, 0) *
                               src.ptr<uint8_t>()[n * src.layout.stride[0] +
                                                  y * src.layout.stride[1] +
                                                  x * src.layout.stride[2] +
                                                  c * src.layout.stride[3]];
                    }
                }
            }
            dst.ptr<uint8_t>()[n * dst.layout.stride[0] +
                               h * dst.layout.stride[1] +
                               w * dst.layout.stride[2] +
                               c * dst.layout.stride[3]] = cast_op(val);
        }
    }
};

template <typename T>
void SeparableFilterForwardImpl::exec_internal(_megdnn_tensor_in src,
                                               _megdnn_tensor_in kx,
                                               _megdnn_tensor_in ky,
                                               _megdnn_tensor_out dst) {
    switch (param().borderMode) {
#define cb(bmode)                                                             \
    case param::WarpPerspective::BorderMode::bmode:                           \
        return remap_func_holder<T,                                           \
                                 param::WarpPerspective::BorderMode::bmode>:: \
                border_interpolate_exec(src, kx, ky, dst);
        cb(BORDER_REPLICATE);
        cb(BORDER_REFLECT);
        cb(BORDER_REFLECT_101);
        cb(BORDER_WRAP);
        cb(BORDER_CONSTANT);
        cb(BORDER_TRANSPARENT);
        cb(BORDER_ISOLATED);
#undef cb
        default:
            megdnn_throw(megdnn_mangle("Unexpected border mode"));
    }
}

void SeparableFilterForwardImpl::exec(_megdnn_tensor_in src,
                                      _megdnn_tensor_in filter_x,
                                      _megdnn_tensor_in filter_y,
                                      _megdnn_tensor_in dst,
                                      _megdnn_workspace workspace) {
    check_exec(src.layout, filter_x.layout, filter_y.layout, dst.layout,
               workspace.size);

#define cb(dt)                                                       \
    if (src.layout.dtype == dt()) {                                  \
        using ctype = typename DTypeTrait<dt>::ctype;                \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                \
                exec_internal<ctype>(src, filter_x, filter_y, dst)); \
        return;                                                      \
    }
    cb(dtype::Uint8);
    cb(dtype::Float32);
#undef cb
    megdnn_assert_internal(0);
}

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen
