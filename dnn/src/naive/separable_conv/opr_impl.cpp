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
 * \file dnn/src/naive/separable_conv/opr_impl.cpp
 *
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2014-2021 Megvii Inc. All rights reserved.
 *
 * ---------------------------------------------------------------------------
 */

#include "src/naive/separable_conv/opr_impl.h"
#include <cstring>
#include "src/common/utils.h"
#include "src/naive/handle.h"

namespace megdnn {
namespace naive {
// using namespace sep_conv;

size_t SeparableConvForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& filter_x,
        const TensorLayout& filter_y, const TensorLayout& dst) {
    MEGDNN_MARK_USED_VAR(src);
    MEGDNN_MARK_USED_VAR(filter_y);
    MEGDNN_MARK_USED_VAR(dst);
    auto kw = filter_x.shape[3];
    auto kh = kw;
    auto ic = filter_x.shape[1];
    auto oc = filter_x.shape[0];
    TensorLayout layout({oc, ic, kh, kw}, dtype::Float32());
    return oc * ic * kh * kw * sizeof(float);
}

void SeparableConvForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in filter_x, _megdnn_tensor_in filter_y,
        _megdnn_tensor_in dst, _megdnn_workspace workspace) {
    check_exec(
            src.layout, filter_x.layout, filter_y.layout, dst.layout, workspace.size);

    auto kw = filter_x.layout.shape[3];
    auto kh = kw;
    auto ic = filter_x.layout.shape[1];
    auto oc = filter_x.layout.shape[0];

    auto transform_filter_2d = [=]() {
        auto kerx = static_cast<float*>(filter_x.raw_ptr());
        auto kery = static_cast<float*>(filter_y.raw_ptr());
        auto filter2d_ptr = workspace.ptr<float>();
        // Generate 2D-filter
        size_t k_pos = 0;
        for (size_t cn = 0; cn < ic * oc; ++cn) {
            for (size_t h = 0; h < kh; ++h) {
                for (size_t w = 0; w < kw; ++w) {
                    filter2d_ptr[k_pos++] = kerx[w] * kery[h];
                }
            }
            kerx += kw;
            kery += kw;
        }
    };

    MEGDNN_DISPATCH_CPU_KERN_OPR(transform_filter_2d());

    //! construct filter 2D tensor
    TensorLayout layout({oc, ic, kh, kw}, dtype::Float32());
    TensorND filter2d(workspace.raw_ptr, layout);

    auto conv_opr = handle()->create_operator<ConvolutionForward>();
    Workspace empty_wsp;
    conv_opr->exec(src, filter2d, dst, nullptr, empty_wsp);
}

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
