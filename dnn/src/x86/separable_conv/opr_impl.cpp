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
 * \file dnn/src/x86/separable_conv/opr_impl.cpp
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

#include "src/x86/separable_conv/opr_impl.h"
#include "./sep_conv_filter.h"
#include "src/common/utils.h"
#include "src/x86/utils.h"
#include "src/x86/handle.h"
#include <cstring>

namespace megdnn {
namespace x86 {
using namespace sep_conv;

void SeparableConvImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in filter_x,
        _megdnn_tensor_in filter_y,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, filter_x.layout, filter_y.layout, dst.layout, workspace.size);
    int ih = src.layout.shape[2];
    int iw = src.layout.shape[3];
    int oh = dst.layout.shape[2];
    int ow = dst.layout.shape[3];

	filter_engine_ = new FilterEngine(ih, iw, oh, ow,
                     param().ksize_h,  param().ksize_w,
                     param().anchor_h, param().anchor_w,
                     param().borderMode, param().is_symm_kernel);

	MEGDNN_DISPATCH_CPU_KERN_OPR(
  		filter_engine_->exec(src, filter_x, filter_y, dst);
  	);

	delete(filter_engine_);

}

} // namespace x86
} // namespace megdnn

// vim: syntax=cpp.doxygen
