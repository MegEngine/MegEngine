/**
 * \file dnn/src/arm_common/separable_conv/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/arm_common/separable_conv/opr_impl.h"
#include "./sep_conv_filter.h"
#include "src/common/utils.h"
//#include "src/arm_common/profile.h"
#include "src/arm_common/handle.h"
#include <cstring>

namespace megdnn {
namespace arm_common {
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

} // namespace arm_common
} // namespace megdnn

// vim: syntax=cpp.doxygen
