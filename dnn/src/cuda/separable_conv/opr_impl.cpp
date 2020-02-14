/**
 * \file dnn/src/cuda/separable_conv/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/separable_conv/opr_impl.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"
#include <cstring>

namespace megdnn {
namespace cuda {
//using namespace sep_conv;

void SeparableConvForwardImpl::exec(_megdnn_tensor_in src,
                                    _megdnn_tensor_in filter_x,
                                    _megdnn_tensor_in filter_y,
                                    _megdnn_tensor_in dst,
                                    _megdnn_workspace workspace) {
    check_exec(src.layout, filter_x.layout, filter_y.layout, dst.layout,
               workspace.size);
    megdnn_assert(false, "SeparableConv is not supported in CUDA");
}

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
