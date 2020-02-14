/**
 * \file dnn/src/cuda/max_tensor_diff/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/max_tensor_diff/opr_impl.h"
#include "src/common/utils.h"

using namespace megdnn;
using namespace cuda;

float MaxTensorDiffImpl::exec(_megdnn_tensor_in, _megdnn_tensor_in,
                          _megdnn_workspace) {
    megdnn_throw("MaxTensorDiff not support in cuda");
}

// vim: syntax=cpp.doxygen
