/**
 * \file dnn/src/opencl/cuda/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/remap/opr_impl.h"
#include "megdnn/config/config.h"
#include "src/common/utils.h"

using namespace megdnn;
using namespace cuda;

void RemapImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out map_xy,
                     _megdnn_tensor_in dst, _megdnn_workspace workspace) {
    check_exec(src.layout, map_xy.layout, dst.layout, workspace.size);
    megdnn_throw("megdnn currently do not support remap in cuda");
}

// vim: syntax=cpp.doxygen
