/**
 * \file dnn/src/common/conv_bias.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/handle.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs/general.h"
#include "megdnn/oprs/nn_int.h"
#include "src/common/utils.h"

namespace megdnn {

void handle_bias_and_nonlinear(Handle* handle, param::ConvBias args,
                               const TensorND* conv_dst_tensor,
                               const TensorND* dst_tensor,
                               const TensorND* bias_tensor);
}  // namespace megdnn

// vim: syntax=cpp.doxygen
