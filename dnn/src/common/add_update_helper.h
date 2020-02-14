/**
 * \file dnn/src/common/add_update_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs.h"

#include "src/common/elemwise_helper.cuh"

namespace megdnn {

class AddUpdateForwardHelper : public AddUpdateForward {
    using AddUpdateForward::AddUpdateForward;

protected:
    ElemwiseOpParamN<2> make_param(_megdnn_tensor_inout dst,
                                   _megdnn_tensor_in delta);
};

}  // namespace megdnn

// vim: syntax=cpp.doxygen
