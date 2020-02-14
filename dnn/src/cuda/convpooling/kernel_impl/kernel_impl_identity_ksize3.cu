/**
 * \file dnn/src/cuda/convpooling/kernel_impl/kernel_impl_identity_ksize3.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./kernel_impl.h"
#include "../conv_pooling_utils.cuh"

namespace megdnn {
namespace cuda {
namespace conv_pool {

DISPATCH_POOLSHAPE(Identity, 3)

} // namespace conv_pool
} // namespace cuda
} // namespace megdnn
// vim: syntax=cpp.doxygen
