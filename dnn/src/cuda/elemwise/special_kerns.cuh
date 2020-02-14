/**
 * \file dnn/src/cuda/elemwise/special_kerns.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/cuda/elemwise_helper.cuh"

namespace megdnn {
namespace cuda {

    template<bool c_is_scalar, typename ctype>
    void kern_fuse_mul_add3(ctype *dest,
            const ElemwiseOpParamN<3> &param, cudaStream_t stream);

    template<typename ctype>
    void kern_fuse_mul_add4(ctype *dest,
            const ElemwiseOpParamN<4> &param, cudaStream_t stream);

} // namespace cuda
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen

