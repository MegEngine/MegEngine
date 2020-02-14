/**
 * \file dnn/src/cuda/convpooling/conv_pooling.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cuda_runtime_api.h>
#include "./conv_pooling.h"

namespace megdnn {
namespace cuda {
namespace conv_pool {

template<int kern_h, int kern_w, int pool_shape_h, int pool_shape_w,
            class Nonlin, class Pooler, class IdxGetter>
__global__ void kern_xcorr_smallkern_pool(
        float *input,
        const float *filter,
        float *output,
        const float *output_bias,
        cudaTextureObject_t m_tex,
        int IC, int IH, int IW,
        int OH, int OW);

} // namespace conv_pool
} // namespace cuda
} // namespace megdnn
// vim: syntax=cpp.doxygen
