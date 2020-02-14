/**
 * \file dnn/src/cuda/concat/concat.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <stdint.h>

namespace megdnn {
namespace cuda {
namespace concat {

template <typename T>
void forward_proxy(const T **srcs,
        T *dst,
        size_t nr_srcs,
        size_t A, size_t B, size_t C,
        const size_t *Bv,
        const size_t *table_outer,
        const size_t *table_inner,
        cudaStream_t stream);

} // namespace concat
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen

