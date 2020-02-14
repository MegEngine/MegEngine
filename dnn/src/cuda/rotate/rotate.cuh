/**
 * \file dnn/src/cuda/rotate/rotate.cuh
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
#include <cstddef>

namespace megdnn {
namespace cuda {
namespace rotate {

template <typename T, bool clockwise>
void rotate(const T* src, T* dst, size_t N, size_t IH, size_t IW,
            size_t CH, size_t istride0, size_t istride1, size_t istride2,
            size_t OH, size_t OW, size_t ostride0, size_t ostride1,
            size_t ostride2, cudaStream_t stream);

} // namespace rotate
} // namespace cuda
} // namespace cuda

// vim: syntax=cpp.doxygen
