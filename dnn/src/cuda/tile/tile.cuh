/**
 * \file dnn/src/cuda/tile/tile.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

namespace megdnn {
namespace cuda {
namespace tile {

template <typename T>
void forward_proxy(const T *src, T *dst, size_t ndim,
        const size_t *sshape, const size_t *dshape, const size_t *tshape,
        cudaStream_t stream);

} // namespace tile
} // namespace cuda
} // namespace megdnn
// vim: syntax=cpp.doxygen

