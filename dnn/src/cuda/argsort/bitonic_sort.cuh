/**
 * \file dnn/src/cuda/argsort/bitonic_sort.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace megdnn {
namespace cuda {

const uint32_t BITONIC_SORT_MAX_LENGTH = 2048;
// cub radix sort seems to be faster with lengths > 2048

/*!
 * \brief bitonic sort for k/v pairs
 *
 * Requires \p length no larger than 4 times of cuda thread num. \p key_inp
 * and \p key_out can be identical, and so are \p value_inp and \p value_out.
 */
template <typename Key, typename Value>
cudaError_t bitonic_sort(uint32_t batch, uint32_t length, const Key* key_inp,
                         const Value* value_inp, Key* key_out, Value* value_out,
                         bool ascending, cudaStream_t stream);

}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen

