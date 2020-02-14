/**
 * \file dnn/src/cuda/cond_take/kern.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/dtype.h"
#include "src/common/cond_take/predicate.cuh"
#include <cuda_runtime_api.h>

namespace megdnn {
namespace cuda {
namespace cond_take {

typedef dt_int32 IdxType;

/*!
 * \brief generate indices to take according to mask
 * \param dest_idx output index, must be size+1 long
 * \param size number of elements in mask
 * \return output size; i.e. number of elements taken
 */
template<typename T>
size_t gen_idx(
        void *workspace, size_t workspace_size,
        IdxType *dest_idx, const T *mask, size_t size,
        uint32_t mode, const megdnn::cond_take::KParam &kparam,
        cudaStream_t stream);

//! get workspace size in bytes for gen_idx()
size_t gen_idx_get_workspace_size(size_t size);

/*!
 * \brief copy to final output
 * \param[out] dest_data data output, size is returned by gen_idx()
 * \param[out] dest_idx index output, size is returned by gen_idx()
 * \param src_data data input
 * \param src_idx index input, must have been filled by gen_idx()
 * \param size size of original mask
 */
template<typename T>
void copy_output(T *dest_data, IdxType *dest_idx,
        const T *src_data, IdxType *src_idx, uint32_t size,
        cudaStream_t stream);

} // namespace cond_take
} // namespace cuda
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
