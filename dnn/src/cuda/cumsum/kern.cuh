/**
 * \file dnn/src/cuda/cumsum/kern.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/cuda/utils.cuh"

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace megdnn {
namespace cuda {
namespace cumsum {

//! compute conventional sum of elements
template <typename T>
struct SumOp {
    const T* data;
    typedef SumOp ContigOp;

    SumOp(const T* d) : data(d) {}

    __host__ __device__ static T init() { return T(0); }
    __device__ static T apply(T lhs, T rhs) { return lhs + rhs; }
    __device__ T visit(uint32_t idx) const { return data[idx]; }

    static SumOp make_contig(const T* data) { return SumOp(data); }
};

/*!
 * \brief cumsum kernel launcher; defined in kern_impl.cuinl
 * \tparam T output data type
 * \tparam Op reduction operator class, which must provide following interface:
 *      typdef ContigOp
 *      static T init(): the identity element
 *      static T apply(T lhs, T rhs): the reduction operation
 *      T visit(uint32_t idx) const: access input
 *      static ContigOp make_contig(const T *data): make an Oo to continue
 *          reduction on temp buffer
 *
 * Note that Op::init() must be accessible from both host and device.
 *
 * In exclusive mode, Op::init() would be filled to the boundary
 *
 * The buffer in *op* and *dst* should not have identical memory addresses.
 */
template <typename T, typename Op, bool exclusive, bool reverse>
void run_kern(T* dst, void* workspace, uint32_t workspace_size, uint32_t A,
              uint32_t B, uint32_t C, const Op& op, cudaStream_t stream);

/*!
 * \brief get required workspace size for cumsum, in bytes
 * \param item_size size of item; i.e. sizeof(T) in run_kern
 *
 * Note: cuda device must be set to the computing device before calling this
 * function.
 */
uint32_t get_workspace_in_bytes(uint32_t A, uint32_t B, uint32_t C,
                                uint32_t item_size);

}  // namespace cumsum
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
