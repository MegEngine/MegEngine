/**
 * \file dnn/src/cuda/reduce_helper.cuh
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

namespace megdnn {
namespace cuda {

/*!
 * \brief run reduce for custom op on (A, B, C) tensor and reduce on the B axis
 * \tparam PublicOperator
 *      must have typedef for wtype (workspace type)
 *      must have const member wtype INIT (the initial value for reduction)
 *      must have method wtype read(uint32_t idx) (load and cast to workspace type)
 *      must have method wtype apply(wtype, wtype) (apply reduction)
 *      must have method void write(uint32_t idx, wtype) (write back)
 *  \tparam sync_within_warp always do a __syncthreads(), even when the reduction falls in a warp. Turn on this to make argmxx work.
 */
template <class PublicOperator, bool sync_within_warp>
void run_reduce(typename PublicOperator::wtype *workspace,
        size_t A, size_t B, size_t C,
        cudaStream_t stream, const PublicOperator &opr);
template <typename wtype>
size_t get_reduce_workspace_in_bytes(size_t A, size_t B, size_t C);

#define INST_REDUCE(Op, sync_within_warp) \
template void run_reduce<Op, sync_within_warp>( \
        typename Op::wtype *, size_t, size_t, size_t, \
        cudaStream_t, const Op &); \
template size_t get_reduce_workspace_in_bytes<Op>(size_t, size_t, size_t)

} // namespace cuda
} // namespace megdnn

#include "src/cuda/reduce_helper.cuinl"

// vim: ft=cpp syntax=cpp.doxygen
