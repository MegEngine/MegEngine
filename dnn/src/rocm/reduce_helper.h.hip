/**
 * \file src/rocm/reduce_helper.h.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \brief helper for implementing reduce operators
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#pragma once

#include "hip_header.h"
#include "src/rocm/utils.h.hip"

namespace megdnn {
namespace rocm {

/*!
 * \brief run reduce for custom op on (A, B, C) tensor and reduce on the B axis
 * \tparam PublicOperator
 *      must have typedef for wtype (workspace type)
 *      must have const member wtype INIT (the initial value for reduction)
 *      must have method wtype read(uint32_t idx) (load and cast to workspace
 * type) must have method wtype apply(wtype, wtype) (apply reduction) must have
 * method void write(uint32_t idx, wtype) (write back) \tparam
 * sync_within_warp always do a __syncthreads(), even when the reduction falls
 * in a warp. Turn on this to make argmxx work.
 */
template <class PublicOperator, bool sync_within_warp>
void run_reduce(typename PublicOperator::wtype* workspace, size_t A, size_t B,
                size_t C, hipStream_t stream, const PublicOperator& opr);
template <typename wtype>
size_t get_reduce_workspace_in_bytes(size_t A, size_t B, size_t C);

#define INST_REDUCE(Op, sync_within_warp)                                   \
    template void run_reduce<Op, sync_within_warp>(typename Op::wtype*,     \
                                                   size_t, size_t, size_t,  \
                                                   hipStream_t, const Op&); \
    template size_t get_reduce_workspace_in_bytes<Op>(size_t, size_t, size_t)

}  // namespace rocm
}  // namespace megdnn

#include "src/rocm/reduce_helper.hipinl"

// vim: ft=cpp syntax=cpp.doxygen
