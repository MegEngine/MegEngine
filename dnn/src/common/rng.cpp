/**
 * \file dnn/src/common/rng.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn {

void PermutationRNG::check_exec(
        const TensorLayout &dst, size_t workspace_in_bytes) {
    megdnn_assert((dst.dtype == dtype::Float32() || 
                   dst.dtype == dtype::Int32()   ||
                   dst.dtype == dtype::Int16() ) &&
                   dst.dtype.enumv() == param().dtype &&
                   dst.is_contiguous());
    megdnn_assert(workspace_in_bytes >= get_workspace_in_bytes(dst));
}

void PoissonRNG::check_exec(const TensorLayout &lam, const TensorLayout &dst, 
                    size_t workspace_in_bytes){
    megdnn_assert( dst.dtype.category() == DTypeCategory::FLOAT &&
                   lam.dtype == dst.dtype);
    megdnn_assert(dst.is_contiguous() && lam.is_contiguous());
    megdnn_assert(lam.total_nr_elems() == dst.total_nr_elems());
    megdnn_assert(workspace_in_bytes >= get_workspace_in_bytes(lam, dst));
}

void GammaRNG::check_exec(const TensorLayout &shape,const TensorLayout &scale, 
                        const TensorLayout &dst, size_t workspace_in_bytes){
    megdnn_assert(dst.dtype.category() == DTypeCategory::FLOAT &&
                  shape.dtype == dst.dtype &&
                  scale.dtype == dst.dtype);
    megdnn_assert(shape.is_contiguous() && scale.is_contiguous() 
                  && dst.is_contiguous());
    megdnn_assert(shape.total_nr_elems() == dst.total_nr_elems() &&
                  scale.total_nr_elems() ==  dst.total_nr_elems());
    megdnn_assert(workspace_in_bytes >= get_workspace_in_bytes(shape,scale,dst));
}

void BetaRNG::check_exec(const TensorLayout &alpha,const TensorLayout &beta, 
                        const TensorLayout &dst, size_t workspace_in_bytes){
    megdnn_assert(dst.dtype.category() == DTypeCategory::FLOAT &&
                  alpha.dtype == dst.dtype &&
                  beta.dtype == dst.dtype);
    megdnn_assert(alpha.is_contiguous() && beta.is_contiguous() 
                  && dst.is_contiguous());
    megdnn_assert(alpha.total_nr_elems() == dst.total_nr_elems() &&
                  beta.total_nr_elems() ==  dst.total_nr_elems());
    megdnn_assert(workspace_in_bytes >= get_workspace_in_bytes(alpha,beta, dst));
}

#define INST_CHECK_EXEC(RNG_NAME)                                           \
    void RNG_NAME::check_exec(                                              \
            const TensorLayout &dst, size_t workspace_in_bytes) {           \
        megdnn_assert(dst.dtype.category() == DTypeCategory::FLOAT &&       \
                    dst.dtype.enumv() == param().dtype &&                   \
                    dst.is_contiguous());                                   \
        megdnn_assert(workspace_in_bytes >= get_workspace_in_bytes(dst));   \
    }

INST_CHECK_EXEC(UniformRNG)
INST_CHECK_EXEC(GaussianRNG)
#undef INST_CHECK_EXEC

} // namespace megdnn

// vim: syntax=cpp.doxygen

