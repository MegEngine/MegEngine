/**
 * \file src/opr/impl/blas.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/blas.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace serialization {

template <>
struct OprMaker<opr::SVD, 1> {
    using Param = opr::SVD::Param;
    static cg::OperatorNodeBase* make(const Param& param,
                                      const cg::VarNodeArray& i,
                                      ComputingGraph& graph,
                                      const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        return opr::SVD::make(i[0], param, config)[0].node()->owner_opr();
    }
};

}  // namespace serialization

namespace opr {

using MatrixMulV2 = MatrixMul;
using BatchedMatrixMulV2 = BatchedMatrixMul;
MGB_SEREG_OPR(MatrixMulV2, 2);
MGB_SEREG_OPR(BatchedMatrixMulV2, 2);
MGB_SEREG_OPR(Dot, 2);
MGB_SEREG_OPR(MatrixInverse, 1);
MGB_SEREG_OPR(SVD, 1);

}  // namespace opr


}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
