/**
 * \file src/opr/impl/blas.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/blas.h"
#include "megbrain/opr/param_defs.h"
#include "megbrain/serialization/sereg.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs/linalg.h"

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

template <class MegDNNConv = megdnn::MatrixMul>
struct MakeMatrixMulCaller {
    template <typename Opr>
    static VarNode* make(const cg::VarNodeArray& inputs,
                         const typename MegDNNConv::Param& param,
                         const megdnn::param::ExecutionPolicy& execution_policy,
                         const OperatorNodeConfig& config) {
        if (inputs.size() == 2) {
            return Opr::make(inputs[0], inputs[1], param, execution_policy,
                             config)
                    .node();
        }
        return nullptr;
    }
};

template <class Opr, class Maker, class MegDNNMatrixMul>
struct MatrixMulLoadDumpImpl {
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr_) {
        auto&& opr = opr_.cast_final_safe<Opr>();
        ctx.write_param<megdnn::param::MatrixMul>(opr.param());
    }

    static VarNode* make(const cg::VarNodeArray& inputs,
                         const megdnn::param::MatrixMul& param,
                         const megdnn::param::ExecutionPolicy& execution_policy,
                         const OperatorNodeConfig& config) {
        VarNode* ret = Maker::template make<Opr>(inputs, param,
                                                 execution_policy, config);
        mgb_assert(ret);
        return ret;
    }

    static cg::OperatorNodeBase* load(OprLoadContext& ctx,
                                      const cg::VarNodeArray& inputs,
                                      const OperatorNodeConfig& config) {
        auto param = ctx.read_param<megdnn::param::MatrixMul>();
        return make(inputs, param, {}, config)->owner_opr();
    }
};

template <>
struct OprLoadDumpImpl<opr::MatrixMul, 2>
        : public MatrixMulLoadDumpImpl<opr::MatrixMul,
                                       MakeMatrixMulCaller<megdnn::MatrixMul>,
                                       megdnn::MatrixMul> {};
template <>
struct OprLoadDumpImpl<opr::BatchedMatrixMul, 2>
        : public MatrixMulLoadDumpImpl<
                  opr::BatchedMatrixMul,
                  MakeMatrixMulCaller<megdnn::BatchedMatrixMul>,
                  megdnn::BatchedMatrixMul> {};

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
