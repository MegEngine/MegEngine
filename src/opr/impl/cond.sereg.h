/**
 * \file src/opr/impl/cond.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/cond.h"
#include "megbrain/opr/utility.h"
#include "megbrain/serialization/sereg.h"

#if MGB_ENABLE_COND_EXEC

namespace mgb {

namespace opr {
//! an empty class to be registered as a python opr
class CondExecMarkIfNeed final : public DynTypeObj {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

public:
    using Param = CondExecMark::Param;
    static Param param() { mgb_trap(); }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(CondExecMarkIfNeed);
}  // namespace opr

namespace serialization {
template <>
struct OprMaker<opr::CondExecPred, 0> {
    using Param = opr::CondExecPred::Param;
    static cg::OperatorNodeBase* make(const Param& param,
                                      const cg::VarNodeArray& inputs,
                                      ComputingGraph& graph,
                                      const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        return opr::CondExecPred::make_opr(
                inputs.back(),
                {inputs.data(), inputs.data() + inputs.size() - 1}, param,
                config);
    }
};

template <>
struct OprMaker<opr::CondExecPredLogical, 0> {
    using Param = opr::CondExecPredLogical::Param;
    static cg::OperatorNodeBase* make(const Param& param,
                                      const cg::VarNodeArray& inputs,
                                      ComputingGraph& graph,
                                      const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        return opr::CondExecPredLogical::make(inputs, param, config)
                .node()
                ->owner_opr();
    }
};

template <>
struct OprMaker<opr::CondExecMark, 0> {
    using Param = opr::CondExecMark::Param;
    static cg::OperatorNodeBase* make(const Param& param,
                                      const cg::VarNodeArray& inputs,
                                      ComputingGraph& graph,
                                      const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        return opr::CondExecMark::make_opr(
                inputs.back(),
                {inputs.data(), inputs.data() + inputs.size() - 1}, param,
                config);
    }
};

template <>
struct OprMaker<opr::CondExecMarkIfNeed, 0> {
    using Param = opr::CondExecMarkIfNeed::Param;
    static cg::OperatorNodeBase* make(const Param& param,
                                      const cg::VarNodeArray& inputs,
                                      ComputingGraph& graph,
                                      const OperatorNodeConfig& config) {
        mgb_assert(inputs.size() == 2);
        auto out = opr::CondExecMark::mark_if_need(inputs[0], inputs[1], param,
                                                   config)
                           .node();
        if (out->owner_opr()->output().size() != 1) {
            out = opr::Identity::make(out).node();
        }
        return out->owner_opr();
    }
};

template <>
struct OprMaker<opr::CondExecMerge, 0> {
    using Param = opr::CondExecMerge::Param;
    static cg::OperatorNodeBase* make(const Param& param,
                                      const cg::VarNodeArray& inputs,
                                      ComputingGraph& graph,
                                      const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        size_t nr_val_inp = inputs.size();
        Maybe<VarNodeArrayView> out_shapes_storage;
        VarNodeArrayView* out_shapes = nullptr;
        if (param.mode == Param::Mode::SUM_COND_OUT) {
            --nr_val_inp;
        } else if (param.mode == Param::Mode::SUM) {
            nr_val_inp -= param.nr_output;
            out_shapes = &out_shapes_storage.emplace(
                    inputs.data() + nr_val_inp, inputs.data() + inputs.size());
        }
        if (!out_shapes) {
            out_shapes = &out_shapes_storage.emplace();
        }
        return opr::CondExecMerge::make_opr(
                {inputs.data(), inputs.data() + nr_val_inp}, *out_shapes, param,
                config);
    }
};
}  // namespace serialization

namespace opr {
MGB_SEREG_OPR(CondExecPred, 0);
MGB_SEREG_OPR(CondExecPredLogical, 0);
MGB_SEREG_OPR(CondExecMark, 0);
MGB_SEREG_OPR(CondExecMarkIfNeed, 0);
MGB_SEREG_OPR(CondExecMerge, 0);
}  // namespace opr

}  // namespace mgb

#endif  // MGB_ENABLE_COND_EXEC

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
