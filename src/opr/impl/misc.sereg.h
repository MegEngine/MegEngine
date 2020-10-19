/**
 * \file src/opr/impl/misc.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/misc.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {

namespace serialization {

    template<>
    struct OprMaker<opr::Argsort, 1> {
        using Opr = opr::Argsort;
        using Param = Opr::Param;
        static cg::OperatorNodeBase* make(
                const Param &param, const cg::VarNodeArray &inputs,
                ComputingGraph &graph, const OperatorNodeConfig &config) {
            MGB_MARK_USED_VAR(graph);
            auto out = Opr::make(inputs[0], param, config);
            return out[0].node()->owner_opr();
        }
    };

    template<>
    struct OprMaker<opr::CondTake, 2> {
        using Opr = opr::CondTake;
        using Param = Opr::Param;
        static cg::OperatorNodeBase* make(
                const Param &param, const cg::VarNodeArray &inputs,
                ComputingGraph &graph, const OperatorNodeConfig &config) {
            MGB_MARK_USED_VAR(graph);
            auto out = Opr::make(inputs[0], inputs[1], param, config);
            return out[0].node()->owner_opr();
        }
    };

    template<>
    struct OprMaker<opr::TopK, 2> {
        using Opr = opr::TopK;
        using Param = Opr::Param;
        static cg::OperatorNodeBase* make(
                const Param &param, const cg::VarNodeArray &inputs,
                ComputingGraph &graph, const OperatorNodeConfig &config) {
            MGB_MARK_USED_VAR(graph);
            auto out = Opr::make(inputs[0], inputs[1], param, config);
            return out[0].node()->owner_opr();
        }
    };

} // namespace serialization


namespace opr {

    MGB_SEREG_OPR(Argmax, 1);
    MGB_SEREG_OPR(Argmin, 1);
    MGB_SEREG_OPR(Argsort, 1);
    MGB_SEREG_OPR(ArgsortBackward, 3);
    MGB_SEREG_OPR(CondTake, 2);
    MGB_SEREG_OPR(TopK, 2);
    //! current cumsum version
    using CumsumV1 = opr::Cumsum;
    MGB_SEREG_OPR(CumsumV1, 1);

#if MGB_CUDA
    MGB_SEREG_OPR(NvOf, 1);
#endif

} // namespace opr
} // namespace mgb


// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

