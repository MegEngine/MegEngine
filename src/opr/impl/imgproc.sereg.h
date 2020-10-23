/**
 * \file src/opr/impl/imgproc.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/imgproc.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace serialization {
    //! OprMaker implementation for operators with variadic arguments
    template<>
    struct OprMaker<opr::WarpPerspective, 0> {
        using Opr = opr::WarpPerspective;
        using Param = Opr::Param;
        static cg::OperatorNodeBase* make(const Param& param,
                                          const cg::VarNodeArray& inputs,
                                          ComputingGraph& graph,
                                          const OperatorNodeConfig& config) {
            MGB_MARK_USED_VAR(graph);
            if (inputs.size() == 3) {
                return Opr::make(inputs[0], inputs[1], inputs[2], param, config)
                        .node()
                        ->owner_opr();
            } else {
                mgb_assert(inputs.size() == 4);
                return Opr::make(inputs[0], inputs[1], inputs[2], inputs[3],
                                 param, config)
                        .node()
                        ->owner_opr();
            }
        }
    };

    template <>
    struct OprMaker<opr::DctChannelSelectForward, 0> {
        using Opr = opr::DctChannelSelectForward;
        using Param = Opr::Param;
        static cg::OperatorNodeBase* make(const Param& param,
                                          const cg::VarNodeArray& inputs,
                                          ComputingGraph& graph,
                                          const OperatorNodeConfig& config) {
            MGB_MARK_USED_VAR(graph);
            if (inputs.size() == 3) {
                return Opr::make(inputs[0], inputs[1], inputs[2], param, config)
                        .node()
                        ->owner_opr();
            } else {
                mgb_assert(inputs.size() == 1);
                return Opr::make(inputs[0], param, config).node()->owner_opr();
            }
        }
    };

    template<>
    struct OprMaker<opr::WarpPerspectiveBackwardData, 0> {
        using Opr = opr::WarpPerspectiveBackwardData;
        using Param = Opr::Param;
        static cg::OperatorNodeBase* make(const Param& param,
                                          const cg::VarNodeArray& inputs,
                                          ComputingGraph& graph,
                                          const OperatorNodeConfig& config) {
            MGB_MARK_USED_VAR(graph);
            if (inputs.size() == 3) {
                return Opr::make(inputs[0], inputs[1], inputs[2], param, config)
                        .node()
                        ->owner_opr();
            } else {
                mgb_assert(inputs.size() == 4);
                return Opr::make(inputs[0], inputs[1], inputs[2], inputs[3],
                                 param, config)
                        .node()
                        ->owner_opr();
            }
        }
    };

    template<>
    struct OprMaker<opr::WarpPerspectiveBackwardMat, 0> {
        using Opr = opr::WarpPerspectiveBackwardMat;
        using Param = Opr::Param;
        static cg::OperatorNodeBase* make(const Param& param,
                                          const cg::VarNodeArray& inputs,
                                          ComputingGraph& graph,
                                          const OperatorNodeConfig& config) {
            MGB_MARK_USED_VAR(graph);
            if (inputs.size() == 3) {
                return Opr::make(inputs[0], inputs[1], inputs[2], param, config)
                        .node()
                        ->owner_opr();
            } else {
                mgb_assert(inputs.size() == 4);
                return Opr::make(inputs[0], inputs[1], inputs[2], inputs[3],
                                 param, config)
                        .node()
                        ->owner_opr();
            }
        }
    };
} // namespace serialization

namespace opr {

    MGB_SEREG_OPR(WarpPerspective, 0);
    MGB_SEREG_OPR(WarpPerspectiveBackwardData, 0);
    MGB_SEREG_OPR(WarpPerspectiveBackwardMat, 0);

    MGB_SEREG_OPR(Rotate, 1);
    MGB_SEREG_OPR(CvtColor, 1);
    MGB_SEREG_OPR(GaussianBlur, 1);

    MGB_SEREG_OPR(ResizeBackward, 2);
    MGB_SEREG_OPR(Remap, 2);
    MGB_SEREG_OPR(RemapBackwardData, 3);
    MGB_SEREG_OPR(RemapBackwardMat, 3);

    //! current warp affine version
    using WarpAffineV1 = opr::WarpAffine;
    MGB_SEREG_OPR(WarpAffineV1, 3);

    //! current resize version
    using ResizeV1 = opr::Resize;
    MGB_SEREG_OPR(ResizeV1, 2);

    MGB_SEREG_OPR(DctChannelSelect, 0);
} // namespace opr


} // namespace mgb


// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
