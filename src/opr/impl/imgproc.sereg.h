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

#include <type_traits>
#include "megbrain/opr/imgproc.h"
#include "megbrain/serialization/sereg.h"
#include "megdnn/opr_param_defs.h"

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
    struct OprMaker<opr::Remap, 0> {
        using Opr = opr::Remap;
        using Param = Opr::Param;
        static cg::OperatorNodeBase* make(const Param& param,
                                          const cg::VarNodeArray& inputs,
                                          ComputingGraph& graph,
                                          const OperatorNodeConfig& config) {
            MGB_MARK_USED_VAR(graph);
            if (inputs.size() == 2) {
                return Opr::make(inputs[0], inputs[1], param, config)
                        .node()
                        ->owner_opr();
            } else {
                return nullptr;
            }
        }
    };

    template<>
    struct OprMaker<opr::RemapBackwardMat, 0> {
        using Opr = opr::RemapBackwardMat;
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
                return nullptr;
            }
        }
    };
    
    template<>
    struct OprMaker<opr::RemapBackwardData, 0> {
        using Opr = opr::RemapBackwardData;
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
                return nullptr;
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
    using WarpPerspectiveV2=WarpPerspective;
    using WarpPerspectiveBackwardDataV2=WarpPerspectiveBackwardData;
    using WarpPerspectiveBackwardMatV2=WarpPerspectiveBackwardMat;
    MGB_SEREG_OPR(WarpPerspectiveV2, 0);
    MGB_SEREG_OPR(WarpPerspectiveBackwardDataV2, 0);
    MGB_SEREG_OPR(WarpPerspectiveBackwardMatV2, 0);

    MGB_SEREG_OPR(Rotate, 1);
    MGB_SEREG_OPR(CvtColor, 1);
    MGB_SEREG_OPR(GaussianBlur, 1);

    MGB_SEREG_OPR(ResizeBackward, 2);
    using RemapV1=Remap;
    using RemapBackwardDataV1=RemapBackwardData;
    using RemapBackwardMatV1=RemapBackwardMat;
    MGB_SEREG_OPR(RemapV1, 2);
    MGB_SEREG_OPR(RemapBackwardDataV1, 3);
    MGB_SEREG_OPR(RemapBackwardMatV1, 3);

    //! current warp affine version
    using WarpAffineV2 = opr::WarpAffine;
    MGB_SEREG_OPR(WarpAffineV2, 3);

    //! current resize version
    using ResizeV2 = opr::Resize;
    MGB_SEREG_OPR(ResizeV2, 2);

    using DctChannelSelectV1 = opr::DctChannelSelect;
    MGB_SEREG_OPR(DctChannelSelectV1, 0);
} // namespace opr


} // namespace mgb


// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
