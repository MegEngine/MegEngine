/**
 * \file src/opr/impl/dnn/dnn.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/dnn/batch_norm.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/images2neibs.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/dnn/adaptive_pooling.h"
#include "megbrain/opr/dnn/roi_pooling.h"
#include "megbrain/opr/dnn/roi_align.h"
#include "megbrain/opr/dnn/local.h"
#include "megbrain/opr/dnn/lrn.h"
#include "megbrain/opr/dnn/fake_quant.h"

#include "megbrain/serialization/sereg.h"

namespace mgb {

namespace serialization {
    template<class MegDNNConv = megdnn::Convolution>
    struct MakeConvCaller2 {
        template<typename Opr>
        static VarNode* make(const cg::VarNodeArray &inputs,
                const typename MegDNNConv::Param &param,
                const megdnn::param::ExecutionPolicy &execution_policy,
                const OperatorNodeConfig &config) {
            if (inputs.size() == 2) {
                return Opr::make(
                        inputs[0], inputs[1], param,
                        execution_policy, config).node();
            }
            return nullptr;
        }
    };
    template<class MegDNNConv = megdnn::Convolution>
    struct MakeConvCaller3 {
        template<typename Opr>
        static VarNode* make(const cg::VarNodeArray &inputs,
                const typename MegDNNConv::Param &param,
                const megdnn::param::ExecutionPolicy &execution_policy,
                const OperatorNodeConfig &config) {
            if (inputs.size() == 3) {
                return Opr::make(
                        inputs[0], inputs[1], inputs[2], param,
                        execution_policy, config).node();
            }
            return nullptr;
        }
    };
    template<class MegDNNConv = megdnn::Convolution>
    struct MakeConvCaller4 {
        template<typename Opr>
        static VarNode* make(const cg::VarNodeArray &inputs,
                const typename MegDNNConv::Param &param,
                const megdnn::param::ExecutionPolicy &execution_policy,
                const OperatorNodeConfig &config) {
            if (inputs.size() == 4) {
                return Opr::make(
                        inputs[0], inputs[1], inputs[2], inputs[3], param,
                        execution_policy, config).node();
            }
            return nullptr;
        }
    };
    template<class MegDNNConv = megdnn::Convolution>
    struct MakeConvCaller5 {
        template <typename Opr>
        static VarNode* make(
                const cg::VarNodeArray& inputs,
                const typename MegDNNConv::Param& param,
                const megdnn::param::ExecutionPolicy& execution_policy,
                const OperatorNodeConfig& config) {
            if (inputs.size() == 5) {
                return Opr::make(inputs[0], inputs[1], inputs[2], inputs[3],
                                 inputs[4], param, execution_policy, config)
                        .node();
            }
            return nullptr;
        }
    };

    template<class MegDNNConv = megdnn::Convolution>
    struct MakeConvCallerEmpty {
        template<typename Opr>
        static VarNode* make(const cg::VarNodeArray &,
                const typename MegDNNConv::Param &,
                const megdnn::param::ExecutionPolicy &,
                const OperatorNodeConfig &) {
            return nullptr;
        }
    };
    

    template<class Opr, class Maker0, class MegDNNConv,
         class Maker1=MakeConvCallerEmpty<MegDNNConv>,
         class Maker2=MakeConvCallerEmpty<MegDNNConv>,
         typename ConvParam = megdnn::param::Convolution >
    struct ConvLoadDumpImpl {
        static void dump(OprDumpContext &ctx,
                const cg::OperatorNodeBase &opr_) {
            auto &&opr = opr_.cast_final_safe<Opr>();
            ctx.write_param<ConvParam>(opr.param());
            ctx.write_param<megdnn::param::ExecutionPolicy>(
                    opr.execution_policy());
        }

        static VarNode* make(
                const cg::VarNodeArray& inputs, const ConvParam& param,
                const megdnn::param::ExecutionPolicy& execution_policy,
                const OperatorNodeConfig& config) {
            VarNode* ret = Maker0::template make<Opr>(inputs, param,
                                                      execution_policy, config);
            if (!ret) {
                ret = Maker1::template make<Opr>(inputs, param,
                                                 execution_policy, config);
            }
            if (!ret) {
                ret = Maker2::template make<Opr>(inputs, param,
                                                 execution_policy, config);
            }
            mgb_assert(ret);
            return ret;
        }

        static cg::OperatorNodeBase* load(
                OprLoadContext &ctx, const cg::VarNodeArray &inputs,
                const OperatorNodeConfig &config) {
            auto param = ctx.read_param<ConvParam>();
            auto execution_policy =
                ctx.read_param<megdnn::param::ExecutionPolicy>();
            return make(inputs, param, execution_policy, config)->owner_opr();
        }
    };

    template<>
    struct OprLoadDumpImpl<opr::Convolution, 0>:
        public ConvLoadDumpImpl<opr::Convolution,
               MakeConvCaller2<megdnn::Convolution>,
               megdnn::Convolution>
    {};
    template<>
    struct OprLoadDumpImpl<opr::ConvolutionBackwardData, 0>:
        public ConvLoadDumpImpl<opr::ConvolutionBackwardData,
               MakeConvCaller2<megdnn::Convolution>,
               megdnn::Convolution,
               MakeConvCaller3<megdnn::Convolution> >
    {};
    template<>
    struct OprLoadDumpImpl<opr::ConvolutionBackwardFilter, 0>:
        public ConvLoadDumpImpl<opr::ConvolutionBackwardFilter,
               MakeConvCaller3<megdnn::Convolution>,
               megdnn::Convolution>
    {};

    template<>
    struct OprLoadDumpImpl<opr::Convolution3D, 0>:
        public ConvLoadDumpImpl<opr::Convolution3D,
               MakeConvCaller2<megdnn::Convolution3D>,
               megdnn::Convolution3D,
               MakeConvCallerEmpty<megdnn::Convolution3D>,
               MakeConvCallerEmpty<megdnn::Convolution3D>,
               megdnn::param::Convolution3D>
    {};
    template<>
    struct OprLoadDumpImpl<opr::Convolution3DBackwardData, 0>:
        public ConvLoadDumpImpl<opr::Convolution3DBackwardData,
               MakeConvCaller2<megdnn::Convolution3D>,
               megdnn::Convolution3D,
               MakeConvCaller3<megdnn::Convolution3D>,
               MakeConvCallerEmpty<megdnn::Convolution3D>,
               megdnn::param::Convolution3D>
    {};
    template<>
    struct OprLoadDumpImpl<opr::Convolution3DBackwardFilter, 0>:
        public ConvLoadDumpImpl<opr::Convolution3DBackwardFilter,
               MakeConvCaller3<megdnn::Convolution3D>,
               megdnn::Convolution3D,
               MakeConvCallerEmpty<megdnn::Convolution3D>,
               MakeConvCallerEmpty<megdnn::Convolution3D>,
               megdnn::param::Convolution3D>
    {};
    template<>
    struct OprLoadDumpImpl<opr::ConvBiasForward, 0>:
        public ConvLoadDumpImpl<opr::ConvBiasForward,
               MakeConvCaller2<megdnn::ConvBiasForward>,
               megdnn::ConvBiasForward,
               MakeConvCaller3<megdnn::ConvBiasForward>,
               MakeConvCaller4<megdnn::ConvBiasForward>,
               megdnn::param::ConvBias>
    {};
    template <>
    struct OprLoadDumpImpl<opr::BatchConvBiasForward, 0>
            : public ConvLoadDumpImpl<
                      opr::BatchConvBiasForward,
                      MakeConvCaller2<megdnn::BatchConvBiasForward>,
                      megdnn::BatchConvBiasForward,
                      MakeConvCaller3<megdnn::BatchConvBiasForward>,
                      MakeConvCaller4<megdnn::BatchConvBiasForward>,
                      megdnn::param::BatchConvBias> {};

    template <>
    struct OprMaker<opr::BatchNorm, 0> {
        using Param = opr::BatchNorm::Param;
        static cg::OperatorNodeBase* make(const Param& param,
                                          const cg::VarNodeArray& i,
                                          ComputingGraph& graph,
                                          const OperatorNodeConfig& config) {
            MGB_MARK_USED_VAR(graph);
            if (i.size() == 3) {
                return opr::BatchNorm::make(i[0], i[1], i[2],
                    param, config)[0].node()->owner_opr();
            } else {
                mgb_assert(i.size() == 5);
                return opr::BatchNorm::make(i[0], i[1], i[2], i[3], i[4],
                    param, config)[0].node()->owner_opr();
            }
        }
    };

    template <>
    struct OprMaker<opr::BatchNormBackward, 5> {
        using Param = opr::BatchNormBackward::Param;
        static cg::OperatorNodeBase* make(const Param& param,
                                          const cg::VarNodeArray& i,
                                          ComputingGraph& graph,
                                          const OperatorNodeConfig& config) {
            MGB_MARK_USED_VAR(graph);
            return opr::BatchNormBackward::make(i[0], i[1], i[2], i[3], i[4],
                param, config)[0].node()->owner_opr();
        }
    };

    template<class MegDNNConv = megdnn::LocalShare>
    struct MakeLocalShareCaller2 {
        template<typename Opr>
        static VarNode* make(const cg::VarNodeArray &inputs,
                const typename MegDNNConv::Param &param,
                const megdnn::param::ExecutionPolicy &execution_policy,
                const OperatorNodeConfig &config) {
            if (inputs.size() == 2) {
                return Opr::make(
                        inputs[0], inputs[1], param,
                        execution_policy, config).node();
            }
            return nullptr;
        }
    };
    template<class MegDNNConv = megdnn::LocalShare>
    struct MakeLocalShareCaller3 {
        template<typename Opr>
        static VarNode* make(const cg::VarNodeArray &inputs,
                const typename MegDNNConv::Param &param,
                const megdnn::param::ExecutionPolicy &execution_policy,
                const OperatorNodeConfig &config) {
            if (inputs.size() == 3) {
                return Opr::make(
                        inputs[0], inputs[1], inputs[2], param,
                        execution_policy, config).node();
            }
            return nullptr;
        }
    };
    template<class MegDNNConv = megdnn::LocalShare>
    struct MakeLocalShareCallerEmpty {
        template<typename Opr>
        static VarNode* make(const cg::VarNodeArray &,
                const typename MegDNNConv::Param &,
                const megdnn::param::ExecutionPolicy &,
                const OperatorNodeConfig &) {
            return nullptr;
        }
    };
 
    template<class Opr, class Maker0, class MegDNNConv,
         class Maker1=MakeLocalShareCallerEmpty<MegDNNConv>,
         class Maker2=MakeLocalShareCallerEmpty<MegDNNConv>,
         typename LocalShareParam = megdnn::param::LocalShare >
    struct LocalShareLoadDumpImpl {
        static void dump(OprDumpContext &ctx,
                const cg::OperatorNodeBase &opr_) {
            auto &&opr = opr_.cast_final_safe<Opr>();
            ctx.write_param<LocalShareParam>(opr.param());
            ctx.write_param<megdnn::param::ExecutionPolicy>(
                    opr.execution_policy());
        }

        static VarNode* make(
                const cg::VarNodeArray& inputs, const LocalShareParam& param,
                const megdnn::param::ExecutionPolicy& execution_policy,
                const OperatorNodeConfig& config) {
            VarNode* ret = Maker0::template make<Opr>(inputs, param,
                                                      execution_policy, config);
            if (!ret) {
                ret = Maker1::template make<Opr>(inputs, param,
                                                 execution_policy, config);
            }
            if (!ret) {
                ret = Maker2::template make<Opr>(inputs, param,
                                                 execution_policy, config);
            }
            mgb_assert(ret);
            return ret;
        }

        static cg::OperatorNodeBase* load(
                OprLoadContext &ctx, const cg::VarNodeArray &inputs,
                const OperatorNodeConfig &config) {
            auto param = ctx.read_param<LocalShareParam>();
            auto execution_policy =
                ctx.read_param<megdnn::param::ExecutionPolicy>();
            return make(inputs, param, execution_policy, config)->owner_opr();
        }
    };

    template <>
    struct OprLoadDumpImpl<opr::LocalShare, 0>
            : public LocalShareLoadDumpImpl<
                      opr::LocalShare,
                      MakeLocalShareCaller2<megdnn::LocalShare>,
                      megdnn::LocalShare> {};
    template <>
    struct OprLoadDumpImpl<opr::LocalShareBackwardData, 0>
            : public LocalShareLoadDumpImpl<
                      opr::LocalShareBackwardData,
                      MakeLocalShareCaller3<megdnn::LocalShare>,
                      megdnn::LocalShare> {};
    template <>
    struct OprLoadDumpImpl<opr::LocalShareBackwardFilter, 0>
            : public LocalShareLoadDumpImpl<
                      opr::LocalShareBackwardFilter,
                      MakeLocalShareCaller3<megdnn::LocalShare>,
                      megdnn::LocalShare> {};
    template<>
    struct OprLoadDumpImpl<opr::DeformableConvForward, 0>:
        public ConvLoadDumpImpl<opr::DeformableConvForward,
               MakeConvCaller4<megdnn::DeformableConvForward>,
               megdnn::Convolution>
    {};
    template<>
    struct OprLoadDumpImpl<opr::DeformableConvBackwardData, 0>:
        public ConvLoadDumpImpl<opr::DeformableConvBackwardData,
               MakeConvCaller5<megdnn::DeformableConvBackwardData>,
               megdnn::Convolution>
    {};
    template<>
    struct OprLoadDumpImpl<opr::DeformableConvBackwardFilter, 0>:
        public ConvLoadDumpImpl<opr::DeformableConvBackwardFilter,
               MakeConvCaller5<megdnn::DeformableConvBackwardFilter>,
               megdnn::Convolution>
    {};
} // namespace serialization

namespace opr {

    using ConvolutionV1 = Convolution;
    using ConvolutionBackwardDataV1 = ConvolutionBackwardData;
    using ConvolutionBackwardFilterV1 = ConvolutionBackwardFilter;
    MGB_SEREG_OPR(ConvolutionV1, 0);
    MGB_SEREG_OPR(ConvolutionBackwardDataV1, 0);
    MGB_SEREG_OPR(ConvolutionBackwardFilterV1, 0);

    MGB_SEREG_OPR(Images2Neibs, 1);
    MGB_SEREG_OPR(Images2NeibsBackward, 2);

    using LocalV1 = Local;
    using LocalBackwardDataV1 = LocalBackwardData;
    using LocalBackwardFilterV1 = LocalBackwardFilter;
    MGB_SEREG_OPR(LocalV1, 2);
    MGB_SEREG_OPR(LocalBackwardDataV1, 3);
    MGB_SEREG_OPR(LocalBackwardFilterV1, 3);

    using GroupLocalV1 = GroupLocal;
    using GroupLocalBackwardDataV1 = GroupLocalBackwardData;
    using GroupLocalBackwardFilterV1 = GroupLocalBackwardFilter;
    MGB_SEREG_OPR(GroupLocalV1, 2);
    MGB_SEREG_OPR(GroupLocalBackwardDataV1, 3);
    MGB_SEREG_OPR(GroupLocalBackwardFilterV1, 3);

    MGB_SEREG_OPR(LRN, 1);
    MGB_SEREG_OPR(LRNBackward, 3);

    MGB_SEREG_OPR(Pooling, 1);
    MGB_SEREG_OPR(PoolingBackward, 3);

    MGB_SEREG_OPR(AdaptivePooling, 2);
    MGB_SEREG_OPR(AdaptivePoolingBackward, 4);

    MGB_SEREG_OPR(ROIPooling, 3);
    MGB_SEREG_OPR(ROIPoolingBackward, 4);

    using MaskConvolutionV1 = MaskConvolution;
    MGB_SEREG_OPR(MaskConvolutionV1, 3);
    MGB_SEREG_OPR(MaskPropagate, 1);

    MGB_SEREG_OPR(Convolution3D, 0);
    MGB_SEREG_OPR(Convolution3DBackwardData, 0);
    MGB_SEREG_OPR(Convolution3DBackwardFilter, 0);

    using ConvBiasForwardV3 = ConvBiasForward;
    MGB_SEREG_OPR(ConvBiasForwardV3, 0);

    MGB_SEREG_OPR(BatchNorm, 0);
    MGB_SEREG_OPR(BatchNormBackward, 5);

    MGB_SEREG_OPR(LocalShareForward, 0);
    MGB_SEREG_OPR(LocalShareBackwardData, 0);
    MGB_SEREG_OPR(LocalShareBackwardFilter, 0);

    MGB_SEREG_OPR(ROIAlign, 2);
    MGB_SEREG_OPR(ROIAlignBackward, 4);
    MGB_SEREG_OPR(DeformableConvForward, 0);
    MGB_SEREG_OPR(DeformableConvBackwardData, 0);
    MGB_SEREG_OPR(DeformableConvBackwardFilter, 0);

    MGB_SEREG_OPR(DeformablePSROIPoolingForward, 3);
    MGB_SEREG_OPR(DeformablePSROIPoolingBackward, 5);

    MGB_SEREG_OPR(BatchConvBiasForward, 0);
    MGB_SEREG_OPR(FakeQuant, 3);
    MGB_SEREG_OPR(FakeQuantBackward, 4);
} // namespace opr


} // namespace mgb


// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
