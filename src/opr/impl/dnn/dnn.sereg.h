#include "megbrain/opr/dnn/adaptive_pooling.h"
#include "megbrain/opr/dnn/batch_norm.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/correlation.h"
#include "megbrain/opr/dnn/fake_quant.h"
#include "megbrain/opr/dnn/images2neibs.h"
#include "megbrain/opr/dnn/layer_norm.h"
#include "megbrain/opr/dnn/local.h"
#include "megbrain/opr/dnn/lrn.h"
#include "megbrain/opr/dnn/lsq.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/dnn/rnn.h"
#include "megbrain/opr/dnn/roi_align.h"
#include "megbrain/opr/dnn/roi_pooling.h"
#include "megbrain/opr/dnn/sliding_window_transpose.h"
#include "megbrain/opr/dnn/softmax.h"
#include "megbrain/opr/dnn/tqt.h"
#include "megbrain/serialization/sereg.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs/nn.h"

namespace mgb {

namespace serialization {
template <class MegDNNPooling = megdnn::Pooling>
struct MakePoolingCaller1 {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray& inputs, const typename MegDNNPooling::Param& param,
            const megdnn::param::ExecutionPolicy& execution_policy,
            const OperatorNodeConfig& config) {
        if (inputs.size() == 1) {
            return Opr::make(inputs[0], param, execution_policy, config).node();
        }
        return nullptr;
    }
};

template <class MegDNNROIALIGN = megdnn::ROIAlign>
struct MakeROIAlignCaller1 {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray& inputs, const typename MegDNNROIALIGN::Param& param,
            const OperatorNodeConfig& config) {
        if (inputs.size() == 2) {
            return Opr::make(inputs[0], inputs[1], param, config).node();
        } else {
            return nullptr;
        }
    }
};

template <class MegDNNROIALIGN = megdnn::ROIAlignBackward>
struct MakeROIAlignCaller4 {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray& inputs, const typename MegDNNROIALIGN::Param& param,
            const OperatorNodeConfig& config) {
        if (inputs.size() == 4) {
            return Opr::make(inputs[0], inputs[1], inputs[2], inputs[3], param, config)
                    .node();
        } else {
            return nullptr;
        }
    }
};

template <class MegDNNPooling = megdnn::PoolingBackward>
struct MakePoolingBackwardCaller3 {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray& inputs, const typename MegDNNPooling::Param& param,
            const megdnn::param::ExecutionPolicy& execution_policy,
            const OperatorNodeConfig& config) {
        if (inputs.size() == 3) {
            return Opr::make(
                           inputs[0], inputs[1], inputs[2], param, execution_policy,
                           config)
                    .node();
        }
        return nullptr;
    }
};

template <class MegDNNPooling = megdnn::AdaptivePoolingBackward>
struct MakeAdaptivePoolingBackwardCaller3 {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray& inputs, const typename MegDNNPooling::Param& param,
            const OperatorNodeConfig& config) {
        if (inputs.size() == 4) {
            return Opr::make(inputs[0], inputs[1], inputs[2], inputs[3], param, config)
                    .node();
        }
        return nullptr;
    }
};

template <class MegDNNConv = megdnn::Convolution>
struct MakeConvCaller2 {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray& inputs, const typename MegDNNConv::Param& param,
            const megdnn::param::ExecutionPolicy& execution_policy,
            const OperatorNodeConfig& config) {
        if (inputs.size() == 2) {
            return Opr::make(inputs[0], inputs[1], param, execution_policy, config)
                    .node();
        }
        return nullptr;
    }
};

template <class MegDNNConv = megdnn::Convolution>
struct MakeConvCaller3 {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray& inputs, const typename MegDNNConv::Param& param,
            const megdnn::param::ExecutionPolicy& execution_policy,
            const OperatorNodeConfig& config) {
        if (inputs.size() == 3) {
            return Opr::make(
                           inputs[0], inputs[1], inputs[2], param, execution_policy,
                           config)
                    .node();
        }
        return nullptr;
    }
};

template <class MegDNNConv = megdnn::Convolution>
struct MakeConvCaller4 {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray& inputs, const typename MegDNNConv::Param& param,
            const megdnn::param::ExecutionPolicy& execution_policy,
            const OperatorNodeConfig& config) {
        if (inputs.size() == 4) {
            return Opr::make(
                           inputs[0], inputs[1], inputs[2], inputs[3], param,
                           execution_policy, config)
                    .node();
        }
        return nullptr;
    }
};

template <class MegDNNConv = megdnn::Convolution>
struct MakeConvCaller5 {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray& inputs, const typename MegDNNConv::Param& param,
            const megdnn::param::ExecutionPolicy& execution_policy,
            const OperatorNodeConfig& config) {
        if (inputs.size() == 5) {
            return Opr::make(
                           inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], param,
                           execution_policy, config)
                    .node();
        }
        return nullptr;
    }
};

template <class MegDNNConv = megdnn::Convolution>
struct MakeConvCallerEmpty {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray&, const typename MegDNNConv::Param&,
            const megdnn::param::ExecutionPolicy&, const OperatorNodeConfig&) {
        return nullptr;
    }
};

template <
        class Opr, class Maker0, class MegDNNConv,
        class Maker1 = MakeConvCallerEmpty<MegDNNConv>,
        class Maker2 = MakeConvCallerEmpty<MegDNNConv>,
        typename ConvParam = megdnn::param::Convolution>
struct ConvLoadDumpImpl {
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr_) {
        auto&& opr = opr_.cast_final_safe<Opr>();
        ctx.write_param<ConvParam>(opr.param());
        ctx.write_param<megdnn::param::ExecutionPolicy>(
                opr.execution_policy_transient());
    }

    static VarNode* make(
            const cg::VarNodeArray& inputs, const ConvParam& param,
            const megdnn::param::ExecutionPolicy& execution_policy,
            const OperatorNodeConfig& config) {
        VarNode* ret =
                Maker0::template make<Opr>(inputs, param, execution_policy, config);
        if (!ret) {
            ret = Maker1::template make<Opr>(inputs, param, execution_policy, config);
        }
        if (!ret) {
            ret = Maker2::template make<Opr>(inputs, param, execution_policy, config);
        }
        mgb_assert(ret);
        return ret;
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        auto param = ctx.read_param<ConvParam>();
        auto execution_policy = ctx.read_param<megdnn::param::ExecutionPolicy>();
        return make(inputs, param, execution_policy, config)->owner_opr();
    }
};

template <class Opr, class Maker0, typename PoolingParam = megdnn::param::Pooling>
struct PoolingLoadDumpImpl {
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr_) {
        auto&& opr = opr_.cast_final_safe<Opr>();
        ctx.write_param<PoolingParam>(opr.param());
    }

    static VarNode* make(
            const cg::VarNodeArray& inputs, const PoolingParam& param,
            const megdnn::param::ExecutionPolicy& execution_policy,
            const OperatorNodeConfig& config) {
        VarNode* ret =
                Maker0::template make<Opr>(inputs, param, execution_policy, config);
        mgb_assert(ret);
        return ret;
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        auto param = ctx.read_param<PoolingParam>();
        return make(inputs, param, {}, config)->owner_opr();
    }
};

template <class Opr, class Maker0, typename GeneralOprParam = megdnn::param::ROIAlign>
struct GeneralOprLoadDumpImpl {
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr_) {
        auto&& opr = opr_.cast_final_safe<Opr>();
        ctx.write_param<GeneralOprParam>(opr.param());
    }

    static VarNode* make(
            const cg::VarNodeArray& inputs, const GeneralOprParam& param,
            const OperatorNodeConfig& config) {
        VarNode* ret = Maker0::template make<Opr>(inputs, param, config);
        mgb_assert(ret);
        return ret;
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        auto param = ctx.read_param<GeneralOprParam>();
        return make(inputs, param, config)->owner_opr();
    }
};

template <>
struct OprMaker<opr::TQTBackward, 3> {
    using Param = opr::TQTBackward::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        return opr::TQTBackward::make(i[0], i[1], i[2], param, config)[0]
                .node()
                ->owner_opr();
    }
};

template <>
struct OprMaker<opr::LSQBackward, 5> {
    using Param = opr::LSQBackward::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        return opr::LSQBackward::make(i[0], i[1], i[2], i[3], i[4], param, config)[0]
                .node()
                ->owner_opr();
    }
};

template <>
struct OprMaker<opr::RNNCellForward, 6> {
    using Param = opr::RNNCellForward::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        return opr::RNNCellForward::make(
                       i[0], i[1], i[2], i[3], i[4], i[5], param, config)
                .node()
                ->owner_opr();
    }
};

template <>
struct OprMaker<opr::LSTMCellForward, 7> {
    using Param = opr::LSTMCellForward::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        return opr::LSTMCellForward::make(
                       i[0], i[1], i[2], i[3], i[4], i[5], i[6], param, config)
                .node()
                ->owner_opr();
    }
};

template <>
struct OprMaker<opr::RNNBackward, 7> {
    using Param = opr::RNNBackward::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        return opr::RNNBackward::make(
                       i[0], i[1], i[2], i[3], i[4], i[5], i[6], param, config)[0]
                .node()
                ->owner_opr();
    }
};

template <>
struct OprMaker<opr::LSTMBackward, 9> {
    using Param = opr::LSTMBackward::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        return opr::LSTMBackward::make(
                       i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], param,
                       config)[0]
                .node()
                ->owner_opr();
    }
};

template <>
struct OprMaker<opr::SoftmaxBackward, 2> {
    using Param = opr::SoftmaxBackward::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        return opr::SoftmaxBackward::make(i[0], i[1], param, config)
                .node()
                ->owner_opr();
    }
};

template <>
struct OprLoadDumpImpl<opr::AdaptivePoolingBackward, 0>
        : public GeneralOprLoadDumpImpl<
                  opr::AdaptivePoolingBackward,
                  MakeAdaptivePoolingBackwardCaller3<megdnn::AdaptivePoolingBackward>,
                  megdnn::param::AdaptivePooling> {};

template <>
struct OprLoadDumpImpl<opr::AdaptivePooling, 0>
        : public GeneralOprLoadDumpImpl<
                  opr::AdaptivePooling, MakeROIAlignCaller1<megdnn::AdaptivePooling>,
                  megdnn::param::AdaptivePooling> {};

template <>
struct OprLoadDumpImpl<opr::ROIAlign, 0>
        : public GeneralOprLoadDumpImpl<
                  opr::ROIAlign, MakeROIAlignCaller1<megdnn::ROIAlign>,
                  megdnn::param::ROIAlign> {};

template <>
struct OprLoadDumpImpl<opr::ROIAlignBackward, 0>
        : public GeneralOprLoadDumpImpl<
                  opr::ROIAlignBackward, MakeROIAlignCaller4<megdnn::ROIAlignBackward>,
                  megdnn::param::ROIAlign> {};

template <>
struct OprLoadDumpImpl<opr::Pooling, 0>
        : public PoolingLoadDumpImpl<
                  opr::Pooling, MakePoolingCaller1<megdnn::Pooling>,
                  megdnn::param::Pooling> {};

template <>
struct OprLoadDumpImpl<opr::PoolingBackward, 0>
        : public PoolingLoadDumpImpl<
                  opr::PoolingBackward,
                  MakePoolingBackwardCaller3<megdnn::PoolingBackward>,
                  megdnn::param::Pooling> {};

template <>
struct OprLoadDumpImpl<opr::Convolution, 0>
        : public ConvLoadDumpImpl<
                  opr::Convolution, MakeConvCaller2<megdnn::Convolution>,
                  megdnn::Convolution> {};
template <>
struct OprLoadDumpImpl<opr::ConvolutionBackwardData, 0>
        : public ConvLoadDumpImpl<
                  opr::ConvolutionBackwardData, MakeConvCaller2<megdnn::Convolution>,
                  megdnn::Convolution, MakeConvCaller3<megdnn::Convolution>> {};
template <>
struct OprLoadDumpImpl<opr::ConvolutionBackwardFilter, 0>
        : public ConvLoadDumpImpl<
                  opr::ConvolutionBackwardFilter, MakeConvCaller3<megdnn::Convolution>,
                  megdnn::Convolution> {};

template <>
struct OprLoadDumpImpl<opr::Convolution3D, 0>
        : public ConvLoadDumpImpl<
                  opr::Convolution3D, MakeConvCaller2<megdnn::Convolution3D>,
                  megdnn::Convolution3D, MakeConvCallerEmpty<megdnn::Convolution3D>,
                  MakeConvCallerEmpty<megdnn::Convolution3D>,
                  megdnn::param::Convolution3D> {};
template <>
struct OprLoadDumpImpl<opr::Convolution3DBackwardData, 0>
        : public ConvLoadDumpImpl<
                  opr::Convolution3DBackwardData,
                  MakeConvCaller2<megdnn::Convolution3D>, megdnn::Convolution3D,
                  MakeConvCaller3<megdnn::Convolution3D>,
                  MakeConvCallerEmpty<megdnn::Convolution3D>,
                  megdnn::param::Convolution3D> {};
template <>
struct OprLoadDumpImpl<opr::Convolution3DBackwardFilter, 0>
        : public ConvLoadDumpImpl<
                  opr::Convolution3DBackwardFilter,
                  MakeConvCaller3<megdnn::Convolution3D>, megdnn::Convolution3D,
                  MakeConvCallerEmpty<megdnn::Convolution3D>,
                  MakeConvCallerEmpty<megdnn::Convolution3D>,
                  megdnn::param::Convolution3D> {};

template <>
struct OprLoadDumpImpl<opr::ConvBiasForward, 0>
        : public ConvLoadDumpImpl<
                  opr::ConvBiasForward, MakeConvCaller2<megdnn::ConvBiasForward>,
                  megdnn::ConvBiasForward, MakeConvCaller3<megdnn::ConvBiasForward>,
                  MakeConvCaller4<megdnn::ConvBiasForward>, megdnn::param::ConvBias> {};
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
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        if (i.size() == 3) {
            return opr::BatchNorm::make(i[0], i[1], i[2], param, config)[0]
                    .node()
                    ->owner_opr();
        } else {
            mgb_assert(i.size() == 5);
            return opr::BatchNorm::make(i[0], i[1], i[2], i[3], i[4], param, config)[0]
                    .node()
                    ->owner_opr();
        }
    }
};

// OprMaker in MGB_SEREG_OPR only support unique output opr
template <>
struct OprMaker<opr::BatchNormBackward, 6> {
    using Param = opr::BatchNormBackward::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        return opr::BatchNormBackward::make(
                       i[0], i[1], i[2], i[3], i[4], i[5], param, config)[0]
                .node()
                ->owner_opr();
    }
};

template <>
struct OprMaker<opr::LayerNorm, 0> {
    using Param = opr::LayerNorm::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        if (i.size() == 3) {
            return opr::LayerNorm::make(i[0], i[1], i[2], param, config)[0]
                    .node()
                    ->owner_opr();
        } else {
            mgb_assert(i.size() == 1);
            return opr::LayerNorm::make(i[0], param, config)[0].node()->owner_opr();
        }
    }
};

// OprMaker in MGB_SEREG_OPR only support unique output opr
template <>
struct OprMaker<opr::LayerNormBackward, 0> {
    using Param = opr::LayerNormBackward::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        if (i.size() == 5) {
            return opr::LayerNormBackward::make(
                           i[0], i[1], i[2], i[3], i[4], param, config)[0]
                    .node()
                    ->owner_opr();
        } else {
            mgb_assert(i.size() == 4);
            return opr::LayerNormBackward::make(
                           i[0], i[1], i[2], i[3], param, config)[0]
                    .node()
                    ->owner_opr();
        }
    }
};

template <class MegDNNConv = megdnn::LocalShare>
struct MakeLocalShareCaller2 {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray& inputs, const typename MegDNNConv::Param& param,
            const megdnn::param::ExecutionPolicy& execution_policy,
            const OperatorNodeConfig& config) {
        if (inputs.size() == 2) {
            return Opr::make(inputs[0], inputs[1], param, execution_policy, config)
                    .node();
        }
        return nullptr;
    }
};
template <class MegDNNConv = megdnn::LocalShare>
struct MakeLocalShareCaller3 {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray& inputs, const typename MegDNNConv::Param& param,
            const megdnn::param::ExecutionPolicy& execution_policy,
            const OperatorNodeConfig& config) {
        if (inputs.size() == 3) {
            return Opr::make(
                           inputs[0], inputs[1], inputs[2], param, execution_policy,
                           config)
                    .node();
        }
        return nullptr;
    }
};
template <class MegDNNConv = megdnn::LocalShare>
struct MakeLocalShareCallerEmpty {
    template <typename Opr>
    static VarNode* make(
            const cg::VarNodeArray&, const typename MegDNNConv::Param&,
            const megdnn::param::ExecutionPolicy&, const OperatorNodeConfig&) {
        return nullptr;
    }
};

template <
        class Opr, class Maker0, class MegDNNConv,
        class Maker1 = MakeLocalShareCallerEmpty<MegDNNConv>,
        class Maker2 = MakeLocalShareCallerEmpty<MegDNNConv>,
        typename LocalShareParam = megdnn::param::LocalShare>
struct LocalShareLoadDumpImpl {
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr_) {
        auto&& opr = opr_.cast_final_safe<Opr>();
        ctx.write_param<LocalShareParam>(opr.param());
        ctx.write_param<megdnn::param::ExecutionPolicy>(opr.execution_policy());
    }

    static VarNode* make(
            const cg::VarNodeArray& inputs, const LocalShareParam& param,
            const megdnn::param::ExecutionPolicy& execution_policy,
            const OperatorNodeConfig& config) {
        VarNode* ret =
                Maker0::template make<Opr>(inputs, param, execution_policy, config);
        if (!ret) {
            ret = Maker1::template make<Opr>(inputs, param, execution_policy, config);
        }
        if (!ret) {
            ret = Maker2::template make<Opr>(inputs, param, execution_policy, config);
        }
        mgb_assert(ret);
        return ret;
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        auto param = ctx.read_param<LocalShareParam>();
        auto execution_policy = ctx.read_param<megdnn::param::ExecutionPolicy>();
        return make(inputs, param, execution_policy, config)->owner_opr();
    }
};

template <>
struct OprLoadDumpImpl<opr::LocalShare, 0>
        : public LocalShareLoadDumpImpl<
                  opr::LocalShare, MakeLocalShareCaller2<megdnn::LocalShare>,
                  megdnn::LocalShare> {};
template <>
struct OprLoadDumpImpl<opr::LocalShareBackwardData, 0>
        : public LocalShareLoadDumpImpl<
                  opr::LocalShareBackwardData,
                  MakeLocalShareCaller3<megdnn::LocalShare>, megdnn::LocalShare> {};
template <>
struct OprLoadDumpImpl<opr::LocalShareBackwardFilter, 0>
        : public LocalShareLoadDumpImpl<
                  opr::LocalShareBackwardFilter,
                  MakeLocalShareCaller3<megdnn::LocalShare>, megdnn::LocalShare> {};
template <>
struct OprLoadDumpImpl<opr::DeformableConvForward, 0>
        : public ConvLoadDumpImpl<
                  opr::DeformableConvForward,
                  MakeConvCaller4<megdnn::DeformableConvForward>, megdnn::Convolution> {
};
template <>
struct OprLoadDumpImpl<opr::DeformableConvBackwardData, 0>
        : public ConvLoadDumpImpl<
                  opr::DeformableConvBackwardData,
                  MakeConvCaller5<megdnn::DeformableConvBackwardData>,
                  megdnn::Convolution> {};
template <>
struct OprLoadDumpImpl<opr::DeformableConvBackwardFilter, 0>
        : public ConvLoadDumpImpl<
                  opr::DeformableConvBackwardFilter,
                  MakeConvCaller5<megdnn::DeformableConvBackwardFilter>,
                  megdnn::Convolution> {};

template <typename Opr>
cg::OperatorNodeBase* opr_shallow_copy_conv(
        const serialization::OprShallowCopyContext& ctx,
        const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
    MGB_MARK_USED_VAR(ctx);
    auto&& opr = opr_.cast_final_safe<Opr>();
    return OprLoadDumpImpl<Opr, 0>::make(
                   inputs, opr.param(), opr.execution_policy_transient(), config)
            ->owner_opr();
}

}  // namespace serialization

namespace opr {
using ConvolutionV2 = Convolution;
using ConvolutionBackwardDataV2 = ConvolutionBackwardData;
using ConvolutionBackwardFilterV2 = ConvolutionBackwardFilter;
MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(ConvolutionV2, 0, opr_shallow_copy_conv);
MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(ConvolutionBackwardDataV2, 0, opr_shallow_copy_conv);
MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(
        ConvolutionBackwardFilterV2, 0, opr_shallow_copy_conv);

MGB_SEREG_OPR(Images2Neibs, 1);
MGB_SEREG_OPR(Images2NeibsBackward, 2);

MGB_SEREG_OPR(SlidingWindowTranspose, 1);
MGB_SEREG_OPR(SlidingWindowTransposeBackward, 2);

using LocalV2 = Local;
using LocalBackwardDataV2 = LocalBackwardData;
using LocalBackwardFilterV2 = LocalBackwardFilter;
MGB_SEREG_OPR(LocalV2, 2);
MGB_SEREG_OPR(LocalBackwardDataV2, 3);
MGB_SEREG_OPR(LocalBackwardFilterV2, 3);

using GroupLocalV2 = GroupLocal;
using GroupLocalBackwardDataV2 = GroupLocalBackwardData;
using GroupLocalBackwardFilterV2 = GroupLocalBackwardFilter;
MGB_SEREG_OPR(GroupLocalV2, 2);
MGB_SEREG_OPR(GroupLocalBackwardDataV2, 3);
MGB_SEREG_OPR(GroupLocalBackwardFilterV2, 3);

MGB_SEREG_OPR(LRN, 1);
MGB_SEREG_OPR(LRNBackward, 3);
using PoolingV1 = Pooling;
using PoolingBackwardV1 = PoolingBackward;
MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(PoolingV1, 0, opr_shallow_copy_conv);
MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(PoolingBackwardV1, 0, opr_shallow_copy_conv);
using AdaptivePoolingV1 = AdaptivePooling;
using AdaptivePoolingBackwardV1 = AdaptivePoolingBackward;
MGB_SEREG_OPR(AdaptivePoolingV1, 2);
MGB_SEREG_OPR(AdaptivePoolingBackwardV1, 4);

MGB_SEREG_OPR(ROIPooling, 3);
MGB_SEREG_OPR(ROIPoolingBackward, 4);

using MaskConvolutionV2 = MaskConvolution;
MGB_SEREG_OPR(MaskConvolutionV2, 3);
MGB_SEREG_OPR(MaskPropagate, 1);

MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(Convolution3D, 0, opr_shallow_copy_conv);
MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(Convolution3DBackwardData, 0, opr_shallow_copy_conv);
MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(
        Convolution3DBackwardFilter, 0, opr_shallow_copy_conv);

using ConvBiasForwardV4 = ConvBiasForward;
MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(ConvBiasForwardV4, 0, opr_shallow_copy_conv);

using BatchNormV1 = BatchNorm;
using BatchNormBackwardV1 = BatchNormBackward;
MGB_SEREG_OPR(BatchNormV1, 0);
MGB_SEREG_OPR(BatchNormBackwardV1, 6);

using LocalShareForwardV1 = LocalShareForward;
using LocalShareBackwardDataV1 = LocalShareBackwardData;
using LocalShareBackwardFilterV1 = LocalShareBackwardFilter;
MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(LocalShareForwardV1, 0, opr_shallow_copy_conv);
MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(LocalShareBackwardDataV1, 0, opr_shallow_copy_conv);
MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(
        LocalShareBackwardFilterV1, 0, opr_shallow_copy_conv);

using ROIAlignV1 = ROIAlign;
using ROIAlignBackwardV1 = ROIAlignBackward;
MGB_SEREG_OPR(ROIAlignV1, 2);
MGB_SEREG_OPR(ROIAlignBackwardV1, 4);
using DeformableConvForwardV1 = DeformableConvForward;
using DeformableConvBackwardDataV1 = DeformableConvBackwardData;
using DeformableConvBackwardFilterV1 = DeformableConvBackwardFilter;
MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(DeformableConvForwardV1, 0, opr_shallow_copy_conv);
MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(
        DeformableConvBackwardDataV1, 0, opr_shallow_copy_conv);
MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(
        DeformableConvBackwardFilterV1, 0, opr_shallow_copy_conv);

MGB_SEREG_OPR(CorrelationForward, 2);
MGB_SEREG_OPR(CorrelationBackwardData1, 3);
MGB_SEREG_OPR(CorrelationBackwardData2, 3);

MGB_SEREG_OPR(DeformablePSROIPoolingForward, 3);
MGB_SEREG_OPR(DeformablePSROIPoolingBackward, 5);

using BatchConvBiasForwardV1 = BatchConvBiasForward;
MGB_SEREG_OPR_AND_REG_SHALLOW_COPY(BatchConvBiasForwardV1, 0, opr_shallow_copy_conv);
MGB_SEREG_OPR(FakeQuant, 3);
MGB_SEREG_OPR(FakeQuantBackward, 4);
MGB_SEREG_OPR(TQT, 2);
MGB_SEREG_OPR(TQTBackward, 3);
MGB_SEREG_OPR(LSQ, 4);
MGB_SEREG_OPR(LSQBackward, 5);
MGB_SEREG_OPR(LayerNorm, 0);
MGB_SEREG_OPR(LayerNormBackward, 0);
MGB_SEREG_OPR(RNNCellForward, 6);
MGB_SEREG_OPR(LSTMCellForward, 7);
MGB_SEREG_OPR(RNNForward, 3);
MGB_SEREG_OPR(RNNBackward, 7);
MGB_SEREG_OPR(LSTMForward, 4);
MGB_SEREG_OPR(LSTMBackward, 9);
MGB_SEREG_OPR(Softmax, 1);
MGB_SEREG_OPR(SoftmaxBackward, 2);
}  // namespace opr

}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
