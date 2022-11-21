#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/dnn/adaptive_pooling.h"
#include "megbrain/opr/dnn/batch_norm.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/correlation.h"
#include "megbrain/opr/dnn/fake_quant.h"
#include "megbrain/opr/dnn/group_norm.h"
#include "megbrain/opr/dnn/images2neibs.h"
#include "megbrain/opr/dnn/instance_norm.h"
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
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/serialization/oss_opr_load_dump.h"
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

template <>
struct OprMaker<opr::GroupNorm, 0> {
    using Param = opr::GroupNorm::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        if (i.size() == 3) {
            return opr::GroupNorm::make(i[0], i[1], i[2], param, config)[0]
                    .node()
                    ->owner_opr();
        } else {
            mgb_assert(i.size() == 1);
            return opr::GroupNorm::make(i[0], param, config)[0].node()->owner_opr();
        }
    }
};

template <>
struct OprLoadDumpImplV2<opr::GroupNorm, 0> {
    using Opr = opr::GroupNorm;
    using Param = opr::GroupNorm::Param;
    using ElemwiseParam = opr::Elemwise::Param;
    using ReduceParam = opr::Reduce::Param;
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr) {
        ctx.write_param<Param>(opr.cast_final_safe<Opr>().param());
    }

    static cg::OperatorNodeBase* replace_opr(
            cg::OperatorNodeBase* opr, const VarNodeArray& inputs) {
        auto graph = inputs[0]->owner_graph();
        auto comp_node = inputs[0]->comp_node();
        // std::unique_ptr<StaticInferManager> m_static_infer_manager;
        auto opr_param = opr->cast_final_safe<opr::GroupNorm>().param();
        float eps = opr_param.eps;
        auto half = DTypeScalar(static_cast<megdnn::dt_float32>(0.5));
        auto param_eps = DTypeScalar(static_cast<megdnn::dt_float32>(eps));
        auto half_node = opr::ImmutableTensor::make(*graph, half, {comp_node});
        auto eps_node = opr::ImmutableTensor::make(*graph, param_eps, {comp_node});

        auto origin_shape = opr::GetVarShape::make(inputs[0]).node();

        TensorShape input_shape =
                inputs[0]->owner_graph()->static_infer_manager().infer_shape(inputs[0]);
        size_t N = input_shape[0];
        size_t inner_size = input_shape[1] * input_shape[2] * input_shape[3];
        int group = opr_param.group;
        int size = inner_size / group;
        HostTensorND hv = HostTensorND(inputs[0]->comp_node(), {3}, dtype::Int32());
        auto* ptr = hv.ptr<dt_int32>();
        ptr[0] = N;
        ptr[1] = group;
        ptr[2] = size;
        auto target_shape = opr::ImmutableTensor::make(*graph, hv, {comp_node});
        auto inp = opr::Reshape::make(inputs[0], target_shape);

        auto mean = opr::Reduce::make(inp, {ReduceParam::Mode::MEAN, 2});
        auto elemwise1 = opr::Elemwise::make({inp, inp}, {ElemwiseParam::Mode::MUL});
        auto temp_var = opr::Reduce::make(elemwise1, {ReduceParam::Mode::MEAN, 2});
        auto elemwise2 = opr::Elemwise::make({mean, mean}, {ElemwiseParam::Mode::MUL});
        auto var =
                opr::Elemwise::make({temp_var, elemwise2}, {ElemwiseParam::Mode::SUB});
        auto add_var = opr::Elemwise::make({var, eps_node}, {ElemwiseParam::Mode::ADD});
        auto sqrt =
                opr::Elemwise::make({add_var, half_node}, {ElemwiseParam::Mode::POW});
        auto div = opr::Elemwise::make({inp, mean}, {ElemwiseParam::Mode::SUB});
        auto temp_inp =
                opr::Elemwise::make({div, sqrt}, {ElemwiseParam::Mode::TRUE_DIV});
        auto res = opr::Reshape::make(temp_inp, origin_shape);

        if (inputs.size() == 3) {
            auto mul_temp =
                    opr::Elemwise::make({res, inputs[1]}, {ElemwiseParam::Mode::MUL});
            auto res = opr::Elemwise::make(
                    {mul_temp, inputs[2]}, {ElemwiseParam::Mode::ADD});
            return res.node()->owner_opr();
        } else {
            return res.node()->owner_opr();
        }
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        // auto& fbs_ctx = CAST_TO_FBS_V2_CTX(ctx);
        return OprMaker<opr::GroupNorm, 0>::make(
                ctx.read_param<Param>(), inputs, ctx.graph(), config);
    }
};

// OprMaker in MGB_SEREG_OPR only support unique output opr
template <>
struct OprMaker<opr::GroupNormBackward, 0> {
    using Param = opr::GroupNormBackward::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        if (i.size() == 5) {
            return opr::GroupNormBackward::make(
                           i[0], i[1], i[2], i[3], i[4], param, config)[0]
                    .node()
                    ->owner_opr();
        } else {
            mgb_assert(i.size() == 4);
            return opr::GroupNormBackward::make(
                           i[0], i[1], i[2], i[3], param, config)[0]
                    .node()
                    ->owner_opr();
        }
    }
};

template <>
struct OprLoadDumpImplV2<opr::GroupNormBackward, 0> {
    using Opr = opr::GroupNormBackward;
    using Param = opr::GroupNormBackward::Param;
    using ElemwiseParam = opr::Elemwise::Param;
    using ReduceParam = opr::Reduce::Param;
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr) {
        ctx.write_param<Param>(opr.cast_final_safe<Opr>().param());
    }

    static cg::OperatorNodeBase* replace_opr(
            cg::OperatorNodeBase* opr, const VarNodeArray& inputs) {
        auto rstd = inputs[4];
        auto graph = inputs[1]->owner_graph();
        auto comp_node = inputs[1]->comp_node();
        auto opr_param = opr->cast_final_safe<opr::GroupNormBackward>().param();
        float eps = opr_param.eps;
        auto half = DTypeScalar(static_cast<megdnn::dt_float32>(0.5));
        auto param_eps = DTypeScalar(static_cast<megdnn::dt_float32>(eps));
        auto half_node = opr::ImmutableTensor::make(*graph, half, {comp_node});
        auto eps_node = opr::ImmutableTensor::make(*graph, param_eps, {comp_node});
        auto const_node =
                opr::ImmutableTensor::make(*graph, DTypeScalar(1), {comp_node});

        TensorShape input_shape =
                inputs[1]->owner_graph()->static_infer_manager().infer_shape(inputs[0]);
        auto origin_shape = opr::GetVarShape::make(inputs[1]).node();
        size_t N = input_shape[0];
        size_t C = input_shape[1];
        size_t inner_size = input_shape[1] * input_shape[2] * input_shape[3];
        int group = opr_param.group;
        int size = inner_size / group;
        HostTensorND hv = HostTensorND(inputs[1]->comp_node(), {3}, dtype::Int32());
        auto* ptr = hv.ptr<dt_int32>();
        ptr[0] = N;
        ptr[1] = group;
        ptr[2] = size;
        auto target_shape = opr::ImmutableTensor::make(*graph, hv, {comp_node});
        auto inp = opr::Reshape::make(inputs[1], target_shape);

        auto temp_rstd =
                opr::Elemwise::make({rstd, eps_node}, {ElemwiseParam::Mode::ADD});
        auto sqrt =
                opr::Elemwise::make({temp_rstd, half_node}, {ElemwiseParam::Mode::POW});
        auto slice_std = opr::Elemwise::make(
                {const_node, sqrt}, {ElemwiseParam::Mode::TRUE_DIV});
        auto sub_mean =
                opr::Elemwise::make({inp, inputs[3]}, {ElemwiseParam::Mode::SUB});
        auto x_hat =
                opr::Elemwise::make({sub_mean, slice_std}, {ElemwiseParam::Mode::MUL});
        x_hat = opr::Reshape::make(x_hat, origin_shape);
        auto size_node =
                opr::ImmutableTensor::make(*graph, DTypeScalar(size), {comp_node});
        auto temp1 = opr::Elemwise::make(
                {slice_std, size_node}, {ElemwiseParam::Mode::TRUE_DIV});

        auto dx_hat =
                opr::Elemwise::make({inputs[0], inputs[2]}, {ElemwiseParam::Mode::MUL});
        HostTensorND tshape = HostTensorND(inputs[1]->comp_node(), {5}, dtype::Int32());
        auto* ptr2 = tshape.ptr<dt_int32>();
        ptr2[0] = N;
        ptr2[1] = group;
        ptr2[2] = C / group;
        ptr2[3] = input_shape[2];
        ptr2[4] = input_shape[3];
        target_shape = opr::ImmutableTensor::make(*graph, tshape, {comp_node});
        x_hat = opr::Reshape::make(x_hat, target_shape);
        dx_hat = opr::Reshape::make(dx_hat, target_shape);
        auto temp2 =
                opr::Elemwise::make({size_node, dx_hat}, {ElemwiseParam::Mode::MUL});
        ptr2[2] = 1;
        ptr2[3] = 1;
        ptr2[4] = 1;
        target_shape = opr::ImmutableTensor::make(*graph, tshape, {comp_node});
        auto temp3 = opr::Reduce::make(dx_hat, {ReduceParam::Mode::SUM}, target_shape);
        auto sum_dx_hat =
                opr::Reduce::make(temp2, {ReduceParam::Mode::SUM}, target_shape);
        auto temp4 =
                opr::Elemwise::make({x_hat, sum_dx_hat}, {ElemwiseParam::Mode::MUL});
        auto temp5 = opr::Elemwise::make({temp2, temp3}, {ElemwiseParam::Mode::SUB});
        auto temp6 = opr::Elemwise::make({temp5, temp4}, {ElemwiseParam::Mode::SUB});
        auto dx_temp = opr::Elemwise::make({temp1, temp6}, {ElemwiseParam::Mode::MUL});
        auto dx = opr::Reshape::make(dx_temp, origin_shape);
        return dx.node()->owner_opr();
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        return OprMaker<opr::GroupNormBackward, 0>::make(
                ctx.read_param<Param>(), inputs, ctx.graph(), config);
    }
};

template <>
struct OprMaker<opr::InstanceNorm, 0> {
    using Param = opr::GroupNorm::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        if (i.size() == 3) {
            return opr::InstanceNorm::make(i[0], i[1], i[2], param, config)[0]
                    .node()
                    ->owner_opr();
        } else {
            mgb_assert(i.size() == 1);
            return opr::InstanceNorm::make(i[0], param, config)[0].node()->owner_opr();
        }
    }
};

template <>
struct OprLoadDumpImplV2<opr::InstanceNorm, 0> {
    using Opr = opr::InstanceNorm;
    using Param = opr::GroupNorm::Param;
    using ElemwiseParam = opr::Elemwise::Param;
    using ReduceParam = opr::Reduce::Param;
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr) {
        ctx.write_param<Param>(opr.cast_final_safe<Opr>().param());
    }

    static cg::OperatorNodeBase* replace_opr(
            cg::OperatorNodeBase* opr, const VarNodeArray& inputs) {
        auto graph = inputs[0]->owner_graph();
        auto comp_node = inputs[0]->comp_node();
        // std::unique_ptr<StaticInferManager> m_static_infer_manager;
        auto opr_param = opr->cast_final_safe<opr::InstanceNorm>().param();
        float eps = opr_param.eps;
        auto half = DTypeScalar(static_cast<megdnn::dt_float32>(0.5));
        auto param_eps = DTypeScalar(static_cast<megdnn::dt_float32>(eps));
        auto half_node = opr::ImmutableTensor::make(*graph, half, {comp_node});
        auto eps_node = opr::ImmutableTensor::make(*graph, param_eps, {comp_node});

        auto origin_shape = opr::GetVarShape::make(inputs[0]).node();

        TensorShape input_shape =
                inputs[0]->owner_graph()->static_infer_manager().infer_shape(inputs[0]);
        size_t N = input_shape[0];
        size_t C = input_shape[1];
        size_t inner_size = input_shape[1] * input_shape[2] * input_shape[3];
        int size = inner_size / C;
        HostTensorND hv = HostTensorND(inputs[0]->comp_node(), {3}, dtype::Int32());
        auto* ptr = hv.ptr<dt_int32>();
        ptr[0] = N;
        ptr[1] = C;
        ptr[2] = size;
        auto target_shape = opr::ImmutableTensor::make(*graph, hv, {comp_node});
        auto inp = opr::Reshape::make(inputs[0], target_shape);

        auto mean = opr::Reduce::make(inp, {ReduceParam::Mode::MEAN, 2});
        auto elemwise1 = opr::Elemwise::make({inp, inp}, {ElemwiseParam::Mode::MUL});
        auto temp_var = opr::Reduce::make(elemwise1, {ReduceParam::Mode::MEAN, 2});
        auto elemwise2 = opr::Elemwise::make({mean, mean}, {ElemwiseParam::Mode::MUL});
        auto var =
                opr::Elemwise::make({temp_var, elemwise2}, {ElemwiseParam::Mode::SUB});
        auto add_var = opr::Elemwise::make({var, eps_node}, {ElemwiseParam::Mode::ADD});
        auto sqrt =
                opr::Elemwise::make({add_var, half_node}, {ElemwiseParam::Mode::POW});
        auto div = opr::Elemwise::make({inp, mean}, {ElemwiseParam::Mode::SUB});
        auto temp_inp =
                opr::Elemwise::make({div, sqrt}, {ElemwiseParam::Mode::TRUE_DIV});
        auto res = opr::Reshape::make(temp_inp, origin_shape);

        if (inputs.size() == 3) {
            auto mul_temp =
                    opr::Elemwise::make({res, inputs[1]}, {ElemwiseParam::Mode::MUL});
            auto res = opr::Elemwise::make(
                    {mul_temp, inputs[2]}, {ElemwiseParam::Mode::ADD});
            return res.node()->owner_opr();
        } else {
            return res.node()->owner_opr();
        }
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        // auto& fbs_ctx = CAST_TO_FBS_V2_CTX(ctx);

        // return OprMaker<opr::AdaptivePooling,0>::make(ctx.read_param<Param>(),
        // inputs, ctx.graph(), config);
        return OprMaker<opr::InstanceNorm, 0>::make(
                ctx.read_param<Param>(), inputs, ctx.graph(), config);
    }
};

// OprMaker in MGB_SEREG_OPR only support unique output opr
template <>
struct OprMaker<opr::InstanceNormBackward, 0> {
    using Param = opr::GroupNormBackward::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        if (i.size() == 5) {
            return opr::InstanceNormBackward::make(
                           i[0], i[1], i[2], i[3], i[4], param, config)[0]
                    .node()
                    ->owner_opr();
        } else {
            mgb_assert(i.size() == 4);
            return opr::InstanceNormBackward::make(
                           i[0], i[1], i[2], i[3], param, config)[0]
                    .node()
                    ->owner_opr();
        }
    }
};

template <>
struct OprLoadDumpImplV2<opr::InstanceNormBackward, 0> {
    using Opr = opr::InstanceNormBackward;
    using Param = opr::GroupNormBackward::Param;
    using ElemwiseParam = opr::Elemwise::Param;
    using ReduceParam = opr::Reduce::Param;
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr) {
        ctx.write_param<Param>(opr.cast_final_safe<Opr>().param());
    }

    static cg::OperatorNodeBase* replace_opr(
            cg::OperatorNodeBase* opr, const VarNodeArray& inputs) {
        auto rstd = inputs[4];
        auto graph = inputs[1]->owner_graph();
        auto comp_node = inputs[1]->comp_node();
        auto opr_param = opr->cast_final_safe<opr::InstanceNormBackward>().param();
        float eps = opr_param.eps;
        auto half = DTypeScalar(static_cast<megdnn::dt_float32>(0.5));
        auto param_eps = DTypeScalar(static_cast<megdnn::dt_float32>(eps));
        auto half_node = opr::ImmutableTensor::make(*graph, half, {comp_node});
        auto eps_node = opr::ImmutableTensor::make(*graph, param_eps, {comp_node});
        auto const_node =
                opr::ImmutableTensor::make(*graph, DTypeScalar(1), {comp_node});

        TensorShape input_shape =
                inputs[1]->owner_graph()->static_infer_manager().infer_shape(inputs[0]);
        auto origin_shape = opr::GetVarShape::make(inputs[1]).node();
        size_t N = input_shape[0];
        size_t C = input_shape[1];
        size_t inner_size = input_shape[1] * input_shape[2] * input_shape[3];
        int size = inner_size / C;
        HostTensorND hv = HostTensorND(inputs[1]->comp_node(), {3}, dtype::Int32());
        auto* ptr = hv.ptr<dt_int32>();
        ptr[0] = N;
        ptr[1] = C;
        ptr[2] = size;
        auto target_shape = opr::ImmutableTensor::make(*graph, hv, {comp_node});
        auto inp = opr::Reshape::make(inputs[1], target_shape);

        auto temp_rstd =
                opr::Elemwise::make({rstd, eps_node}, {ElemwiseParam::Mode::ADD});
        auto sqrt =
                opr::Elemwise::make({temp_rstd, half_node}, {ElemwiseParam::Mode::POW});
        auto slice_std = opr::Elemwise::make(
                {const_node, sqrt}, {ElemwiseParam::Mode::TRUE_DIV});
        auto sub_mean =
                opr::Elemwise::make({inp, inputs[3]}, {ElemwiseParam::Mode::SUB});
        auto x_hat =
                opr::Elemwise::make({sub_mean, slice_std}, {ElemwiseParam::Mode::MUL});
        x_hat = opr::Reshape::make(x_hat, origin_shape);
        auto size_node =
                opr::ImmutableTensor::make(*graph, DTypeScalar(size), {comp_node});
        auto temp1 = opr::Elemwise::make(
                {slice_std, size_node}, {ElemwiseParam::Mode::TRUE_DIV});

        auto dx_hat =
                opr::Elemwise::make({inputs[0], inputs[2]}, {ElemwiseParam::Mode::MUL});
        HostTensorND tshape = HostTensorND(inputs[1]->comp_node(), {5}, dtype::Int32());
        auto* ptr2 = tshape.ptr<dt_int32>();
        ptr2[0] = N;
        ptr2[1] = C;
        ptr2[2] = 1;
        ptr2[3] = input_shape[2];
        ptr2[4] = input_shape[3];
        target_shape = opr::ImmutableTensor::make(*graph, tshape, {comp_node});
        x_hat = opr::Reshape::make(x_hat, target_shape);
        dx_hat = opr::Reshape::make(dx_hat, target_shape);
        auto temp2 =
                opr::Elemwise::make({size_node, dx_hat}, {ElemwiseParam::Mode::MUL});
        ptr2[2] = 1;
        ptr2[3] = 1;
        ptr2[4] = 1;
        target_shape = opr::ImmutableTensor::make(*graph, tshape, {comp_node});
        auto temp3 = opr::Reduce::make(dx_hat, {ReduceParam::Mode::SUM}, target_shape);
        auto sum_dx_hat =
                opr::Reduce::make(temp2, {ReduceParam::Mode::SUM}, target_shape);
        auto temp4 =
                opr::Elemwise::make({x_hat, sum_dx_hat}, {ElemwiseParam::Mode::MUL});
        auto temp5 = opr::Elemwise::make({temp2, temp3}, {ElemwiseParam::Mode::SUB});
        auto temp6 = opr::Elemwise::make({temp5, temp4}, {ElemwiseParam::Mode::SUB});
        auto dx_temp = opr::Elemwise::make({temp1, temp6}, {ElemwiseParam::Mode::MUL});
        auto dx = opr::Reshape::make(dx_temp, origin_shape);
        return dx.node()->owner_opr();
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        return OprMaker<opr::InstanceNormBackward, 0>::make(
                ctx.read_param<Param>(), inputs, ctx.graph(), config);
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
MGB_SEREG_OPR(GroupNorm, 0);
MGB_SEREG_OPR(GroupNormBackward, 0);
MGB_SEREG_OPR(InstanceNorm, 0);
MGB_SEREG_OPR(InstanceNormBackward, 0);
MGB_SEREG_OPR(RNNCellForward, 6);
MGB_SEREG_OPR(LSTMCellForward, 7);
MGB_SEREG_OPR(RNNForward, 3);
MGB_SEREG_OPR(RNNBackward, 7);
MGB_SEREG_OPR(LSTMForward, 4);
MGB_SEREG_OPR(LSTMBackward, 9);
MGB_SEREG_OPR(Softmax, 1);
MGB_SEREG_OPR(SoftmaxBackward, 2);
MGB_SEREG_OPR_V2(
        GroupNorm, 0,
        (mgb::serialization::OprLoadDumpImplV2<opr::GroupNorm, 0>::replace_opr),
        VERSION_2, CURRENT_VERSION);
MGB_SEREG_OPR_V2(
        GroupNormBackward, 0,
        (mgb::serialization::OprLoadDumpImplV2<opr::GroupNormBackward, 0>::replace_opr),
        VERSION_2, CURRENT_VERSION);
MGB_SEREG_OPR_V2(
        InstanceNorm, 0,
        (mgb::serialization::OprLoadDumpImplV2<opr::InstanceNorm, 0>::replace_opr),
        VERSION_2, CURRENT_VERSION);
MGB_SEREG_OPR_V2(
        InstanceNormBackward, 0,
        (mgb::serialization::OprLoadDumpImplV2<
                opr::InstanceNormBackward, 0>::replace_opr),
        VERSION_2, CURRENT_VERSION);
}  // namespace opr

}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
