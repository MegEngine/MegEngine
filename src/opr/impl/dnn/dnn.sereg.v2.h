#include "megbrain/graph/symbol_var.h"
#include "megdnn/oprs/general.h"
#if MGB_ENABLE_FBS_SERIALIZATION
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/dnn/softmax.h"
#include "megbrain/serialization/oss_opr_load_dump.h"
#include "megbrain/serialization/sereg.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs/nn.h"

namespace mgb {
namespace serialization {

template <>
struct OprLoadDumpImplV2<opr::Softmax, 1> {
    using Opr = opr::Softmax;
    using PersisParam = opr::Softmax::Param;
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr) {
        ctx.write_param<PersisParam>(opr.cast_final_safe<Opr>().param());
    }

    static cg::OperatorNodeBase* replace_opr(
            cg::OperatorNodeBase* opr, const VarNodeArray& inputs) {
        int32_t axis = opr->cast_final_safe<Opr>().param().axis;
        auto input_var = inputs[0];
        auto max_reduce_out =
                opr::Reduce::make(input_var, {megdnn::Reduce::Mode::MAX, axis});
        auto elemwise_sub_out = opr::Elemwise::make(
                {input_var, max_reduce_out}, {megdnn::Elemwise::Mode::SUB});
        auto elemwise_exp_out =
                opr::Elemwise::make({elemwise_sub_out}, {megdnn::Elemwise::Mode::EXP});
        auto sum_reduce_out =
                opr::Reduce::make(elemwise_exp_out, {megdnn::Reduce::Mode::SUM, axis});
        auto out = opr::Elemwise::make(
                {elemwise_exp_out, sum_reduce_out}, {megdnn::Elemwise::Mode::TRUE_DIV});
        return out.node()->owner_opr();
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        auto& fbs_ctx = CAST_TO_FBS_V2_CTX(ctx);
        auto param = fbs_ctx.read_param<PersisParam>(0);
        return Opr::make(inputs[0], param, config).node()->owner_opr();
    }
};

template <
        class Opr, class Maker0, class MegDNNConv,
        class Maker1 = MakeConvCallerEmpty<MegDNNConv>,
        class Maker2 = MakeConvCallerEmpty<MegDNNConv>,
        typename ConvParam = megdnn::param::Convolution>
struct WithPolicyOprLoadDumpImpl {
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
        auto& fbs_ctx = CAST_TO_FBS_V2_CTX(ctx);
        auto fopr = reinterpret_cast<const fbs::v2::Operator*>(
                fbs_ctx.get_current_opr_data());
        auto conv_param = fbs_ctx.read_param<ConvParam>(0);
        megdnn::param::ExecutionPolicy policy;
        if (fopr->additional_params() && fopr->additional_params()->size()) {
            policy = fbs_ctx.read_param<megdnn::param::ExecutionPolicy>(1);
        }
        return make(inputs, conv_param, policy, config)->owner_opr();
    }
};

template <>
struct OprLoadDumpImplV2<opr::Convolution, 0>
        : public WithPolicyOprLoadDumpImpl<
                  opr::Convolution, MakeConvCaller2<megdnn::Convolution>,
                  megdnn::Convolution> {};
template <>
struct OprLoadDumpImplV2<opr::ConvolutionBackwardData, 0>
        : public WithPolicyOprLoadDumpImpl<
                  opr::ConvolutionBackwardData, MakeConvCaller2<megdnn::Convolution>,
                  megdnn::Convolution, MakeConvCaller3<megdnn::Convolution>> {};
template <>
struct OprLoadDumpImplV2<opr::ConvolutionBackwardFilter, 0>
        : public WithPolicyOprLoadDumpImpl<
                  opr::ConvolutionBackwardFilter, MakeConvCaller3<megdnn::Convolution>,
                  megdnn::Convolution> {};

template <>
struct OprLoadDumpImplV2<opr::Convolution3D, 0>
        : public WithPolicyOprLoadDumpImpl<
                  opr::Convolution3D, MakeConvCaller2<megdnn::Convolution3D>,
                  megdnn::Convolution3D, MakeConvCallerEmpty<megdnn::Convolution3D>,
                  MakeConvCallerEmpty<megdnn::Convolution3D>,
                  megdnn::param::Convolution3D> {};
template <>
struct OprLoadDumpImplV2<opr::Convolution3DBackwardData, 0>
        : public WithPolicyOprLoadDumpImpl<
                  opr::Convolution3DBackwardData,
                  MakeConvCaller2<megdnn::Convolution3D>, megdnn::Convolution3D,
                  MakeConvCaller3<megdnn::Convolution3D>,
                  MakeConvCallerEmpty<megdnn::Convolution3D>,
                  megdnn::param::Convolution3D> {};
template <>
struct OprLoadDumpImplV2<opr::Convolution3DBackwardFilter, 0>
        : public WithPolicyOprLoadDumpImpl<
                  opr::Convolution3DBackwardFilter,
                  MakeConvCaller3<megdnn::Convolution3D>, megdnn::Convolution3D,
                  MakeConvCallerEmpty<megdnn::Convolution3D>,
                  MakeConvCallerEmpty<megdnn::Convolution3D>,
                  megdnn::param::Convolution3D> {};
template <>
struct OprLoadDumpImplV2<opr::ConvBiasForward, 0>
        : public WithPolicyOprLoadDumpImpl<
                  opr::ConvBiasForward, MakeConvCaller2<megdnn::ConvBiasForward>,
                  megdnn::ConvBiasForward, MakeConvCaller3<megdnn::ConvBiasForward>,
                  MakeConvCaller4<megdnn::ConvBiasForward>, megdnn::param::ConvBias> {};
template <>
struct OprLoadDumpImplV2<opr::BatchConvBiasForward, 0>
        : public WithPolicyOprLoadDumpImpl<
                  opr::BatchConvBiasForward,
                  MakeConvCaller2<megdnn::BatchConvBiasForward>,
                  megdnn::BatchConvBiasForward,
                  MakeConvCaller3<megdnn::BatchConvBiasForward>,
                  MakeConvCaller4<megdnn::BatchConvBiasForward>,
                  megdnn::param::BatchConvBias> {};

template <>
struct OprLoadDumpImplV2<opr::LocalShare, 0>
        : public WithPolicyOprLoadDumpImpl<
                  opr::LocalShare, MakeLocalShareCaller2<megdnn::LocalShare>,
                  megdnn::LocalShare, MakeLocalShareCallerEmpty<megdnn::LocalShare>,
                  MakeLocalShareCallerEmpty<megdnn::LocalShare>,
                  megdnn::param::LocalShare> {};
template <>
struct OprLoadDumpImplV2<opr::LocalShareBackwardData, 0>
        : public WithPolicyOprLoadDumpImpl<
                  opr::LocalShareBackwardData,
                  MakeLocalShareCaller3<megdnn::LocalShare>, megdnn::LocalShare,
                  MakeLocalShareCallerEmpty<megdnn::LocalShare>,
                  MakeLocalShareCallerEmpty<megdnn::LocalShare>,
                  megdnn::param::LocalShare> {};
template <>
struct OprLoadDumpImplV2<opr::LocalShareBackwardFilter, 0>
        : public WithPolicyOprLoadDumpImpl<
                  opr::LocalShareBackwardFilter,
                  MakeLocalShareCaller3<megdnn::LocalShare>, megdnn::LocalShare,
                  MakeLocalShareCallerEmpty<megdnn::LocalShare>,
                  MakeLocalShareCallerEmpty<megdnn::LocalShare>,
                  megdnn::param::LocalShare> {};
template <>
struct OprLoadDumpImplV2<opr::DeformableConvForward, 0>
        : public WithPolicyOprLoadDumpImpl<
                  opr::DeformableConvForward,
                  MakeConvCaller4<megdnn::DeformableConvForward>, megdnn::Convolution> {
};
template <>
struct OprLoadDumpImplV2<opr::DeformableConvBackwardData, 0>
        : public WithPolicyOprLoadDumpImpl<
                  opr::DeformableConvBackwardData,
                  MakeConvCaller5<megdnn::DeformableConvBackwardData>,
                  megdnn::Convolution> {};
template <>
struct OprLoadDumpImplV2<opr::DeformableConvBackwardFilter, 0>
        : public WithPolicyOprLoadDumpImpl<
                  opr::DeformableConvBackwardFilter,
                  MakeConvCaller5<megdnn::DeformableConvBackwardFilter>,
                  megdnn::Convolution> {};

}  // namespace serialization

namespace opr {
#define SERGE_OPR_V2_CONVERTER(_cls, _arity, _converter) \
    MGB_SEREG_OPR_V2(_cls, _arity, _converter, VERSION_2, CURRENT_VERSION);

#define SERGE_OPR_V2_NO_CONVERTER(_cls, _arity) \
    MGB_SEREG_OPR_V2(_cls, _arity, nullptr, VERSION_2, CURRENT_VERSION);

SERGE_OPR_V2_CONVERTER(
        Softmax, 1,
        (mgb::serialization::OprLoadDumpImplV2<opr::Softmax, 1>::replace_opr));

SERGE_OPR_V2_NO_CONVERTER(ConvBiasForward, 0)
SERGE_OPR_V2_NO_CONVERTER(BatchConvBiasForward, 0);

SERGE_OPR_V2_NO_CONVERTER(Convolution, 0)
SERGE_OPR_V2_NO_CONVERTER(ConvolutionBackwardData, 0)
SERGE_OPR_V2_NO_CONVERTER(ConvolutionBackwardFilter, 0)

SERGE_OPR_V2_NO_CONVERTER(Convolution3D, 0);
SERGE_OPR_V2_NO_CONVERTER(Convolution3DBackwardData, 0);
SERGE_OPR_V2_NO_CONVERTER(Convolution3DBackwardFilter, 0);

SERGE_OPR_V2_NO_CONVERTER(LocalShareForward, 0);
SERGE_OPR_V2_NO_CONVERTER(LocalShareBackwardData, 0);
SERGE_OPR_V2_NO_CONVERTER(LocalShareBackwardFilter, 0);

SERGE_OPR_V2_NO_CONVERTER(DeformableConvForward, 0);
SERGE_OPR_V2_NO_CONVERTER(DeformableConvBackwardData, 0);
SERGE_OPR_V2_NO_CONVERTER(DeformableConvBackwardFilter, 0);

#undef SERGE_OPR_V2_CONVERTER
#undef SERGE_OPR_V2_NO_CONVERTER
}  // namespace opr

}  // namespace mgb

#endif

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
