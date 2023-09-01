#include "megbrain/opr/rand.h"
#include "megbrain/serialization/oss_opr_load_dump.h"
#include "megbrain/serialization/sereg.h"
#include "megdnn/opr_param_defs.h"

namespace mgb {

namespace serialization {

template <>
struct OprMaker<opr::ShuffleRNG, 1> {
    using Opr = opr::ShuffleRNG;
    using Param = Opr::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& inputs, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        auto out = Opr::make(inputs[0], param, config);
        return out[0].node()->owner_opr();
    }
};

// OprMaker in MGB_SEREG_OPR only support unique output opr
template <>
struct OprMaker<opr::DropoutForward, 1> {
    using Param = opr::DropoutForward::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        return opr::DropoutForward::make(i[0], param, config)[0].node()->owner_opr();
    }
};

template <>
struct OprMaker<opr::MultiHeadAttn, 0> {
    using Param = opr::MultiHeadAttn::Param;
    using InputType = Param::TensorCombinationType;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        if (i.size() == 7) {
            mgb_assert(InputType::ALL == param.tensor_combination_type);
            return opr::MultiHeadAttn::make(
                           i[0], i[1], i[2], i[3], i[4], i[5], i[6], param, config)[0]
                    .node()
                    ->owner_opr();
        } else if (i.size() == 6) {
            mgb_assert(InputType::ONLY_BIASKV == param.tensor_combination_type);
            return opr::MultiHeadAttn::make(
                           i[0], i[1], i[2], i[3], i[4], i[5], param, config)[0]
                    .node()
                    ->owner_opr();
        } else if (i.size() == 5) {
            mgb_assert(InputType::ONLY_MASK == param.tensor_combination_type);
            return opr::MultiHeadAttn::make(
                           i[0], i[1], i[2], i[3], i[4], param, config)[0]
                    .node()
                    ->owner_opr();
        } else {
            mgb_assert(InputType::NONE == param.tensor_combination_type);
            return opr::MultiHeadAttn::make(i[0], i[1], i[2], i[3], param, config)[0]
                    .node()
                    ->owner_opr();
        }
    }
};

// OprMaker in MGB_SEREG_OPR only support unique output opr
template <>
struct OprMaker<opr::MultiHeadAttnBackward, 0> {
    using Param = opr::MultiHeadAttnBackward::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);

        if (i.size() == 8)
            return opr::MultiHeadAttnBackward::make(
                           i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], param,
                           config)[0]
                    .node()
                    ->owner_opr();
        else
            return opr::MultiHeadAttnBackward::make(
                           i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], param,
                           config)[0]
                    .node()
                    ->owner_opr();
    }
};

template <>
struct OprLoadDumpImpl<opr::DropoutForward, 1> {
    using Opr = opr::DropoutForward;
    using PersisParam = opr::DropoutForward::Param;
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr) {
        ctx.write_param<PersisParam>(opr.cast_final_safe<Opr>().param());
    }

    static cg::OperatorNodeBase* replace_opr(
            cg::OperatorNodeBase* opr, const VarNodeArray& inputs) {
        float prob = opr->cast_final_safe<Opr>().param().drop_prob;
        uint64_t seed = opr->cast_final_safe<Opr>().param().seed;
        auto input_var = inputs[0];
        auto cn = inputs[0]->comp_node();
        OperatorNodeConfig config{cn};
        auto get_shape_out = opr::GetVarShape::make(input_var);
        auto uniform_out = opr::UniformRNG::make(get_shape_out, {seed});
        auto prob_var = opr::ImmutableTensor::make(
                *input_var->owner_graph(), DTypeScalar(prob), config);
        auto mask_out = opr::Elemwise::make(
                {prob_var, uniform_out}, {megdnn::Elemwise::Mode::LT});
        auto as_bool_out = opr::TypeCvt::make(mask_out, dtype::Bool(), {});
        auto as_fp32_out = opr::TypeCvt::make(as_bool_out, dtype::Float32(), {});
        auto drop_out = opr::Elemwise::make(
                {input_var, as_fp32_out}, {megdnn::Elemwise::Mode::MUL});
        auto inv_prob_var = opr::ImmutableTensor::make(
                *input_var->owner_graph(), DTypeScalar(1 / (1 - prob)), config);
        auto out = opr::Elemwise::make(
                {drop_out, inv_prob_var}, {megdnn::Elemwise::Mode::MUL});
        return out.node()->owner_opr();
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        return OprMaker<opr::DropoutForward, 1>::make(
                ctx.read_param<PersisParam>(), inputs, ctx.graph(), config);
    }
};

}  // namespace serialization

namespace opr {

using UniformRNGV1 = opr::UniformRNG;
MGB_SEREG_OPR(UniformRNGV1, 1);
using GaussianRNGV1 = opr::GaussianRNG;
MGB_SEREG_OPR(GaussianRNGV1, 1);
MGB_SEREG_OPR(GammaRNG, 2);
MGB_SEREG_OPR(PoissonRNG, 1);
MGB_SEREG_OPR(PermutationRNG, 1);
MGB_SEREG_OPR(BetaRNG, 2);
MGB_SEREG_OPR(ShuffleRNG, 1);
MGB_SEREG_OPR(ShuffleRNGBackward, 3);
MGB_SEREG_OPR_WITH_CONVERTER(
        Dropout, 1,
        (mgb::serialization::OprLoadDumpImpl<opr::DropoutForward, 1>::replace_opr));
MGB_SEREG_OPR(DropoutBackward, 2);
MGB_SEREG_OPR(MultiHeadAttn, 0);
MGB_SEREG_OPR(MultiHeadAttnBackward, 0);

}  // namespace opr
}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
