#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/misc.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {

namespace serialization {

template <>
struct OprMaker<opr::Argsort, 1> {
    using Opr = opr::Argsort;
    using Param = Opr::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& inputs, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        auto out = Opr::make(inputs[0], param, config);
        return out[0].node()->owner_opr();
    }
};

template <>
struct OprMaker<opr::CondTake, 2> {
    using Opr = opr::CondTake;
    using Param = Opr::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& inputs, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        auto out = Opr::make(inputs[0], inputs[1], param, config);
        return out[0].node()->owner_opr();
    }
};

template <>
struct OprMaker<opr::NonZero, 1> {
    using Opr = opr::NonZero;
    using Param = Opr::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& inputs, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        auto out = Opr::make(inputs[0], param, config);
        return out.node()->owner_opr();
    }
};

template <>
struct OprMaker<opr::TopK, 2> {
    using Opr = opr::TopK;
    using Param = Opr::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& inputs, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        auto out = Opr::make(inputs[0], inputs[1], param, config);
        return out[0].node()->owner_opr();
    }
};

template <>
struct OprMaker<opr::CheckNonFinite, 0> {
    using Opr = opr::CheckNonFinite;
    using Param = Opr::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& inputs, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        auto out = Opr::make(inputs, param, config);
        return out[0].node()->owner_opr();
    }
};

template <>
struct OprLoadDumpImpl<opr::CheckNonFinite, 0> {
    using Opr = opr::CheckNonFinite;
    using PersisParam = opr::CheckNonFinite::Param;
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr) {
        ctx.write_param<PersisParam>(opr.cast_final_safe<Opr>().param());
    }

    static void dump_checknonfinite_v0(
            serialization::OprDumpContext& ctx, const cg::OperatorNodeBase& opr) {
        using EmptyParam = megdnn::param::Empty;
        if (ctx.config().compat_older_version == "8.14") {
            EmptyParam empty;
            ctx.write_param<EmptyParam>(empty);
        } else {
            ctx.write_param<PersisParam>(opr.cast_final_safe<Opr>().param());
        }
    }

    static cg::OperatorNodeBase* replace_opr(
            cg::OperatorNodeBase* opr, const VarNodeArray& inputs) {
        for (size_t i = 0; i < inputs.size(); i++) {
            opr->output(i)->add_flag(VarNode::Flag::VOLATILE_CONTENT);
        }
        return opr;
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        return OprMaker<opr::CheckNonFinite, 0>::make(
                ctx.read_param<PersisParam>(), inputs, ctx.graph(), config);
    }
};

template <>
struct OprMaker<opr::Where, 3> {
    using Opr = opr::Where;
    using Param = Opr::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& inputs, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        auto out = Opr::make(inputs[0], inputs[1], inputs[2], param, config);
        return out.node()->owner_opr();
    }
};

template <>
struct OprMaker<opr::WhereBackward, 4> {
    using Opr = opr::WhereBackward;
    using Param = opr::WhereBackward::Param;
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& inputs, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        auto out = Opr::make(inputs[0], inputs[1], param, config);
        return out[0].node()->owner_opr();
    }
};

template <>
struct OprLoadDumpImplV2<opr::Where, 3> {
    using Opr = opr::Where;
    using Mode = opr::Elemwise::Mode;
    using PersisParam = opr::Where::Param;
    using PersisWhereParam = opr::Where::Param;
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr) {
        ctx.write_param<PersisParam>(opr.cast_final_safe<Opr>().param());
    }

    static cg::OperatorNodeBase* replace_opr(
            cg::OperatorNodeBase* opr, const VarNodeArray& inputs) {
        auto mask = SymbolVar(inputs[0]);
        auto x = SymbolVar(inputs[1]);
        mask = opr::TypeCvt::make(mask, x.dtype());
        auto y = SymbolVar(inputs[2]);
        auto oup = opr::Elemwise::make({mask, x}, Mode::SWITCH_GT0);
        auto ksam = 1.0f - mask;
        oup = oup + opr::Elemwise::make({ksam, y}, Mode::SWITCH_GT0);
        return oup.node()->owner_opr();
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        return OprMaker<opr::Where, 3>::make(
                ctx.read_param<PersisParam>(), inputs, ctx.graph(), config);
    }
};

}  // namespace serialization

// OprMaker in MGB_SEREG_OPR only support unique output opr

namespace opr {

MGB_SEREG_OPR(Argmax, 1);
MGB_SEREG_OPR(Argmin, 1);
MGB_SEREG_OPR(Argsort, 1);
MGB_SEREG_OPR(ArgsortBackward, 3);
MGB_SEREG_OPR(CondTake, 2);
MGB_SEREG_OPR(NonZero, 1);
MGB_SEREG_OPR(TopK, 2);
MGB_SEREG_OPR_V1_WITH_CONVERTER(
        Where, 3, (mgb::serialization::OprLoadDumpImplV2<opr::Where, 3>::replace_opr),
        nullptr)
MGB_SEREG_OPR_V2_HASH_WITHOUT_TAIL_0(
        Where, 3, (mgb::serialization::OprLoadDumpImplV2<opr::Where, 3>::replace_opr),
        VERSION_1, VERSION_1);
MGB_SEREG_OPR(WhereBackward, 4)

//! current cumsum version
using CumsumV1 = opr::Cumsum;
MGB_SEREG_OPR(CumsumV1, 1);

#if MGB_CUDA
MGB_SEREG_OPR(NvOf, 1);
#endif
MGB_SEREG_OPR_V1_WITH_CONVERTER(
        CheckNonFinite, 0,
        (mgb::serialization::OprLoadDumpImpl<opr::CheckNonFinite, 0>::replace_opr),
        (mgb::serialization::OprLoadDumpImpl<
                opr::CheckNonFinite, 0>::dump_checknonfinite_v0));
MGB_SEREG_OPR_V2_HASH_WITHOUT_TAIL_0(CheckNonFinite, 0, nullptr, VERSION_1, VERSION_1);
}  // namespace opr
}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
