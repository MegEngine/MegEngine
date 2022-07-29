#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace serialization {
template <>
struct OprMaker<opr::ElemwiseMultiType, 0>
        : public OprMakerVariadic<opr::ElemwiseMultiType> {};

template <>
struct OprLoadDumpImplV2<opr::ElemwiseMultiType, 0> {
    using Opr = opr::ElemwiseMultiType;
    using PersisParam = opr::ElemwiseMultiType::Param;
    using PersisElemwseiParam = opr::Elemwise::Param;
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr) {
        ctx.write_param<PersisParam>(opr.cast_final_safe<Opr>().param());
    }

    static cg::OperatorNodeBase* replace_opr(
            cg::OperatorNodeBase* opr, const VarNodeArray& inputs) {
        auto mode = opr->cast_final_safe<Opr>().param().mode;
        auto change_to_elemwise_mode = [&](PersisParam::Mode multitype_mode) {
            if (multitype_mode == PersisParam::Mode::EQ) {
                return PersisElemwseiParam::Mode::EQ;
            } else if (multitype_mode == PersisParam::Mode::LT) {
                return PersisElemwseiParam::Mode::LT;
            } else if (multitype_mode == PersisParam::Mode::LEQ) {
                return PersisElemwseiParam::Mode::LEQ;
            }
            mgb_assert(0, "no supported model.");
        };
        if (PersisParam::Mode::EQ == mode || PersisParam::Mode::LT == mode ||
            PersisParam::Mode::LEQ == mode) {
            auto elemwise_mode = change_to_elemwise_mode(mode);
            auto elemiwse_out = opr::Elemwise::make(inputs, {elemwise_mode});
            return opr::TypeCvt::make(elemiwse_out, dtype::Bool()).node()->owner_opr();
        } else if (PersisParam::Mode::NEQ == mode) {
            auto elemiwse_out =
                    opr::Elemwise::make(inputs, {PersisElemwseiParam::Mode::EQ});
            auto bool_out = opr::TypeCvt::make(elemiwse_out, dtype::Bool());
            return opr::Elemwise::make({bool_out}, {PersisElemwseiParam::Mode::NOT})
                    .node()
                    ->owner_opr();
        } else if (PersisParam::Mode::ISNAN == mode) {
            auto elemiwse_out = opr::Elemwise::make(
                    {inputs[0], inputs[0]}, {PersisElemwseiParam::Mode::EQ});
            auto bool_out = opr::TypeCvt::make(elemiwse_out, dtype::Bool());
            return opr::Elemwise::make({bool_out}, {PersisElemwseiParam::Mode::NOT})
                    .node()
                    ->owner_opr();
        } else if (PersisParam::Mode::ISINF == mode) {
            auto input_var = SymbolVar{inputs[0]};
            auto inf_var = input_var.make_scalar(INFINITY);
            auto float_out = opr::TypeCvt::make(inputs[0], dtype::Float32());
            auto elemiwse_out = opr::Elemwise::make(
                    {float_out, inf_var}, {PersisElemwseiParam::Mode::EQ});
            return opr::TypeCvt::make(elemiwse_out, dtype::Bool()).node()->owner_opr();
        }
        return opr;
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        return OprMaker<opr::ElemwiseMultiType, 0>::make(
                ctx.read_param<PersisParam>(), inputs, ctx.graph(), config);
    }
};

}  // namespace serialization

namespace opr {
MGB_SEREG_OPR_CONDITION(ElemwiseMultiType, 0, false);
MGB_SEREG_OPR_V2(
        ElemwiseMultiType, 0,
        (mgb::serialization::OprLoadDumpImplV2<opr::ElemwiseMultiType, 0>::replace_opr),
        VERSION_1, VERSION_1);
MGB_SEREG_OPR(AffineInt, 3);
}  // namespace opr
}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
