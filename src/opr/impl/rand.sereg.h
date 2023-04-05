#include "megbrain/opr/rand.h"
#include "megbrain/serialization/sereg.h"

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
    static cg::OperatorNodeBase* make(
            const Param& param, const cg::VarNodeArray& i, ComputingGraph& graph,
            const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(graph);
        return opr::MultiHeadAttn::make(i[0], i[1], i[2], i[3], param, config)[0]
                .node()
                ->owner_opr();
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

        return opr::MultiHeadAttnBackward::make(
                       i[0], i[1], i[2], i[3], i[4], i[5], param, config)[0]
                .node()
                ->owner_opr();
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
MGB_SEREG_OPR(Dropout, 1);
MGB_SEREG_OPR(DropoutBackward, 2);
MGB_SEREG_OPR(MultiHeadAttn, 0);
MGB_SEREG_OPR(MultiHeadAttnBackward, 0);

}  // namespace opr
}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
