#include "megbrain/opr/dnn/lrn.h"
#include "megbrain/graph/grad_impl.h"

#include "../internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;

namespace mgb {
namespace opr {
namespace intl {
template <>
struct MegDNNOprInitPostCtor<LRNForward> {
    static void apply(cg::OperatorNodeBase& opr) {
        opr.output(0)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    }
};

}  // namespace intl
}  // namespace opr
}  // namespace mgb

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LRNForward);
MEGDNN_OPR_INIT1(LRNForward, "lrn")

void LRNForward::scn_do_execute() {
    if (input(0)->dev_tensor().empty()) {
        mgb_assert(output(0)->dev_tensor().empty());
        return;
    }
    Super::scn_do_execute();
}

MAKE_NODE_PROP_WITH_ZERO_SHAPE_1(LRNForward, 0)

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(LRNForward) {
    mgb_assert(wrt_idx == 0);
    SymbolVar grad =
            LRNBackward::make(opr.input(0), opr.output(0), out_grad[0], opr.param());
    return grad.node();
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LRNBackward);
MEGDNN_OPR_INIT3(LRNBackward, "lrn_bwd", 0, true);

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
