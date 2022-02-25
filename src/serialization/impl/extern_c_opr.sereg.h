#include "megbrain/serialization/extern_c_opr_io.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {

namespace serialization {
template <>
struct OprLoadDumpImpl<opr::ExternCOprRunner, 0> {
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr) {
        opr::ExternCOprRunner::dump(ctx, opr);
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        return opr::ExternCOprRunner::load(ctx, inputs, config);
    }
};

using ExternCOprRunner = opr::ExternCOprRunner;
MGB_SEREG_OPR(ExternCOprRunner, 0);
MGB_REG_OPR_SHALLOW_COPY(ExternCOprRunner, ExternCOprRunner::shallow_copy);
}  // namespace serialization
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
