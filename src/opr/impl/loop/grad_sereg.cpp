#include "./grad_sereg.h"
#include "./grad.h"
#include "./impl.h"
#include "megbrain/opr/internal/param_tag_defs.h"
#include "megbrain/serialization/sereg.h"

using namespace mgb;
using namespace mgb::serialization;
using namespace mgb::opr::intl;

namespace mgb {
namespace opr {
namespace intl {

//! this is a friend class of LoopImpl and LoopGrad
class LoopGradSerializer {
    template <class Opr>
    static cg::OperatorNodeBase* wrap_shallow_copy(
            const OprShallowCopyContext& ctx, const cg::OperatorNodeBase& opr,
            const VarNodeArray& inputs, const OperatorNodeConfig& config) {
        MGB_MARK_USED_VAR(ctx);
        return opr.cast_final_safe<Opr>().shallow_copy(inputs, config);
    }

public:
    static void reg_all();
};

}  // namespace intl
}  // namespace opr
}  // namespace mgb

void LoopGradSerializer::reg_all() {
#define REG(_opr) MGB_REG_OPR_SHALLOW_COPY_IMPL(_opr, wrap_shallow_copy<_opr>)

    REG(LoopGrad);
    REG(LoopGrad::AssignorGradOpr);
    REG(LoopImpl::DepTensorUpdator);

#undef REG
}

void LoopGradSerializerReg::entry() {
    LoopGradSerializer::reg_all();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
