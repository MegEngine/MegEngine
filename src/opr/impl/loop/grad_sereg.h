#pragma once

#include "./grad.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace serialization {
struct LoopGradSerializerReg {
    //! entry for registering serializers related to loop grad
    static void entry();
};

cg::OperatorNodeBase* opr_shallow_copy_loop_grad(
        const OprShallowCopyContext& ctx, const cg::OperatorNodeBase& opr,
        const VarNodeArray& inputs, const OperatorNodeConfig& config);
}  // namespace serialization
}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
