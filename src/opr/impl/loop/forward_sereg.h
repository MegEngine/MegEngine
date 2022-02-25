#pragma once

#include "megbrain/opr/loop.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace serialization {
template <>
struct OprLoadDumpImpl<opr::Loop, 0> {
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr);
    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config);
};

struct LoopSerializerReg {
    //! entry for registering serializers related to loop
    static void entry();
};

cg::OperatorNodeBase* opr_shallow_copy_loop(
        const OprShallowCopyContext& ctx, const cg::OperatorNodeBase& opr,
        const VarNodeArray& inputs, const OperatorNodeConfig& config);

}  // namespace serialization
}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
