#include "megbrain/opr/loop.h"

#include "./forward_sereg.h"

namespace mgb {
namespace opr {
MGB_SEREG_OPR_INTL_CALL_ENTRY(Loop, serialization::LoopSerializerReg);
MGB_REG_OPR_SHALLOW_COPY(Loop, serialization::opr_shallow_copy_loop);
}  // namespace opr
}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
