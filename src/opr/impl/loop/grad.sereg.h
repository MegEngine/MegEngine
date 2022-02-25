#include "./grad_sereg.h"
#include "megbrain/opr/loop.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace opr {
namespace intl {
MGB_SEREG_OPR_INTL_CALL_ENTRY(LoopGrad, serialization::LoopGradSerializerReg);
}  // namespace intl
}  // namespace opr
}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
