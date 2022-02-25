#include "megbrain/opr/tensor_gen.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace opr {
MGB_SEREG_OPR(Alloc, 1);
MGB_SEREG_OPR(Linspace, 3);
MGB_SEREG_OPR(Eye, 1);
}  // namespace opr
}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
