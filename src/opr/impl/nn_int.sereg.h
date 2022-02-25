#include "megbrain/opr/nn_int.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace serialization {
template <>
struct OprMaker<opr::ElemwiseMultiType, 0>
        : public OprMakerVariadic<opr::ElemwiseMultiType> {};

}  // namespace serialization

namespace opr {
MGB_SEREG_OPR(ElemwiseMultiType, 0);
MGB_SEREG_OPR(AffineInt, 3);
}  // namespace opr
}  // namespace mgb

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
