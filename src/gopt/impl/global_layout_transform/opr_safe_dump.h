#pragma once
#include "megbrain/graph.h"

namespace mgb {
namespace gopt {
namespace intl {
#define FOREACH_SUPPORTED_OPR_WITHOUT_EXECUTION_POLICY(cb)                       \
    cb(WarpPerspective) cb(Resize) cb(Elemwise) cb(ElemwiseMultiType) cb(Concat) \
            cb(PowC) cb(TypeCvt)

#define FOREACH_SUPPORTED_OPR_WITH_EXECUTION_POLICY(cb) \
    cb(Convolution) cb(ConvBiasForward) cb(ConvolutionBackwardData) cb(PoolingForward)

#define FOREACH_SUPPORTED_OPR(cb)                      \
    FOREACH_SUPPORTED_OPR_WITHOUT_EXECUTION_POLICY(cb) \
    FOREACH_SUPPORTED_OPR_WITH_EXECUTION_POLICY(cb)

std::string opr_safe_dump(const cg::OperatorNodeBase* opr);

}  // namespace intl
}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen
