#include "megbrain/imperative/transformations/tangent.h"

namespace mgb {
namespace imperative {

ValueRefList TangentTransformation::apply_transformation(
        const Operator& op, Span<ValueRef> inputs) {
    if (auto* apply_op = op.as<ApplyOp>()) {
    }
    mgb_assert(false);
}

}  // namespace imperative
}  // namespace mgb
