#include "megbrain/imperative/transformation.h"

namespace mgb {
namespace imperative {

TransformationContext& Transformation::get_context() {
    thread_local TransformationContext tl_context;
    return tl_context;
}

}  // namespace imperative
}  // namespace mgb
