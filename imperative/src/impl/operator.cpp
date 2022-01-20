#include "megbrain/imperative/operator.h"

namespace mgb {
namespace imperative {

ValueRefList Operator::fallback(Span<ValueRef> inputs) const {
    mgb_throw(MegBrainError, "no fallback implementation for %s", to_string().c_str());
}

size_t Operator::register_type(std::type_index type) {
    auto& types = const_cast<std::vector<std::type_index>&>(registered_types());
    types.push_back(type);
    return types.size() - 1;
}

const std::vector<std::type_index>& Operator::registered_types() {
    static std::vector<std::type_index> sm_registered_types;
    return sm_registered_types;
}

}  // namespace imperative
}  // namespace mgb
