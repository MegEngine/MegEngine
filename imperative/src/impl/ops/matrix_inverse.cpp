#include "../op_trait.h"

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/blas.h"

namespace mgb {
namespace imperative {

namespace {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<MatrixInverse>();
    mgb_assert(inputs.size() == 1);
    OperatorNodeConfig config{op.make_name()};
    return opr::MatrixInverse::make(inputs[0], {}, config);
}
OP_TRAIT_REG(MatrixInverse, MatrixInverse)
        .apply_on_var_node(apply_on_var_node)
        .fallback();
}  // anonymous namespace

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
