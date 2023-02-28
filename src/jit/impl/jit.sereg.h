#include "megbrain/graph/helper.h"
#include "megbrain/jit/executor_opr.h"
#include "megbrain/serialization/opr_shallow_copy.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace jit {
cg::OperatorNodeBase* opr_shallow_copy_jit_executor_opr(
        const serialization::OprShallowCopyContext& ctx,
        const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
    auto&& opr = opr_.cast_final_safe<JITExecutor>();
    auto* shape_infer = opr.internal_graph().shape_infer();
    auto* value_infer = opr.internal_graph().value_infer();
    ThinHashMap<cg::VarNode*, cg::VarNode*> var_replace_map;
    mgb_assert(inputs.size() == opr.input().size());
    auto on_opr = [&](cg::OperatorNodeBase* opr) {
        auto&& inputs = opr->input();
        cg::VarNodeArray new_inputs;
        for (auto&& input : inputs) {
            new_inputs.push_back(var_replace_map.at(input));
        }
        auto* new_opr = opr;
        if (new_inputs != inputs) {
            auto&& config = opr->config();
            new_opr =
                    mgb::serialization::copy_opr_shallow(*opr, new_inputs, config, ctx);
        }
        auto&& outputs = opr->output();
        auto&& new_outputs = new_opr->output();
        mgb_assert(outputs.size() == new_outputs.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
            var_replace_map[outputs.at(i)] = new_outputs.at(i);
        }
    };
    cg::DepOprIter iter{on_opr};
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto input_opr = opr.input(i)->owner_opr();
        for (size_t j = 0; j < input_opr->output().size(); j++) {
            var_replace_map[input_opr->output(j)] = input_opr->output(j);
        }
        iter.set_visited(opr.input(i)->owner_opr());
        var_replace_map[opr.input(i)] = inputs[i];
    }
    if (shape_infer) {
        iter.add(shape_infer);
        shape_infer = var_replace_map.at(shape_infer);
    }
    if (value_infer) {
        iter.add(value_infer);
        value_infer = var_replace_map.at(value_infer);
    }
    auto internal_graph = opr.internal_graph_ptr();
    internal_graph = std::make_shared<InternalGraph>(
            internal_graph->output(), shape_infer, value_infer,
            internal_graph->placeholders());
    return JITExecutor::make(internal_graph, inputs, config).node()->owner_opr();
}

MGB_REG_OPR_SHALLOW_COPY(JITExecutor, opr_shallow_copy_jit_executor_opr);
}  // namespace jit
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
