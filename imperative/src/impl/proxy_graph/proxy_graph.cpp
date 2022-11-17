#include "../mgb_cg_impl.h"
#include "./mini_graph.h"
#include "megbrain/opr/io.h"

using LayoutConstraintLevel = mgb::cg::VarNodeMemManager::LayoutConstraintLevel;
using LayoutConstraintCallback = mgb::VarNode::LayoutConstraintCallback;
namespace mgb::imperative::proxy_graph {
MGB_DYN_TYPE_OBJ_FINAL_IMPL(ProxyGraph::InputPlaceholder);

thread_local std::unique_ptr<ProxyGraphTypeI> ProxyGraphTypeI::sm_instance = {};
}  // namespace mgb::imperative::proxy_graph

namespace mgb::imperative::proxy_graph_detail {

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto ret = proxy_graph::ProxyGraphTypeI::inst().infer_output_attrs_fallible(
            def, inputs);
    return ret;
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, SmallVector<TensorPtr> inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto ret = proxy_graph::ProxyGraphTypeI::inst().apply_on_physical_tensor(
            def, inputs, output_descs, validated);
    return ret;
}

std::unordered_map<size_t, SmallVector<LayoutConstraintCallback>>
        input_layout_constraints_cache;

SmallVector<LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    auto get_input_layout_constraint_hash_key =
            [](const OpDef& def, const SmallVector<TensorPtr>& inputs) {
                XXHash state;
                size_t length = 0, data[1 + inputs.size()];
                data[length++] = def.hash();
                for (auto&& i : inputs) {
                    data[length++] = mgb::hash(i->comp_node());
                }
                state.update(data, length * sizeof(size_t));
                return state.digest();
            };
    auto hash_key = get_input_layout_constraint_hash_key(def, inputs);
    auto&& iter = input_layout_constraints_cache.find(hash_key);
    if (iter != input_layout_constraints_cache.end()) {
        return iter->second;
    }
    static cg::ComputingGraphImpl* graph =
            imperative::ResourceManager::create_global<cg::ComputingGraphImpl>();
    VarNodeArray vinputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        OperatorNodeConfig config;
        auto layout = inputs[i]->layout();
        layout.init_contiguous_stride();
        vinputs[i] = graph->insert_opr(std::make_unique<mgb::opr::SharedDeviceTensor>(
                                               *graph,
                                               std::make_shared<DeviceTensorND>(
                                                       inputs[i]->comp_node(), layout),
                                               false, config))
                             ->output(0);
    }
    auto&& opr = OpDef::apply_on_var_node(def, vinputs)[0]->owner_opr();
    opr->add_input_layout_constraint();

    SmallVector<LayoutConstraintCallback> res(inputs.size());
    auto& mem_mgr = graph->var_node_mem_manager();
    for (size_t i = 0; i < vinputs.size(); ++i) {
        auto& trait = mem_mgr.get_var_node_mem_trait(vinputs[i]);
        switch (trait.layout_constraint.level) {
            case LayoutConstraintLevel::CONTIG:
                res[i] = [](const TensorLayout& layout) {
                    return layout.is_contiguous();
                };
                break;
            case LayoutConstraintLevel::MONOTONE:
                res[i] = [&trait](const TensorLayout& layout) {
                    if (!layout.is_abs_monotonous_allow_brdcst()) {
                        return false;
                    }
                    for (auto&& i : trait.layout_constraint.custom)
                        if (!i(layout))
                            return false;
                    return true;
                };
                break;
            case LayoutConstraintLevel::NONE:
                if (!trait.layout_constraint.custom.empty()) {
                    res[i] = [&trait](const TensorLayout& layout) {
                        for (auto&& i : trait.layout_constraint.custom)
                            if (!i(layout))
                                return false;
                        return true;
                    };
                }
                break;
            default:
                mgb_throw(InternalError, "invalid layout_constraint_level");
        }
    }
    input_layout_constraints_cache.emplace(hash_key, res);
    return res;
}

}  // namespace mgb::imperative::proxy_graph_detail
