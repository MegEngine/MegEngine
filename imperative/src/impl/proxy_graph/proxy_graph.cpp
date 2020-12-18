#include "./mini_graph.h"
// #include "../proxy_graph.h"

namespace mgb::imperative::proxy_graph {
    MGB_DYN_TYPE_OBJ_FINAL_IMPL(ProxyGraph::InputPlaceholder);

    thread_local std::unique_ptr<ProxyGraphTypeI> ProxyGraphTypeI::sm_instance = {};
} // namespace mgb::imperative::proxy_graph

namespace mgb::imperative::proxy_graph_detail {

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs) {
    auto ret = proxy_graph::ProxyGraphTypeI::inst().infer_output_attrs_fallible(def, inputs);
    // auto ref = ProxyGraph::get_default_graph()->infer_output_attrs_fallible(def, inputs);
    // auto& [a, _1] = ret;
    // auto& [b, _2] = ref;
    // if (a.size() != b.size()) mgb_trap();
    // for (size_t i = 0; i < a.size(); ++i) {
    //     if (a[i].layout.dtype != b[i].layout.dtype) mgb_trap();
    //     if (a[i].comp_node != b[i].comp_node) mgb_trap();
    //     if (!a[i].layout.eq_shape(b[i].layout)) mgb_trap();
    // }
    return ret;
}

} // namespace mgb::imperative::proxy_graph_detail
