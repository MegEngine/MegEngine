#include "megbrain/imperative/proxy_graph_detail.h"
#include "./proxy_graph.h"
#include "megbrain/imperative/ops/autogen.h"

namespace mgb {
namespace imperative {
namespace proxy_graph_detail {

EncodedSubgraph make_backward_graph(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad) {
    return ProxyGraph::get_default_graph()->make_backward_graph(
            def, inputs, input_requires_grad, output_has_grad);
}

}  // namespace proxy_graph_detail
}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
