#include "./imperative_rt.h"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <future>
#include <unordered_map>
#include <variant>

#include "./common.h"
#include "./helper.h"
#include "megbrain/imperative.h"
#include "megbrain/imperative/interpreter.h"
#include "megbrain/imperative/ops/opr_attr.h"

namespace py = pybind11;

using namespace mgb;
using namespace imperative;
using namespace interpreter;

void init_imperative_rt(py::module m) {
    auto make_backward_graph = [](const OpDef& def,
                                  const SmallVector<LogicalTensorDesc>& inputs,
                                  const SmallVector<bool>& input_requires_grad,
                                  const SmallVector<bool>& output_has_grad) {
        auto result = OpDef::make_backward_graph(
                def, inputs, input_requires_grad, output_has_grad);
        return std::make_tuple("backward_graph", result.input_mask, result.output_mask);
    };
    m.def("make_backward_graph", make_backward_graph);
}
