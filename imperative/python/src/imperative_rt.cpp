/**
 * \file imperative/python/src/imperative_rt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./imperative_rt.h"

#include <future>
#include <variant>
#include <unordered_map>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include "megbrain/imperative.h"
#include "megbrain/imperative/interpreter.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "./helper.h"
#include "./common.h"

namespace py = pybind11;

using namespace mgb;
using namespace imperative;
using namespace interpreter;


namespace {

std::optional<std::tuple<std::shared_ptr<OpDef>, std::vector<bool>, std::vector<bool>>>
make_backward_graph(
    const OpDef& opdef, std::vector<LogicalTensorDesc> inputs,
    std::vector<bool> input_requires_grad,
    std::vector<bool> output_has_grad) {
    auto res = OpDef::make_backward_graph(opdef,
        SmallVector<LogicalTensorDesc>(inputs.begin(), inputs.end()),
        SmallVector<bool>(input_requires_grad.begin(), input_requires_grad.end()),
        SmallVector<bool>(output_has_grad.begin(), output_has_grad.end()));
    if (res.backward) {
        return std::optional<std::tuple<std::shared_ptr<OpDef>, std::vector<bool>, std::vector<bool>>>{
                std::in_place, res.backward, res.save_for_backward, res.input_has_grad};
    } else {
        return {};
    }
}
} // namespace

void init_imperative_rt(py::module m) {
    m.def("make_backward_graph", &make_backward_graph);
}
