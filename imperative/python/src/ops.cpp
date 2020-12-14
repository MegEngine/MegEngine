/**
 * \file imperative/python/src/ops.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./ops.h"

#include "megbrain/imperative.h"
#include "megbrain/imperative/ops/backward_graph.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/ops/autogen.h"

namespace py = pybind11;

namespace {
auto normalize_enum(const std::string& in) {
    std::string ret;
    for (auto&& c : in) {
        ret += toupper(c);
    }
    return ret;
}
} // anonymous namespace

void init_ops(py::module m) {
    using namespace mgb::imperative;

    py::class_<BackwardGraph, std::shared_ptr<BackwardGraph>, OpDef>(m, "BackwardGraph")
        .def("interpret", [](BackwardGraph& self, py::object pyf, py::object pyc,
                             const mgb::SmallVector<py::object>& inputs) {
                auto f = [pyf](OpDef& op, const mgb::SmallVector<py::object>& inputs) {
                    return py::cast<mgb::SmallVector<py::object>>(pyf(op.shared_from_this(), inputs));
                };
                auto c = [pyc](const TensorPtr& tensor) {
                    return pyc(tensor->dev_tensor());
                };
                return self.graph().interpret<py::object>(f, c, inputs);
            });

    #include "opdef.py.inl"
}
