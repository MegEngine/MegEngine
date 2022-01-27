/**
 * \file imperative/python/src/module_trace.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/transformations/trace.h"
#include "megbrain/imperative/utils/map.h"
#include "megbrain/imperative/utils/stats.h"

#include "./tensor.h"

namespace mgb::imperative::python {

namespace py = pybind11;

class ModuleTraceTransformation final : public Transformation {
private:
    py::function m_hook_fn;
    int m_enabled = 0;

    ValueRefList apply_module_trace_hook(const OpDef& op, Span<ValueRef> input_values) {
        py::list input_tws;
        for (auto&& input_value : input_values) {
            input_tws.append(TensorWrapper::make(py_tensor_type, input_value));
        }
        py::list output_tws = m_hook_fn(py::cast(op.shared_from_this()), *input_tws);
        ValueRefList outputs(output_tws.size());
        auto it = outputs.begin();
        for (auto&& output_tw : output_tws) {
            *(it++) = TensorWrapper::try_cast(output_tw.ptr())->m_tensor->data();
        }
        return outputs;
    }

public:
    ModuleTraceTransformation(py::function hook_fn) : m_hook_fn(hook_fn) {}
    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override {
        if (op.is<ApplyOp>() && m_enabled > 0) {
            auto outputs = apply_module_trace_hook(op.cast<ApplyOp>().op(), inputs);
            return outputs;
        } else {
            return imperative::apply(op, inputs);
        }
    }

    void enable() { m_enabled = 1; }

    void disable() { m_enabled = 0; }

    bool enabled() const { return m_enabled; }

    ValueRef unwrap(ValueRef value) override { return value; }

    std::string name() const override { return "ModuleTraceTransformation"; }
};

}  // namespace mgb::imperative::python
