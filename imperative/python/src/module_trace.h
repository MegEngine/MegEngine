#pragma once

#include <list>
#include "megbrain/imperative/transformations/trace.h"
#include "megbrain/imperative/utils/map.h"

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
    inline static WeakKeyMap<ValueWeakRef, py::object> module_trace_info_map;
    ModuleTraceTransformation(py::function hook_fn) : m_hook_fn(hook_fn) {}
    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override {
        if (op.is<ApplyOp>() && m_enabled > 0) {
            auto outputs = apply_module_trace_hook(op.cast<ApplyOp>().op(), inputs);
            return outputs;
        } else if (op.is<RenameValue>()) {
            auto outputs = imperative::apply(op, inputs);
            if (auto module_trace_info = module_trace_info_map.try_get(inputs[0])) {
                if (module_trace_info->ptr()) {
                    auto node = module_trace_info.value();
                    module_trace_info_map[outputs[0]] = module_trace_info.value();
                }
            }
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
