#pragma once

#include <list>
#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/utils/map.h"

#include "./tensor.h"

namespace mgb::imperative::python {

namespace py = pybind11;

class CreateExternalWrapper final : public OperatorImpl<CreateExternalWrapper> {
private:
    py::object m_object;
    CompNode m_device;

public:
    CreateExternalWrapper(py::object obj, CompNode device)
            : m_object(obj), m_device(device) {}

    py::object object() const { return m_object; }

    CompNode device() const { return m_device; }

    std::string raw_type() const { return "CreateExternalWrapper"; }

    std::string to_string() const { return "CreateExternalWrapper"; };
};

class GetExternalVal final
        : public OperatorImpl<GetExternalVal, Operator::GetAttrLike> {
public:
    std::string to_string() const { return "GetExternalVal"; };
    std::string raw_type() const { return "GetExternalVal"; }
};

class PyobjectStorage {
private:
    py::object m_object;

public:
    PyobjectStorage() = default;
    PyobjectStorage(py::object object) : m_object(object) {}
    py::object object() const { return m_object; }
    std::string to_string() const { return "PyobjectStorage"; }
};

class PyobjectValue final : public PrimitiveValue<PyobjectValue, PyobjectStorage> {
public:
    using PrimitiveValue::PrimitiveValue;

    std::string to_string() const override { return PyobjectStorage::to_string(); }
};

class ExternalValue final : public ObjectValue<ExternalValue> {
private:
    py::object m_obj;
    mutable CompNodeValue::ref_t m_device;

public:
    ExternalValue(py::object obj, CompNode device)
            : m_obj(obj), m_device(CompNodeValue::make(device)) {}

    py::object object() const { return m_obj; }

    CompNodeValue::ref_t device() const { return m_device; }

    std::string to_string() const override { return "ExternalValue"; }

    void clear() override {}
};

class ExternalConvertTransformation final : public Transformation {
private:
    py::function m_hook_fn;
    int m_enabled = 0;
    ObjectType<ExternalValue> m_value_type{"ExternalValue"};

public:
    ValueRefList apply_external_imperative_hook(
            const Operator& op, Span<ValueRef> input_values) {
        for (int i = 0; i < input_values.size(); i++) {
            if (auto* val = input_values[i].as(m_value_type)) {
                CompNode cn = *(val->device());
                py::object fn_res = m_hook_fn(val->object(), cn);
                auto* tw = TensorWrapper::try_cast(fn_res.ptr());
                mgb_assert(tw, "expect Tensor");
                auto external_input = input_values[i].as_ref(m_value_type);
                external_input.reset(tw->m_tensor->data());
            }
        }
        auto outputs = imperative::apply(op, input_values);
        return outputs;
    }

    ExternalConvertTransformation(py::function hook_fn) : m_hook_fn(hook_fn) {}
    ValueRefList apply_transformation(
            const Operator& op, Span<ValueRef> inputs) override {
        if (!m_enabled) {
            return imperative::apply(op, inputs);
        }
        bool has_external_inp = false;
        if (auto* obj_value = op.as<CreateExternalWrapper>()) {
            return m_value_type.make(obj_value->object(), obj_value->device());
        }
        for (auto&& input : inputs) {
            if (input.is(m_value_type)) {
                has_external_inp = true;
                break;
            }
        }
        if (!has_external_inp) {
            return imperative::apply(op, inputs);
        } else if (op.is<GetExternalVal>()) {
            py::object m_object = inputs.item().cast(m_value_type).object();
            PyobjectStorage inp_obj = PyobjectStorage(m_object);
            return {PyobjectValue::make(inp_obj)};
        } else if (op.is<RenameValue>()) {
            return {inputs[0]};
        } else if (auto* get_attr = op.as<GetAttr>()) {
            auto& input = inputs.item().cast(m_value_type);
            ValueRefList outputs;
            switch (get_attr->attr()) {
                case GetAttr::Device:
                    outputs = {input.device()};
                    break;
                default:
                    outputs = apply_external_imperative_hook(op, inputs);
                    break;
            }
            return outputs;
        } else {
            auto outputs = apply_external_imperative_hook(op, inputs);
            return outputs;
        }
    }

    void enable() { m_enabled = 1; }

    void disable() { m_enabled = 0; }

    bool enabled() const { return m_enabled; }

    ValueRef unwrap(ValueRef value) override { return value; }

    const Type<ExternalValue>& value_type() const { return m_value_type; }

    std::string name() const override { return "ExternalConvertTransformation"; }
};

}  // namespace mgb::imperative::python