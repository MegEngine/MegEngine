/**
 * \file imperative/python/src/tensor.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

#include <variant>

#include <string>
#include <unordered_map>
#include "megbrain/imperative/interpreter.h"
#include "pybind11/pybind11.h"

#include "./pyext17.h"
#include "megbrain/imperative/dispatch.h"
#include "megbrain/imperative/utils/span.h"

namespace mgb::imperative::python {

template <typename T, typename B = pybind11::object>
struct ObjectPtr : B {
    using B::B;
    T& operator*() { return reinterpret_cast<T&>(*B::ptr()); }
    T* operator->() { return reinterpret_cast<T*>(B::ptr()); }
};

}  // namespace mgb::imperative::python

namespace mgb::imperative::python {

extern interpreter::Interpreter::Channel* interpreter_for_py;
extern PyTypeObject* py_tensor_type;

struct Tensor : std::enable_shared_from_this<Tensor>, NonCopyableObj {
private:
    std::string m_name;
    ValueRef m_data;

public:
    using Handle = interpreter::Interpreter::Handle;

    inline explicit Tensor(ValueRef data) : m_data{data} {}

    ~Tensor() = default;

    inline std::shared_ptr<Tensor> copy() {
        auto ret = std::make_shared<Tensor>(m_data.unwrap());
        ret->m_name = m_name;
        return ret;
    }

    inline DType dtype() { return *data().dtype(); }
    inline CompNode comp_node() { return *data().device(); }
    inline std::optional<ValueShape> shape() {
        auto shape = data().shape();
        if (!shape) {
            return {};
        }
        return *shape;
    }
    inline HostValue::ref_t numpy() { return data().numpy(); }
    inline void reset(ValueRef value) {
        m_data = value;
        if (!m_name.empty()) {
            set_name(m_name);
        }
    }
    inline ValueRef data() { return m_data.unwrap(); }
    bool is_scalar() { return data().is_scalar(); }
    inline std::string name() { return m_name; }
    inline void set_name(std::string name) {
        m_name = name;
        if (!name.empty()) {
            auto output = imperative::apply(RenameValue(name), m_data)[0];
            m_data = output;
        }
    }
};

struct TensorWrapper {
public:
    std::shared_ptr<Tensor> m_tensor;

    inline TensorWrapper(std::shared_ptr<Tensor> tensor = {})
            : m_tensor(std::move(tensor)) {
        mgb_assert(tensor, "empty storage");
    }

    inline TensorWrapper(ValueRef value) : m_tensor(std::make_shared<Tensor>(value)) {}
    TensorWrapper(PyObject* args, PyObject* kwargs);
    ~TensorWrapper() = default;

    static constexpr auto tp_name = pybind11::detail::_("Tensor");

    using wrap_t = pyext17::wrap<TensorWrapper>;
    friend wrap_t;

    inline static TensorWrapper* cast(PyObject* obj) {
        return reinterpret_cast<wrap_t*>(obj)->inst();
    }
    inline static TensorWrapper* try_cast(PyObject* obj) {
        if (!wrap_t::type().isinstance(obj))
            return nullptr;
        return cast(obj);
    }
    inline ObjectPtr<TensorWrapper, pybind11::handle> self() {
        return wrap_t::pycast(this);
    }

    template <typename... Args>
    static ObjectPtr<Tensor> make(Args&&... args) {
        auto* op = wrap_t::cnew(std::forward<Args>(args)...);
        return pybind11::reinterpret_steal<ObjectPtr<Tensor>>(op);
    }

    template <typename... Args>
    static ObjectPtr<Tensor> make(PyTypeObject* pytype, Args&&... args) {
        auto* op = wrap_t::cnew_with_type(pytype, std::forward<Args>(args)...);
        return pybind11::reinterpret_steal<ObjectPtr<Tensor>>(op);
    }

    PyObject* shape();
    PyObject* dtype();
    PyObject* device();
    PyObject* numpy();
    void reset(PyObject*);
    PyObject* detach();
    PyObject* isscalar();
    PyObject* _dev_tensor();
    void _drop();
    PyObject* varnode();
    PyObject* recording();
    PyObject* copied();
    PyObject* module_trace_info();
    void set_module_trace_info(PyObject*);
    void _set_name(PyObject*);
    PyObject* _use_cnt() { return PyLong_FromSize_t(m_tensor.use_count()); };
    PyObject* _detail();
    void _watch();
};

struct PySymbolVar {
    cg::VarNode* m_node = nullptr;
    bool is_scalar = false;
    PySymbolVar() = default;
    PySymbolVar(VarNode* m) : m_node(m) {}
};

PyObject* py_apply(
        PyObject* self, PyObject* const* args, size_t nargs /* , PyObject* kwnames */);

void init_tensor(pybind11::module);

extern PyObject* cpp_apply_module_trace;

}  // namespace mgb::imperative::python

namespace pybind11::detail {

template <>
struct type_caster<mgb::imperative::python::TensorWrapper>
        : mgb::imperative::python::TensorWrapper::wrap_t::caster {};

}  // namespace pybind11::detail
