/**
 * \file imperative/python/src/tensor.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <variant>

#include "megbrain/imperative/interpreter.h"
#include "pybind11/pybind11.h"

#include "./pyext17.h"

namespace mgb::imperative::python {

template<typename T, typename B = pybind11::object>
struct ObjectPtr : B {
    using B::B;
    T& operator*() {return reinterpret_cast<T&>(*B::ptr());}
    T* operator->() {return reinterpret_cast<T*>(B::ptr());}
};

} // namespace mgb::imperative::python

#include "./grad_info.h" // for struct GradInfo

namespace mgb::imperative::python {

struct TraceInfo {

};

extern std::unique_ptr<interpreter::Interpreter::Channel> interpreter_for_py;

class SharedHandle {
    using Handle = interpreter::Interpreter::Handle;
    static_assert(std::is_pointer_v<Handle>);
    std::shared_ptr<std::remove_pointer_t<Handle>> holder;

public:
    inline explicit SharedHandle(Handle handle) : holder(handle, [](auto* h){
        interpreter_for_py->del(h);
    }) {}
    SharedHandle(const SharedHandle&) = default;
    SharedHandle& operator=(const SharedHandle&) = default;
    SharedHandle(SharedHandle&&) = default;
    SharedHandle& operator=(SharedHandle&&) = default;

    inline Handle get() {return holder.get();}
};


struct Tensor : std::enable_shared_from_this<Tensor>, NonCopyableObj {
    using flags_t = uint64_t;

    struct Flags {
        static constexpr flags_t SCALAR = 1;
        static constexpr flags_t GRAD = 1 << 1;
        static constexpr flags_t TRACE = 1 << 2;
    };

    flags_t m_flags = 0;

    GradInfo m_grad_info;
    TraceInfo m_trace_info;
    SharedHandle m_handle;

    using Handle = interpreter::Interpreter::Handle;

    inline explicit Tensor(Handle handle) : m_handle(handle) {}
    inline explicit Tensor(SharedHandle handle) : m_handle(std::move(handle)) {}
    ~Tensor() = default;

    inline std::shared_ptr<Tensor> copy() {
        auto ret = std::make_shared<Tensor>(m_handle);
        ret->m_flags = m_flags;
        ret->m_grad_info = m_grad_info;
        ret->m_trace_info = m_trace_info;
        return ret;
    }

    inline DType dtype() {return interpreter_for_py->get_dtype(m_handle.get());}
    inline CompNode comp_node() {return interpreter_for_py->get_device(m_handle.get());}
    inline TensorShape shape() {return interpreter_for_py->get_shape(m_handle.get());}
};


struct TensorWrapper {
    std::shared_ptr<Tensor> m_tensor;

    inline TensorWrapper(std::shared_ptr<Tensor> tensor = {}) : m_tensor(std::move(tensor)) {}
    TensorWrapper(PyObject* args, PyObject* kwargs);
    ~TensorWrapper() = default;

    static constexpr auto tp_name = pybind11::detail::_("Tensor");

    using wrap_t = pyext17::wrap<TensorWrapper>;
    friend wrap_t;

    inline static TensorWrapper* cast(PyObject* op) {return reinterpret_cast<wrap_t*>(op)->inst();}
    inline static TensorWrapper* cast_safe(PyObject* op) {
        if (!wrap_t::type().isinstance(op)) return nullptr;
        return cast(op);
    }
    inline ObjectPtr<TensorWrapper, pybind11::handle> self() {return wrap_t::pycast(this);}

    template <typename... Args>
    static ObjectPtr<Tensor> make(Args&&... args) {
        auto* op = wrap_t::cnew(std::forward<Args>(args)...);
        return pybind11::reinterpret_steal<ObjectPtr<Tensor>>(op);
    }

    template <typename... Args>
    static ObjectPtr<Tensor> make(PyTypeObject* pytype, Args&&... args) {
        auto* op = wrap_t::cnew_with_type(pytype,std::forward<Args>(args)...);
        return pybind11::reinterpret_steal<ObjectPtr<Tensor>>(op);
    }

    PyObject* shape();
    PyObject* dtype();
    PyObject* device();
    PyObject* numpy();
    void reset(PyObject*);
    PyObject* isscalar();
    void setscalar();
};


PyObject* py_apply(PyObject* self, PyObject*const* args, size_t nargs/* , PyObject* kwnames */);

struct ApplyContext {
    Tensor::flags_t flags;
    std::shared_ptr<OpDef> op;
    Tensor*const* args;
    size_t nargs;
};

using apply_result_t = SmallVector<std::shared_ptr<Tensor>, 8>;

apply_result_t apply(ApplyContext& ctx);

void init_tensor(pybind11::module);

} // namespace mgb::imperative::python

namespace pybind11::detail {

template<> struct type_caster<mgb::imperative::python::TensorWrapper> : mgb::imperative::python::TensorWrapper::wrap_t::caster {};

} // namespace pybind11::detail
