/**
 * \file imperative/python/src/tensor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/dtype.h"
#include "megbrain/common.h"
#include "megbrain/imperative/ops/utility.h"
#include "megbrain/imperative/ops/backward_graph.h"
#include "megbrain/imperative/profiler.h"
#include "megbrain/opr/io.h"

#include "./tensor.h"
#include "./grad.h"
#include "./trace.h"
#include "./common.h"
#include "./numpy_dtypes.h"
#include "./graph_rt.h"
#include "./helper.h"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <range/v3/all.hpp>
#include <string>

#include <unordered_map>

namespace py = pybind11;
namespace views = ranges::views;

namespace mgb::imperative::python {

interpreter::Interpreter::Channel* interpreter_for_py;

PyObject *cpp_apply_with_tracing, *cpp_apply_const_with_tracing;
PyObject *cpp_apply_backward_varnode;

std::shared_ptr<Tensor> make_const(imperative::TensorPtr value) {
    if (!(ApplyContext::global_enable & Tensor::Flags::TRACE)) {
        return std::make_shared<Tensor>(interpreter_for_py->put(value->dev_tensor()));
    }
    py::tuple tup(6);
    auto data = value->get_value();
    tup[0] = py::reinterpret_steal<py::array>(ndarray_from_tensor(data, npy::ShareType::MUST_SHARE));
    tup[1] = value->dtype();
    tup[2] = value->comp_node();
    tup[3] = true;
    tup[4] = false;
    tup[5] = py::none{};
    auto py_ret = PyObject_Call(cpp_apply_const_with_tracing, tup.ptr(), nullptr);
    if (!py_ret) throw py::error_already_set();
    auto py_list = py::reinterpret_steal<py::list>(py_ret);
    auto* tensor_wrapper = TensorWrapper::try_cast(py_list[0].ptr());
    auto tensor = tensor_wrapper->m_tensor;
    return tensor_wrapper->m_tensor;
}

#define REGISTE_APPLY_FUNC(mode)            \
        void set_##mode(py::object pyf) {   \
            mode = pyf.ptr();               \
        }

REGISTE_APPLY_FUNC(cpp_apply_with_tracing)
REGISTE_APPLY_FUNC(cpp_apply_const_with_tracing)
REGISTE_APPLY_FUNC(cpp_apply_backward_varnode)

#undef REGISTE_APPLY_FUNC

Tensor::flags_t ApplyContext::global_disable = 0;
Tensor::flags_t ApplyContext::global_enable = 0;

void set_tracing() { ApplyContext::global_enable |= Tensor::Flags::TRACE; }
void unset_tracing() { ApplyContext::global_enable &= ~Tensor::Flags::TRACE; }

bool skip_tracing = false;

apply_result_t apply(ApplyContext& ctx) {
    // emulating scalar should be put to specific op's apply, e.g.,
    // elementwise, reduce, typecvt. Currently it's still handled at python
    // side. It could be move to C++ side if it has an impact on performance
    auto flags = ctx.flags & ~ApplyContext::global_disable;
    flags = flags | ApplyContext::global_enable;

    if (flags & Tensor::Flags::SCALAR) {
        // TODO: emulate scalar
    }

    if (flags & Tensor::Flags::GRAD) {
        return apply_grad(ctx);
    }

    if (auto* op = ctx.op->try_cast_final<GenericPyOp>()) {
        py::tuple pyin(ctx.nargs);
        for (size_t i = 0; i < ctx.nargs; ++i) {
            pyin[i] = TensorWrapper::make(ctx.pytype, ctx.args[i]->shared_from_this());
        }
        auto f = py::getattr(op->obj, "_default_rule");
        auto pyout = py::reinterpret_steal<py::object>(PyObject_Call(f.ptr(), pyin.ptr(), nullptr));
        if (!pyout) throw py::error_already_set();
        if (auto* tw = TensorWrapper::try_cast(pyout.ptr())) {
            return {tw->m_tensor};
        }
        apply_result_t ret;
        ret.reserve(py::len(pyout));
        for (auto&& i : pyout) {
            auto* tw = TensorWrapper::try_cast(i.ptr());
            mgb_assert(tw);
            ret.push_back(tw->m_tensor);
        }
        return ret;
    }

    if (flags & Tensor::Flags::TRACE) {
        return apply_trace(ctx);
    } else {
        SmallVector<interpreter::Interpreter::Handle> handles(ctx.nargs);
        for (size_t i = 0; i < ctx.nargs; ++i) {
            handles[i] = ctx.args[i]->m_handle.get();
        }

        apply_result_t outputs;

        // fast copy without really applying
        if (ctx.op->same_type<FastpathCopy>()) {
            mgb_assert(ctx.nargs == 1);
            outputs.reserve(ctx.nargs);
            outputs.emplace_back(std::make_shared<Tensor>(ctx.args[0]->m_handle));
            return outputs;
        }

        auto output_handles = interpreter_for_py->apply_op(ctx.op, handles);

        outputs.reserve(output_handles.size());
        for (auto h : output_handles) {
            outputs.emplace_back(std::make_shared<Tensor>(h));
        }
        return outputs;
    }

    mgb_assert(0);
}

PyObject* py_apply(PyObject* self, PyObject*const* args, size_t nargs/* , PyObject* kwnames */) {
    try {
        // if (kwnames && PyTuple_GET_SIZE(kwnames)) {
        //     PyErr_SetString(PyExc_TypeError, "keyword argument not allowed");
        //     return nullptr;
        // }
        if (nargs < 2) {
            PyErr_SetString(PyExc_TypeError,
                            "py_apply expects one Op and at least one tensor "
                            "as argument");
            return nullptr;
        }

        auto* op = args[0];

        PyTypeObject* pytype = args[1]->ob_type;

        // check if pytype is Parameter(and all other python Tensor's derived class),
        // if yes, using it's tp_base(python Tensor)
        if (TensorWrapper::wrap_t::type().same_pytype(pytype->tp_base->tp_base)) {
            pytype = pytype->tp_base;
        }
        ++args;
        --nargs;

        ApplyContext ctx;
        ctx.flags = 0;
        ctx.op = py::handle(op).cast<std::shared_ptr<OpDef>>();
        SmallVector<Tensor*, 64> tensors(nargs);
        ctx.args = &tensors[0];
        ctx.nargs = nargs;
        ctx.pytype = pytype;

        if (py::isinstance<PySymbolVar>(py::handle(args[0]))){
            SmallVector<cg::VarNode*> vinputs(nargs);
            for (size_t i = 0; i < nargs; ++i) {
                    vinputs[i] = py::handle(args[i]).cast<PySymbolVar*>()->m_node;   
            }
            auto op = ctx.op.get();
            auto rst = OpDef::apply_on_var_node(*op, vinputs);
            auto ret = pybind11::tuple(rst.size());
            auto typeobj = py::handle(args[0]).get_type();
            for (size_t i = 0; i<rst.size(); ++i) {
                ret[i] = typeobj(pybind11::cast(rst[i], pybind11::return_value_policy::automatic));
            }
            return ret.release().ptr();
        }

        for (size_t i = 0; i < nargs; ++i) {
            if (TensorWrapper* tw = TensorWrapper::try_cast(args[i])) {
                auto* t = tensors[i] = tw->m_tensor.get();
                ctx.flags |= t->m_flags;
            } else {
                PyErr_SetString(PyExc_TypeError,
                    ssprintf("op %s expect type Tensor as inputs, got %s actually",
                        ctx.op->make_name().c_str(), Py_TYPE(args[i])->tp_name).c_str());
                return nullptr;
            }
        }

        auto outputs = apply(ctx);
        size_t nout = outputs.size();
        auto ret = py::tuple(nout);
        for (size_t i = 0; i < nout; ++i) {
            ret[i] = TensorWrapper::make(pytype, std::move(outputs[i]));
        }
        return ret.release().ptr();
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}


TensorWrapper::TensorWrapper(PyObject* args, PyObject* kwargs) {
    if (kwargs && PyDict_Size(kwargs)) {
        throw py::type_error("keyword argument not allowed");
    }
    auto nargs = PyTuple_Size(args);
    auto tup = py::reinterpret_borrow<py::tuple>(args);
    if (nargs == 0) {
        throw py::type_error("too few arguments");
    }
    if (auto* t = try_cast(tup[0].ptr())) {
        if (nargs > 1) {
            throw py::type_error("expect 1 argument");
        }
        m_tensor = t->m_tensor;
    } else {
        if (nargs == 1) {
            auto arg0 = PyTuple_GetItem(args, 0);
            // for lazy_eval_tensor
            if (strstr(arg0->ob_type->tp_name, "VarNode")) {
                if (PyObject_HasAttrString(arg0, "_node")) {
                    arg0 = PyObject_GetAttrString(arg0, "_node");
                }
                m_tensor = std::make_shared<Tensor>(py::handle(arg0).cast<cg::VarNode *>());
            } else {
                // for DeviceTensorND
                if (strstr(arg0->ob_type->tp_name, "DeviceTensorND")) {
                    auto dv = py::handle(arg0).cast<DeviceTensorND>();
                    interpreter::Interpreter::Handle handle = interpreter_for_py->put(dv);
                    m_tensor = std::make_shared<Tensor>(handle);
                } else {
                    throw py::type_error("single argument is not tensor, varnode or devicetensor");
                }
            }
        } else {
            py::detail::loader_life_support life_sup; // FIXME!!!required to cast DType
            if (nargs != 5 && nargs != 6) {
                throw py::type_error("expect 5 or 6 arguments");
            }
            auto data = tup[0].cast<py::array>();
            DType dtype = tup[1].cast<DType>();
            CompNode cn = tup[2].cast<CompNode>();
            bool is_const = tup[3].cast<bool>();
            bool no_cache = nargs == 6 ? tup[4].cast<bool>() : false;
            std::string name;
            if (tup[nargs - 1].ptr() != Py_None) name = tup[nargs - 1].cast<std::string>();

            // const op
            if (is_const && (ApplyContext::global_enable == Tensor::Flags::TRACE)) {
                auto py_ret = PyObject_Call(cpp_apply_const_with_tracing, tup.ptr(), nullptr);
                if (!py_ret) throw py::error_already_set();
                auto py_list = py::reinterpret_steal<py::list>(py_ret);
                if (auto* t = try_cast(py_list[0].ptr())) {
                    m_tensor = t->m_tensor;
                }
                return;
            }

            interpreter::Interpreter::Handle handle;
            {
                HostTensorND ret(cn);
                handle = interpreter_for_py->put(npy::np2tensor(data.ptr(), npy::Meth::copy_into(&ret), dtype), no_cache);
            }

            m_tensor = std::make_shared<Tensor>(handle);
            m_tensor->user_custom_name = name;

            if (data.ndim() == 0) {
                m_tensor->m_flags |= Tensor::Flags::SCALAR;
            }
        }
    }
}


#define REGISTE_TENSORWRAPPER_FUNC(type, member)                                    \
        PyObject* TensorWrapper::member() {                                         \
            return py::cast(m_tensor->m_trace_info.member).release().ptr();         \
        }                                                                           \
        void TensorWrapper::set_##member(PyObject* dest) {                          \
            auto py_dest = py::reinterpret_borrow<py::object>(dest);                \
            type real_dest = py_dest.cast<type>();                                  \
            m_tensor->m_trace_info.member = real_dest;                              \
        }

REGISTE_TENSORWRAPPER_FUNC(int64_t, mixin_handle)
REGISTE_TENSORWRAPPER_FUNC(bool, recording)

#undef REGISTE_TENSORWRAPPER_FUNC


#define REGISTE_TENSORWRAPPER_PYOBJECT_FUNC(member)                                 \
        PyObject* TensorWrapper::member() {                                         \
            if (m_tensor->m_trace_info.member) {                                    \
                return m_tensor->m_trace_info.member;                               \
            } else {                                                                \
                Py_RETURN_NONE;                                                     \
            }                                                                       \
        }                                                                           \
        void TensorWrapper::set_##member(PyObject* dest) {                          \
            if (dest == Py_None) {                                                  \
                Py_XDECREF(m_tensor->m_trace_info.member);                          \
                m_tensor->m_trace_info.member = nullptr;                            \
            } else {                                                                \
                Py_INCREF(dest);                                                    \
                m_tensor->m_trace_info.member = dest;                               \
            }                                                                       \
        }

REGISTE_TENSORWRAPPER_PYOBJECT_FUNC(compiled_info)
REGISTE_TENSORWRAPPER_PYOBJECT_FUNC(trace_mixin_info)

#undef REGISTE_TENSORWRAPPER_PYOBJECT_FUNC


#define SET_GET_NAME(member)                                     \
    PyObject* TensorWrapper::member() {                          \
        return py::cast(m_tensor->member).release().ptr();       \
    }                                                            \
    void TensorWrapper::set_##member(PyObject* dest) {           \
        auto py_dest = py::reinterpret_borrow<py::object>(dest); \
        m_tensor->member = py_dest.cast<std::string>();          \
    }
SET_GET_NAME(user_custom_name)
SET_GET_NAME(automatic_name)
#undef SET_GET_NAME


PyObject* TensorWrapper::handle() {
    return py::cast(m_tensor->m_handle).release().ptr();
}


void TensorWrapper::set_handle(PyObject* dest) {
    auto py_dest = py::reinterpret_borrow<py::object>(dest);
    SharedHandle real_dest = py_dest.cast<SharedHandle>();
    m_tensor->m_handle = std::move(real_dest);
}


PyObject* TensorWrapper::shape() {
    // if it's tracing compiled mode, get value from compiled_info 
    if (m_tensor->m_trace_info.compiled_info != nullptr) {
        if (m_tensor->m_flags & Tensor::Flags::SCALAR) {
            return PyTuple_New(0);
        }
        PyObject *shp = PyObject_GetAttrString(m_tensor->m_trace_info.compiled_info, "shape");
        if (shp == Py_None) {
            throw TraceReadError("shape of this tensor is not read in trace");
        }
        return shp;
    }

    // inside trace, if tensor shape is useful for other operations, set shape_read = true
    if (m_tensor->m_trace_info.recording && !skip_tracing) {
        PyObject_SetAttrString(m_tensor->m_trace_info.trace_mixin_info, "shape_read", py::cast(true).release().ptr());
    }

    if (m_tensor->m_flags & Tensor::Flags::SCALAR) {
        return PyTuple_New(0);
    }

    TensorShape shape;
    if (m_tensor->m_var) {      // get shape from m_var
        auto&& mgr = m_tensor->m_var->owner_graph()->static_infer_manager();
        auto *tshp = mgr.infer_shape_fallible(m_tensor->m_var);
        if (!tshp) {
            Py_RETURN_NONE;
        }
        shape = *tshp;
    } else {
        py::gil_scoped_release _;
        shape = m_tensor->shape();
    }

    if (!shape.ndim) {
        Py_RETURN_NONE;
    }
    py::tuple ret(shape.ndim);
    for (size_t i = 0; i < shape.ndim; ++i) {
        ret[i] = shape[i];
    }
    return ret.release().ptr();
}


PyObject* TensorWrapper::dtype() {
    if (m_tensor->m_var) {
        return py::cast(m_tensor->m_var->dtype()).release().ptr();
    }
    return py::cast(m_tensor->dtype()).release().ptr();
}


PyObject* TensorWrapper::device() {
    if (m_tensor->m_var) {
        return py::cast(m_tensor->m_var->comp_node()).release().ptr();
    }
    return py::cast(m_tensor->comp_node()).release().ptr();
}


PyObject* TensorWrapper::numpy() {
    if (m_tensor->m_trace_info.compiled_info != nullptr) {
        PyObject* np_val = PyObject_CallMethod(m_tensor->m_trace_info.compiled_info, "numpy", nullptr);
        if (!np_val) throw py::error_already_set();
        if (np_val == Py_None) {
            throw TraceReadError("value of this tensor is not read in trace");
        }
        if (m_tensor->m_flags & Tensor::Flags::SCALAR) {
            PyObject *np_scalar = PyArray_Squeeze(reinterpret_cast<PyArrayObject*>(np_val));
            Py_DECREF(np_val);
            return np_scalar;
        }
        return np_val;
    }

    if (m_tensor->m_trace_info.recording && !skip_tracing) {
        PyObject_SetAttrString(m_tensor->m_trace_info.trace_mixin_info, "value_read", py::cast(true).release().ptr());
    }

    if (m_tensor->m_handle.get() == nullptr && m_tensor->m_var != nullptr) {
        auto&& mgr = m_tensor->m_var->owner_graph()->static_infer_manager();
        auto&& type = mgr.get_infer_type(m_tensor->m_var);
        using InferType = cg::static_infer::InferType;
        if (!(type.value & (InferType::CONST | InferType::RT_STATIC))) {
            PyErr_SetString(PyExc_ValueError, "tensor invalid");
            return nullptr;
        }
        auto* val = mgr.infer_value_fallible(m_tensor->m_var);
        if (!val) {
            PyErr_SetString(PyExc_ValueError, "tensor invalid");
            return nullptr;
        }
        auto np_val = py::cast(*val).attr("numpy")();
        if (m_tensor->m_flags & Tensor::Flags::SCALAR) {
            return PyArray_Squeeze(reinterpret_cast<PyArrayObject*>(np_val.release().ptr()));
        }
        return np_val.release().ptr();
    }
    auto&& hv = [&]() {
        py::gil_scoped_release _;
        return interpreter_for_py->get_value(m_tensor->m_handle.get());
    }();
    auto arr = py::reinterpret_steal<py::array>(npy::ndarray_from_tensor(hv, npy::ShareType::TRY_SHARE));
    if (!arr) {
        PyErr_SetString(PyExc_ValueError, "tensor invalid");
        return nullptr;
    }

    if (m_tensor->m_flags & Tensor::Flags::SCALAR) {
        mgb_assert(PyArray_Check(arr.ptr()));
        return PyArray_Squeeze(reinterpret_cast<PyArrayObject*>(arr.ptr()));
    }
    return arr.release().ptr();
}

PyObject* TensorWrapper::varnode() {
    if (m_tensor->m_var) {
        return py::cast(m_tensor->m_var).release().ptr();
    }
    Py_RETURN_NONE;
}

void TensorWrapper::reset(PyObject* tensor) {
    TensorWrapper* t = TensorWrapper::try_cast(tensor);
    if (!t) {
        throw py::type_error("expect Tensor");
    }
    std::string user_custom_name = m_tensor->user_custom_name;
    std::string automatic_name = m_tensor->automatic_name;
    m_tensor = t->m_tensor;
    m_tensor->user_custom_name = user_custom_name;
    m_tensor->automatic_name = automatic_name;
}

void TensorWrapper::reset_varnode() {
    m_tensor->m_var = nullptr;
}

PyObject* TensorWrapper::detach() {
    PyObject* self = wrap_t::pycast(this);
    PyTypeObject* pytype = self->ob_type;

    std::shared_ptr<Tensor> new_tensor;
    if (m_tensor->m_handle.get()) {
        new_tensor = std::make_shared<Tensor>(m_tensor->m_handle);
    } else {
        new_tensor = std::make_shared<Tensor>(m_tensor->m_var);
    }
    new_tensor->m_trace_info = m_tensor->m_trace_info;

    new_tensor->m_flags = m_tensor->m_flags;
    auto ret = TensorWrapper::make(pytype, std::move(new_tensor));
    return ret.release().ptr();
}

PyObject* TensorWrapper::_dev_tensor(){
    if (m_tensor->m_trace_info.compiled_info != nullptr) {
        auto *dev_tensor = PyObject_CallMethod(m_tensor->m_trace_info.compiled_info, "_dev_tensor", nullptr);
        if (!dev_tensor) throw py::error_already_set();
        if (dev_tensor == Py_None) {
            throw TraceReadError("raw data of this tensor is not read in trace");
        }

        // set m_handle to make it a real tensor
        auto py_dev_tensor = py::reinterpret_borrow<py::object>(dev_tensor);
        auto sh = interpreter_for_py->put(py_dev_tensor.cast<DeviceTensorND>());
        m_tensor->m_handle = std::move(SharedHandle(sh));

        // compiled info is useless after m_handle is set
        Py_DECREF(m_tensor->m_trace_info.compiled_info);
        m_tensor->m_trace_info.compiled_info = nullptr;

        return dev_tensor;
    }
    if (m_tensor->m_trace_info.recording && !skip_tracing) {
        PyObject_SetAttrString(m_tensor->m_trace_info.trace_mixin_info, "data_read", py::cast(true).release().ptr());
    }
    auto dev_tensor = [&](){
        py::gil_scoped_release _;
        return interpreter_for_py->get_dev_tensor(m_tensor->m_handle.get());
    }();
    return py::cast(dev_tensor).release().ptr();
}

void TensorWrapper::_swap_out() {
    interpreter_for_py->swap_out(m_tensor->m_handle.get());
}

void TensorWrapper::_swap_in() {
    interpreter_for_py->swap_in(m_tensor->m_handle.get());
}

void TensorWrapper::_drop() {
    interpreter_for_py->drop(m_tensor->m_handle.get());
}


PyObject* TensorWrapper::isscalar() {
    if(m_tensor->m_flags & Tensor::Flags::SCALAR) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}


void TensorWrapper::setscalar() {
    m_tensor->m_flags |= Tensor::Flags::SCALAR;
}


void TensorWrapper::unsetscalar() {
    m_tensor->m_flags &= ~Tensor::Flags::SCALAR;
}


struct TensorWeakRef {
    std::weak_ptr<Tensor> wptr;

    TensorWeakRef(const TensorWrapper& tw) : wptr(tw.m_tensor) {}

    py::object operator()() {
        if (auto p = wptr.lock()) {
            return TensorWrapper::make(p);
        }
        return py::none();
    }
    int _use_cnt() { return wptr.use_count(); }
};

/* ============== convert inputs ============== */

// map numpy.dtype.kind to priority
inline uint8_t category_priority(char c) {
    switch (c) {
        case 'f': return 3; // floating-point
        case 'i': return 2; // signed integer
        case 'u': return 2; // unsigned integer
        case 'b': return 1; // boolean
        default: return 0;
    }
}

// Returns the maximum value of the priority of each type in the list `types`.
uint8_t max_priority(SmallVector<PyArray_Descr*> types) {
    if (types.size() == 0) {
        return 0;
    } else {
        uint8_t max_p = 0;
        for (auto&& desc: types) {
            max_p = std::max(max_p, category_priority(desc->kind));
        }
        return max_p;
    }
}

// Returns the data type with sufficient size to hold all types of
// category `cat` in the list `types`.
PyArray_Descr* promote_types(SmallVector<PyArray_Descr*> types, uint8_t cat) {
    // Return value: New reference
    SmallVector<PyArray_Descr*> used_types;
    for (auto&& desc: types) {
        auto&& v = category_priority(desc->kind);
        if (v == cat) {
            used_types.emplace_back(desc);
        }
    }
    mgb_assert(used_types.size() > 0, "size of used_types is 0");
    PyArray_Descr* res = used_types[0];
    Py_INCREF(res);

    for (size_t i = 1; i < used_types.size(); ++i) {
        PyArray_Descr* tmp = PyArray_PromoteTypes(used_types[i], res);
        Py_DECREF(res);
        res = tmp;
    }
    return res;
}

PyArray_Descr* scalar2dtype(PyObject* arg) {
    // Return value: New reference
    if (PyBool_Check(arg)) {
        auto&& descr = PyArray_DescrFromType(NPY_BOOL);
        return descr;
    }
    if (PyLong_CheckExact(arg)) {
        auto&& descr = PyArray_DescrFromType(NPY_INT32);
        return descr;
    }
    if (PyFloat_CheckExact(arg)) {
        auto&& descr = PyArray_DescrFromType(NPY_FLOAT32);
        return descr;
    }
    return nullptr;
}

PyArray_Descr* _dtype_promotion(PyObject*const* args, size_t nargs) {
    // Return value: New reference
    SmallVector<PyArray_Descr*> tensors;
    SmallVector<PyArray_Descr*> scalars;

    bool is_tuple = false;
    PyObject* tuple = nullptr;
    if (nargs == 1 && (PyTuple_Check(args[0]) || PyList_Check(args[0]))) {
        if (PyList_Check(args[0])) {
            tuple = PyList_AsTuple(args[0]);
        } else {
            tuple = args[0];
            Py_INCREF(tuple);
        }
        nargs = PyTuple_Size(tuple);
        is_tuple = true;
    }

    for (size_t i = 0; i < nargs; ++i) {
        PyObject* handle = is_tuple ? PyTuple_GetItem(tuple, i): args[i];
        if (handle == Py_None) continue;
        TensorWrapper* tw = TensorWrapper::try_cast(handle);
        if (tw) {
            mgb::DType type = tw->m_tensor->dtype();
            auto&& descr = npy::dtype_mgb2np_descr(type);
            Py_INCREF(descr.get());
            tensors.emplace_back(descr.get());
        }else{
            if (PyArray_Check(handle) || PyArray_CheckScalar(handle)) {
                auto&& descr = PyArray_DescrFromObject(handle, nullptr);
                tensors.emplace_back(descr);
                continue;
            }

            if (py::isinstance<PySymbolVar>(py::handle(handle))){
                auto var = py::handle(handle).cast<PySymbolVar*>();
                mgb::DType type = var->m_node->dtype();
                auto && descr = npy::dtype_mgb2np_descr(type);
                Py_INCREF(descr.get());
                tensors.emplace_back(descr.get());
                continue;
            }

            PyArray_Descr* descr = scalar2dtype(handle);
            if (descr) {
                scalars.emplace_back(descr);
                continue;
            }
        }
    }

    auto max_pri_scalars = max_priority(scalars);
    auto max_pri_tensors = max_priority(tensors);

    if (max_pri_scalars <= 0 && max_pri_tensors <= 0) {
        throw py::value_error("invalid input, no dtype avaliable");
    }
    PyArray_Descr* res;
    if (max_pri_scalars > max_pri_tensors) {
        res = promote_types(scalars, max_pri_scalars);
    }else{
        res = promote_types(tensors, max_pri_tensors);
    }
    for (auto *p: tensors) { Py_DECREF(p); }
    for (auto *p: scalars) { Py_DECREF(p); }
    Py_XDECREF(tuple);
    return res;
}

CompNode _get_device(PyObject*const* args, size_t nargs) {
    bool is_tuple = false;
    PyObject* tuple = nullptr;
    if (nargs == 1 && (PyTuple_Check(args[0]) || PyList_Check(args[0]))) {
        if (PyList_Check(args[0])) {
            tuple = PyList_AsTuple(args[0]);
        } else {
            tuple = args[0];
            Py_INCREF(tuple);
        }
        nargs = PyTuple_Size(tuple);
        is_tuple = true;
    }
    bool valid = false;
    CompNode cn;
    for (size_t i = 0; i < nargs; ++i) {
        PyObject* handle = is_tuple ? PyTuple_GetItem(tuple, i) : args[i];
        TensorWrapper* tw = TensorWrapper::try_cast(handle);

        bool is_symvar = py::isinstance<PySymbolVar>(py::handle(handle));
        if (tw || is_symvar) {
            if (!valid) {
                cn = tw ? tw->m_tensor->comp_node()
                        : py::handle(handle)
                                     .cast<PySymbolVar*>()
                                     ->m_node->comp_node();
                valid = true;
            } else {
                CompNode cn1 = tw ? tw->m_tensor->comp_node()
                                  : py::handle(handle)
                                               .cast<PySymbolVar*>()
                                               ->m_node->comp_node();
                if (cn1 != cn) {
                    throw py::value_error(ssprintf("ambiguous device: %s vs %s",
                                                   cn.to_string().c_str(),
                                                   cn1.to_string().c_str()));
                }
            }
        }
    }
    if (!valid) {
        return CompNode::load(get_default_device());
    }
    Py_XDECREF(tuple);
    return cn;
}

// Returns the dtype that would result from performing an arithmetic
// operation on the provided input tensors and scalars.
PyObject* dtype_promotion(PyObject* self, PyObject*const* args, size_t nargs) {
    if (!nargs) {
        PyErr_SetString(PyExc_TypeError, "empty input is not allowed");
        return nullptr;
    }
    try {
        PyArray_Descr* res = _dtype_promotion(args, nargs);
        return py::cast(npy::dtype_np2mgb_descr(res)).release().ptr();
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

PyObject* get_device(PyObject* self, PyObject*const* args, size_t nargs) {
    if (!nargs) {
        PyErr_SetString(PyExc_TypeError, "empty input is not allowed");
        return nullptr;
    }
    try {
        CompNode cn = _get_device(args, nargs);
        return py::cast(cn).release().ptr();
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

#ifdef METH_FASTCALL
#define MGE_PY_INTERFACE(NAME, FUNC) \
    { #NAME, (PyCFunction)FUNC, METH_FASTCALL, nullptr }
#else
#define WRAP_FUNC_PY35(FUNC)                                \
    PyObject* py35_##FUNC(PyObject* self, PyObject* args) { \
        auto* arr = &PyTuple_GET_ITEM(args, 0);             \
        auto size = PyTuple_GET_SIZE(args);                 \
        return FUNC(self, arr, size);                       \
    }
WRAP_FUNC_PY35(py_apply);
WRAP_FUNC_PY35(dtype_promotion);
WRAP_FUNC_PY35(get_device);
#undef WRAP_FUNC_PY35
#define MGE_PY_INTERFACE(NAME, FUNC) \
    { #NAME, (PyCFunction)py35_##FUNC, METH_VARARGS, nullptr }
#endif


void init_tensor(py::module m) {
    imperative::Tensor::static_initialize();
    static auto sl_interpreter_for_py = interpreter::Interpreter::inst().create_channel();
    interpreter_for_py = sl_interpreter_for_py.get();

    auto* tensor_type = TensorWrapper::wrap_t::type()
        .def<&TensorWrapper::numpy>("numpy")
        .def_getset<&TensorWrapper::shape>("shape")
        .def_getset<&TensorWrapper::dtype>("dtype")
        .def_getset<&TensorWrapper::device>("device")
        .def<&TensorWrapper::reset>("_reset")
        .def<&TensorWrapper::isscalar>("_isscalar")
        .def<&TensorWrapper::setscalar>("_setscalar")
        .def<&TensorWrapper::unsetscalar>("_unsetscalar")
        .def<&TensorWrapper::detach>("detach")
        .def<&TensorWrapper::_dev_tensor>("_dev_tensor")
        .def<&TensorWrapper::_swap_out>("_swap_out")
        .def<&TensorWrapper::_swap_in>("_swap_in")
        .def<&TensorWrapper::_drop>("_drop")
        .def<&TensorWrapper::reset_varnode>("_reset_varnode")
        .def<&TensorWrapper::_use_cnt>("_use_cnt")
        .def_getset<&TensorWrapper::varnode>("_varnode")
        .def_getset<&TensorWrapper::mixin_handle, &TensorWrapper::set_mixin_handle>("_mixin_handle")
        .def_getset<&TensorWrapper::recording, &TensorWrapper::set_recording>("_recording")
        .def_getset<&TensorWrapper::handle, &TensorWrapper::set_handle>("_handle")
        .def_getset<&TensorWrapper::compiled_info, &TensorWrapper::set_compiled_info>("_compiled_info")
        .def_getset<&TensorWrapper::trace_mixin_info, &TensorWrapper::set_trace_mixin_info>("_trace_mixin_info")
        .def_getset<&TensorWrapper::user_custom_name, &TensorWrapper::set_user_custom_name>("c_name")
        .def_getset<&TensorWrapper::automatic_name, &TensorWrapper::set_automatic_name>("_name")
        .finalize();
    if (!tensor_type) throw py::error_already_set();
    py::setattr(m, "Tensor", tensor_type);

    py::class_<TensorWeakRef>(m, "TensorWeakRef")
        .def(py::init<const TensorWrapper&>())
        .def("__call__", &TensorWeakRef::operator())
        .def("_use_cnt", &TensorWeakRef::_use_cnt);

    py::class_<PySymbolVar, std::shared_ptr<PySymbolVar>>(m, "SymbolVar")
            .def_property_readonly(
                    "dtype", [](PySymbolVar* v) { return v->m_node->dtype(); })
            .def_property("var", [](PySymbolVar* v) { return v->m_node; },
                          [](PySymbolVar* s, cg::VarNode* v) { s->m_node = v; })
            .def_property_readonly(
                    "device",
                    [](PySymbolVar* v) { return v->m_node->comp_node(); })
            .def_property_readonly(
                    "graph",
                    [](PySymbolVar* v) { return v->m_node->owner_graph(); })
            .def_property_readonly(
                    "shape",
                    [](PySymbolVar* v) -> const TensorShape* {
                        auto&& mgr = v->m_node->owner_graph()
                                             ->static_infer_manager();
                        return mgr.infer_shape_fallible(v->m_node);
                    })
            .def("_isscalar", [](PySymbolVar* v) { return v->is_scalar; })
            .def("_setscalar",
                 [](PySymbolVar* v) { return v->is_scalar = true; })
            .def(py::init([](cg::VarNode* node) {
                     return std::make_shared<PySymbolVar>(node);
                 }),
                 py::arg() = nullptr);

    static PyMethodDef method_defs[] = {
            MGE_PY_INTERFACE(apply, py_apply),
            MGE_PY_INTERFACE(dtype_promotion, dtype_promotion),
            MGE_PY_INTERFACE(get_device, get_device),
            {nullptr, nullptr, 0, nullptr}};
    for (auto&& def: method_defs) {
        if (def.ml_meth != nullptr) {
            auto* func = PyCFunction_NewEx(&def, nullptr, nullptr);
            if (!func) throw py::error_already_set();
            py::setattr(m, def.ml_name, func);
        }
    }

    static constexpr auto sync_py_task_q = []{
        py_task_q.wait_all_task_finish();
    };

    m.def("set_option",
          [](std::string name, size_t value){ interpreter_for_py->set_option(name, value); });
    m.def("get_option",
          [](std::string name){ return interpreter_for_py->get_option(name); });
    m.def("_set_swap_flag",
          [](bool flag) { interpreter_for_py->set_option("enable_swap", flag); });
    m.def("_set_drop_flag",
          [](bool flag) { interpreter_for_py->set_option("enable_drop", flag); });
    m.def("config_async_level",
          [](int level) {
              mgb_assert(level >= 0 and level <= 2, "async_level should be 0, 1 or 2");
              interpreter_for_py->set_option("async_level", level);
          });
    m.def("get_async_level",
          []() { return interpreter_for_py->get_option("async_level"); });
    m.def("set_buffer_length",
          [](int length) {
              mgb_assert(length >= 0 and length < 100, "buffer_length should be in [0, 100)");
              interpreter_for_py->set_option("buffer_length", length);
          });
    m.def("push_scope",
          [](std::string name) { interpreter_for_py->push_scope(name); });
    m.def("pop_scope",
          [](std::string name) { interpreter_for_py->pop_scope(name); });
    m.def("start_profile",
          [](imperative::Profiler::options_t options) {
              interpreter_for_py->sync();
              imperative::Profiler::load_options(std::move(options));
              imperative::Profiler::start_profile();
              interpreter_for_py->start_profile();
          }, py::call_guard<py::gil_scoped_release>());
    m.def("stop_profile",
          []() -> std::function<void(std::string, std::string)> {
              interpreter_for_py->stop_profile();
              interpreter_for_py->sync();
              imperative::Profiler::stop_profile();
              auto results = imperative::Profiler::collect();
              auto options = imperative::Profiler::get_options();
              return [results=std::move(results), options=std::move(options)](std::string basename, std::string format){
                  imperative::Profiler::dump_profile(basename, format, results, options);
              };
          }, py::call_guard<py::gil_scoped_release>());
    m.def("sync",
          []() {
              interpreter_for_py->sync();
              sync_py_task_q();
          }, py::call_guard<py::gil_scoped_release>());
    m.def("full_sync",
          []() {
              interpreter_for_py->sync();
              CompNode::sync_all();
              sync_py_task_q();
          }, py::call_guard<py::gil_scoped_release>());
    m.def("close",
          []() {
              interpreter_for_py->close();
              sync_py_task_q();
          }, py::call_guard<py::gil_scoped_release>());

    py::handle grad_key_type = GradKeyWrapper::wrap_t::type()
        .def<&GradKeyWrapper::attach>("attach")
        .def<&GradKeyWrapper::is_attached_to>("is_attached_to")
        .def_getset<&GradKeyWrapper::get_name, &GradKeyWrapper::set_name>("name")
        .finalize();
    if (!grad_key_type) throw py::error_already_set();
    py::setattr(m, "GradKey", grad_key_type);
    m.def("backward", &GradKeyWrapper::backward);

    m.def("set_cpp_apply_with_tracing", &set_cpp_apply_with_tracing);
    m.def("set_cpp_apply_const_with_tracing", &set_cpp_apply_const_with_tracing);
    m.def("set_cpp_apply_backward_varnode", &set_cpp_apply_backward_varnode);

    m.attr("skip_tracing") = &skip_tracing;

    py::class_<SharedHandle>(m, "SharedHandle")
        .def(py::init<const SharedHandle&>())
        .def("__eq__", [](SharedHandle &thish, SharedHandle &thath) {
            return (thish.get() == thath.get());
        })
        .def("__hash__", [](SharedHandle &sh) {
            return reinterpret_cast<int64_t>(sh.get());
        })
        ;

    m.def("set_tracing", &set_tracing);
    m.def("unset_tracing", &unset_tracing);
}

#undef MGE_PY_INTERFACE

} // namespace mgb::imperative::python
