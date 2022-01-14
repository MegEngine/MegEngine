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

#include "megbrain/common.h"
#include "megbrain/dtype.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/backward_graph.h"
#include "megbrain/imperative/ops/utility.h"
#include "megbrain/imperative/profiler.h"
#include "megbrain/imperative/transformations/eval.h"
#include "megbrain/imperative/transformations/lazy.h"
#include "megbrain/imperative/transformations/scalar.h"
#include "megbrain/imperative/transformations/symbol.h"
#include "megbrain/imperative/transformations/trace.h"
#include "megbrain/imperative/utils/map.h"
#include "megbrain/opr/io.h"
#include "megbrain/plugin/profiler.h"

#include "./common.h"
#include "./grad.h"
#include "./graph_rt.h"
#include "./helper.h"
#include "./module_trace.h"
#include "./numpy_dtypes.h"
#include "./tensor.h"
#include "./transformation.h"

#include <object.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pytypes.h>
#include <pyerrors.h>
#include <range/v3/all.hpp>
#include <string>

#include <unordered_map>

#include "../../src/impl/mgb_cg_impl.h"

namespace py = pybind11;
namespace views = ranges::views;

namespace mgb::imperative::python {

namespace {
WeakKeyMap<ValueWeakRef, py::object> module_trace_info_map;
}

interpreter::Interpreter::Channel* interpreter_for_py = nullptr;
PyTypeObject* py_tensor_type = nullptr;

PyObject* py_apply(
        PyObject* self, PyObject* const* args, size_t nargs /* , PyObject* kwnames */) {
    try {
        // if (kwnames && PyTuple_GET_SIZE(kwnames)) {
        //     PyErr_SetString(PyExc_TypeError, "keyword argument not allowed");
        //     return nullptr;
        // }
        if (nargs < 2) {
            PyErr_SetString(
                    PyExc_TypeError,
                    "py_apply expects one Op and at least one tensor "
                    "as argument");
            return nullptr;
        }

        auto* py_op = args[0];

        ++args;
        --nargs;

        auto op = py::handle(py_op).cast<std::shared_ptr<OpDef>>();
        SmallVector<ValueRef, 64> tensors(nargs);

        if (py::isinstance<PySymbolVar>(py::handle(args[0]))) {
            // swap to a special context to reuse scalar handle
            TransformationContext symbol_var_context;
            Transformation::swap_context(symbol_var_context);
            CleanupGuard _{[&] { Transformation::swap_context(symbol_var_context); }};
            auto* graph =
                    py::handle(args[0]).cast<PySymbolVar*>()->m_node->owner_graph();
            std::make_shared<SymbolTransformation>(graph)->register_at(
                    Transformation::top());
            std::make_shared<ScalarTransformation>()->register_at(
                    Transformation::top());
            SmallVector<ValueRef> inputs(nargs);
            for (size_t i = 0; i < nargs; ++i) {
                auto* py_input = py::handle(args[i]).cast<PySymbolVar*>();
                ValueRef input = SymbolValue::make(py_input->m_node);
                if (py_input->is_scalar) {
                    input = ScalarValue::make(input);
                }
                inputs[i] = input;
            }
            auto outputs = imperative::apply(*op, inputs);
            auto ret = pybind11::tuple(outputs.size());
            auto typeobj = py::handle(args[0]).get_type();
            for (size_t i = 0; i < outputs.size(); ++i) {
                bool is_scalar = false;
                if (auto* scalar_value = outputs[i].as<ScalarValue>()) {
                    outputs[i] = scalar_value->value();
                    is_scalar = true;
                }
                auto* node = outputs[i].cast<SymbolValue>().node();
                ret[i] = typeobj(
                        pybind11::cast(node, pybind11::return_value_policy::automatic));
                py::handle(ret[i]).cast<PySymbolVar*>()->is_scalar = is_scalar;
            }
            return ret.release().ptr();
        }

        for (size_t i = 0; i < nargs; ++i) {
            if (TensorWrapper* tw = TensorWrapper::try_cast(args[i])) {
                tensors[i] = tw->m_tensor->data();
            } else {
                PyErr_SetString(
                        PyExc_TypeError,
                        ssprintf(
                                "op %s expect type Tensor as inputs, got %s actually",
                                op->make_name().c_str(), Py_TYPE(args[i])->tp_name)
                                .c_str());
                return nullptr;
            }
        }

        auto outputs = imperative::apply(ApplyOp(*op), {tensors.data(), nargs});
        size_t nout = outputs.size();
        auto ret = py::tuple(nout);
        for (size_t i = 0; i < nout; ++i) {
            ret[i] = TensorWrapper::make(py_tensor_type, std::move(outputs[i]));
        }
        return ret.release().ptr();
    }
    PYEXT17_TRANSLATE_EXC_RET(nullptr)
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
        m_tensor = t->m_tensor->copy();
    } else {
        if (nargs == 1) {
            auto arg0 = PyTuple_GetItem(args, 0);
            // for DeviceTensorND
            if (strstr(arg0->ob_type->tp_name, "DeviceTensorND")) {
                auto dv = py::handle(arg0).cast<DeviceTensorND>();
                m_tensor = std::make_shared<Tensor>(imperative::apply(
                        CreateTensor(CreateTensor::Common, dv.comp_node(), dv.layout()),
                        DeviceStorage::make(dv.storage()))[0]);
            } else {
                throw py::type_error(
                        "single argument is not tensor, varnode or devicetensor");
            }
        } else {
            py::detail::loader_life_support life_sup;  // FIXME!!!required to cast DType
            if (nargs != 5 && nargs != 6) {
                throw py::type_error("expect 5 or 6 arguments");
            }
            auto data = tup[0].cast<py::array>();
            DType dtype = tup[1].cast<DType>();
            CompNode cn = tup[2].cast<CompNode>();
            bool is_const = tup[3].cast<bool>();
            bool no_cache = nargs == 6 ? tup[4].cast<bool>() : false;
            std::string name;
            if (tup[nargs - 1].ptr() != Py_None)
                name = tup[nargs - 1].cast<std::string>();

            // const op
            {
                CreateTensor::Kind kind = is_const ? CreateTensor::Const
                                        : no_cache ? CreateTensor::Unique
                                                   : CreateTensor::Common;
                HostTensorND ret(cn);
                ret = npy::np2tensor(data.ptr(), npy::Meth::copy_into(&ret), dtype);
                mgb_assert(
                        ret.layout().is_empty() || ret.layout().is_contiguous(),
                        "host value should be continuous");
                ValueShape shape;
                for (size_t i = 0; i < data.ndim(); ++i) {
                    shape[shape.ndim++] = data.shape(i);
                }
                m_tensor = std::make_shared<Tensor>(imperative::apply(
                        CreateTensor(kind, cn, ret.dtype(), shape),
                        HostStorage::make(ret.storage()))[0]);
            }

            if (!name.empty()) {
                m_tensor->reset(
                        imperative::apply(RenameValue(name), m_tensor->data())[0]);
                mgb_assert(
                        ((std::string&)*m_tensor->data().name()) == name,
                        "result name incorrect");
            }

            if (data.ndim() == 0) {
                mgb_assert(m_tensor->is_scalar(), "result should be scalar");
            }
        }
    }
}

PyObject* TensorWrapper::module_trace_info() {
    if (auto module_trace_info = module_trace_info_map.try_get(m_tensor->data())) {
        return module_trace_info->inc_ref().ptr();
    } else {
        PyErr_SetString(
                PyExc_AttributeError,
                "Has no attribute named \'_NodeMixin__node\', please "
                "set it first");
        return nullptr;
    }
}

void TensorWrapper::set_module_trace_info(PyObject* obj) {
    module_trace_info_map[m_tensor->data()] = py::reinterpret_borrow<py::object>(obj);
}

void TensorWrapper::_set_name(PyObject* dest) {
    auto py_dest = py::reinterpret_borrow<py::object>(dest);
    auto name = py_dest.cast<std::string>();
    m_tensor->set_name(name);
}

PyObject* TensorWrapper::_detail() {
    return py::str(m_tensor->data().unwrap().to_string()).release().ptr();
}

void TensorWrapper::_watch() {
    m_tensor->data().watch();
}

PyObject* TensorWrapper::shape() {
    auto shape = m_tensor->shape();

    if (!shape) {
        Py_RETURN_NONE;
    }
    py::tuple ret(shape->ndim);
    for (size_t i = 0; i < shape->ndim; ++i) {
        ret[i] = shape->at(i);
    }
    return ret.release().ptr();
}

PyObject* TensorWrapper::dtype() {
    return py::cast(m_tensor->dtype()).release().ptr();
}

PyObject* TensorWrapper::device() {
    return py::cast(m_tensor->comp_node()).release().ptr();
}

PyObject* TensorWrapper::numpy() {
    auto hv = m_tensor->numpy();
    // if (!hv) {
    //     PyErr_SetString(PyExc_ValueError, "tensor invalid");
    //     return nullptr;
    // }
    auto arr = py::reinterpret_steal<py::array>(
            npy::ndarray_from_tensor(hv->as_nd(true), npy::ShareType::TRY_SHARE));
    if (!arr) {
        PyErr_SetString(PyExc_ValueError, "tensor invalid");
        return nullptr;
    }
    if (hv->shape().is_scalar()) {
        mgb_assert(PyArray_Check(arr.ptr()));
        return PyArray_Squeeze(reinterpret_cast<PyArrayObject*>(arr.ptr()));
    }
    return arr.release().ptr();
}

void TensorWrapper::reset(PyObject* tensor) {
    TensorWrapper* t = TensorWrapper::try_cast(tensor);
    if (!t) {
        throw py::type_error("expect Tensor");
    }
    m_tensor->reset(t->m_tensor->data());
}

PyObject* TensorWrapper::detach() {
    auto detached = imperative::apply(DetachGrad(), m_tensor->data())[0];
    return TensorWrapper::make(py_tensor_type, detached).release().ptr();
}

PyObject* TensorWrapper::_dev_tensor() {
    auto dv = m_tensor->data().dev_tensor();
    // TODO: handle scalar
    return py::cast(dv->as_nd(true)).release().ptr();
}

void TensorWrapper::_drop() {
    imperative::apply(DTRCommand(DTRCommand::Drop), m_tensor->data());
}

PyObject* TensorWrapper::isscalar() {
    if (m_tensor->is_scalar()) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

struct TensorWeakRef {
    std::weak_ptr<Tensor> wptr;

    TensorWeakRef(const TensorWrapper& tw) : wptr(tw.m_tensor) {}

    py::object operator()() {
        if (auto p = wptr.lock()) {
            return TensorWrapper::make(py_tensor_type, p);
        }
        return py::none();
    }
    int _use_cnt() { return wptr.use_count(); }
};

/* ============== convert inputs ============== */

// map numpy.dtype.kind to priority
inline uint8_t category_priority(char c) {
    switch (c) {
        case 'f':
            return 3;  // floating-point
        case 'i':
            return 2;  // signed integer
        case 'u':
            return 2;  // unsigned integer
        case 'b':
            return 1;  // boolean
        default:
            return 0;
    }
}

// Returns the maximum value of the priority of each type in the list `types`.
uint8_t max_priority(SmallVector<PyArray_Descr*> types) {
    if (types.size() == 0) {
        return 0;
    } else {
        uint8_t max_p = 0;
        for (auto&& desc : types) {
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
    for (auto&& desc : types) {
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

PyArray_Descr* _dtype_promotion(PyObject* const* args, size_t nargs) {
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
        PyObject* handle = is_tuple ? PyTuple_GetItem(tuple, i) : args[i];
        if (handle == Py_None)
            continue;
        TensorWrapper* tw = TensorWrapper::try_cast(handle);
        if (tw) {
            mgb::DType type = tw->m_tensor->dtype();
            auto&& descr = npy::dtype_mgb2np_descr(type);
            Py_INCREF(descr.get());
            tensors.emplace_back(descr.get());
        } else {
            if (PyArray_Check(handle) || PyArray_CheckScalar(handle)) {
                auto&& descr = PyArray_DescrFromObject(handle, nullptr);
                tensors.emplace_back(descr);
                continue;
            }

            if (py::isinstance<PySymbolVar>(py::handle(handle))) {
                auto var = py::handle(handle).cast<PySymbolVar*>();
                mgb::DType type = var->m_node->dtype();
                auto&& descr = npy::dtype_mgb2np_descr(type);
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
    } else {
        res = promote_types(tensors, max_pri_tensors);
    }
    for (auto* p : tensors) {
        Py_DECREF(p);
    }
    for (auto* p : scalars) {
        Py_DECREF(p);
    }
    Py_XDECREF(tuple);
    return res;
}

CompNode _get_device(PyObject* const* args, size_t nargs) {
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
                        : py::handle(handle).cast<PySymbolVar*>()->m_node->comp_node();
                valid = true;
            } else {
                CompNode cn1 = tw ? tw->m_tensor->comp_node()
                                  : py::handle(handle)
                                               .cast<PySymbolVar*>()
                                               ->m_node->comp_node();
                if (cn1 != cn) {
                    throw py::value_error(ssprintf(
                            "ambiguous device: %s vs %s", cn.to_string().c_str(),
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
PyObject* dtype_promotion(PyObject* self, PyObject* const* args, size_t nargs) {
    if (!nargs) {
        PyErr_SetString(PyExc_TypeError, "empty input is not allowed");
        return nullptr;
    }
    try {
        PyArray_Descr* res = _dtype_promotion(args, nargs);
        return py::cast(npy::dtype_np2mgb_descr(res)).release().ptr();
    }
    PYEXT17_TRANSLATE_EXC_RET(nullptr)
}

PyObject* get_device(PyObject* self, PyObject* const* args, size_t nargs) {
    if (!nargs) {
        PyErr_SetString(PyExc_TypeError, "empty input is not allowed");
        return nullptr;
    }
    try {
        CompNode cn = _get_device(args, nargs);
        return py::cast(cn).release().ptr();
    }
    PYEXT17_TRANSLATE_EXC_RET(nullptr)
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

    static auto& transformations = TransformationManager::get_instance();

    using Segment = TransformationManager::Segment;

    auto* channel = interpreter::Interpreter::inst().create_channel().release();
    interpreter_for_py = channel;
    transformations.register_at<Segment::Eval>(
            std::make_shared<InterpreterTransformation>(
                    std::unique_ptr<interpreter::Interpreter::Channel>(channel)));
    transformations.register_at<Segment::Scalar>(
            std::make_shared<ScalarTransformation>());

    static py::exception<interpreter::AsyncError> py_async_error(
            m, "AsyncError", PyExc_RuntimeError);
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (const interpreter::AsyncError& e) {
            pyext17::pybind11_translate_exception(e.nested_ptr());
            if (PyErr_Occurred()) {
                PyObject *exc, *val, *tb;
                PyErr_Fetch(&exc, &val, &tb);
                PyErr_NormalizeException(&exc, &val, &tb);
                if (tb) {
                    PyException_SetTraceback(val, tb);
                }
                auto val2 = py_async_error.py::object::operator()(
                        "An async error is reported. See above for the actual cause."
                        " Hint: This is where it is reported, not where it happened."
                        " You may call `megengine.config.async_level = 0 "
                        "to get better error reporting.");
                PyException_SetCause(
                        val2.ptr(), val);  // PyException_SetCause steals reference
                Py_XDECREF(exc);
                Py_XDECREF(tb);
                PyErr_Restore(
                        py_async_error.inc_ref().ptr(), val2.release().ptr(), nullptr);
            } else {
                py_async_error("Unkown async error");
            }
        }
    });

    auto* tensor_type =
            TensorWrapper::wrap_t::type()
                    .def<&TensorWrapper::numpy>("numpy")
                    .def_getset<&TensorWrapper::shape>("shape")
                    .def_getset<&TensorWrapper::dtype>("dtype")
                    .def_getset<&TensorWrapper::device>("device")
                    .def<&TensorWrapper::reset>("_reset")
                    .def<&TensorWrapper::isscalar>("_isscalar")
                    .def<&TensorWrapper::detach>("detach")
                    // TODO: remove this
                    .def<&TensorWrapper::_dev_tensor>("_dev_tensor")
                    .def<&TensorWrapper::_drop>("_drop")
                    .def<&TensorWrapper::_use_cnt>("_use_cnt")
                    .def<&TensorWrapper::_detail>("_detail")
                    .def<&TensorWrapper::_set_name>("_set_name")
                    .def<&TensorWrapper::_watch>("_watch")
                    .def_getset<
                            &TensorWrapper::module_trace_info,
                            &TensorWrapper::set_module_trace_info>("_NodeMixin__node")
                    .finalize();
    if (!tensor_type)
        throw py::error_already_set();
    py::setattr(m, "Tensor", tensor_type);

    py::class_<TensorWeakRef>(m, "TensorWeakRef")
            .def(py::init<const TensorWrapper&>())
            .def("__call__", &TensorWeakRef::operator())
            .def("_use_cnt", &TensorWeakRef::_use_cnt);

    py::class_<PySymbolVar, std::shared_ptr<PySymbolVar>>(m, "SymbolVar")
            .def_property_readonly(
                    "dtype", [](PySymbolVar* v) { return v->m_node->dtype(); })
            .def_property(
                    "var", [](PySymbolVar* v) { return v->m_node; },
                    [](PySymbolVar* s, cg::VarNode* v) { s->m_node = v; })
            .def_property_readonly(
                    "device", [](PySymbolVar* v) { return v->m_node->comp_node(); })
            .def_property_readonly(
                    "graph", [](PySymbolVar* v) { return v->m_node->owner_graph(); })
            .def_property_readonly(
                    "shape",
                    [](PySymbolVar* v) -> const TensorShape* {
                        auto&& mgr = v->m_node->owner_graph()->static_infer_manager();
                        return mgr.infer_shape_fallible(v->m_node);
                    })
            .def("numpy",
                 [](PySymbolVar* v) {
                     auto&& mgr = v->m_node->owner_graph()->static_infer_manager();
                     auto&& type = mgr.get_infer_type(v->m_node);
                     using InferType = cg::static_infer::InferType;
                     if (!(type.value & (InferType::CONST | InferType::RT_STATIC))) {
                         throw py::value_error("value invalid!");
                     }
                     auto* val = mgr.infer_value_fallible(v->m_node);
                     if (!val) {
                         throw py::value_error("value invalid!");
                     }
                     auto np_val = py::cast(*val).attr("numpy")();
                     return np_val;
                 })
            .def("_isscalar", [](PySymbolVar* v) { return v->is_scalar; })
            .def(py::init([](cg::VarNode* node) {
                     return std::make_shared<PySymbolVar>(node);
                 }),
                 py::arg() = nullptr);

    static PyMethodDef method_defs[] = {
            MGE_PY_INTERFACE(apply, py_apply),
            MGE_PY_INTERFACE(dtype_promotion, dtype_promotion),
            MGE_PY_INTERFACE(get_device, get_device),
            {nullptr, nullptr, 0, nullptr}};
    for (auto&& def : method_defs) {
        if (def.ml_meth != nullptr) {
            auto* func = PyCFunction_NewEx(&def, nullptr, nullptr);
            if (!func)
                throw py::error_already_set();
            py::setattr(m, def.ml_name, func);
        }
    }

    static constexpr auto sync_py_task_q = [] {
        py::gil_scoped_release _;
        py_task_q.wait_all_task_finish();
    };

    m.def("clear_candidates", [channel]() { channel->clear_candidates(); });
    m.def("set_option", [channel](std::string name, size_t value) {
        channel->set_option(name, value);
    });
    m.def("get_option",
          [channel](std::string name) { return channel->get_option(name); });
    m.def("set_buffer_length", [channel](int length) {
        mgb_assert(length >= 0 and length < 100, "buffer_length should be in [0, 100)");
        channel->set_option("buffer_length", length);
    });
    m.def("push_scope", [channel](std::string name) {
        Transformation::push_scope(name);
        channel->push_scope(name);
    });
    m.def("pop_scope", [channel](std::string name) {
        channel->pop_scope(name);
        Transformation::pop_scope(name);
    });
    m.def("start_profile", [channel](imperative::Profiler::options_t options) {
        channel->sync();
        imperative::Profiler::load_options(std::move(options));
        imperative::Profiler::start_profile();
        channel->start_profile();
    });
    m.def("stop_profile", [channel]() -> std::function<void(std::string, std::string)> {
        channel->stop_profile();
        channel->sync();
        imperative::Profiler::stop_profile();
        auto results = std::make_shared<imperative::Profiler::bundle_t>(
                imperative::Profiler::collect());
        return [results = results](std::string basename, std::string format) mutable {
            imperative::Profiler::dump_profile(basename, format, std::move(*results));
            results = nullptr;
        };
    });
    m.def("sync", [channel]() {
        if (channel->check_available()) {
            channel->sync();
        }
        sync_py_task_q();
    });
    m.def("full_sync", [channel]() {
        if (channel->check_available()) {
            channel->sync();
        }
        CompNode::sync_all();
        CompNode::foreach ([](CompNode cn) {
            auto err = cn.check_async_error();
            mgb_assert(!err, "%s", err->what());
        });
        sync_py_task_q();
    });
    m.def("close", [channel]() {
        channel->close();
        sync_py_task_q();
    });

    py::handle grad_key_type =
            GradKeyWrapper::wrap_t::type()
                    .def<&GradKeyWrapper::attach>("attach")
                    .def<&GradKeyWrapper::is_attached_to>("is_attached_to")
                    .def_getset<&GradKeyWrapper::get_name, &GradKeyWrapper::set_name>(
                            "name")
                    .def<&GradKeyWrapper::enter>("enter")
                    .def<&GradKeyWrapper::exit>("exit")
                    .def<&GradKeyWrapper::suppress>("suppress")
                    .def<&GradKeyWrapper::resume>("resume")
                    .finalize();
    if (!grad_key_type)
        throw py::error_already_set();
    py::setattr(m, "GradKey", grad_key_type);
    m.def("backward", &GradKeyWrapper::backward);
    m.def("get_backward_closure", &GradKeyWrapper::get_backward_closure);

    m.def("set_py_tensor_type", [](py::object type_obj) {
        py_tensor_type = reinterpret_cast<PyTypeObject*>(type_obj.inc_ref().ptr());
    });

    /**
     * \brief trace proxy
     *
     */
    struct Trace {
        bool symbolic = false;
        bool no_exec = false;
        bool capture_as_const = false;
        bool profile = false;
        bool record_input_shapes = false;
        py::function options_visitor;
        std::shared_ptr<TracingTransformation> tracing;
        std::shared_ptr<CompiledTransformation> compiled;
        std::shared_ptr<LazyEvalTransformation> lazy_eval;
        std::pair<size_t, std::shared_ptr<GraphProfiler>> profiler;
        std::optional<TraceResult> trace_result;
        std::function<bool(py::object, py::object)> array_comparator;

        bool compare_value(ValueRef lhs, ValueRef rhs) {
            if (!lhs.shape()->eq(*rhs.shape())) {
                return false;
            }
            HostTensorND lvalue = lhs.numpy()->as_nd(true);
            HostTensorND rvalue = rhs.numpy()->as_nd(true);
            auto larr = py::reinterpret_steal<py::array>(
                    npy::ndarray_from_tensor(lvalue, npy::ShareType::TRY_SHARE));
            auto rarr = py::reinterpret_steal<py::array>(
                    npy::ndarray_from_tensor(rvalue, npy::ShareType::TRY_SHARE));
            return array_comparator(larr, rarr);
        }

        void enter() {
            auto& self = *this;
            if (!self.trace_result) {  // untraced
                self.tracing = std::make_shared<TracingTransformation>(
                        self.capture_as_const, self.record_input_shapes);
                if (self.symbolic) {
                    self.lazy_eval =
                            std::make_shared<LazyEvalTransformation>(self.no_exec);
                    self.options_visitor(py::cast(&self.lazy_eval->options()));
                }
            } else if (!self.compiled) {  // traced but not compiled
                using namespace std::placeholders;
                self.compiled = std::make_shared<CompiledTransformation>(
                        *self.trace_result, self.record_input_shapes);
                self.compiled->set_value_comparator(
                        std::bind(&Trace::compare_value, this, _1, _2));
                self.options_visitor(py::cast(&self.compiled->options()));
                self.compiled->compile();
            }
            // register transformations
            if (self.compiled) {
                if (self.profile) {
                    auto& current_graph = self.compiled->graph();
                    if (self.profiler.first != self.compiled->graph().id()) {
                        // graph changed
                        self.profiler = std::make_pair(
                                current_graph.id(),
                                std::make_shared<GraphProfiler>(&current_graph));
                    }
                }
                transformations.register_at<Segment::Trace>(self.compiled);
                // start execute because InputCallback depends
                self.compiled->execute();
            } else if (self.tracing) {
                transformations.register_at<Segment::Trace>(self.tracing);
                if (self.lazy_eval) {
                    transformations.register_at<Segment::Eval>(self.lazy_eval);
                }
            } else {
                mgb_throw(MegBrainError, "invalid state: neither tracing nor compiled");
            }
        }

        void exit() {
            auto& self = *this;
            if (self.tracing) {
                transformations.unregister<Segment::Trace>(self.tracing);
                self.trace_result = self.tracing->get_result();
                self.tracing.reset();
                if (self.lazy_eval) {
                    auto lazy_eval = std::move(self.lazy_eval);
                    transformations.unregister<Segment::Eval>(lazy_eval);
                    lazy_eval->check_exception();
                }
            } else if (self.compiled) {
                transformations.unregister<Segment::Trace>(self.compiled);
                self.compiled->wait();
            } else {
                mgb_throw(MegBrainError, "invalid state: neither tracing nor compiled");
            }
        }

        VarNodeArray dump(
                std::shared_ptr<ComputingGraph> graph,
                std::vector<std::tuple<std::string, std::string, TensorShape>> inputs,
                std::vector<std::pair<std::string, std::string>> outputs,
                bool prefer_input_names) {
            auto& self = *this;
            mgb_assert(self.trace_result);
            // mark is like "arg_0", "kwarg_xxx", "output_0" ...
            std::unordered_map<std::string, size_t> mark2var;
            for (size_t i = 0; i < self.trace_result->vars.size(); ++i) {
                auto& name = self.trace_result->vars[i].mark;
                if (!name.empty()) {
                    mark2var[name] = i;
                }
            }
            std::vector<std::tuple<size_t, std::string, TensorShape>> input_vars;
            std::vector<std::pair<size_t, std::string>> output_vars;
            for (auto&& [input_mark, input_name, input_shape] : inputs) {
                mgb_assert(input_shape.ndim, "input shape invalid");
                input_vars.push_back(
                        {mark2var.at(input_mark), input_name, input_shape});
            }
            for (auto&& [output_name, repr] : outputs) {
                output_vars.push_back({mark2var.at(output_name), repr});
            }
            self.options_visitor(py::cast(&graph->options()));
            auto vars = self.trace_result->dump(
                    *graph, input_vars, output_vars, prefer_input_names);
            return vars;
        }
    };

    py::class_<Trace>(m, "Trace")
            .def(py::init<>())
            .def_readwrite("record_input_shapes", &Trace::record_input_shapes)
            .def_readwrite("array_comparator", &Trace::array_comparator)
            .def_readwrite("profile", &Trace::profile)
            .def_property_readonly(
                    "options",
                    [](Trace& self) {
                        if (self.compiled) {
                            return &self.compiled->options();
                        } else {
                            return (ComputingGraph::Options*)nullptr;
                        }
                    })
            .def("get_profile",
                 [](Trace& self) -> py::object {
                     if (self.profiler.second && self.compiled) {
                         auto json = self.profiler.second->to_json_full(
                                 self.compiled->graph().current_comp_seq());
                         return py::str(json->to_string());
                     } else {
                         return py::none();
                     }
                 })
            .def_readwrite("symbolic", &Trace::symbolic)
            .def_readwrite("capture_as_const", &Trace::capture_as_const)
            .def_readwrite("no_exec", &Trace::no_exec)
            .def_readwrite("options_visitor", &Trace::options_visitor)
            .def("enter", &Trace::enter)
            .def("exit", &Trace::exit)
            .def("dump", &Trace::dump)
            .def("begin_excluded_region",
                 [](Trace& self) {
                     mgb_assert(bool(self.tracing) ^ bool(self.compiled));
                     if (self.tracing) {
                         transformations.unregister<Segment::Trace>(self.tracing);
                     } else if (self.compiled) {
                         transformations.unregister<Segment::Trace>(self.compiled);
                     }
                 })
            .def("end_excluded_region", [](Trace& self) {
                mgb_assert(bool(self.tracing) ^ bool(self.compiled));
                if (self.tracing) {
                    transformations.register_at<Segment::Trace>(self.tracing);
                } else if (self.compiled) {
                    transformations.register_at<Segment::Trace>(self.compiled);
                }
            });

    m.def("name_tensor", [](std::string name, py::object tensor) {
        auto* tw = TensorWrapper::try_cast(tensor.ptr());
        auto output = imperative::apply(TraceMarkVar(name), tw->m_tensor->data())[0];
        tw->m_tensor->reset(output);
    });

    m.def("is_grad_attached", [](std::vector<py::object> tensors) -> bool {
        SmallVector<ValueRef> values;
        for (auto&& tensor : tensors) {
            values.push_back(tensor.cast<TensorWrapper>().m_tensor->data());
        }
        auto outputs = imperative::apply(GetGradKey(), values);
        if (outputs[0].is<GradKeyValue>()) {
            return true;
        } else {
            return false;
        }
    });

    m.def("get_grad_key", [](std::vector<py::object> tensors) -> py::object {
        SmallVector<ValueRef> values;
        for (auto&& tensor : tensors) {
            values.push_back(tensor.cast<TensorWrapper>().m_tensor->data());
        }
        auto outputs = imperative::apply(GetGradKey(), values);
        if (auto* grad_key_val = outputs[0].as<GradKeyValue>()) {
            return py::reinterpret_borrow<py::object>(
                    GradKeyWrapper::wrap_t::pycast(GradKeyWrapper::get(*grad_key_val)));
        } else {
            return py::none();
        }
    });

    m.def("set_grad", [](py::object py_key, py::function backward_fn,
                         std::vector<py::object> inputs,
                         std::vector<py::object> outputs) {
        mgb_assert(GradKeyWrapper::wrap_t::type().isinstance(py_key.ptr()));
        auto* key = reinterpret_cast<GradKeyWrapper::wrap_t*>(py_key.ptr())->inst();
        GenericFunction generic_backward_fn =
                [backward_fn](Span<ValueRef> output_grads) -> std::vector<ValueRef> {
            py::list output_grad_tws;
            for (auto&& output_grad : output_grads) {
                if (output_grad) {
                    output_grad_tws.append(
                            TensorWrapper::make(py_tensor_type, output_grad));
                } else {
                    output_grad_tws.append(py::none());
                }
            }
            py::tuple input_grad_tws = backward_fn(*output_grad_tws);
            std::vector<ValueRef> input_grads;
            for (auto&& input_grad_tw : input_grad_tws) {
                if (!input_grad_tw.is_none()) {
                    input_grads.push_back(
                            py::cast<TensorWrapper>(input_grad_tw).m_tensor->data());
                } else {
                    input_grads.push_back({});
                }
            }
            return input_grads;
        };
        SmallVector<ValueRef> values;
        for (auto&& input : inputs) {
            values.push_back(input.cast<TensorWrapper>().m_tensor->data());
        }
        for (auto&& output : outputs) {
            values.push_back(output.cast<TensorWrapper>().m_tensor->data());
        }
        auto wrapped_output_values = imperative::apply(
                SetGrad(key->m_key, generic_backward_fn, inputs.size()), values);
        std::vector<py::object> wrapped_outputs;
        mgb_assert(wrapped_output_values.size() == outputs.size());
        for (auto&& output_value : wrapped_output_values) {
            wrapped_outputs.push_back(
                    TensorWrapper::make(py_tensor_type, output_value));
        }
        return wrapped_outputs;
    });

    static py::function module_trace_hook;

    static std::shared_ptr<ModuleTraceTransformation> module_trace_transformation;
    static int module_tracing = 0;

    m.def("set_module_tracing", [=] {
        if (!module_trace_transformation) {
            mgb_assert(module_trace_hook);
            module_trace_transformation =
                    std::make_shared<ModuleTraceTransformation>(module_trace_hook);
        }
        if (++module_tracing == 1) {
            transformations.register_at<TransformationManager::ModuleTrace>(
                    module_trace_transformation);
        }
    });

    m.def("unset_module_tracing", [=] {
        if (--module_tracing == 0) {
            transformations.unregister<TransformationManager::ModuleTrace>(
                    module_trace_transformation);
        }
    });

    m.def("is_tracing_module", [=] { return module_tracing > 0; });

    m.def("set_module_trace_hook",
          [](py::function function) { module_trace_hook = function; });

    m.def("begin_record_values", [] { Value::begin_record_values(); });

    m.def("end_record_values", [] {
        std::vector<std::pair<size_t, std::string>> reprs;
        auto values = Value::end_record_values();
        for (auto&& value : values) {
            reprs.push_back({value.id(), value.to_string()});
        }
        return reprs;
    });

    py::register_exception<TraceError>(m, "TraceError");
}

#undef MGE_PY_INTERFACE

}  // namespace mgb::imperative::python
