/**
 * \file imperative/python/src/ops.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./ops.h"

#include "megbrain/imperative.h"
#include "megbrain/imperative/ops/backward_graph.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/ops/utility.h"
#include "megbrain/imperative/ops/autogen.h"

#include <Python.h>
#include <unordered_map>

namespace py = pybind11;
using namespace mgb::imperative;

namespace {
auto normalize_enum(const std::string& in) {
    std::string ret;
    for (auto&& c : in) {
        ret += toupper(c);
    }
    return ret;
}
} // anonymous namespace

#define CATCH_ALL(RETVAL) \
    catch(py::error_already_set& e) { \
        e.restore(); \
        return RETVAL; \
    } catch(py::builtin_exception& e) { \
        e.set_error(); \
        return RETVAL; \
    } catch(std::exception& e) { \
        PyErr_SetString(PyExc_RuntimeError, e.what()); \
        return RETVAL; \
    } \

namespace {
#define PyOp(name) Py##name
#define PyOpType(name) PyOp(name)::py_type

#define PyOpDefBegin(name) \
struct PyOp(name) : PyOpDef { \
    using Ty = name; \
    Ty& inst() { return op->cast_final_safe<Ty>(); } \
    static PyTypeObject py_type;

#define PyOpDefEnd(name) \
}; \
PyTypeObject PyOpType(name);

#define RETURN_RICHCOMPARE(val1, val2, op)                               \
    do {                                                                    \
        switch (op) {                                                       \
        case Py_EQ: if ((val1) == (val2)) Py_RETURN_TRUE; Py_RETURN_FALSE;  \
        case Py_NE: if ((val1) != (val2)) Py_RETURN_TRUE; Py_RETURN_FALSE;  \
        case Py_LT: if ((val1) < (val2)) Py_RETURN_TRUE; Py_RETURN_FALSE;   \
        case Py_GT: if ((val1) > (val2)) Py_RETURN_TRUE; Py_RETURN_FALSE;   \
        case Py_LE: if ((val1) <= (val2)) Py_RETURN_TRUE; Py_RETURN_FALSE;  \
        case Py_GE: if ((val1) >= (val2)) Py_RETURN_TRUE; Py_RETURN_FALSE;  \
        default:                                                            \
            Py_FatalError("Unreachable C code path reached");               \
        }                                                                   \
    } while (0)

template <typename T, typename SFINAE = void>
struct pyobj_convert_generic {
    static T from(PyObject* obj) {
        // TODO: remove this guard which is used for pybind11 implicit conversion
        py::detail::loader_life_support guard{};
        return py::cast<T>(py::handle(obj));
    }
    template<typename U,
        typename = std::enable_if_t<std::is_same_v<T, std::decay_t<U>>>>
    static PyObject* to(U&& t) {
        return py::cast(std::forward<U>(t)).release().ptr();
    }
};

template<typename T, typename SFINAE=void>
struct EnumTrait;

template <typename T>
struct EnumTrait<T, std::enable_if_t<std::is_enum_v<T>>> {
    static constexpr bool is_bit_combined = false;
    static constexpr std::underlying_type_t<T> max = 0;
};

template <typename T>
PyObject* py_new_generic(PyTypeObject* type, PyObject*, PyObject*) {
    PyObject* obj = type->tp_alloc(type, 0);
    T* self = reinterpret_cast<T*>(obj);
    if (self != NULL) {
        self->op = T::Ty::make();
    }
    return obj;
}

template<typename T>
void py_dealloc_generic(PyObject* obj) {
    reinterpret_cast<T*>(obj)->op.reset();
    Py_TYPE(obj)->tp_free(obj);
}

template<typename T, typename U, U T::Ty::*attr>
PyObject* py_get_generic_impl(PyObject* obj, void* /* closure */) {
    auto& op = reinterpret_cast<T*>(obj)->inst();
    return pyobj_convert_generic<U>::to(op.*attr);
}
#define py_get_generic(name, attr) \
    py_get_generic_impl<PyOp(name), decltype(std::declval<name>().attr), &name::attr>

template<typename T, typename U, U T::Ty::*attr>
int py_set_generic_impl(PyObject* obj, PyObject* value, void* /* closure */) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    auto& op = reinterpret_cast<T*>(obj)->inst();
    try {
        op.*attr = pyobj_convert_generic<U>::from(value);
    } CATCH_ALL(-1)
    return 0;
}
#define py_set_generic(name, attr) \
    py_set_generic_impl<PyOp(name), decltype(std::declval<name>().attr), &name::attr>

struct PyOpDef {
    PyObject_HEAD
    std::shared_ptr<OpDef> op;
    static PyTypeObject py_type;
    static std::unordered_map<mgb::Typeinfo*, PyTypeObject*> ctype2pytype;
    static PyGetSetDef py_getsetters[];
    static Py_hash_t tp_hash(PyObject *obj);
    static PyObject* tp_richcompare(PyObject *self, PyObject *other, int op);
};
PyTypeObject PyOpType(OpDef);
std::unordered_map<mgb::Typeinfo*, PyTypeObject*> PyOp(OpDef)::ctype2pytype;

PyObject* py_get_scope(PyObject* obj, void* /* closure */) {
    return pyobj_convert_generic<std::string>::to(
        reinterpret_cast<PyOp(OpDef)*>(obj)->op->scope());
}

int py_set_scope(PyObject* obj, PyObject* value, void* /* closure */) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    try {
        reinterpret_cast<PyOp(OpDef)*>(obj)->op
            ->set_scope(pyobj_convert_generic<std::string>::from(value));
    } CATCH_ALL(-1)
    return 0;
}

PyGetSetDef PyOp(OpDef)::py_getsetters[] = {
    {const_cast<char*>("scope"), py_get_scope, py_set_scope, "scope", NULL},
    {NULL}
};

Py_hash_t PyOp(OpDef)::tp_hash(PyObject *obj) {
    return static_cast<Py_hash_t>(
        reinterpret_cast<PyOp(OpDef)*>(obj)->op->hash());
}

PyObject* PyOp(OpDef)::tp_richcompare(PyObject *self, PyObject *other, int op) {
    bool same = reinterpret_cast<PyOp(OpDef)*>(self)->op->is_same(
        *reinterpret_cast<PyOp(OpDef)*>(other)->op);
    if (op == Py_EQ || op == Py_NE) {
        RETURN_RICHCOMPARE(same, true, op);
    }
    Py_RETURN_NOTIMPLEMENTED;
}

template<typename T>
struct EnumWrapper {
    static_assert(std::is_enum_v<T>);
    PyObject_HEAD
    T value;
    static const char* name;
    static PyTypeObject type;
    static std::unordered_map<T, std::string> type2str;
    static std::unordered_map<std::string, T> str2type;
    EnumWrapper() = default;
    EnumWrapper(T v): value(v) {}
    EnumWrapper(std::string&& str): EnumWrapper(str2type.at(normalize_enum(str))) {}
    std::string to_string() const {
        return type2str.at(value);
    }
    static PyObject* py_repr(PyObject* self) {
        return pyobj_convert_generic<std::string>::to(
            std::string(name) + "." + reinterpret_cast<EnumWrapper*>(self)->to_string());
    }
    static PyObject* tp_richcompare(PyObject *self, PyObject *other, int op) {
        T lhs = reinterpret_cast<EnumWrapper*>(self)->value,
          rhs = reinterpret_cast<EnumWrapper*>(other)->value;
        if (op == Py_EQ || op == Py_NE) {
            RETURN_RICHCOMPARE(lhs, rhs, op);
        }
        Py_RETURN_NOTIMPLEMENTED;
    }
};

template <typename T>
struct pyobj_convert_generic<T,
                             std::enable_if_t<std::is_enum_v<std::decay_t<T>> &&
                                              !EnumTrait<T>::is_bit_combined>> {
    using Wrapper = EnumWrapper<T>;
    static T from(PyObject* obj) {
        if (PyObject_TypeCheck(obj, &Wrapper::type)) {
            return reinterpret_cast<Wrapper*>(obj)->value;
        }
        // try as string
        // TODO: type checkcd
        return Wrapper(pyobj_convert_generic<std::string>::from(obj)).value;
    }
    static PyObject* to(T t) {
        PyTypeObject* pytype = &Wrapper::type;
        PyObject* obj = pytype->tp_alloc(pytype, 0);
        reinterpret_cast<Wrapper*>(obj)->value = t;
        return obj;
    }
};

template<typename T>
struct BitCombinedEnumWrapper {
    static_assert(std::is_enum_v<T>);
    PyObject_HEAD
    T value;
    static const char* name;
    static PyTypeObject type;
    static std::unordered_map<T, std::string> type2str;
    static std::unordered_map<std::string, T> str2type;
    static PyNumberMethods number_methods;
    BitCombinedEnumWrapper() = default;
    BitCombinedEnumWrapper(T v): value(v) {}
    BitCombinedEnumWrapper(std::string&& str)
            : BitCombinedEnumWrapper(str2type.at(normalize_enum(str))) {}
    std::string to_string() const {
        if (static_cast<uint32_t>(value) == 0) {
            return "None";
        } else {
            auto ret = std::string();
            bool first = true;
            for (uint32_t i = 0; i < 32; i++) {
                uint32_t value_int = static_cast<uint32_t>(value);
                auto it = type2str.find(static_cast<T>((1 << i) & value_int));
                if (it != type2str.end()) {
                    if (!first) {
                        ret += " + ";
                    } else {
                        first = false;
                    }
                    ret += (std::string(name) + "." + it->second);
                }
            }
            return ret;
        }
    }
    static PyObject* py_new_combined_enum(PyTypeObject* type, PyObject* args, PyObject*) {
        if (!PyTuple_Size(args)) {
            PyObject* obj = type->tp_alloc(type, 0);
            reinterpret_cast<BitCombinedEnumWrapper*>(obj)->value = T();
            return obj;
        }
        else {
            PyObject* input;
            if (!PyArg_ParseTuple(args, "|O", &input)) {
                return nullptr;
            }
            T value;
            try {
                value = pyobj_convert_generic<T>::from(input);
            } CATCH_ALL(nullptr);
            PyObject* obj = type->tp_alloc(type, 0);
            reinterpret_cast<BitCombinedEnumWrapper*>(obj)->value = value;
            return obj;
        }
    }
    static PyObject* py_repr(PyObject* self) {
        return pyobj_convert_generic<std::string>::to(
                reinterpret_cast<BitCombinedEnumWrapper*>(self)->to_string());
    }
    static PyObject* py_or(PyObject* self, PyObject* other) {
        if(!(self->ob_type == other->ob_type)){
            return PyErr_Format(
                    PyExc_RuntimeError,
                    "Operand in or operator must be the same type.");
        }
        PyObject* obj = type.tp_alloc(&type, 0);
        T lhs = reinterpret_cast<BitCombinedEnumWrapper*>(self)->value,
          rhs = reinterpret_cast<BitCombinedEnumWrapper*>(other)->value;
        reinterpret_cast<BitCombinedEnumWrapper*>(obj)->value = static_cast<T>(
                static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
        return obj;
    }
    static PyObject* py_and(PyObject* self, PyObject* other) {
        if (!(self->ob_type == other->ob_type)) {
            return PyErr_Format(
                    PyExc_RuntimeError,
                    "Operand in and operator must be the same type.");
        }
        PyObject* obj = type.tp_alloc(&type, 0);
        T lhs = reinterpret_cast<BitCombinedEnumWrapper*>(self)->value,
          rhs = reinterpret_cast<BitCombinedEnumWrapper*>(other)->value;
        reinterpret_cast<BitCombinedEnumWrapper*>(obj)->value = static_cast<T>(
                static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
        return obj;
    }
    static PyObject* tp_richcompare(PyObject* self, PyObject* other, int op) {
        T lhs = reinterpret_cast<BitCombinedEnumWrapper*>(self)->value,
          rhs = reinterpret_cast<BitCombinedEnumWrapper*>(other)->value;
        if (op == Py_EQ || op == Py_NE) {
            RETURN_RICHCOMPARE(lhs, rhs, op);
        }
        Py_RETURN_NOTIMPLEMENTED;
    }
};

template <typename T>
struct pyobj_convert_generic<T,
                             std::enable_if_t<std::is_enum_v<std::decay_t<T>> &&
                                              EnumTrait<T>::is_bit_combined>> {
    using Wrapper = BitCombinedEnumWrapper<T>;
    static T from(PyObject* obj) {
        if (PyObject_TypeCheck(obj, &Wrapper::type)) {
            return reinterpret_cast<Wrapper*>(obj)->value;
        } else if(PyLong_Check(obj)) {
            auto value = pyobj_convert_generic<std::underlying_type_t<T>>::from(obj);
            mgb_throw_if(value > EnumTrait<T>::max, mgb::MegBrainError,
                    "out of range, cannot convert %zu to %s",
                    static_cast<uint32_t>(value), Wrapper::name);
            return static_cast<T>(value);
        }
        // try as string
        // TODO: type checkcd
        return Wrapper(pyobj_convert_generic<std::string>::from(obj)).value;
    }
    static PyObject* to(T t) {
        PyTypeObject* pytype = &Wrapper::type;
        PyObject* obj = pytype->tp_alloc(pytype, 0);
        reinterpret_cast<Wrapper*>(obj)->value = t;
        return obj;
    }
};

void _init_py_op_def(py::module m) {
    using py_op = PyOp(OpDef);
    auto& py_type = PyOpType(OpDef);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.OpDef";
    py_type.tp_basicsize = sizeof(PyOp(OpDef));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "OpDef";
    py_type.tp_base = &PyBaseObject_Type;
    py_type.tp_hash = PyOp(OpDef)::tp_hash;
    py_type.tp_richcompare = PyOp(OpDef)::tp_richcompare;
    py_type.tp_getset = py_op::py_getsetters;
    mgb_assert(PyType_Ready(&py_type) >= 0);
    m.add_object("OpDef", reinterpret_cast<PyObject*>(&py_type));
}

/*********** begin of hand-write opdefs **************/

PyOpDefBegin(BackwardGraph) // {{
// };
PyOpDefEnd(BackwardGraph)

void _init_py_backward_graph(py::module m) {
    using py_op = PyOp(BackwardGraph);
    auto& py_type = PyOpType(BackwardGraph);
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.BackwardGraph";
    py_type.tp_basicsize = sizeof(PyOp(BackwardGraph));
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "BackwardGraph";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_new_generic<py_op>;
    mgb_assert(PyType_Ready(&py_type) >= 0);
    // FIXME: rewrite interpret function in cpython instead wrap directly by pybind11::cppfunction
    auto interpret = py::cpp_function(
        [](OpDef& self, py::object pyf, py::object pyc,
                const mgb::SmallVector<py::object>& inputs) {
            auto f = [pyf](OpDef& op, const mgb::SmallVector<py::object>& inputs) {
                return py::cast<mgb::SmallVector<py::object>>(pyf(op.shared_from_this(), inputs));
            };
            auto c = [pyc](const TensorPtr& tensor) {
                return pyc(tensor->dev_tensor());
            };
            return self.cast_final_safe<BackwardGraph>().graph().interpret<py::object>(f, c, inputs);
        });
    mgb_assert(PyDict_SetItemString(
        py_type.tp_dict, "interpret", interpret.release().ptr()) >= 0);
    PyType_Modified(&py_type);
    m.add_object("BackwardGraph", reinterpret_cast<PyObject*>(&py_type));
    mgb_assert(PyOp(OpDef)::ctype2pytype.emplace(BackwardGraph::typeinfo(), &py_type).second);
}

struct PyOpBase : PyOpDef {
    static PyTypeObject py_type;

    static PyObject* tp_new(PyTypeObject* type, PyObject*, PyObject*) {
        auto* obj = type->tp_alloc(type, 0);
        if (obj) {
            auto* self = reinterpret_cast<PyOpBase*>(obj);
            new(&self->op) decltype(self->op);
        }
        return obj;
    }
};
PyTypeObject PyOpBase::py_type;

void _init_py_op_base(py::module m) {
    using py_op = PyOpBase;
    auto& py_type = PyOpBase::py_type;
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.ops.PyOpBase";
    py_type.tp_basicsize = sizeof(py_op);
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "PyOpBase";
    py_type.tp_base = &PyOpType(OpDef);
    py_type.tp_dealloc = py_dealloc_generic<py_op>;
    py_type.tp_new = py_op::tp_new;
    mgb_assert(PyType_Ready(&py_type) >= 0);
    m.add_object("PyOpBase", reinterpret_cast<PyObject*>(&py_type));
}

/*********** end of hand-write opdefs **************/

// auto generated opdefs
#include "opdef.cpy.inl"

#undef CATCH_ALL

} // anonymous namespace

namespace PYBIND11_NAMESPACE {
namespace detail {
bool type_caster<OpDef>::load(handle src, bool convert) {
    PyObject* obj = src.ptr();
    if (!PyObject_TypeCheck(obj, &PyOpType(OpDef))) {
        return false;
    }
    value = reinterpret_cast<PyOp(OpDef)*>(obj)->op;
    if (!value) {
        // opdef only defined in Python
        value = std::make_shared<GenericPyOp>(reinterpret_borrow<object>(src));
    }
    return true;
}
handle type_caster<OpDef>::cast(const OpDef& op, return_value_policy, handle) {
    if (auto* pyop = op.try_cast_final<GenericPyOp>()) {
        return object(pyop->obj).release();
    }
    PyTypeObject* pytype;
    auto& c2p = PyOp(OpDef)::ctype2pytype;
    auto&& iter = c2p.find(op.dyn_typeinfo());
    if (iter != c2p.end()) { // FIXME: should always meet this condition
        pytype = iter->second;
    } else { // which means unregistered op type, jsut make it as an opaque op type
        // currently, only OprAttr goes into this branch
        pytype = &PyOpType(OpDef);
    }
    PyObject* obj = pytype->tp_alloc(pytype, 0);
    mgb_assert(PyObject_TypeCheck(obj, &PyOpType(OpDef)));
    reinterpret_cast<PyOp(OpDef)*>(obj)->op = const_cast<OpDef&>(op).shared_from_this();
    return py::handle(obj);
}
} // detail
} // PYBIND11_NAMESPACE

void init_ops(py::module m) {
    _init_py_op_def(m);
    _init_py_backward_graph(m);
    _init_py_op_base(m);
    INIT_ALL_OP(m)
}
