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
#include "./helper.h"
#include "./tensor.h"

#include "megbrain/common.h"
#include "megbrain/imperative.h"
#include "megbrain/imperative/graph_builder.h"
#include "megbrain/imperative/ops/backward_graph.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/ops/utility.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/rng.h"

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
    return py::cast(op.*attr).release().ptr();
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
        // TODO: remove this guard which is used for pybind11 implicit conversion
        py::detail::loader_life_support guard{};
        op.*attr = py::cast<U>(py::handle(value));
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
    return py::cast(
            reinterpret_cast<PyOp(OpDef)*>(obj)->op->scope()).release().ptr();
}

int py_set_scope(PyObject* obj, PyObject* value, void* /* closure */) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    try {
        reinterpret_cast<PyOp(OpDef)*>(obj)->op
            ->set_scope(py::cast<std::string>(py::handle(value)));
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
struct EnumTrait;

#define PyEnumHead \
    static_assert(std::is_enum_v<T>); \
    PyObject_HEAD \
    T value; \
    constexpr static const char *name = EnumTrait<T>::name; \
    static PyTypeObject* type; \
    static const char* members[]; \
    static std::unordered_map<std::string, T> mem2value; \
    static PyObject* pyobj_insts[];

template<typename T>
struct EnumWrapper {
    PyEnumHead
    std::string to_string() const {
        return members[static_cast<size_t>(value)];
    }
    static PyObject* py_repr(PyObject* self) {
        return py::cast(
            std::string(name) + "." + reinterpret_cast<EnumWrapper*>(self)->to_string())
                .release().ptr();
    }
    static PyObject* tp_richcompare(PyObject *self, PyObject *other, int op) {
        if (op == Py_EQ || op == Py_NE) {
            T lhs, rhs;
            if (load(other, rhs) && load(self, lhs)) {
                RETURN_RICHCOMPARE(lhs, rhs, op);
            } else {
                RETURN_RICHCOMPARE(0, 1, op);
            }
        }
        Py_RETURN_NOTIMPLEMENTED;
    }
    static bool load(py::handle src, T& value) {
        PyObject* obj = src.ptr();
        if (PyObject_TypeCheck(obj, type)) {
            value = reinterpret_cast<EnumWrapper*>(obj)->value;
            return true;
        }
        if (py::isinstance<py::str>(src)) {
            auto&& iter = mem2value.find(
                normalize_enum(py::cast<std::string>(src)));
            if (iter != mem2value.end()) {
                value = iter->second;
                return true;
            } else {
                return false;
            }
        }
        return false;
    }
    static PyObject* cast(const T& value) {
        auto v = static_cast<std::underlying_type_t<T>>(value);
        mgb_assert(v <= EnumTrait<T>::max);
        PyObject* obj = pyobj_insts[v];
        Py_INCREF(obj);
        return obj;
    }
};

template<typename T>
struct BitCombinedEnumWrapper {
    PyEnumHead
    std::string to_string() const {
        uint32_t value_int = static_cast<uint32_t>(value);
        if (value_int == 0) {
            return "None";
        } else {
            std::string ret;
            bool first = true;
            for (uint32_t i = 0; i < 32; i++) {
                if (value_int >> i & 1) {
                    if (!first) {
                        ret += " + ";
                    } else {
                        first = false;
                    }
                    ret += (std::string(name) + "." + members[i]);
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
            if (load(input, value)) {
                return cast(value);
            } else {
                PyErr_SetString(PyExc_RuntimeError,
                    mgb::ssprintf("Cannot convert type %s to type %s\n",
                        input->ob_type->tp_name, name).c_str());
                return nullptr;
            }
        }
    }
    static PyObject* py_repr(PyObject* self) {
        return py::cast(
                reinterpret_cast<BitCombinedEnumWrapper*>(self)->to_string())
                        .release().ptr();
    }
    static PyObject* py_or(PyObject* self, PyObject* other) {
        if(!(self->ob_type == other->ob_type)){
            return PyErr_Format(
                    PyExc_RuntimeError,
                    "Operand in or operator must be the same type.");
        }
        T lhs = reinterpret_cast<BitCombinedEnumWrapper*>(self)->value,
          rhs = reinterpret_cast<BitCombinedEnumWrapper*>(other)->value;
        return cast(lhs | rhs);
    }
    static PyObject* py_and(PyObject* self, PyObject* other) {
        if (!(self->ob_type == other->ob_type)) {
            return PyErr_Format(
                    PyExc_RuntimeError,
                    "Operand in and operator must be the same type.");
        }
        T lhs = reinterpret_cast<BitCombinedEnumWrapper*>(self)->value,
          rhs = reinterpret_cast<BitCombinedEnumWrapper*>(other)->value;
        return cast(lhs & rhs);
    }
    static PyObject* tp_richcompare(PyObject* self, PyObject* other, int op) {
        if (op == Py_EQ || op == Py_NE) {
            T lhs, rhs;
            if (load(other, rhs) && load(self, lhs)) {
                RETURN_RICHCOMPARE(lhs, rhs, op);
            } else {
                RETURN_RICHCOMPARE(0, 1, op);
            }
        }
        Py_RETURN_NOTIMPLEMENTED;
    }
    static bool load(py::handle src, T& value) {
        PyObject* obj = src.ptr();
        if (PyObject_TypeCheck(obj, type)) {
            value = reinterpret_cast<BitCombinedEnumWrapper*>(obj)->value;
            return true;
        }
        if (py::isinstance<py::str>(src)) {
            auto&& iter = mem2value.find(
                normalize_enum(py::cast<std::string>(src)));
            if (iter != mem2value.end()) {
                value = iter->second;
                return true;
            } else {
                return false;
            }
        }
        if (py::isinstance<py::int_>(obj)) {
            auto v = py::cast<std::underlying_type_t<T>>(src);
            if(v > EnumTrait<T>::max) {
                return false;
            }
            value = static_cast<T>(v);
            return true;
        }
        return false;
    }
    static PyObject* cast(const T& value) {
        auto v = static_cast<std::underlying_type_t<T>>(value);
        mgb_assert(v <= EnumTrait<T>::max);
        if ((!v) || (v & (v - 1))) {
            PyObject* obj = type->tp_alloc(type, 0);
            reinterpret_cast<BitCombinedEnumWrapper*>(obj)->value = value;
            return obj;
        } else {
            PyObject* obj = pyobj_insts[__builtin_ctz(v)];
            Py_INCREF(obj);
            return obj;
        }
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

#define ENUM_CASTER_IMPL(T) \
bool type_caster<T>::load(handle src, bool) { \
    return EnumWrapper<T>::load(src, value); \
} \
handle type_caster<T>::cast(const T& value, return_value_policy, handle) { \
    return EnumWrapper<T>::cast(value); \
}
FOR_EACH_ENUM_PARAM(ENUM_CASTER_IMPL)

#define BIT_COMBINED_ENUM_CASTER_IMPL(T) \
bool type_caster<T>::load(handle src, bool) { \
    return BitCombinedEnumWrapper<T>::load(src, value); \
} \
handle type_caster<T>::cast(const T& value, return_value_policy, handle) { \
    return BitCombinedEnumWrapper<T>::cast(value); \
}
FOR_EACH_BIT_COMBINED_ENUM_PARAM(BIT_COMBINED_ENUM_CASTER_IMPL)

} // detail
} // PYBIND11_NAMESPACE

void init_ops(py::module m) {
    _init_py_op_def(m);
    _init_py_op_base(m);
    INIT_ALL_OP(m)

    m.def("new_rng_handle", &rng::new_handle);
    m.def("delete_rng_handle", [](size_t handle){
        // RNG op might execute after handle released due to async dispatch, so
        // we need sync before delete a handle to avoid memory leak or use-after-free
        if(python::interpreter_for_py->check_available()){
            python::interpreter_for_py->sync();
        }
        mgb::CompNode::sync_all();
        py_task_q.wait_all_task_finish();
        rng::delete_handle(handle);
    }, py::call_guard<py::gil_scoped_release>());
    m.def("set_global_rng_seed", &rng::set_global_rng_seed);
    m.def("get_global_rng_seed", &rng::get_global_rng_seed);
    m.def("get_rng_handle_compnode", &rng::get_rng_handle_compnode);

    struct PySubgraphBuilder {
        explicit PySubgraphBuilder(std::string name) : name{name}{}
        std::string name;
        std::shared_ptr<Subgraph> graph_storage = std::make_shared<Subgraph>();
        std::shared_ptr<UniqueKey> graph_key = std::make_shared<UniqueKey>();
        Subgraph& graph = *graph_storage;
        mgb::SmallVector<bool> output_grad_mask;
        Subgraph::var_t next_var = 1;

        std::shared_ptr<OpDef> build() const {
            return SubgraphOp::make(name, graph_storage, output_grad_mask, graph_key);
        }
    };

    py::class_<PySubgraphBuilder>(m, "SubgraphBuilder")
        .def(py::init<std::string>())
        .def("input", [](PySubgraphBuilder& self){
            auto var = self.next_var++;
            self.graph.inputs.push_back(var);
            return var;
        })
        .def("apply", [](PySubgraphBuilder& self, std::shared_ptr<OpDef> op, Subgraph::vars_t inputs, size_t nr_outputs){
            Subgraph::vars_t outputs;
            for (size_t i = 0; i < nr_outputs; ++i) {
                outputs.push_back(self.next_var++);
            }
            self.graph.exprs.push_back({op, inputs, outputs});
            return outputs;
        })
        .def("apply_const", [](PySubgraphBuilder& self, py::object value, mgb::DType dtype, mgb::CompNode cn){
            auto var = self.next_var++;
            mgb::HostTensorND hvalue(cn);
            npy::np2tensor(value.cast<py::array>().ptr(), npy::Meth::copy_into(&hvalue), dtype);
            self.graph.constants.push_back({var, Tensor::make(hvalue)});
            return var;
        })
        .def("outputs", [](PySubgraphBuilder& self, Subgraph::vars_t outputs){
            self.graph.outputs = outputs;
            self.output_grad_mask.resize(outputs.size(), true);
        })
        .def("outputs_has_grad", [](PySubgraphBuilder& self, mgb::SmallVector<bool> outputs_has_grad){
            mgb_assert(self.graph.outputs.size() == self.output_grad_mask.size());
            self.output_grad_mask = outputs_has_grad;
        })
        .def("get", [](PySubgraphBuilder& self){
            return (std::shared_ptr<OpDef>)self.build();
        })
        .def("compile", [](PySubgraphBuilder& self, int gopt_level){
            return (std::shared_ptr<OpDef>)CompiledOp::make(self.build(), gopt_level);
        });
}
