/**
 * \file imperative/python/src/numpy_dtypes_bfloat16.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./numpy_dtypes.h"

#include <Python.h>
#include <structmember.h>
#include <pybind11/operators.h>

#include "megbrain/common.h"
#include "megbrain/dtype.h"

#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

namespace {

struct BFloat16Type {
    static int npy_typenum;
    mgb::dt_bfloat16 value;

    struct PyObj;
    struct NpyType;

    template <typename S, typename T>
    struct NpyCast;
};

int BFloat16Type::npy_typenum;

/* ==================== BFloat16Type::NpyCast ==================== */

template <typename S>
struct BFloat16Type::NpyCast<S, BFloat16Type> {
    static void apply(void* from_, void* to_, npy_intp n, void* /*fromarr*/,
                      void* /*toarr*/) {
        auto from = static_cast<S*>(from_);
        auto to = static_cast<BFloat16Type*>(to_);
        for (npy_intp i = 0; i < n; ++i) {
            float cur = static_cast<float>(from[i]);
            to[i].value = cur;
        }
    }
};

template <typename T>
struct BFloat16Type::NpyCast<BFloat16Type, T> {
    static void apply(void* from_, void* to_, npy_intp n, void* /*fromarr*/,
                      void* /*toarr*/) {
        auto from = static_cast<BFloat16Type*>(from_);
        auto to = static_cast<T*>(to_);
        for (npy_intp i = 0; i < n; ++i) {
            to[i] = from[i].value;
        }
    }
};

/* ==================== BFloat16Type::PyObj ==================== */
struct BFloat16Type::PyObj {
    PyObject_HEAD BFloat16Type obj;

    static PyTypeObject py_type;

    static PyObject* from_bfloat16(BFloat16Type val) {
        auto p = reinterpret_cast<PyObj*>(py_type.tp_alloc(&py_type, 0));
        p->obj.value = val.value;
        return reinterpret_cast<PyObject*>(p);
    }

    static PyObject* py_new(PyTypeObject* type, PyObject* args, PyObject* kwds);
    static PyObject* py_repr(PyObject* obj);
    static PyObject* py_richcompare(PyObject* a, PyObject* b, int op);
};
PyTypeObject BFloat16Type::PyObj::py_type;

PyObject* BFloat16Type::PyObj::py_new(PyTypeObject* type, PyObject* args,
                                      PyObject* kwds) {
    PyObj* self;
    Py_ssize_t size;

    self = (PyObj*)type->tp_alloc(type, 0);

    size = PyTuple_GET_SIZE(args);
    if (size > 1) {
        PyErr_SetString(PyExc_TypeError, "BFloat16Type Only has 1 parameter");
        return NULL;
    }
    PyObject* x = PyTuple_GET_ITEM(args, 0);
    if (PyObject_IsInstance(x, (PyObject*)&py_type)) {
        Py_INCREF(x);
        return x;
    }

    if (!PyFloat_Check(x)) {
        PyErr_SetString(PyExc_TypeError,
                        "BFloat16Type must be initialized wit float");
        return NULL;
    }

    const float s = PyFloat_AsDouble(x);

    self->obj.value = s;

    return (PyObject*)self;
}

PyObject* BFloat16Type::PyObj::py_repr(PyObject* obj) {
    float fval = static_cast<float>(((PyObj*)obj)->obj.value);
    return PyUnicode_FromString(mgb::ssprintf("%f", fval).c_str());
}

PyObject* BFloat16Type::PyObj::py_richcompare(PyObject* a, PyObject* b,
                                              int op) {
    mgb_assert(PyObject_IsInstance(a, (PyObject*)&py_type));
    auto bval = PyFloat_AsDouble(b);
    if (bval == -1 && PyErr_Occurred()) {
        return NULL;
    }
    double aval = ((PyObj*)a)->obj.value;
#define OP(py, op)           \
    case py: {               \
        if (aval op bval) {  \
            Py_RETURN_TRUE;  \
        } else {             \
            Py_RETURN_FALSE; \
        }                    \
    }
    switch (op) {
        OP(Py_LT, <)
        OP(Py_LE, <=)
        OP(Py_EQ, ==)
        OP(Py_NE, !=)
        OP(Py_GT, >)
        OP(Py_GE, >=)
    };
#undef OP
    return Py_NotImplemented;
}

/* ==================== BFloat16Type<N>::NpyType ==================== */
struct BFloat16Type::NpyType {
    static PyArray_ArrFuncs funcs;
    static PyArray_Descr descr;

    static bool init();

    static void copyswap(void* dst, void* src, int swap, void* /*arr*/) {
        if (src) {
            mgb_assert(!swap);
            memcpy(dst, src, sizeof(BFloat16Type));
        }
    }
    static PyObject* getitem(void* data, void* ap) {
        return BFloat16Type::PyObj::from_bfloat16(
                *static_cast<BFloat16Type*>(data));
    }
    static int setitem(PyObject* op, void* ov, void* ap);
};

PyArray_ArrFuncs BFloat16Type::NpyType::funcs;
PyArray_Descr BFloat16Type::NpyType::descr;

int BFloat16Type::NpyType::setitem(PyObject* op, void* ov, void* ap) {
    if (PyLong_Check(op)) {
        int a = PyLong_AsLong(op);
        static_cast<BFloat16Type*>(ov)->value = a;
    } else if (PyFloat_Check(op)) {
        float a = PyFloat_AsDouble(op);
        static_cast<BFloat16Type*>(ov)->value = a;
    } else if (PyObject_IsInstance(
                       op, (PyObject*)(&(BFloat16Type::PyObj::py_type)))) {
        static_cast<BFloat16Type*>(ov)->value = ((PyObj*)op)->obj.value;
    } else {
        PyErr_SetString(PyExc_ValueError,
                        "input type must be int/float/bfloat16");
        return -1;
    }
    return 0;
}

bool BFloat16Type::NpyType::init() {
    descr = {PyObject_HEAD_INIT(0) & BFloat16Type::PyObj::py_type,
             'V',  // kind
             'f',  // type
             '=',  // byteorder
             NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM,
             1,  // type num
             sizeof(BFloat16Type),
             alignof(BFloat16Type),
             NULL,
             NULL,
             NULL,
             &funcs};
    Py_TYPE(&descr) = &PyArrayDescr_Type;
    PyArray_InitArrFuncs(&funcs);
    funcs.copyswap = copyswap;
    funcs.getitem = getitem;
    funcs.setitem = setitem;
    npy_typenum = PyArray_RegisterDataType(&descr);

#define REGISTER_CAST(From, To, From_descr, To_typenum, safe)         \
    {                                                                 \
        PyArray_Descr* from_descr = (From_descr);                     \
        if (PyArray_RegisterCastFunc(from_descr, (To_typenum),        \
                                     NpyCast<From, To>::apply) < 0) { \
            return false;                                             \
        }                                                             \
        if (safe && PyArray_RegisterCanCast(from_descr, (To_typenum), \
                                            NPY_NOSCALAR) < 0) {      \
            return false;                                             \
        }                                                             \
    }
#define REGISTER_INT_CASTS(bits)                                         \
    REGISTER_CAST(npy_int##bits, BFloat16Type,                           \
                  PyArray_DescrFromType(NPY_INT##bits),                  \
                  BFloat16Type::npy_typenum, 1)                          \
    REGISTER_CAST(BFloat16Type, npy_int##bits, &descr, NPY_INT##bits, 0) \
    REGISTER_CAST(npy_uint##bits, BFloat16Type,                          \
                  PyArray_DescrFromType(NPY_UINT##bits),                 \
                  BFloat16Type::npy_typenum, 1)                          \
    REGISTER_CAST(BFloat16Type, npy_uint##bits, &descr, NPY_UINT##bits, 0)

    REGISTER_INT_CASTS(8)
    REGISTER_INT_CASTS(16)
    REGISTER_INT_CASTS(32)
    REGISTER_INT_CASTS(64)
    REGISTER_CAST(BFloat16Type, float, &descr, NPY_FLOAT, 0)
    REGISTER_CAST(float, BFloat16Type, PyArray_DescrFromType(NPY_FLOAT),
                  BFloat16Type::npy_typenum, 0)
    REGISTER_CAST(BFloat16Type, double, &descr, NPY_DOUBLE, 1)
    REGISTER_CAST(double, BFloat16Type, PyArray_DescrFromType(NPY_DOUBLE),
                  BFloat16Type::npy_typenum, 0)
    return true;
}

}  // anonymous namespace

// define a new python type: pybfloat16
bool init_pytype_bfloat16() {
    auto& py_type = BFloat16Type::PyObj::py_type;
    py_type = {PyVarObject_HEAD_INIT(NULL, 0)};
    py_type.tp_name = "megengine.core._imperative_rt.pybfloat16";
    py_type.tp_basicsize = sizeof(BFloat16Type::PyObj);
    py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    py_type.tp_doc = "bfloat16 type";
    py_type.tp_new = BFloat16Type::PyObj::py_new;
    py_type.tp_str = BFloat16Type::PyObj::py_repr;
    py_type.tp_repr = BFloat16Type::PyObj::py_repr;
    py_type.tp_richcompare = BFloat16Type::PyObj::py_richcompare;
    py_type.tp_base = &PyGenericArrType_Type;
    return PyType_Ready(&py_type) >= 0;
}

int mgb::npy_num_bfloat16() {
    return BFloat16Type::npy_typenum;
}

namespace py = pybind11;

void mgb::init_npy_num_bfloat16(py::module m) {
    mgb_assert(init_pytype_bfloat16());
    mgb_assert(BFloat16Type::NpyType::init());
    m.add_object("pybfloat16", reinterpret_cast<PyObject*>(
        &BFloat16Type::PyObj::py_type));
    m.add_object("bfloat16", reinterpret_cast<PyObject*>(
        PyArray_DescrFromType(npy_num_bfloat16())));
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
