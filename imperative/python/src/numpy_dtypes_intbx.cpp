/**
 * \file imperative/python/src/numpy_dtypes_intbx.cpp
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

#include "megbrain/common.h"

#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

namespace {

template <size_t N>
struct LowBitType {
    static_assert(N < 8, "low bit only supports less than 8 bits");
    static int npy_typenum;
    //! numerical value (-3, -1, 1, 3)
    int8_t value;

    struct PyObj;
    struct NpyType;

    const static int32_t max_value = (1 << N) - 1;

    //! check whether val is (-3, -1, 1, 3) and set python error
    static bool check_value_set_err(int val) {
        int t = val + max_value;
        if ((t & 1) || t < 0 || t > (max_value << 1)) {
            PyErr_SetString(PyExc_ValueError,
                            mgb::ssprintf("low bit dtype number error: "
                                          "value=%d; allowed {-3, -1, 1, 3}",
                                          val)
                                    .c_str());
            return false;
        }

        return true;
    }

    template <typename S, typename T>
    struct NpyCast;
};

template <size_t N>
int LowBitType<N>::npy_typenum;

/* ==================== LowBitType::NpyCast ==================== */

template <size_t N>
template <typename S>
struct LowBitType<N>::NpyCast<S, LowBitType<N>> {
    static void apply(void* from_, void* to_, npy_intp n, void* /*fromarr*/,
                      void* /*toarr*/) {
        auto from = static_cast<S*>(from_);
        auto to = static_cast<LowBitType<N>*>(to_);
        for (npy_intp i = 0; i < n; ++i) {
            int cur = static_cast<int>(from[i]);
            if (!LowBitType<N>::check_value_set_err(cur))
                return;
            to[i].value = cur;
        }
    }
};

template <size_t N>
template <typename T>
struct LowBitType<N>::NpyCast<LowBitType<N>, T> {
    static void apply(void* from_, void* to_, npy_intp n, void* /*fromarr*/,
                      void* /*toarr*/) {
        auto from = static_cast<LowBitType<N>*>(from_);
        auto to = static_cast<T*>(to_);
        for (npy_intp i = 0; i < n; ++i) {
            to[i] = from[i].value;
        }
    }
};

/* ==================== LowBitType::PyObj ==================== */
template <size_t N>
struct LowBitType<N>::PyObj {
    PyObject_HEAD LowBitType<N> obj;

    static PyTypeObject py_type;

    static PyObject* from_lowbit(LowBitType<N> val) {
        auto p = reinterpret_cast<PyObj*>(py_type.tp_alloc(&py_type, 0));
        p->obj.value = val.value;
        return reinterpret_cast<PyObject*>(p);
    }

    static PyObject* py_new(PyTypeObject* type, PyObject* args, PyObject* kwds);
    static PyObject* py_repr(PyObject* obj);
    static PyObject* py_richcompare(PyObject* a, PyObject* b, int op);
};
template <size_t N>
PyTypeObject LowBitType<N>::PyObj::py_type;

template <size_t N>
PyObject* LowBitType<N>::PyObj::py_new(PyTypeObject* type, PyObject* args,
                                       PyObject* kwds) {
    PyObj* self;
    Py_ssize_t size;

    self = (PyObj*)type->tp_alloc(type, 0);

    size = PyTuple_GET_SIZE(args);
    if (size > 1) {
        PyErr_SetString(PyExc_TypeError, "LowBitType Only has 1 parameter");
        return NULL;
    }
    PyObject* x = PyTuple_GET_ITEM(args, 0);
    if (PyObject_IsInstance(x, (PyObject*)&py_type)) {
        Py_INCREF(x);
        return x;
    }

    if (!PyLong_Check(x)) {
        PyErr_SetString(PyExc_TypeError,
                        "LowBitType must be initialized wit int");
        return NULL;
    }

    const long s = PyLong_AsLong(x);

    self->obj.value = s;

    return (PyObject*)self;
}

template <size_t N>
PyObject* LowBitType<N>::PyObj::py_repr(PyObject* obj) {
    return PyUnicode_FromFormat("%d", ((PyObj*)obj)->obj.value);
}

template <size_t N>
PyObject* LowBitType<N>::PyObj::py_richcompare(PyObject* a, PyObject* b,
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

/* ==================== LowBitType<N>::NpyType ==================== */
template <size_t N>
struct LowBitType<N>::NpyType {
    static PyArray_ArrFuncs funcs;
    static PyArray_Descr descr;

    static bool init();

    static void copyswap(void* dst, void* src, int swap, void* /*arr*/) {
        if (src) {
            mgb_assert(!swap);
            memcpy(dst, src, sizeof(LowBitType<N>));
        }
    }
    static PyObject* getitem(void* data, void* ap) {
        return LowBitType<N>::PyObj::from_lowbit(
                *static_cast<LowBitType<N>*>(data));
    }
    static int setitem(PyObject* op, void* ov, void* ap);
    static int fill(void* data_, npy_intp length, void* arr);
};

template <size_t N>
PyArray_ArrFuncs LowBitType<N>::NpyType::funcs;
template <size_t N>
PyArray_Descr LowBitType<N>::NpyType::descr;

template <size_t N>
int LowBitType<N>::NpyType::setitem(PyObject* op, void* ov, void* ap) {
    if (!PyLong_Check(op)) {
        PyErr_SetString(PyExc_ValueError, "input type must be int");
        return -1;
    }

    int a = PyLong_AsLong(op);
    if (!check_value_set_err(a))
        return -1;

    static_cast<LowBitType<N>*>(ov)->value = a;
    return 0;
}

template <size_t N>
int LowBitType<N>::NpyType::fill(void* data_, npy_intp length, void* arr) {
    auto data = static_cast<LowBitType<N>*>(data_);
    int8_t delta = data[1].value - data[0].value, r = data[1].value;
    if (!check_value_set_err(data[0].value) ||
        !check_value_set_err(data[1].value))
        return -1;
    for (npy_intp i = 2; i < length; i++) {
        r += delta;
        if (r > max_value)
            r = -max_value;
        else if (r < -max_value)
            r = max_value;
        data[i].value = r;
    }
    return 0;
}

template <size_t N>
bool LowBitType<N>::NpyType::init() {
    descr = {PyObject_HEAD_INIT(0) & LowBitType<N>::PyObj::py_type,
             'V',  // kind
             'r',  // type
             '=',  // byteorder
             NPY_NEEDS_PYAPI | NPY_USE_GETITEM | NPY_USE_SETITEM,
             0,  // type num
             sizeof(LowBitType<N>),
             alignof(LowBitType<N>),
             NULL,
             NULL,
             NULL,
             &funcs};
    Py_TYPE(&descr) = &PyArrayDescr_Type;
    PyArray_InitArrFuncs(&funcs);
    funcs.copyswap = copyswap;
    funcs.getitem = getitem;
    funcs.setitem = setitem;
    funcs.fill = fill;
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
#define REGISTER_INT_CASTS(bits)                                          \
    REGISTER_CAST(npy_int##bits, LowBitType<N>,                           \
                  PyArray_DescrFromType(NPY_INT##bits),                   \
                  LowBitType<N>::npy_typenum, 1)                          \
    REGISTER_CAST(LowBitType<N>, npy_int##bits, &descr, NPY_INT##bits, 0) \
    REGISTER_CAST(npy_uint##bits, LowBitType<N>,                          \
                  PyArray_DescrFromType(NPY_UINT##bits),                  \
                  LowBitType<N>::npy_typenum, 1)                          \
    REGISTER_CAST(LowBitType<N>, npy_uint##bits, &descr, NPY_UINT##bits, 0)

    REGISTER_INT_CASTS(8)
    REGISTER_INT_CASTS(16)
    REGISTER_INT_CASTS(32)
    REGISTER_INT_CASTS(64)
    REGISTER_CAST(LowBitType<N>, float, &descr, NPY_FLOAT, 0)
    REGISTER_CAST(float, LowBitType<N>, PyArray_DescrFromType(NPY_FLOAT),
                  LowBitType<N>::npy_typenum, 0)
    REGISTER_CAST(LowBitType<N>, double, &descr, NPY_DOUBLE, 1)
    REGISTER_CAST(double, LowBitType<N>, PyArray_DescrFromType(NPY_DOUBLE),
                  LowBitType<N>::npy_typenum, 0)
    return true;
}

}  // anonymous namespace

#define DEFINE_INTBX(n) using IntB##n = LowBitType<n>;
FOREACH_MGB_LOW_BIT(DEFINE_INTBX)
#undef DEFINE_INTBX

// define a new python type: pyintb1/2/4
#define DEFINE_INIT_PYTYPE(n)                                        \
    bool init_pytype_intb##n() {                                     \
        auto& py_type = IntB##n::PyObj::py_type;                     \
        py_type = {PyVarObject_HEAD_INIT(NULL, 0)};                  \
        py_type.tp_name = "megengine.core._imperative_rt.pyintb" #n; \
        py_type.tp_basicsize = sizeof(IntB##n::PyObj);               \
        py_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE; \
        py_type.tp_doc = "an low bit int type";                      \
        py_type.tp_new = IntB##n::PyObj::py_new;                     \
        py_type.tp_str = IntB##n::PyObj::py_repr;                    \
        py_type.tp_repr = IntB##n::PyObj::py_repr;                   \
        py_type.tp_richcompare = IntB##n::PyObj::py_richcompare;     \
        py_type.tp_base = &PyGenericArrType_Type;                    \
        return PyType_Ready(&py_type) >= 0;                          \
    }
FOREACH_MGB_LOW_BIT(DEFINE_INIT_PYTYPE)
#undef DEFINE_INIT_PYTYPE

#define DEFINE_NPY_INTBX(n) \
    int mgb::npy_num_intb##n() { return IntB##n::npy_typenum; }
FOREACH_MGB_LOW_BIT(DEFINE_NPY_INTBX)
#undef DEFINE_NPY_INTBX

namespace py = pybind11;

void mgb::init_npy_num_intbx(py::module m) {
#define ADD_OBJ_INTBX(n)                                             \
    mgb_assert(init_pytype_intb##n());                               \
    mgb_assert(IntB##n::NpyType::init());                            \
    m.add_object("pyintb" #n, reinterpret_cast<PyObject*>(           \
        &IntB##n::PyObj::py_type));                                  \
    m.add_object("intb" #n, reinterpret_cast<PyObject*>(             \
        PyArray_DescrFromType(npy_num_intb##n())));
    FOREACH_MGB_LOW_BIT(ADD_OBJ_INTBX)
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
