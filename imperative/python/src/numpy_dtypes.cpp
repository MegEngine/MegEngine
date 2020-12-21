/**
 * \file imperative/python/src/numpy_dtypes.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./numpy_dtypes.h"
#include "./helper.h"
#include "./pyext17.h"

#include "pybind11/pybind11.h"

#include <cstring>

namespace py = pybind11;

namespace mgb {
namespace {

inline bool _is_quantize(PyArray_Descr* dtype) {
    static PyObject* PY_MGB_DTYPE_KEY = PyUnicode_FromString("mgb_dtype");
    return dtype->metadata &&
           PyDict_CheckExact(dtype->metadata) &&
           PyDict_Contains(dtype->metadata, PY_MGB_DTYPE_KEY) == 1;
}

PyObject* _get_mgb_dtype(PyArray_Descr* dtype) {
    // Return value: New reference.
    if (!_is_quantize(dtype)) {
        throw py::type_error("expact quantize dtype");
    }
    PyObject* ob = PyDict_GetItemString(dtype->metadata, "mgb_dtype");
    if (!PyDict_CheckExact(ob)) {
        throw py::type_error("mgb_dtype is not dict");
    }
    Py_INCREF(ob);
    return ob;
}

double _get_scale(PyArray_Descr* dtype) {
    PyObject* ob = _get_mgb_dtype(dtype);
    PyObject* scale = PyDict_GetItemString(ob, "scale");
    if (!scale) {
        Py_DECREF(ob);
        throw py::key_error("scale");
    }
    if (!PyFloat_Check(scale)) {
        Py_DECREF(ob);
        throw py::type_error("scale is not float");
    }
    double ret = PyFloat_AsDouble(scale);
    Py_DECREF(ob);
    return ret;
}

long _get_zero_point(PyArray_Descr* dtype) {
    PyObject* ob = _get_mgb_dtype(dtype);
    PyObject* name = PyDict_GetItemString(ob, "name");
    if (!name) {
        Py_DECREF(ob);
        throw py::key_error("name");
    }
    const char* s = PyUnicode_AsUTF8(name);
    if (strcmp(s, "Quantized8Asymm") != 0 && strcmp(s, "Quantized4Asymm") != 0) {
        Py_DECREF(ob);
        throw py::value_error(ssprintf("expect name to be \"Quantized8Asymm\" or \"Quantized4Asymm\", got %s", s));
    }
    PyObject* zp = PyDict_GetItemString(ob, "zero_point");
    if (!zp) {
        Py_DECREF(ob);
        throw py::key_error("zero_point");
    }
    long ret = PyLong_AsLong(zp);
    Py_DECREF(ob);
    return ret;
}

bool _is_dtype_equal(PyArray_Descr* dt1, PyArray_Descr* dt2) {
    bool q1 = _is_quantize(dt1),
         q2 = _is_quantize(dt2);
    if (q1 && q2) {
        if (_get_scale(dt1) != _get_scale(dt2)) {
            return false;
        }
        PyObject* zp1 = PyDict_GetItemString(
            PyDict_GetItemString(dt1->metadata, "mgb_dtype"), "zero_point");
        PyObject* zp2 = PyDict_GetItemString(
            PyDict_GetItemString(dt2->metadata, "mgb_dtype"), "zero_point");
        if (!zp1 || !zp2) {
            throw py::key_error("zero_point");
        }
        return PyLong_AsLong(zp1) == PyLong_AsLong(zp2);
    }
    if (!q1 && !q2) {
        return dt1->type_num == dt2->type_num;
    }
    return false;
}

template<auto f>
struct _wrap {
    static constexpr size_t n_args = []() {
        using F = decltype(f);
        using T = PyArray_Descr*;
        static_assert(std::is_pointer<F>::value);
        if constexpr (std::is_invocable<F, T>::value) {
            return 1;
        } else if constexpr (std::is_invocable<F, T, T>::value) {
            return 2;
        } else {
            static_assert(!std::is_same_v<F, F>, "unreachable");
        }
    }();

    static PyObject* impl(PyObject* self, PyObject*const* args, size_t nargs) {
        if (nargs != n_args) {
            PyErr_Format(PyExc_ValueError, "expected %lu arguments", n_args);
            return nullptr;
        }
        for (size_t i=0; i<nargs; ++i) {
            if (args[i] == Py_None) {
                PyErr_SetString(PyExc_ValueError, "can not convert null PyObject to numpy dtype");
                return nullptr;
            }
        }
        try {
            PyArray_Descr *dt1;
            if(!PyArray_DescrConverter(args[0], &dt1)) {
                throw ConversionError(ssprintf("can not convert to numpy.dtype from %s",
                            args[0]->ob_type->tp_name));
            }
            if constexpr (n_args == 1) {
                auto res = (*f)(dt1);
                Py_DECREF(dt1);
                return py::cast(res).release().ptr();
            } else {
                PyArray_Descr *dt2;            
                if(!PyArray_DescrConverter(args[1], &dt2)) {
                    Py_DECREF(dt1);
                    throw ConversionError(ssprintf("can not convert to numpy.dtype from %s",
                                args[1]->ob_type->tp_name));
                }
                auto&& res = (*f)(dt1, dt2);
                Py_DECREF(dt1);
                Py_DECREF(dt2);
                return py::cast(res).release().ptr();
            }
        } catch (std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return nullptr;
        }
    }
};

} // anonymous namespace

void init_dtypes(py::module m) {
    static PyMethodDef method_defs[] = {
        {"is_quantize",     (PyCFunction)_wrap<&_is_quantize>::impl,    METH_FASTCALL, nullptr},
        {"get_scale",       (PyCFunction)_wrap<&_get_scale>::impl,      METH_FASTCALL, nullptr},
        {"get_zero_point",  (PyCFunction)_wrap<&_get_zero_point>::impl, METH_FASTCALL, nullptr},
        {"is_dtype_equal",  (PyCFunction)_wrap<&_is_dtype_equal>::impl, METH_FASTCALL, nullptr},
        {nullptr, nullptr, 0, nullptr}
    };
    for (auto&& def: method_defs) {
        if (def.ml_meth != nullptr) {
            auto* func = PyCFunction_NewEx(&def, nullptr, nullptr);
            if (!func) throw py::error_already_set();
            py::setattr(m, def.ml_name, func);    
        }
    }
}

} // namespace mgb
