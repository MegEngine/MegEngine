/*
 * $File: mgb_exception.i
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * $Copyright: Copyright (c) 2014-2017 Megvii Inc. All rights reserved.
 */


%include "std_string.i"
%include "std_except.i"
%include "pyabc.i"

%{
#include "python_helper.h"
%}

namespace PyMGBExceptionMaker {
    void _reg_exception_class(PyObject *cls);
}

%feature("director:except") {
    if ($error)
        PyExceptionForward::throw_();
}

%include "exception.i"
%allowexception;
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyMGBExceptionMaker::setup_py_exception(e);
        SWIG_fail;
    }  catch(...) {
        SWIG_exception(SWIG_UnknownError, "Unknown exception");
    }
}

// vim: ft=swig
