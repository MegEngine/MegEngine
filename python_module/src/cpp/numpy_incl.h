/**
 * \file python_module/src/cpp/numpy_incl.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \brief import numpy array with proper settings
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */
#pragma once

#define PY_ARRAY_UNIQUE_SYMBOL mgb_numpy_array_api
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define FOREACH_MGB_LOW_BIT(cb) \
    cb(1) \
    cb(2) \
    cb(4) \

#define FOREACH_MGB_DTYPE_PAIR(cb) \
    cb(IntB1, npy_num_intb1()) \
    cb(IntB2, npy_num_intb2()) \
    cb(IntB4, npy_num_intb4()) \

namespace mgb {
    //! numpy type num for intb2 type
#define DEFINE_NPY_INTBX(n) \
    int npy_num_intb##n();
FOREACH_MGB_LOW_BIT(DEFINE_NPY_INTBX)
#undef DEFINE_NPY_INTBX
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
