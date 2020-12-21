/**
 * \file imperative/python/src/numpy_dtypes.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#ifndef DO_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL mgb_numpy_array_api
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <pybind11/pybind11.h>

#define FOREACH_MGB_LOW_BIT(cb) \
    cb(1) \
    cb(2) \
    cb(4) \

#define FOREACH_MGB_DTYPE_PAIR(cb) \
    cb(IntB1, npy_num_intb1()) \
    cb(IntB2, npy_num_intb2()) \
    cb(IntB4, npy_num_intb4()) \
    cb(BFloat16, npy_num_bfloat16()) \

namespace mgb {
    //! numpy type num for intb1/2/4 type
#define DEFINE_NPY_INTBX(n) \
    int npy_num_intb##n();
FOREACH_MGB_LOW_BIT(DEFINE_NPY_INTBX)
#undef DEFINE_NPY_INTBX
    void init_dtypes(pybind11::module m);
    void init_npy_num_intbx(pybind11::module m);

    //! numpy type num for bfloat16 type
    int npy_num_bfloat16();
    void init_npy_num_bfloat16(pybind11::module m);
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
