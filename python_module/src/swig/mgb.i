/*
 * $File: mgb.i
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * $Copyright: Copyright (c) 2014-2017 Megvii Inc. All rights reserved.
 */

%include "symbol_var_array.i"

%include "mgb_exception.i"
%module(directors="1") mgb
%{
#define SWIG_FILE_WITH_INIT 1
void mgb_init_numpy(); // implemented in python_helper.cpp
void _init_intbx_types(PyObject *m); // implemented in intbx.cpp
%}

%init %{
    mgb_init_numpy();
    _init_intbx_types(m);
%}

%include "std_vector.i"
%include "std_pair.i"
%include "stdint.i"
%template(_VectorSizeT) std::vector<size_t>;
%template(_VectorInt) std::vector<int>;
%template(_VectorString) std::vector<std::string>;
%template(_PairStringSizeT) std::pair<std::string, size_t>;
%template(_VectorPairUint64String) std::vector<std::pair<uint64_t, std::string>>;

%pythoncode %{
import numpy as np
import os
intb1 = _mgb.intb1
intb2 = _mgb.intb2
intb4 = _mgb.intb4
%}

%{
#include "megbrain/comp_node.h"
#include "megbrain/tensor.h"
#include "megbrain/graph.h"

#include "megbrain_wrap.h"
#include "megbrain_config.h"
#include "megbrain_serialize.h"
#include "plugin.h"
%}

%include "comp_node.i"
%include "comp_graph.i"
%include "symbol_var.i"
%include "shared_nd.i"
%include "../cpp/megbrain_config.h"
%include "callback.i"
%include "operator.i"
%include "craniotome.i"
%include "misc.i"
%include "loop.i"
%include "../cpp/megbrain_serialize.h"
%include "../cpp/plugin.h"

// vim: ft=swig
