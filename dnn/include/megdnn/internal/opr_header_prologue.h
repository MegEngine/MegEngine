/**
 * \file dnn/include/megdnn/internal/opr_header_prologue.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
// intentional no header guard here

#include "megdnn/handle.h"
#include "megdnn/oprs/base.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/opr_result_defs.h"

#include "./visibility_prologue.h"

#include <limits>
#include <array>

#ifndef _megdnn_in
#define _megdnn_in
#endif

#ifndef _megdnn_out
#define _megdnn_out
#endif

#ifndef _megdnn_tensor_in
#define _megdnn_tensor_in const TensorND &
#endif

#ifndef _megdnn_tensor_out
#define _megdnn_tensor_out const TensorND &
#endif

#ifndef _megdnn_tensor_inout
#define _megdnn_tensor_inout const TensorND &
#endif

#ifndef _megdnn_workspace
#define _megdnn_workspace const Workspace &
#endif

#define DEF_OPR_IMPL_CTOR(_opr_name, _base_name)  \
    public: \
        _opr_name(Handle *handle): _base_name(handle) {} \

#define DEF_OPR_IMPL(_opr_name, _base_name, _nr_inputs, _nr_outputs) \
    DEF_OPR_IMPL_CTOR(_opr_name, _base_name) \
    static MEGDNN_CONSTEXPR int NR_INPUTS = _nr_inputs; \
    static MEGDNN_CONSTEXPR int NR_OUTPUTS = _nr_outputs; \

#define DEF_OPR_PARAM(_pname) \
    public: \
        using Param = param::_pname; \
        Param& param() { return m_param; }  \
        const Param& param() const { return m_param; }  \
    protected: \
        Param m_param

// vim: syntax=cpp.doxygen
