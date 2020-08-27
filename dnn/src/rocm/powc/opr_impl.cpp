/**
 * \file dnn/src/rocm/powc/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "./opr_impl.h"
#include "src/rocm/powc/powc.h.hip"

#include "src/rocm/utils.h"

using namespace megdnn;
using namespace rocm;

void PowCImpl::do_exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                       const float* exp_f, const int* exp_i) {
    powc_kern(dst, src, exp_f, exp_i, hip_stream(handle()));
}

// vim: syntax=cpp.doxygen
