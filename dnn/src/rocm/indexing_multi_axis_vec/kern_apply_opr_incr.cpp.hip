/**
 * \file dnn/src/rocm/indexing_multi_axis_vec/kern_apply_opr_incr.cpp.hip
 *
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "hip_header.h"
#include "megdnn/dtype.h"

__device__ void atomicAdd(megdnn::dt_bfloat16 *, megdnn::dt_bfloat16) {
    asm("s_trap 2;");
    ((int*)0)[0] = 1;
}

#if !MEGDNN_DISABLE_FLOAT16
__device__ void atomicAdd(megdnn::dt_float16 *, megdnn::dt_float16) {
    asm("s_trap 2;");
    ((int*)0)[0] = 1;
}
#endif

__device__ void atomicAdd(megdnn::dt_int8 *, megdnn::dt_int8) {
     asm("s_trap 2;");
     ((int*)0)[0] = 1;
}

__device__ void atomicAdd(megdnn::dt_uint8 *, megdnn::dt_uint8) {
    asm("s_trap 2;");
    ((int*)0)[0] = 1;
}

__device__ void atomicAdd(megdnn::dt_int16 *, megdnn::dt_int16) {
    asm("s_trap 2;");
    ((int*)0)[0] = 1;
}

__device__ void atomicAdd(megdnn::dt_bool *, megdnn::dt_bool) {
    asm("s_trap 2;");
    ((int*)0)[0] = 1;
}

#define KERN_APPLY_OPR_OPR \
    ::megdnn::rocm::indexing_multi_axis_vec::OprAtomicIncr
#include "./kern_apply_opr_impl.hipinl"

// vim: ft=cuda syntax=cpp.doxygen

