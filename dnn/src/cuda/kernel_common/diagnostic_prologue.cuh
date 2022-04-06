/**
 * \file dnn/src/cuda/kernel_common/diagnostic_prologue.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#ifdef MEGDNN_DIAGNOSTIC_PROLOGUE_INCLUDED
#error "diagnostic_prologue.h included twice without including diagnostic_epilogue.h"
#else
#define MEGDNN_DIAGNOSTIC_PROLOGUE_INCLUDED
#endif

//! see
//! https://stackoverflow.com/questions/49836419/how-to-hide-nvccs-function-was-declared-but-never-referenced-warnings
//! for more details.
#ifdef __GNUC__
#if CUDA_VERSION < 9020
#pragma GCC diagnostic push
#pragma diag_suppress 177  // suppress "function was declared but never referenced
                           // warning"
#endif
#endif

// vim: syntax=cpp.doxygen
