/**
 * \file dnn/src/cuda/kernel_common/diagnostic_epilogue.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#ifdef __GNUC__
#if CUDA_VERSION < 9020
#pragma GCC diagnostic pop
#endif
#endif

#ifdef MEGDNN_DIAGNOSTIC_PROLOGUE_INCLUDED
#undef MEGDNN_DIAGNOSTIC_PROLOGUE_INCLUDED
#else
#error "diagnostic_epilogue.h must be included after diagnostic_prologue.h"
#endif

// vim: syntax=cpp.doxygen
