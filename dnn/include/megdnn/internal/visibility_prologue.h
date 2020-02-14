/**
 * \file dnn/include/megdnn/internal/visibility_prologue.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#ifdef MEGDNN_VISIBILITY_PROLOGUE_INCLUDED
#error "visibility_prologue.h included twice without including visibility_epilogue.h"
#else
#define MEGDNN_VISIBILITY_PROLOGUE_INCLUDED
#endif

#if MEGDNN_SHARED_LIB
#pragma GCC visibility push(default)
#endif

// vim: syntax=cpp.doxygen
