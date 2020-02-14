/**
 * \file dnn/include/megdnn/internal/visibility_epilogue.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#if MEGDNN_SHARED_LIB
#pragma GCC visibility pop
#endif

#ifdef MEGDNN_VISIBILITY_PROLOGUE_INCLUDED
#undef MEGDNN_VISIBILITY_PROLOGUE_INCLUDED
#else
#error "visibility_epilogue.h must be included after visibility_prologue.h"
#endif

// vim: syntax=cpp.doxygen

