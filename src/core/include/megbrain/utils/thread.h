/**
 * \file src/core/include/megbrain/utils/thread.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain_build_config.h"
#if MGB_HAVE_THREAD
#include "./thread_impl_1.h"
#else
#include "./thread_impl_0.h"
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

