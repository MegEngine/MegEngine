/**
 * \file src/core/impl/version.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/version.h"

using namespace mgb;

Version mgb::get_version() {
#ifdef MGB_MAJOR
    return {MGB_MAJOR, MGB_MINOR, MGB_PATCH, MGB_IS_DEV};
#else
    return {MGE_MAJOR, MGE_MINOR, MGE_PATCH, MGB_IS_DEV};
#endif
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
