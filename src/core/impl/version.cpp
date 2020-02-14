/**
 * \file src/core/impl/version.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/version.h"

using namespace mgb;

Version mgb::get_version() {
    return {MGB_MAJOR, MGB_MINOR, MGB_PATCH, MGB_IS_DEV};
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
