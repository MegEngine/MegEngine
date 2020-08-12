/**
 * \file src/core/include/megbrain/version.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#define MGB_MAJOR   8
#define MGB_MINOR   9999
#define MGB_PATCH   0
//! whether it is development version
#ifndef MGB_IS_DEV
#define MGB_IS_DEV  0
#endif  // MGB_IS_DEV

namespace mgb {
    struct Version {
        int major, minor, patch, is_dev;
    };

    Version get_version();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
