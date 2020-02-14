/**
 * \file dnn/include/megdnn/version.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#define MEGDNN_MAJOR 9
#define MEGDNN_MINOR 3
#define MEGDNN_PATCH 0

#include "megdnn/internal/visibility_prologue.h"

namespace megdnn {
    struct Version {
        int major, minor, patch;
    };

    //! get megdnn version of the binary
    Version get_version();
}

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
