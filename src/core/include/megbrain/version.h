/**
 * \file src/core/include/megbrain/version.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain_build_config.h"

#define MGE_MAJOR 1
#define MGE_MINOR 7
#define MGE_PATCH 0

// for rc version, could be like "rc1", "rc2", etc
#define MGE_EXTRA_NAME ""

//! whether it is development version
#ifndef MGB_IS_DEV
#define MGB_IS_DEV 0
#endif  // MGB_IS_DEV

namespace mgb {
struct Version {
    int major, minor, patch, is_dev;
};

MGE_WIN_DECLSPEC_FUC Version get_version();
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
