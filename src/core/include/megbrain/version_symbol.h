/**
 * \file src/core/include/megbrain/version_symbol.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#define MGB_VERSION_SYMBOL_(name, ver) \
    int MGB_VSYM_##name##_##ver __attribute__((visibility("default")))

/*!
 * This macro should be placed in a .cpp file. A symbol would be inserted in the
 * produced binary with the name MGB_VERSION_`name`_`ver`
 */
#define MGB_VERSION_SYMBOL(name, ver) MGB_VERSION_SYMBOL_(name, ver)

//! helper macro
#define MGB_VERSION_SYMBOL3_(name, ver0, ver1, ver2) \
    MGB_VERSION_SYMBOL_(name, ver0##_##ver1##_##ver2)

//! concat three symbols (usually used for version major, minor and patch)
#define MGB_VERSION_SYMBOL3(name, ver0, ver1, ver2) \
    MGB_VERSION_SYMBOL3_(name, ver0, ver1, ver2)

// vim: syntax=cpp.doxygen
