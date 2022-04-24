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
#include "megbrain/common.h"

#ifndef __IN_TEE_ENV__
#include "git_full_hash_header.h"
#endif

using namespace mgb;

//! some sdk do not call mgb::get_version explicitly, so we force show version for
//! debug, mgb_log level is info, sdk may config a higher, need export
//! RUNTIME_OVERRIDE_LOG_LEVEL=0 to force change log level to show version
#ifndef __IN_TEE_ENV__
static __attribute__((constructor)) void show_version() {
    auto v = get_version();
    mgb_log("init Engine with version: %d.%d.%d(%d) @(%s)", v.major, v.minor, v.patch,
            v.is_dev, GIT_FULL_HASH);
}
#endif

Version mgb::get_version() {
#ifdef MGB_MAJOR
    return {MGB_MAJOR, MGB_MINOR, MGB_PATCH, MGB_IS_DEV};
#else
    return {MGE_MAJOR, MGE_MINOR, MGE_PATCH, MGB_IS_DEV};
#endif
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
