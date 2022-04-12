/**
 * \file src/parse_info/cache_parse.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "lite/global.h"
#if LITE_BUILD_WITH_MGE
#include "megbrain/utils/infile_persistent_cache.h"
#endif

namespace lite {
//! The LITE_parse_cache parse info function
bool parse_info_cache(
        const uint8_t* cache, size_t cache_length, bool is_fast_run_cache = true,
        const uint8_t* binary_cache = nullptr, size_t binary_cache_length = 0) {
    LITE_MARK_USED_VAR(binary_cache);
    LITE_MARK_USED_VAR(binary_cache_length);
#if LITE_BUILD_WITH_MGE
    if (is_fast_run_cache) {
        mgb::PersistentCache::set_impl(
                std::make_shared<mgb::InFilePersistentCache>(cache, cache_length));
    }
#endif
    return true;
}

}  // namespace lite

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
