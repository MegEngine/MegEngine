/**
 * \file imperative/src/include/megbrain/imperative/persistent_cache.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <memory>
#include "megbrain/utils/persistent_cache.h"

namespace mgb::imperative::persistent_cache {

class ExtendedPersistentCache : public mgb::PersistentCache {
public:
    virtual bool valid() const = 0;
    virtual std::optional<size_t> clear() = 0;
    virtual void flush() = 0;

    static std::shared_ptr<ExtendedPersistentCache> make_from_config(
            std::string type, std::unordered_map<std::string, std::string> args,
            std::string& err_msg);
};

}  // namespace mgb::imperative::persistent_cache
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
