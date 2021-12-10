/**
 * \file imperative/src/impl/persistent_cache.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <fstream>
#include <string>
#include <vector>

#include "megbrain/imperative/persistent_cache.h"
#include "megbrain/imperative/utils/base64.h"
#include "megbrain/utils/infile_persistent_cache.h"

namespace mgb::imperative::persistent_cache {

class ExtendedInFilePersistentCache final : public ExtendedPersistentCache {
private:
    std::string m_path;
    std::unique_ptr<mgb::InFilePersistentCache> m_impl;

public:
    ExtendedInFilePersistentCache() = default;

    bool open(std::string path) {
        std::fstream file;
        file.open(path, std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        std::vector<char> bytes = {
                std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
        if (bytes.size()) {
            m_impl = std::make_unique<mgb::InFilePersistentCache>(
                    (const uint8_t*)bytes.data(), bytes.size());
        } else {
            m_impl = std::make_unique<mgb::InFilePersistentCache>();
        }
        m_path = path;
        return true;
    }

    ~ExtendedInFilePersistentCache() {
        if (m_impl) {
            m_impl->dump_cache(m_path.c_str());
        }
    }

    mgb::Maybe<Blob> get(const std::string& category, const Blob& key) override {
        return m_impl->get(category, key);
    }

    void put(const std::string& category, const Blob& key, const Blob& value) override {
        return m_impl->put(category, key, value);
    }

    std::optional<size_t> clear() override {
        m_impl = std::make_unique<mgb::InFilePersistentCache>();
        m_impl->dump_cache(m_path.c_str());
        return {};
    }

    bool valid() const override { return m_impl != nullptr; }
};

std::shared_ptr<ExtendedPersistentCache> make_in_file(std::string path) {
    auto cache = std::make_shared<ExtendedInFilePersistentCache>();
    if (!cache->open(path)) {
        return nullptr;
    }
    return cache;
}

}  // namespace mgb::imperative::persistent_cache

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
