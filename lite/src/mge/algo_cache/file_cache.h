/**
 * \file lite/src/mge/algo_cache/file_cache.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "lite_build_config.h"
#if LITE_BUILD_WITH_MGE

#include "megbrain/utils/persistent_cache.h"

namespace lite {

/**
 * dump format:
 *
 * all integers in local endian (effectively little endian as I can see)
 *
 * dump format:
 * <nr_category|uint32_t><category_size|uint32_t><category|uint8_t*>
 * <nr_bob|uint32_t>[<key_size|uint32_t><key|uint8_t*><data_size|
 * uint32_t><data|uint8_t*>]*
 */
//! TODO: fix one thread set cache when other threads is using old cache
class InFilePersistentCache final : public mgb::PersistentCache {
    class InputFile;
    class InputMemory;
    class OutputFile;
    struct BlobStorage : public Blob {
        std::unique_ptr<uint8_t[]> data_refhold;
        size_t hash = 0;

        template <typename Input>
        BlobStorage& init_from_input(Input& inp);
        void write_to_file(OutputFile& out_file) const;
        BlobStorage& init_data_ref(const Blob& b);

        BlobStorage& init_hash() {
            hash = mgb::XXHash{}.update(ptr, size).digest();
            return *this;
        }

        bool operator==(const BlobStorage& rhs) const {
            return size == rhs.size && !memcmp(ptr, rhs.ptr, size);
        }

        struct Hash {
            size_t operator()(const BlobStorage& b) const { return b.hash; }
        };
    };
    std::unordered_map<
            std::string,
            std::unordered_map<BlobStorage, BlobStorage, BlobStorage::Hash>>
            m_cache;
    LITE_MUTEX m_mtx;
    std::shared_ptr<OutputFile> m_always_open_file;

    template <typename Input>
    void read_cache(Input& inp);

public:
    InFilePersistentCache() = default;
    InFilePersistentCache(const char* path, bool always_open = false);
    InFilePersistentCache(const uint8_t* bin, size_t size);

    /**
     * \warning You should invoke \c dump_cache mannually to save the cache
     * file.
     */
    void dump_cache(const char* path);
    void dump_cache(OutputFile* out_file);

    mgb::Maybe<Blob> get(const std::string& category, const Blob& key) override;
    void put(const std::string& category, const Blob& key, const Blob& value) override;
};

}  // namespace lite
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
