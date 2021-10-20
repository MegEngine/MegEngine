/**
 * \file src/core/include/megbrain/utils/persistent_cache.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/tensor.h"

namespace mgb {

/*!
 * \brief persistent cache that should be implemented outside of megbrain
 *
 * There should be O(1) categories, whose names are ascii strings. Each
 * category acts like an independent string key-value cache.
 *
 * The implementation must be thread safe.
 */
class PersistentCache {
    static std::shared_ptr<PersistentCache> sm_impl;

public:
    virtual ~PersistentCache() = default;

    struct Blob {
        const void* ptr;
        size_t size;
    };

    virtual Maybe<Blob> get(const std::string& category, const Blob& key) = 0;

    virtual void put(
            const std::string& category, const Blob& key, const Blob& value) = 0;

    //! set an implementation; return the original implementation
    static std::shared_ptr<PersistentCache> set_impl(
            std::shared_ptr<PersistentCache> impl);

    //! get the instance; the default implementation just caches in
    //! memory
    static PersistentCache& inst() { return *sm_impl; }

    //! make a cache category that incorporates all tratis of a comp
    //! node (e.g. device name, library versions)
    static std::string make_category_from_comp_node(CompNode comp_node);
};

/*!
 * \brief persistent cache that keep in memory
 * The implementation is thread safe.
 */
class InMemoryPersistentCache final : public PersistentCache {
    struct BlobStorage : public PersistentCache::Blob {
        std::unique_ptr<uint8_t[]> data_refhold;
        size_t hash = 0;

        BlobStorage& init_data_ref(const Blob& b);

        BlobStorage& init_hash();

        bool operator==(const BlobStorage& rhs) const;

        struct Hash {
            size_t operator()(const BlobStorage& b) const { return b.hash; }
        };
    };

    Maybe<Blob> get(const std::string& category, const Blob& key) override;
    void put(const std::string& category, const Blob& key, const Blob& value) override;

    std::unordered_map<
            std::string,
            std::unordered_map<BlobStorage, BlobStorage, BlobStorage::Hash>>
            m_cache;
    MGB_MUTEX m_mtx;
};

/*!
 * \brief proxy PersistentCache to be better suited for managing profiling
 *      results of operator impl algorithms
 *
 * \param cn comp node on which this operator should run
 * \param opr_type an arbitrary constant string to identify operator type;
 *      can be treated as namespace of the algorithms
 */
class AlgoChooserProfileCache {
    std::string m_category;

public:
    AlgoChooserProfileCache(CompNode cn, const char* opr_type);

    /*!
     * \brief key to identify a profiling run
     *
     * \param param extra param to index the cache
     * \param param_size size of extra cache indexing param, in bytes
     */
    class Key final : public NonCopyableObj {
        mutable std::string m_blob_storage;
        const TensorLayout* m_inp_layouts_ptr;
        size_t m_inp_layouts_size;

        const void* m_param;
        size_t m_param_size;

    public:
        Key(const TensorLayout* inp_layouts_ptr, size_t inp_layouts_size,
            const void* param = nullptr, size_t param_size = 0)
                : m_inp_layouts_ptr{inp_layouts_ptr},
                  m_inp_layouts_size{inp_layouts_size},
                  m_param{param},
                  m_param_size{param_size} {}
        //! build a blob representation to be used as cache key
        PersistentCache::Blob build_blob() const;
    };

    struct ResultEntry {
        std::string algo;    //! serialized algo desc
        uint32_t attribute;  //! algo attribute, e.g. reproducible
        double time;         //! execution time in seconds
        size_t workspace;    //! workspace in bytes
    };

    //! result for a single profiling run
    using Result = std::vector<ResultEntry>;

    /*!
     * \brief try to get result from cache
     *
     * This returned result, if valid, would be sorted by ascending time
     * and descending workspace
     */
    Maybe<Result> get(const Key& key);

    /*!
     * \brief put result to cache
     *
     * Note that result would be sorted and useless entries would be
     * removed.
     */
    void put(const Key& key, Result& result);
};

}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
