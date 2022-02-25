#pragma once

#include "megbrain/utils/persistent_cache.h"

namespace mgb {

/**
 * dump format:
 *
 * all integers in local endian (effectively little endian as I can see)
 *
 * dump format:
 * <nr_category|uint32_t><category_size|uint32_t><category|uint8_t*>
 *  <nr_bob|uint32_t>[<key_size|uint32_t><key|uint8_t*><data_size|uint32_t><data|uint8_t*>]*
 */
class InFilePersistentCache final : public PersistentCache {
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
            hash = XXHash{}.update(ptr, size).digest();
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
    MGB_MUTEX m_mtx;
    std::shared_ptr<OutputFile> m_always_open_file;

    template <typename Input>
    void read_cache(Input& inp);

public:
    MGE_WIN_DECLSPEC_FUC InFilePersistentCache() = default;
    MGE_WIN_DECLSPEC_FUC InFilePersistentCache(
            const char* path, bool always_open = false);
    MGE_WIN_DECLSPEC_FUC InFilePersistentCache(const uint8_t* bin, size_t size);

    /**
     * \warning You should invoke \c dump_cache mannually to save the cache
     * file.
     */
    MGE_WIN_DECLSPEC_FUC void dump_cache(const char* path);
    MGE_WIN_DECLSPEC_FUC void dump_cache(OutputFile* out_file);
    MGE_WIN_DECLSPEC_FUC std::vector<uint8_t> dump_cache();

    MGE_WIN_DECLSPEC_FUC Maybe<Blob> get(
            const std::string& category, const Blob& key) override;
    MGE_WIN_DECLSPEC_FUC void put(
            const std::string& category, const Blob& key, const Blob& value) override;
    bool support_dump_cache() override { return true; }

    std::unordered_map<
            std::string,
            std::unordered_map<BlobStorage, BlobStorage, BlobStorage::Hash>>
    get_cache() {
        return std::move(m_cache);
    }
};
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
