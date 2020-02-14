/**
 * \file sdk/load-and-run/src/infile_persistent_cache.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./infile_persistent_cache.h"

#if defined(_WIN32)
#include <io.h>
#define F_OK 0
#define access(a, b) _access(a, b)
#elif __linux__ || __unix__ || __APPLE__
#include <unistd.h>
#endif

using namespace mgb;

//////////////////////// InFilePersistentCache::InputMemory ///////////////
class InFilePersistentCache::InputMemory {
    const uint8_t* m_ptr;
    size_t m_offset = 0;
    size_t m_size;

public:
    InputMemory(const uint8_t* bin, size_t size) : m_ptr{bin}, m_size{size} {}

    template <typename T>
    void read(T& val) {
        static_assert(std::is_trivially_copyable<T>::value,
                      "only support trivially copyable type");
        mgb_assert(m_offset + sizeof(T) <= m_size);
        memcpy(&val, m_ptr, sizeof(T));
        m_offset += sizeof(T);
        m_ptr += sizeof(T);
    }

    template <typename T>
    void read(T* buf, size_t size) {
        static_assert(std::is_trivially_copyable<T>::value && sizeof(T) == 1,
                      "only support read bytes");
        mgb_assert(m_offset + size <= m_size);
        memcpy(buf, m_ptr, size);
        m_offset += size;
        m_ptr += size;
    }
};

//////////////////////// InFilePersistentCache::InputFile ///////////////
class InFilePersistentCache::InputFile {
    FILE* m_fp;

public:
    InputFile(const char* path) : m_fp{fopen(path, "rb")} {
        mgb_assert(m_fp, "failed to open %s: %s", path, strerror(errno));
    }
    ~InputFile() {
        if (m_fp) {
            fclose(m_fp);
        }
    }

    template <typename T>
    void read(T& val) {
        static_assert(std::is_trivially_copyable<T>::value,
                      "only support trivially copyable type");
        auto ret = fread(&val, sizeof(T), 1, m_fp);
        mgb_assert(ret == 1);
    }

    template <typename T>
    void read(T* buf, size_t size) {
        static_assert(std::is_trivially_copyable<T>::value && sizeof(T) == 1,
                      "only support read bytes");
        auto ret = fread(buf, size, 1, m_fp);
        mgb_assert(ret == 1);
    }

};

//////////////////////// InFilePersistentCache::OutputFile ///////////////
class InFilePersistentCache::OutputFile {
    FILE* m_fp;

public:
    OutputFile(const char* path) : m_fp{fopen(path, "wb")} {
        mgb_assert(m_fp, "failed to open %s: %s", path, strerror(errno));
    }
    ~OutputFile() {
        if (m_fp) {
            fclose(m_fp);
        }
    }

    template <typename T>
    void write(T val) {
        auto ret = fwrite(&val, sizeof(T), 1, m_fp);
        mgb_assert(ret == 1);
    }

    template <typename T>
    void write(const T* buf, size_t size) {
        static_assert(sizeof(T) == 1, "only support write bytes");
        auto ret = fwrite(buf, size, 1, m_fp);
        mgb_assert(ret == 1);
    }
};

//////////////////////// InFilePersistentCache::BlobStorage ///////////////

template <typename Input>
InFilePersistentCache::BlobStorage&
InFilePersistentCache::BlobStorage::init_from_input(Input& inp) {
    uint32_t data_size;
    inp.read(data_size);
    size = data_size;
    data_refhold = std::make_unique<uint8_t[]>(size);
    inp.read(data_refhold.get(), size);
    ptr = data_refhold.get();
    return *this;
}

void InFilePersistentCache::BlobStorage::write_to_file(
        OutputFile& out_file) const {
    uint32_t u_size = size;
    out_file.write(u_size);
    out_file.write(data_refhold.get(), u_size);
}

InFilePersistentCache::BlobStorage&
InFilePersistentCache::BlobStorage::init_data_ref(const Blob& b) {
    data_refhold = std::make_unique<uint8_t[]>(b.size + 1);
    memcpy(data_refhold.get(), b.ptr, b.size);
    data_refhold.get()[b.size] = 0;  // for C-string safety
    ptr = data_refhold.get();
    size = b.size;
    return *this;
}

//////////////////////// InFilePersistentCache //////////////////////

template <typename Input>
void InFilePersistentCache::read_cache(Input& inp) {
    uint32_t nr_category;
    inp.read(nr_category);
    char category_buf[256];
    for (uint32_t i = 0; i < nr_category; i++) {
        uint32_t category_size;
        inp.read(category_size);
        inp.read(category_buf, category_size);
        category_buf[category_size] = '\0';

        std::string category(category_buf);
        mgb_log_debug("load new category: %s", category_buf);

        // read bobs
        uint32_t nr_bobs;
        inp.read(nr_bobs);
        for (uint32_t j = 0; j < nr_bobs; j++) {
            BlobStorage key_storage;
            key_storage.init_from_input(inp).init_hash();
            mgb_log_debug("read key: %zu", key_storage.hash);
            m_cache[category][std::move(key_storage)].init_from_input(inp);
        }
    }
}

InFilePersistentCache::InFilePersistentCache(const char* path) {
    if (!access(path, F_OK)) {
        mgb_log_debug("use fastrun cache: %s", path);
        InputFile inp(path);
        read_cache<InputFile>(inp);
    }
}

InFilePersistentCache::InFilePersistentCache(const uint8_t* bin, size_t size) {
    mgb_assert(bin);
    InputMemory inp(bin, size);
    read_cache<InputMemory>(inp);
}

void InFilePersistentCache::dump_cache(const char* path) {
    OutputFile out_file(path);
    uint32_t nr_category = m_cache.size();
    out_file.write(nr_category);

    for (const auto& cached_category : m_cache) {
        uint32_t category_size = cached_category.first.size();
        out_file.write(category_size);
        out_file.write(cached_category.first.data(), category_size);
        mgb_log_debug("write new category: %s", cached_category.first.c_str());

        uint32_t nr_bobs = cached_category.second.size();
        out_file.write(nr_bobs);
        for (const auto& item : cached_category.second) {
            mgb_log_debug("dump key: %zu", item.first.hash);
            item.first.write_to_file(out_file);
            item.second.write_to_file(out_file);
        }
    }
}

Maybe<InFilePersistentCache::Blob> InFilePersistentCache::get(
        const std::string& category, const Blob& key) {
    decltype(m_cache.begin()) iter0;
    {
        MGB_LOCK_GUARD(m_mtx);
        iter0 = m_cache.find(category);
        if (iter0 == m_cache.end())
            return None;
    }

    BlobStorage key_storage;
    key_storage.Blob::operator=(key);
    key_storage.init_hash();

    MGB_LOCK_GUARD(m_mtx);

    auto iter1 = iter0->second.find(key_storage);
    if (iter1 == iter0->second.end())
        return None;
    return iter1->second;
}

void InFilePersistentCache::put(const std::string& category, const Blob& key,
                                const Blob& value) {
    BlobStorage key_storage;
    key_storage.init_data_ref(key).init_hash();

    MGB_LOCK_GUARD(m_mtx);
    auto size0 = m_cache.size();
    m_cache[category][std::move(key_storage)].init_data_ref(value);
    if (m_cache.size() > size0) {
        mgb_log_debug("new cache category: %s", category.c_str());
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
