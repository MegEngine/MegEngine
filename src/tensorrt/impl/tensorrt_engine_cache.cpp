/**
 * \file src/tensorrt/impl/tensorrt_engine_cache.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/tensorrt/tensorrt_engine_cache.h"

#if MGB_ENABLE_TENSOR_RT

#if defined(_WIN32)
#include <io.h>
#define F_OK 0
#define access(a, b) _access(a, b)
#elif __linux__ || __unix__ || __APPLE__
#include <unistd.h>
#endif

using namespace mgb;

/* ========================== TensorRTEngineCache ========================== */
bool TensorRTEngineCache::sm_enable_engine_cache = false;
std::string TensorRTEngineCache::make_key_from_trt_opr(
        const opr::TensorRTOpr* opr) {
    auto&& env = CompNodeEnv::from_comp_node(opr->output(0)->comp_node());
    mgb_assert(env.property().type == CompNode::DeviceType::CUDA,
               "tensorrt opr only support CompNode with DeviceType::CUDA");
    auto&& prop = env.cuda_env().device_prop;
    std::string key;
    int tensorrt_version = getInferLibVersion();
    key = ssprintf("dev=%s;cap=%d.%d;trt=%d;", prop.name, prop.major,
                   prop.minor, tensorrt_version);
    key.append(opr->cname());
    return key;
}

bool TensorRTEngineCache::enable_engine_cache(bool enable_engine_cache) {
    if (enable_engine_cache)
        sm_enable_engine_cache = enable_engine_cache;
    return sm_enable_engine_cache;
}

void TensorRTEngineCache::disable_engine_cache() {
    sm_enable_engine_cache = false;
}

std::shared_ptr<TensorRTEngineCache> TensorRTEngineCache::set_impl(
        std::shared_ptr<TensorRTEngineCache> impl) {
    mgb_assert(impl);
    sm_impl.swap(impl);
    return impl;
}

/* =================== TensorRTEngineCacheMemory ============= */
class TensorRTEngineCacheMemory final : public TensorRTEngineCache {
    const uint8_t* m_ptr;
    size_t m_size;
    size_t m_offset = 0;

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

    void read_cache() {
        uint32_t nr_engines;
        read(nr_engines);
        for (uint32_t i = 0; i < nr_engines; ++i) {
            uint32_t key_size;
            read(key_size);
            std::string key;
            key.resize(key_size);
            read(const_cast<char*>(key.data()), key.size());
            mgb_log_debug("read key: %s", key.c_str());
            m_cache[std::move(key)].init_from(*this);
        }
    }

    struct EngineStorage : public Engine {
        std::unique_ptr<uint8_t[]> data_refhold;

        EngineStorage& init_from(TensorRTEngineCacheMemory& io) {
            uint32_t data_size;
            io.read(data_size);
            size = data_size;
            data_refhold = std::make_unique<uint8_t[]>(size);
            io.read(data_refhold.get(), size);
            ptr = data_refhold.get();
            return *this;
        };

        EngineStorage& init_from_buf(const void* buf, size_t buf_size) {
            data_refhold = std::make_unique<uint8_t[]>(buf_size);
            memcpy(data_refhold.get(), buf, buf_size);
            size = buf_size;
            ptr = data_refhold.get();
            return *this;
        };
    };

    std::unordered_map<std::string, EngineStorage> m_cache;
    std::mutex m_mtx;

public:
    TensorRTEngineCacheMemory() = default;
    TensorRTEngineCacheMemory(const uint8_t* bin, size_t size)
            : m_ptr{bin}, m_size{size} {
        read_cache();
    }

    Maybe<Engine> get(const std::string& key) override {
        MGB_LOCK_GUARD(m_mtx);
        auto find = m_cache.find(key);
        if (find == m_cache.end())
            return None;
        return find->second;
    }

    void put(const std::string& key, const Engine& value) override {
        MGB_LOCK_GUARD(m_mtx);
        m_cache[key].init_from_buf(value.ptr, value.size);
    }

    void dump_cache() override {}
};

/* =================== TensorRTEngineCacheIO  ============= */
void TensorRTEngineCacheIO::read_cache() {
    uint32_t nr_engines;
    read(nr_engines);
    for (uint32_t i = 0; i < nr_engines; ++i) {
        uint32_t key_size;
        read(key_size);
        std::string key;
        key.resize(key_size);
        read(const_cast<char*>(key.data()), key.size());
        mgb_log_debug("read key: %s", key.c_str());
        m_cache[std::move(key)].init_from(*this);
    }
}

TensorRTEngineCacheIO::TensorRTEngineCacheIO(std::string filename)
        : m_filename{std::move(filename)} {
    mgb_log_debug("create tensorrt engine cache: %s", m_filename.c_str());
    if (access(m_filename.c_str(), F_OK) == 0) {
        mgb_log(
                "tensorrt engine cache %s already exists, read from binary "
                "file.",
                m_filename.c_str());
        m_ptr = fopen(m_filename.c_str(), "r+b");
        mgb_throw_if(m_ptr == nullptr, SystemError,
                     "failed to open tensorrt engine file %s %s",
                     m_filename.c_str(), strerror(errno));
        std::unique_ptr<FILE, int (*)(FILE*)> fptr_close{m_ptr, ::fclose};
        read_cache();
        m_update_cache = true;
    } else {
        mgb_log_debug(
                "tensorrt engine cache %s not exists, will create tensorrt "
                "engine cache file",
                m_filename.c_str());
    }
}

void TensorRTEngineCacheIO::dump_cache() {
    if (m_update_cache)
        mgb_log_debug(
                "tensorrt engine cache %s already exists, and will be "
                "rewritten during dumping cache to this file.",
                m_filename.c_str());
    m_ptr = fopen(m_filename.c_str(), "wb");
    mgb_throw_if(m_ptr == nullptr, SystemError,
                 "failed to open tensorrt engine file %s %s",
                 m_filename.c_str(), strerror(errno));
    std::unique_ptr<FILE, int (*)(FILE*)> fptr_close{m_ptr, ::fclose};
    uint32_t nr_engines = m_cache.size();
    write(nr_engines);
    for (auto&& cached_engine : m_cache) {
        uint32_t key_size = cached_engine.first.size();
        write(key_size);
        write(cached_engine.first.data(), key_size);
        cached_engine.second.write_to(*this);
    }
}

Maybe<TensorRTEngineCache::Engine> TensorRTEngineCacheIO::get(
        const std::string& key) {
    MGB_LOCK_GUARD(m_mtx);
    auto find = m_cache.find(key);
    if (find == m_cache.end())
        return None;
    return find->second;
}

void TensorRTEngineCacheIO::put(const std::string& key, const Engine& value) {
    MGB_LOCK_GUARD(m_mtx);
    m_cache[key].init_from_buf(value.ptr, value.size);
}

std::shared_ptr<TensorRTEngineCache> TensorRTEngineCache::sm_impl =
        std::make_shared<TensorRTEngineCacheMemory>();
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
