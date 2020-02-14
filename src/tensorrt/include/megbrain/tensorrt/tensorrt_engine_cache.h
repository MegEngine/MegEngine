/**
 * \file src/tensorrt/include/megbrain/tensorrt/tensorrt_engine_cache.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/tensorrt/tensorrt_opr.h"

#if MGB_ENABLE_TENSOR_RT
namespace mgb {

/*!
 * \brief a cache for tensorrt engine
 *
 * The cache stores tensorrt engine as key-value pairs. The keys are ascii
 * strings, which include the device name, the compute capability, the tensorrt
 * runtime version and the name of operator. The get and put methods must be
 * thread safe. read_cache() and dump_cache() are not thread safe
 *
 * \note We did not implement the hashing of general graph, and just used the
 * name of tensorrt opr as the key. This implementation is enough, when there is
 * only one megbrain computing graph. Because the names of different tensorrt
 * opr in the same computing graph are different.
 */
class TensorRTEngineCache : public NonCopyableObj {
    static std::shared_ptr<TensorRTEngineCache> sm_impl;
    static bool sm_enable_engine_cache;

public:
    virtual ~TensorRTEngineCache() = default;

    struct Engine {
        const void* ptr;
        size_t size;
    };

    virtual Maybe<Engine> get(const std::string& key) = 0;
    virtual void put(const std::string& key, const Engine& value) = 0;
    virtual void dump_cache() = 0;

    //! get the key of the TensorRTOpr
    static std::string make_key_from_trt_opr(const opr::TensorRTOpr* opr);

    //! enable the tensorrt engine cache, or query whether the cache is used
    static bool enable_engine_cache(bool enable_engine_cache = false);
    //! disable the tensorrt engine cache
    static void disable_engine_cache();

    //! set an implementation; return the original implementation
    static std::shared_ptr<TensorRTEngineCache> set_impl(
            std::shared_ptr<TensorRTEngineCache> impl);

    //! get the instance; the default implementation is an InMemoryCache
    static TensorRTEngineCache& inst() { return *sm_impl; }
};

/*!
 * \brief a infile tensorrt cache implementation
 *
 * dump format:
 *
 * all integers in local endian (effectively little endian as I can see)
 *
 * dump format:
 *  <nr_blob|uint32_t>[<key_size|uint32_t><key|uint8_t*><data_size|uint32_t><data|uint8_t*>]*
 */
class TensorRTEngineCacheIO final : public TensorRTEngineCache {
    std::string m_filename;
    FILE* m_ptr = nullptr;
    bool m_update_cache = false;

    template <typename T>
    void read(T& val) {
        auto ret = fread(&val, sizeof(T), 1, m_ptr);
        MGB_MARK_USED_VAR(ret);
        mgb_throw_if(ret != 1, SystemError,
                     "failed to read block with size (%zu) from file %s %s",
                     sizeof(T), m_filename.c_str(), strerror(errno));
    }

    template <typename T>
    void read(T* buf, size_t size) {
        auto ret = fread(buf, size, 1, m_ptr);
        MGB_MARK_USED_VAR(ret);
        mgb_throw_if(ret != 1, SystemError,
                     "failed to read block with size (%zu) from file %s %s",
                     size, m_filename.c_str(), strerror(errno));
    }

    template <typename T>
    void write(T val) {
        auto ret = fwrite(&val, sizeof(T), 1, m_ptr);
        MGB_MARK_USED_VAR(ret);
        mgb_throw_if(ret != 1, SystemError,
                     "failed to write block with size (%zu) to file %s %s",
                     sizeof(T), m_filename.c_str(), strerror(errno));
    }

    template <typename T>
    void write(const T* buf, size_t size) {
        static_assert(sizeof(T) == 1, "only support write bytes");
        auto ret = fwrite(buf, size, 1, m_ptr);
        MGB_MARK_USED_VAR(ret);
        mgb_throw_if(ret != 1, SystemError,
                     "failed to write block with size (%zu) to file %s %s", size,
                     m_filename.c_str(), strerror(errno));
    }

    void read_cache();

    struct EngineStorage : public Engine {
        std::unique_ptr<uint8_t[]> data_refhold;

        EngineStorage& init_from(TensorRTEngineCacheIO& io) {
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

        void write_to(TensorRTEngineCacheIO& io) {
            uint32_t data_size = size;
            io.write(data_size);
            io.write(data_refhold.get(), size);
        }
    };

    std::unordered_map<std::string, EngineStorage> m_cache;
    std::mutex m_mtx;

public:
    TensorRTEngineCacheIO(std::string filename);
    ~TensorRTEngineCacheIO() = default;

    void dump_cache() override;

    Maybe<Engine> get(const std::string& key) override;

    void put(const std::string& key, const Engine& value) override;
};
}  // namespace mgb
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
