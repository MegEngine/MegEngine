/**
 * \file src/core/impl/utils/persistent_cache.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/persistent_cache.h"
#include "megbrain/comp_node_env.h"

#include <cstdio>
#include <cstring>

#ifdef WIN32
#define snprintf _snprintf
#endif

#if MGB_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace mgb;

namespace {

    class InMemoryPersistentCache final: public PersistentCache {
        struct BlobStorage: public Blob {
            std::unique_ptr<uint8_t[]> data_refhold;
            size_t hash = 0;

            BlobStorage& init_data_ref(const Blob &b) {
                data_refhold = std::make_unique<uint8_t[]>(b.size + 1);
                memcpy(data_refhold.get(), b.ptr, b.size);
                data_refhold.get()[b.size] = 0;   // for C-string safety
                ptr = data_refhold.get();
                size = b.size;
                return *this;
            }

            BlobStorage& init_hash() {
                hash = XXHash{}.update(ptr, size).digest();
                return *this;
            }

            bool operator == (const BlobStorage &rhs) const {
                return size == rhs.size && !memcmp(ptr, rhs.ptr, size);
            }

            struct Hash {
                size_t operator() (const BlobStorage &b) const {
                    return b.hash;
                }
            };
        };
        std::unordered_map<std::string,
            std::unordered_map<BlobStorage, BlobStorage, BlobStorage::Hash>>
                m_cache;
        std::mutex m_mtx;

        Maybe<Blob> get(const std::string& category, const Blob& key) override {
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

        void put(const std::string& category, const Blob& key,
                 const Blob& value) override {
            BlobStorage key_storage;
            key_storage.init_data_ref(key).init_hash();

            MGB_LOCK_GUARD(m_mtx);
            auto size0 = m_cache.size();
            m_cache[category][std::move(key_storage)].init_data_ref(value);
            if (m_cache.size() > size0) {
                mgb_log_debug("new cache category: %s", category.c_str());
            }
        }
    };
}
std::shared_ptr<PersistentCache> PersistentCache::sm_impl =
std::make_shared<InMemoryPersistentCache>();

std::shared_ptr<PersistentCache> PersistentCache::set_impl(
        std::shared_ptr<PersistentCache> impl) {
    mgb_assert(impl);
    sm_impl.swap(impl);
    return impl;
}

std::string PersistentCache::make_category_from_comp_node(CompNode comp_node) {
    auto&& env = CompNodeEnv::from_comp_node(comp_node);
    switch (env.property().type) {
#if MGB_CUDA
        case CompNode::DeviceType::CUDA: {
            int drv = -1, cuda_rt = -1;
            MGB_CUDA_CHECK(cudaDriverGetVersion(&drv));
            MGB_CUDA_CHECK(cudaRuntimeGetVersion(&cuda_rt));
            auto&& prop = env.cuda_env().device_prop;
            // note: we do not contain library versions such as cudnn here. They
            // are handled by opr impls in MegDNN
            return ssprintf("plat=cuda;dev=%s;cap=%d.%d,drv=%d;runtime=%d",
                            prop.name, prop.major, prop.minor, drv, cuda_rt);
            break;
        }
#endif
#if MGB_ROCM
        case CompNode::DeviceType::ROCM: {
            int drv = -1, hip_rt = -1;
            MGB_ROCM_CHECK(hipDriverGetVersion(&drv));
            MGB_ROCM_CHECK(hipRuntimeGetVersion(&hip_rt));
            auto&& prop = env.rocm_env().device_prop;
            return ssprintf("plat=rocm;dev=%s;cap=%d.%d,drv=%d;runtime=%d",
                            prop.name, prop.major, prop.minor, drv, hip_rt);
            break;
        }
#endif
        case CompNode::DeviceType::CPU:
            return "plat=cpu";
        default:
            mgb_throw(MegBrainError,
                      "unsupported comp node for persistent cache category");
    }
}

AlgoChooserProfileCache::AlgoChooserProfileCache(
        CompNode cn, const char *opr_type) {
    m_category = "profile:";
    m_category.append(PersistentCache::make_category_from_comp_node(cn));
    m_category.append(":");
    m_category.append(opr_type);
}

#define ENTRY_FMT ":%d;%lg;%zu:"

Maybe<AlgoChooserProfileCache::Result>
AlgoChooserProfileCache::get(const Key &key) {
    auto raw_buf = PersistentCache::inst().get(m_category, key.build_blob());
    if(!raw_buf.valid())
        return None;
    mgb_assert(raw_buf->size <= 1024 * 1024,
            "buf size too large, maybe corrupted data: %p %zu",
            raw_buf->ptr, raw_buf->size);
    auto buf = static_cast<const uint8_t*>(raw_buf->ptr),
         buf_end = buf + raw_buf->size;
    mgb_assert(buf && buf < buf_end,
            "PersistentCache returned invalid value: ptr=%p size=%zu",
            raw_buf->ptr, raw_buf->size);
    auto read_uint32 = [&]() {
        auto next = buf + sizeof(uint32_t);
        mgb_assert(next <= buf_end);
        auto ret = *reinterpret_cast<const uint32_t*>(buf);
        buf = next;
        return ret;
    };

    auto ret_size = read_uint32();
    mgb_assert(static_cast<ptrdiff_t>(ret_size) < buf_end - buf,
            "result size too large (%u), maybe corrupted data",
            ret_size);
    Result ret(ret_size);
    for (auto &&i: ret) {
        // read algo name
        auto size = read_uint32();
        i.algo.resize(size);
        mgb_assert(buf + size < buf_end);
        memcpy(&i.algo[0], buf, size);
        buf += size;

        auto entry_len = read_uint32();
        mgb_assert(buf + entry_len <= buf_end);
        auto nr = sscanf(reinterpret_cast<const char*>(buf), ENTRY_FMT,
                         &i.reproducible, &i.time, &i.workspace);
        mgb_assert(nr == 3);
        buf += entry_len;
    }
    mgb_assert(buf == buf_end);
    return ret;
}

void AlgoChooserProfileCache::put(const Key &key, Result &result) {
    mgb_assert(!result.empty());
    auto result_cmp = [](const ResultEntry &a, const ResultEntry &b) {
        return a.time < b.time ||
            (a.time == b.time && a.workspace < b.workspace);
    };
    small_sort(result.begin(), result.end(), result_cmp);

    // remove algos that run slower but use more workspace
    for (size_t i = 1; i < result.size(); ) {
        auto &&prev = result[i - 1];
        auto &&cur = result[i];

        if (prev.workspace <= cur.workspace &&
                prev.reproducible == cur.reproducible) {
            result.erase(result.begin() + i);
        } else {
            ++ i;
        }
    }

    std::string val;
    val.reserve((sizeof(ResultEntry) - sizeof(std::string)) * 2 * result.size());
    auto write_uint32 = [&](uint32_t v) {
        val.append(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    write_uint32(result.size());
    constexpr int SPR_SIZE = 100;
    for (auto &&i: result) {
        // write algo
        write_uint32(i.algo.size());
        auto pos = val.size();
        val.resize(pos + i.algo.size());
        memcpy(&val[pos], i.algo.data(), i.algo.size());

        // write others
        write_uint32(0);
        pos = val.size();
        val.resize(pos + SPR_SIZE);
        uint32_t nr = snprintf(&val[pos], SPR_SIZE,
                ENTRY_FMT, i.reproducible, i.time, i.workspace);
        //! for memory boundary failed, snprintf ret do not contain \0
        nr += 1;
        mgb_assert(nr < SPR_SIZE);
        memcpy(&val[pos - sizeof(uint32_t)], &nr, sizeof(nr));
        val.resize(pos + nr);
    }

    PersistentCache::inst().put(m_category, key.build_blob(),
            {val.data(), val.size()});
}

PersistentCache::Blob AlgoChooserProfileCache::Key::build_blob() const {
    auto &&ret = m_blob_storage;
    if (!m_blob_storage.empty())
        return {ret.data(), ret.size()};

    ret.reserve(sizeof(TensorLayout) * 3 * m_inp_layouts_size + m_param_size);
    for (size_t i = 0; i < m_inp_layouts_size; ++ i) {
        auto &&ly = m_inp_layouts_ptr[i];
        for (size_t j = 0; j < ly.ndim; ++ j) {
            if (j)
                ret.push_back(',');
            ret.append(std::to_string(ly.shape[j]));
        }
        if (!ly.is_contiguous()) {
            ret.push_back(';');
            for (size_t j = 0; j < ly.ndim; ++ j) {
                if (j)
                    ret.push_back(',');
                ret.append(std::to_string(ly.stride[j]));
            }
        }
        ret.push_back(';');
        ret.append(ly.dtype.name());
        ret.push_back('|');
        mgb_assert(ly.format.is_default(),
                   "currently only default format is supported");
    }
    if (m_param_size) {
        ret.append(reinterpret_cast<const char*>(m_param), m_param_size);
    }
    return {ret.data(), ret.size()};
}

#undef ENGRY_FMT

#ifdef WIN32
#undef snprintf
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
