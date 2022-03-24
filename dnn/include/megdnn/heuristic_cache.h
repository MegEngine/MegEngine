/**
 * \file dnn/include/megdnn/heuristic_cache.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megdnn/basic_types.h"
#include "megdnn/oprs/base.h"

#include <mutex>
#include <string>
#include <unordered_map>

namespace megdnn {

class HeuristicCache {
private:
    HeuristicCache() = default;

public:
    MGE_WIN_DECLSPEC_FUC static HeuristicCache& instance();

    struct KeyStorage {
        size_t k1, k2;

        bool operator==(const KeyStorage& k) const { return k1 == k.k1 && k2 == k.k2; }
    };

    struct Key {
        Handle* m_handle;
        uint32_t m_opr_type;
        const TensorLayout* m_inp_layouts_ptr;
        size_t m_inp_layouts_size;
        const void* m_param_ptr;
        size_t m_param_size;

        mutable SmallVector<size_t> m_buf;

    public:
        Key(Handle* opr_handle, Algorithm::OprType opr_type,
            const TensorLayout* inp_layouts_ptr, size_t inp_layouts_size,
            const void* param_ptr = nullptr, size_t param_size = 0)
                : m_handle{opr_handle},
                  m_opr_type{static_cast<uint32_t>(opr_type)},
                  m_inp_layouts_ptr{inp_layouts_ptr},
                  m_inp_layouts_size{inp_layouts_size},
                  m_param_ptr{param_ptr},
                  m_param_size{param_size} {}

        KeyStorage build_key_storage() const;
    };

    struct Result {
        ExecutionPolicy policy;
        size_t workspace;

        // for cache collision
        SmallVector<size_t> m_buf;
        SmallVector<char> m_param_buf;
    };

    MGE_WIN_DECLSPEC_FUC void put(const Key& key, Result& result);

    MGE_WIN_DECLSPEC_FUC Result get(const Key& key);

    void clear();

private:
    struct Hash {
        size_t operator()(const KeyStorage& k) const {
            size_t h1 = k.k1;
            size_t h2 = k.k2;
            h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
            return h1;
        }
    };
    std::unordered_map<KeyStorage, Result, Hash> m_heuristic_cache;
#if __DEPLOY_ON_XP_SP2__
    size_t m_mtx;
#else
    std::mutex m_mtx;
#endif
};

}  // namespace megdnn
