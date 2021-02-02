/**
 * \file dnn/test/common/fast_run_cache.h
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

#include "megdnn/oprs.h"
#include "src/common/hash_ct.h"

#include <unordered_map>

namespace megdnn {
namespace test {
class FastRunCache {
    struct SearchItemStorage {
        std::string data_hold;
        size_t hash = 0;

        SearchItemStorage(const Algorithm::SearchItem& item);

        SearchItemStorage& init_hash() {
            hash = XXHash64CT::hash(data_hold.data(), data_hold.size(),
                                    20201225);
            return *this;
        }

        bool operator==(const SearchItemStorage& rhs) const {
            return data_hold == rhs.data_hold;
        }

        struct Hash {
            size_t operator()(const SearchItemStorage& s) const {
                return s.hash;
            }
        };
    };

    std::unordered_map<SearchItemStorage, Algorithm::Info::Desc,
                       SearchItemStorage::Hash>
            m_cache;

public:
    Algorithm::Info::Desc get(const Algorithm::SearchItem& key);
    void put(const Algorithm::SearchItem& key,
             const Algorithm::Info::Desc& val);
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
