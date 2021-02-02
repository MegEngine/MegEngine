/**
 * \file dnn/test/common/fast_run_cache.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "test/common/fast_run_cache.h"
#include "src/common/utils.h"

using namespace megdnn;
using namespace test;

FastRunCache::SearchItemStorage::SearchItemStorage(
        const Algorithm::SearchItem& item) {
    Algorithm::serialize_write_pod(item.opr_type, data_hold);
    for (auto&& layout : item.layouts) {
        data_hold += layout.serialize();
    }
    data_hold += item.param;
}

Algorithm::Info::Desc FastRunCache::get(const Algorithm::SearchItem& key) {
    SearchItemStorage key_storage(key);
    key_storage.init_hash();

    auto iter = m_cache.find(key_storage);
    if (iter == m_cache.end()) {
        return {};
    }
    return iter->second;
}

void FastRunCache::put(const Algorithm::SearchItem& key,
                       const Algorithm::Info::Desc& val) {
    SearchItemStorage key_storage(key);
    key_storage.init_hash();
    megdnn_assert(m_cache.find(key_storage) == m_cache.end());
    m_cache[std::move(key_storage)] = val;
}

// vim: syntax=cpp.doxygen
