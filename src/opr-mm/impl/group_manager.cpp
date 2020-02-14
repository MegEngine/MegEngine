/**
 * \file src/opr-mm/impl/group_manager.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/group_manager.h"

using namespace mgb;
using namespace opr;

/* ================= GroupInfo ================= */

void GroupInfo::add_opr(const std::string& key, size_t nr_expected_devices,
        uint32_t rank, uintptr_t stream) {
    std::unique_lock<std::mutex> lk{m_group_mtx};
    if (m_nr_expected_devs == 0) {
        m_nr_expected_devs = nr_expected_devices;
    } else {
        mgb_assert(m_nr_expected_devs == nr_expected_devices);
    }
    OprInfo opr_info = {rank, stream};
    m_opr_infos.push_back(std::move(opr_info));
    m_nr_registered_devs++;
    m_count++;
    if (m_nr_registered_devs > nr_expected_devices) {
        mgb_log_error(
                "too many opr registered with key %s, expected %zu, actual %u",
                key.c_str(), nr_expected_devices, m_nr_registered_devs);
        mgb_throw(
                MegBrainError,
                "too many opr registered with key %s, expected %zu, actual %u",
                key.c_str(), nr_expected_devices, m_nr_registered_devs);
    }
    if (m_nr_expected_devs == m_nr_registered_devs) {
        m_register_cv.notify_all();
    } else {
        m_register_cv.wait(lk,
                [&] { return m_nr_expected_devs == m_nr_registered_devs; });
    }
}

void GroupInfo::set_output_shape(const std::string& key,
        const TensorShape& shape) {
    MGB_LOCK_GUARD(m_output_shape_mtx);
    m_output_shape = shape;
    m_output_shape_cv.notify_all();
}

TensorShape GroupInfo::get_output_shape(const std::string& key) {
    std::unique_lock<std::mutex> lk{m_output_shape_mtx};
    if (!m_output_shape.valid()) {
        m_output_shape_cv.wait(lk);
        mgb_assert(m_output_shape.valid());
    }
    return m_output_shape.val();
}

void GroupInfo::clear() {
    std::unique_lock<std::mutex> lk{m_group_mtx};
    m_count--;
    if (m_count == 0) {
        m_opr_infos.clear();
        m_nr_expected_devs = 0;
        m_nr_registered_devs = 0;
        m_output_shape.invalidate();
        m_clear_cv.notify_all();
    } else {
        m_clear_cv.wait(lk, [&] { return m_count == 0; });
    }
}

/* ================= GroupManager ================= */

uint64_t GroupManager::opr_register(const std::string& key, size_t nr_devices,
    uint32_t rank, uintptr_t stream) {
    auto&& group = get_group(key);
    group.add_opr(key, nr_devices, rank, stream);
    auto&& opr_infos = group.opr_infos();
    uint64_t hash = get_hash_key(opr_infos, rank);
    group.clear();
    return hash;
}

std::vector<std::string> GroupManager::gather_uid(const std::string& uid,
    const std::string& key, uint32_t size, uint32_t rank) {
    std::unique_lock<std::mutex> lk{m_key2uids_mtx};
    if (m_key2uids_size[key] == 0)
        m_key2uids[key].resize(size);
    m_key2uids[key][rank] = uid;
    m_key2uids_size[key]++;
    if (m_key2uids_size[key] == size) {
        m_key2uids_flag[key] = true;
        m_gather_uid_cv.notify_all();
    } else {
        m_gather_uid_cv.wait(
                lk, [&] { return m_key2uids_flag.count(key) > 0; });
    }
    auto uids = m_key2uids[key];
    m_key2uids_size[key]--;
    if (m_key2uids_size[key] == 0) {
        m_key2uids.erase(key);
        m_key2uids_flag.erase(key);
    }
    return uids;
}

void GroupManager::set_output_shape(const std::string& key,
        const TensorShape& shape) {
    auto&& group = get_group(key);
    group.set_output_shape(key, shape);
}

TensorShape GroupManager::get_output_shape(const std::string& key) {
    auto&& group = get_group(key);
    return group.get_output_shape(key);
}

GroupInfo& GroupManager::get_group(const std::string& key) {
    MGB_LOCK_GUARD(m_key2group_info_mtx);
    return m_key2group_info[key];
}

uint64_t GroupManager::get_hash_key(const std::vector<GroupInfo::OprInfo>& _infos,
        uint32_t rank) {
    auto cmp = [](const GroupInfo::OprInfo& lhs, const GroupInfo::OprInfo& rhs) {
        return lhs.rank < rhs.rank;
    };
    auto infos = _infos;
    std::sort(infos.begin(), infos.end(), cmp);
    auto xxhash = XXHash{};
    for (auto&& opr_info : infos) {
        xxhash.update(&opr_info.rank, sizeof(uint32_t))
                .update(&opr_info.stream, sizeof(uintptr_t));
    }
    xxhash.update(&rank, sizeof(uint32_t));
    return xxhash.digest();
};

uint32_t GroupManager::group_barrier(uint32_t size, uint32_t rank) {
    std::unique_lock<std::mutex> lk{m_barrier_mtx};
    if (m_barrier_set.empty()) {
        m_barrier_size = size;
    } else if (size != m_barrier_size) {
        mgb_log_error("inconsistent size: %d, expect %d", size, m_barrier_size);
        return m_barrier_size;
    } else if (rank >= size) {
        mgb_log_error("invalid rank %d", rank);
        return m_barrier_size;
    }
    if (m_barrier_set.count(rank) > 0) {
        mgb_log_error("rank already registered: %d", rank);
        return 0;
    }
    m_barrier_set.insert(rank);
    if (m_barrier_set.size() == m_barrier_size) {
        m_barrier_set.clear();
        m_barrier_cv.notify_all();
    } else {
        m_barrier_cv.wait(lk, [&] { return m_barrier_set.empty(); });
    }
    return m_barrier_size;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
