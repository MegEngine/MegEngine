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

void GroupInfo::sort_opr_infos() {
    auto cmp = [](const GroupInfo::OprInfo& a, const GroupInfo::OprInfo& b) {
        return a.comp_node_hash < b.comp_node_hash;
    };
    std::sort(m_opr_infos.begin(), m_opr_infos.end(), cmp);
}

void GroupInfo::gen_infos_from_opr_infos() {
    // generate rank
    bool rank_assgined = true;
    for (auto& opr_info:m_opr_infos) {
        if(opr_info.rank < 0) {
            rank_assgined = false;
            break;
        }
    }
    if (!rank_assgined) {
        for (size_t i = 0; i < m_opr_infos.size(); i++) {
            m_opr_infos[i].rank = i;
            m_rank_map.insert({m_opr_infos[i].comp_node_hash, i});
        }
    } else {
        for (size_t i = 0; i < m_opr_infos.size(); i++) {
            m_rank_map.insert(
                    {m_opr_infos[i].comp_node_hash, m_opr_infos[i].rank});
        }
    }

    // generate root rank
    for (auto& opr_info:m_opr_infos) {
        if (opr_info.is_root) {
            m_root_rank = opr_info.rank;
            break;
        }
    }

    // generate group hash
    auto xxhash = XXHash{};
    for (auto&& opr_info : m_opr_infos) {
        xxhash.update(&opr_info.comp_node_hash, sizeof(uint64_t))
                .update(&opr_info.rank, sizeof(int));
    }
    m_hash = xxhash.digest();
}

void GroupInfo::add_opr(const std::string& key, size_t nr_expected_devices,
        bool is_root, int rank, uint64_t comp_node_hash) {
    std::unique_lock<std::mutex> lk{m_group_mtx};
    if (m_nr_expected_devs == 0) {
        m_nr_expected_devs = nr_expected_devices;
    } else {
        mgb_assert(m_nr_expected_devs == nr_expected_devices);
    }
    m_opr_infos.push_back({comp_node_hash, is_root, rank});
    m_nr_registered_devs++;
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
        sort_opr_infos();
        gen_infos_from_opr_infos();
        m_count = m_nr_registered_devs;
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
        m_rank_map.clear();
        m_root_rank = -1;
        m_nr_expected_devs = 0;
        m_nr_registered_devs = 0;
        m_output_shape.invalidate();
        m_clear_cv.notify_all();
    } else {
        m_clear_cv.wait(lk, [&] { return m_count == 0; });
    }
}

/* ================= GroupManager ================= */

GroupManager::RegisterInfo GroupManager::opr_register(const std::string& key,
                                                      size_t nr_devices,
                                                      bool is_root, int rank,
                                                      uint64_t comp_node_hash) {
    GroupManager::RegisterInfo ret{0, 0, 0};
    auto&& group = get_group(key);
    group.add_opr(key, nr_devices, is_root, rank, comp_node_hash);
    ret.rank = group.get_rank(comp_node_hash);
    ret.root_rank = group.get_root_rank();
    ret.hash = group.get_group_hash() + ret.rank;
    group.clear();
    return ret;
}

void GroupManager::bcast_addr(std::string& master_ip, int& port,
    const std::string& key, uint32_t size, uint32_t rank, uint32_t root) {
    std::unique_lock<std::mutex> lk{m_key2addr_mtx};
    if (rank == root) {
        m_key2master_ip[key] = master_ip;
        m_key2port[key] = port;
    }
    m_key2addr_size[key]++;
    if (m_key2addr_size[key] == size) {
        m_key2addr_flag[key] = true;
        m_bcast_cv.notify_all();
    } else {
        m_bcast_cv.wait(
                lk, [&] { return m_key2addr_flag.count(key) > 0; });
    }
    master_ip = m_key2master_ip[key];
    port = m_key2port[key];
    m_key2addr_size[key]--;
    if (m_key2addr_size[key] == 0) {
        m_key2master_ip.erase(key);
        m_key2port.erase(key);
        m_key2addr_flag.erase(key);
    }
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

void RegInfoCache::set_info(const std::string& key,
        const GroupManager::RegisterInfo& info) {
    std::unique_lock<std::mutex> lock(RegInfoCache::mtx);
    RegInfoCache::key2info[key] = info;
}

bool RegInfoCache::has_info(const std::string& key) {
    std::unique_lock<std::mutex> lock(RegInfoCache::mtx);
    return RegInfoCache::key2info.find(key) != RegInfoCache::key2info.end();
}

GroupManager::RegisterInfo RegInfoCache::get_info(const std::string& key) {
    std::unique_lock<std::mutex> lock(RegInfoCache::mtx);
    return RegInfoCache::key2info[key];
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
