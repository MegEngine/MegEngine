/**
 * \file src/opr-mm/include/megbrain/opr/group_manager.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <set>

#include "megbrain/tensor.h"

namespace mgb {
namespace opr {

/*!
 * GroupInfo: stream and shape information from all ranks of a group
 */
class GroupInfo {
    public:
        struct OprInfo {
            uint64_t comp_node_hash;
            bool is_root;
            int rank;
        };

        void add_opr(const std::string& key, size_t nr_expected_devices,
                bool is_root, int rank, uint64_t comp_node_hash);

        void set_output_shape(const std::string& key, const TensorShape& shape);

        TensorShape get_output_shape(const std::string& key);

        void clear();

        const std::vector<OprInfo>& opr_infos() const { return m_opr_infos; }

        int get_root_rank() const { return m_root_rank; }
        int get_rank(uint64_t hash) const { return m_rank_map.at(hash); }
        uint64_t get_group_hash() const { return m_hash; }

    private:
        void sort_opr_infos();
        void gen_infos_from_opr_infos();

        std::vector<OprInfo> m_opr_infos;
        std::unordered_map<uint64_t, int> m_rank_map;
        uint64_t m_hash;
        uint32_t m_nr_registered_devs;
        uint32_t m_nr_expected_devs;
        Maybe<TensorShape> m_output_shape;

        uint32_t m_count = 0;
        int m_root_rank = -1;
        std::mutex m_group_mtx;
        std::condition_variable m_register_cv;
        std::condition_variable m_clear_cv;

        std::mutex m_output_shape_mtx;
        std::condition_variable m_output_shape_cv;
};

/*!
 * GroupManager: build groups and exchange meta information
 */
class GroupManager {
    public:
        ~GroupManager() = default;

        struct RegisterInfo
        {
            uint64_t hash;
            int rank, root_rank;
        };

        //! register oprs' info to server, return deduplicated hash
        RegisterInfo opr_register(const std::string& key, size_t nr_devices,
                                  bool is_root, int rank, uint64_t comp_node_hash);

        //! broadcast master_ip and port
        void bcast_addr(std::string& master_ip, int& port,
            const std::string& key, uint32_t size, uint32_t rank, uint32_t root);
    
        //! Set output shape of this key
        void set_output_shape(const std::string& key, const TensorShape& shape);
    
        //! Get output shape of this key, blocks until output shape is set
        TensorShape get_output_shape(const std::string& key);

        //! Block clients until all ranks reach this barrier
        uint32_t group_barrier(uint32_t size, uint32_t rank);

    private:
        GroupInfo& get_group(const std::string& key);
    
        //! key -> group info.
        std::unordered_map<std::string, GroupInfo> m_key2group_info;
        std::mutex m_key2group_info_mtx;
    
        //! key -> addr
        std::unordered_map<std::string, std::string> m_key2master_ip;
        std::unordered_map<std::string, int> m_key2port;
        std::unordered_map<std::string, uint32_t> m_key2addr_size;
        std::unordered_map<std::string, bool> m_key2addr_flag;
        std::mutex m_key2addr_mtx;
        std::condition_variable m_bcast_cv;

        //! barrier
        uint32_t m_barrier_size;
        std::set<uint32_t> m_barrier_set;
        std::mutex m_barrier_mtx;
        std::condition_variable m_barrier_cv;
};

/*!
 * Client interface to interact with GroupManager.
 * All the methods below should be overrided by subclasses
 * Test cases mock the interface to directly interact with GroupManager
 */
class GroupClient {
    protected:
        virtual ~GroupClient() = default;
    
    public:
        virtual const std::string& get_addr() const = 0;

        virtual GroupManager::RegisterInfo opr_register(const std::string& key,
                                                        size_t nr_devices,
                                                        bool is_root, int rank,
                                                        uint64_t comp_node_hash) = 0;

        virtual void bcast_addr(std::string& master_ip, int& port,
            const std::string& key, uint32_t size, uint32_t rank, uint32_t root) = 0;
    
        virtual void set_output_shape(const std::string& key,
                const TensorShape& shape) = 0;
    
        virtual TensorShape get_output_shape(const std::string& key) = 0;

        virtual uint32_t group_barrier(uint32_t size, uint32_t rank) = 0;
};

/*!
 * Cache RegisterInfo returned from GroupManager. This feature is only enabled
 * in imperative runtime mode, so that multi-machine operators do not have to
 * call opr_register repeatedly in each iter
 */
namespace RegInfoCache {

static std::mutex mtx;
static std::unordered_map<std::string, GroupManager::RegisterInfo> key2info;

void set_info(const std::string& key, const GroupManager::RegisterInfo& info);
bool has_info(const std::string& key);
GroupManager::RegisterInfo get_info(const std::string& key);

}  // namespace RegInfoCache

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
