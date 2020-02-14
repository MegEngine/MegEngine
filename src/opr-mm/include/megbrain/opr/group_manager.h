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
            uint32_t rank;
            uintptr_t stream; 
        };

        void add_opr(const std::string& key, size_t nr_expected_devices,
                uint32_t graph_id, uintptr_t stream);

        void set_output_shape(const std::string& key, const TensorShape& shape);

        TensorShape get_output_shape(const std::string& key);

        void clear();

        const std::vector<OprInfo>& opr_infos() const {return m_opr_infos; }

    private:
        std::vector<OprInfo> m_opr_infos;
        uint32_t m_nr_registered_devs;
        uint32_t m_nr_expected_devs;
        Maybe<TensorShape> m_output_shape;

        uint32_t m_count = 0;
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

        //! register oprs' info to server, return deduplicated hash
        uint64_t opr_register(const std::string& key, size_t nr_devices, uint32_t rank,
                uintptr_t stream);
    
        //! gather uids from all ranks
        std::vector<std::string> gather_uid(const std::string& uid,
                const std::string& key, uint32_t size, uint32_t rank);
    
        //! Set output shape of this key
        void set_output_shape(const std::string& key, const TensorShape& shape);
    
        //! Get output shape of this key, blocks until output shape is set
        TensorShape get_output_shape(const std::string& key);

        //! Block clients until all ranks reach this barrier
        uint32_t group_barrier(uint32_t size, uint32_t rank);

    private:
        GroupInfo& get_group(const std::string& key);

        uint64_t get_hash_key(const std::vector<GroupInfo::OprInfo>& _infos,
                uint32_t rank);
    
        //! key -> group info.
        std::unordered_map<std::string, GroupInfo> m_key2group_info;
        std::mutex m_key2group_info_mtx;
    
        //! key -> uid
        std::unordered_map<std::string, std::vector<std::string>> m_key2uids;
        std::unordered_map<std::string, uint32_t> m_key2uids_size;
        std::unordered_map<std::string, bool> m_key2uids_flag;
        std::mutex m_key2uids_mtx;
        std::condition_variable m_gather_uid_cv;

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
        virtual uint64_t opr_register(const std::string& key, size_t nr_devices,
                uint32_t rank, uintptr_t stream) = 0;
    
        virtual std::vector<std::string> gather_uid(const std::string& uid,
                const std::string& key, uint32_t size, uint32_t rank) = 0;
    
        virtual void set_output_shape(const std::string& key,
                const TensorShape& shape) = 0;
    
        virtual TensorShape get_output_shape(const std::string& key) = 0;

        virtual uint32_t group_barrier(uint32_t size, uint32_t rank) = 0;
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
