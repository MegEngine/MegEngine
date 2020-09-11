/**
 * \file src/opr-mm/include/megbrain/opr/megray_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <mutex>

#include "megbrain/opr/group_manager.h"
#include "megray.h"

namespace mgb {
namespace opr {

MegRay::DType get_megray_dtype(megdnn::DType);

MegRay::Backend get_megray_backend(const std::string& backend);

/*!
 * gather MegRay unique ids and build communicator, use hash for deduplication
 */
class MegRayCommBuilder {
    private:
        bool find(uint64_t hash, std::shared_ptr<MegRay::Communicator>& comm);
        void emplace(uint64_t hash, std::shared_ptr<MegRay::Communicator> comm);

        std::unordered_map<uint64_t, std::shared_ptr<MegRay::Communicator>> m_megray_comms;
        std::mutex m_map_mtx;

        static MegRayCommBuilder* sm_instance;
        static std::mutex sm_instance_mtx;

    public:
        static std::shared_ptr<MegRay::Communicator> get_megray_comm(
                uint64_t hash, std::string key, uint32_t size, uint32_t rank,
                MegRay::Backend backend,
                std::shared_ptr<mgb::opr::GroupClient> group_client);
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
