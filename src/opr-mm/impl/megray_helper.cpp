/**
 * \file src/opr-mm/impl/megray_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/megray_helper.h"

using namespace mgb;
using namespace opr;

bool MegRayCommunicatorBuilder::find(uint64_t hash, std::shared_ptr<MegRay::Communicator>& comm) {
    std::unique_lock<std::mutex> lk(m_mtx);
    auto it = m_megray_comms.find(hash);
    if (it != m_megray_comms.end()) {
        comm = it->second;
        return true;
    }
    return false;
}

void MegRayCommunicatorBuilder::emplace(uint64_t hash,
        std::shared_ptr<MegRay::Communicator> comm) {
    std::unique_lock<std::mutex> lk(m_mtx);
    m_megray_comms.emplace(hash, comm);
}

std::shared_ptr<MegRay::Communicator> MegRayCommunicatorBuilder::get_megray_comm(
        uint64_t hash, std::string key, uint32_t size, uint32_t rank,
        MegRay::Backend backend,
        std::shared_ptr<mgb::opr::GroupClient> group_client) {
    std::shared_ptr<MegRay::Communicator> comm;
    if (!find(hash, comm)) {
        comm = MegRay::get_communicator(size, rank, backend);
        auto uid = comm->get_uid();
        auto uids = group_client->gather_uid(uid, key, size, rank);
        mgb_assert(comm->init(uids) == MegRay::Status::MEGRAY_OK);
        emplace(hash, comm);
    }
    return comm;
}

MGB_TYPEINFO_OBJ_IMPL(MegRayCommunicatorBuilder);

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
