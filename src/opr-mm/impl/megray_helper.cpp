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

MegRay::DType mgb::opr::get_megray_dtype(megdnn::DType dtype) {
    switch(dtype.enumv()) {
        case DTypeEnum::Int8:
            return MegRay::DType::MEGRAY_INT8;
        case DTypeEnum::Int32:
            return MegRay::DType::MEGRAY_INT32;
        case DTypeEnum::Float32:
            return MegRay::DType::MEGRAY_FLOAT32;
#ifndef MEGDNN_DISABLE_FLOAT16
        case DTypeEnum::Float16:
            return MegRay::DType::MEGRAY_FLOAT16;
#endif
        default:
            mgb_throw(MegBrainError, "bad CollectiveComm dtype");
    }
}

MegRay::Backend mgb::opr::get_megray_backend(const std::string& backend) {
    if (backend == "nccl") {
        return MegRay::MEGRAY_NCCL;
    } else if (backend == "ucx") {
        return MegRay::MEGRAY_UCX;
    } else {
        mgb_throw(MegBrainError, "back CollectiveComm backend");
    }
}

bool MegRayCommBuilder::find(uint64_t hash, std::shared_ptr<MegRay::Communicator>& comm) {
    std::unique_lock<std::mutex> lk(m_map_mtx);
    auto it = m_megray_comms.find(hash);
    if (it != m_megray_comms.end()) {
        comm = it->second;
        return true;
    }
    return false;
}

void MegRayCommBuilder::emplace(uint64_t hash,
        std::shared_ptr<MegRay::Communicator> comm) {
    std::unique_lock<std::mutex> lk(m_map_mtx);
    m_megray_comms.emplace(hash, comm);
}

std::shared_ptr<MegRay::Communicator> MegRayCommBuilder::get_megray_comm(
        uint64_t hash, std::string key, uint32_t size, uint32_t rank,
        MegRay::Backend backend,
        std::shared_ptr<mgb::opr::GroupClient> group_client) {
    {
        // singleton pattern
        std::unique_lock<std::mutex> lk(sm_instance_mtx);
        if (sm_instance == nullptr) {
            sm_instance = new MegRayCommBuilder();
        }
    }

    std::shared_ptr<MegRay::Communicator> comm;
    if (!sm_instance->find(hash, comm)) {
        uint32_t root = 0;
        std::string master_ip;
        int port = 0;
        if (rank == root) {
            char* c = MegRay::get_host_ip();
            master_ip = std::string(c);
            delete c;
            port = MegRay::get_free_port();
            auto ret = MegRay::create_server(size, port);
            mgb_assert(ret == MegRay::Status::MEGRAY_OK);
        }
        group_client->bcast_addr(master_ip, port, key, size, rank, root);

        comm = MegRay::get_communicator(size, rank, backend);
        auto ret = comm->init(master_ip.c_str(), port);
        mgb_assert(ret == MegRay::Status::MEGRAY_OK);
        sm_instance->emplace(hash, comm);
    }
    return comm;
}

MegRayCommBuilder* MegRayCommBuilder::sm_instance = nullptr;

std::mutex MegRayCommBuilder::sm_instance_mtx;

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
