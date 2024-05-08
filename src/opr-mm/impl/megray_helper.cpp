#include "megbrain/opr/megray_helper.h"
#include "megbrain/comp_node_env.h"
#include "megray/common.h"

using namespace mgb;
using namespace opr;

MegRay::DType mgb::opr::get_megray_dtype(megdnn::DType dtype) {
    switch (dtype.enumv()) {
        case DTypeEnum::Int8:
            return MegRay::DType::MEGRAY_INT8;
        case DTypeEnum::Uint8:
            return MegRay::DType::MEGRAY_UINT8;
        case DTypeEnum::Int32:
            return MegRay::DType::MEGRAY_INT32;
        case DTypeEnum::Float32:
            return MegRay::DType::MEGRAY_FLOAT32;
#if !MEGDNN_DISABLE_FLOAT16
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
    } else if (backend == "rccl") {
        return MegRay::MEGRAY_RCCL;
    } else if (backend == "ucx") {
        return MegRay::MEGRAY_UCX;
    } else if (backend == "cncl") {
        return MegRay::MEGRAY_CNCL;
    } else if (backend == "hccl") {
        return MegRay::MEGRAY_HCCL;
    } else {
        mgb_throw(MegBrainError, "bad CollectiveComm backend");
    }
}

std::shared_ptr<MegRay::Context> mgb::opr::get_megray_context(CompNode comp_node) {
#if MGB_CUDA
    return MegRay::CudaContext::make(
            CompNodeEnv::from_comp_node(comp_node).cuda_env().stream);
#elif MGB_ROCM
    return MegRay::HipContext::make(
            CompNodeEnv::from_comp_node(comp_node).rocm_env().stream);
#elif MGB_CAMBRICON
    return MegRay::CnrtContext::make(
            CompNodeEnv::from_comp_node(comp_node).cnrt_env().queue);
#elif MGB_ATLAS
    return MegRay::AclrtContext::make(
            CompNodeEnv::from_comp_node(comp_node).atlas_env().stream);
#else
#error "neither CUDA nor ROCm is enabled"
#endif
}

bool MegRayCommBuilder::find(
        uint64_t hash, std::shared_ptr<MegRay::Communicator>& comm) {
    std::unique_lock<std::mutex> lk(m_map_mtx);
    auto it = m_megray_comms.find(hash);
    if (it != m_megray_comms.end()) {
        comm = it->second;
        return true;
    }
    return false;
}

void MegRayCommBuilder::emplace(
        uint64_t hash, std::shared_ptr<MegRay::Communicator> comm) {
    std::unique_lock<std::mutex> lk(m_map_mtx);
    m_megray_comms.emplace(hash, comm);
}

void MegRayCommBuilder::remove(
        uint64_t hash, std::shared_ptr<MegRay::Communicator> comm) {
    std::unique_lock<std::mutex> lk(m_map_mtx);
    auto it = m_megray_comms.find(hash);
    if (it != m_megray_comms.end()) {
        m_megray_comms.erase(hash);
    }
}

std::shared_ptr<MegRay::Communicator> MegRayCommBuilder::get_megray_comm(
        uint64_t hash, std::string key, uint32_t size, uint32_t rank,
        MegRay::Backend backend, std::shared_ptr<mgb::opr::GroupClient> group_client) {
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
            delete[] c;
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
