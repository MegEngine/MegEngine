#include "megbrain/comp_node_api.h"
#include <unordered_map>
#include "megbrain/comp_node.h"
#include "megbrain/comp_node_env.h"
#include "megbrain_build_config.h"

namespace mgb {
namespace pubapi {

std::unordered_map<std::string, mgb::CompNode*>& cn_cache() {
    static std::unordered_map<std::string, mgb::CompNode*> cn_map;
    return cn_map;
}

class CompNodeDepedentObjectInst final : public CompNodeDepedentObject {
    std::shared_ptr<void> on_comp_node_finalize() override { return {}; }

public:
    bool is_finalized() const { return CompNodeDepedentObject::is_finalized(); }
};

bool is_finalize() {
    static CompNodeDepedentObjectInst* obj = new CompNodeDepedentObjectInst;
    return obj->is_finalized();
}

void sync(mgbComputeNode_t cn) {
    if (!is_finalize()) {
        auto* s = reinterpret_cast<mgb::CompNode*>(cn);
        if (s->valid())
            s->sync();
    }
}

mgbComputeNode_t load_cuda_cn(int device_id, int stream) {
    std::string loc = ssprintf("gpu%i:%i", device_id, stream);
    mgb_assert(!is_finalize());
    auto& cache = cn_cache();
    if (cache.find(loc) == cache.end()) {
        auto* cn = new mgb::CompNode;
        (*cn) = mgb::CompNode::load(loc);
        mgb_assert(cn->to_string_physical() == loc);
        cache[loc] = cn;
        cn->activate();
    }
    return reinterpret_cast<mgbComputeNode_t>(cache[loc]);
}

void unload_cuda_cn(mgbComputeNode_t cn) {
    auto* device = reinterpret_cast<mgb::CompNode*>(cn);
    auto& cache = cn_cache();
    mgb_assert(
            cache.find(device->to_string_physical()) != cache.end() &&
            device == cache[device->to_string_physical()]);
    cache.erase(device->to_string_physical());
    delete device;
}

void* alloc(mgbComputeNode_t device, size_t s) {
    if (s == 0)
        return nullptr;
    auto* cn = reinterpret_cast<mgb::CompNode*>(device);
    mgb_assert(!is_finalize());
    return cn->alloc_device(s);
}

void dealloc(mgbComputeNode_t device, void* addr) {
    if (addr != nullptr) {
        auto* cn = reinterpret_cast<mgb::CompNode*>(device);
        if (!is_finalize()) {
            cn->free_device(addr);
        }
    }
}

void* get_cuda_stream(mgbComputeNode_t device) {
    void* rst = nullptr;
#if MGB_CUDA
    auto* cn = reinterpret_cast<mgb::CompNode*>(device);
    MGB_TRY { rst = CompNodeEnv::from_comp_node(*cn).cuda_env().stream; }
    MGB_CATCH(MegBrainError & exc, {
        mgb_log_error("failed to get stream: %s", exc.what());
    })
#else
    mgb_log_error("megbrain compiled without cuda support!");
#endif
    return rst;
}

MGB_API DeviceLocator get_physical_location(mgbComputeNode_t device) {
    auto location = reinterpret_cast<CompNode*>(device)->locator().to_physical();
    return {location.device, location.stream};
}
}  // namespace pubapi
}  // namespace mgb
