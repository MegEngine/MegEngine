#include "megbrain/comp_node_api.h"
#include <unordered_map>
#include "megbrain/comp_node.h"
#include "megbrain/comp_node_env.h"
#include "megbrain_build_config.h"

namespace mgb {
namespace {

class XLAMemStats {
    class MemStatsOfSingleCN {
        size_t m_peak_alloc;
        size_t m_cur_alloc;
        size_t m_max_block_size;
        CompNode m_cn;
        std::unordered_map<void*, size_t> m_allocated;

    public:
        MemStatsOfSingleCN(CompNode cn) : m_cn(cn) {
            m_peak_alloc = 0;
            m_cur_alloc = 0;
            m_max_block_size = 0;
        }
        void on_alloc(void* ptr, size_t sz) {
            m_allocated[ptr] = sz;
            m_cur_alloc += sz;
            if (mgb_unlikely(m_cur_alloc > m_peak_alloc)) {
                m_peak_alloc = m_cur_alloc;
            }
            if (mgb_unlikely(sz > m_max_block_size)) {
                m_max_block_size = sz;
            }
        }
        void on_dealloc(void* ptr) {
            auto it = m_allocated.find(ptr);
            mgb_assert(it != m_allocated.end());
            m_cur_alloc -= it->second;
            m_allocated.erase(it);
        }
        void log_states() {
            printf("%s: peak alloc (%ld KB), max block size: (%ld KB), current alloc: "
                   "(%ld KB)\n",
                   m_cn.to_string().c_str(), (long)((long double)(m_peak_alloc) / 1024),
                   (long)((long double)(m_max_block_size) / 1024),
                   (long)((long double)(m_cur_alloc) / 1024));
        }
    };

    CompNode::UnorderedMap<std::unique_ptr<MemStatsOfSingleCN>> m_cn2stats;
    XLAMemStats() {}

public:
    static XLAMemStats* inst() {
        static XLAMemStats item;
        return &item;
    }

    bool is_xla_used() { return m_cn2stats.size() != 0; }

    void on_alloc(CompNode cn, void* ptr, size_t sz) {
        if (m_cn2stats.find(cn) == m_cn2stats.end()) {
            m_cn2stats[cn] = std::make_unique<MemStatsOfSingleCN>(cn);
        }
        m_cn2stats[cn]->on_alloc(ptr, sz);
    }
    void on_dealloc(CompNode cn, void* ptr) {
        mgb_assert(m_cn2stats.find(cn) != m_cn2stats.end());
        m_cn2stats[cn]->on_dealloc(ptr);
    }
    void log_states(CompNode cn) {
        mgb_assert(m_cn2stats.find(cn) != m_cn2stats.end());
        m_cn2stats[cn]->log_states();
    }
};

}  // namespace

#define XLA_MEM_STATS_ON_ALLOC(cn, ptr, sz) XLAMemStats::inst()->on_alloc(cn, ptr, sz)
#define XLA_MEM_STATS_ON_DEALLOC(cn, ptr)   XLAMemStats::inst()->on_dealloc(cn, ptr)

namespace pubapi {

bool is_xla_used() {
    return XLAMemStats::inst()->is_xla_used();
}

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
    auto ret = cn->alloc_device(s);
    XLA_MEM_STATS_ON_ALLOC(*cn, ret, s);
    return ret;
}

void dealloc(mgbComputeNode_t device, void* addr) {
    if (addr != nullptr) {
        auto* cn = reinterpret_cast<mgb::CompNode*>(device);
        if (!is_finalize()) {
            XLA_MEM_STATS_ON_DEALLOC(*cn, addr);
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
