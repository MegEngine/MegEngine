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

class XLAMemAllocHelper {
    class XLAMemAllocHelperOfSingleCN {
#ifdef XLA_MEM_DEBUG
        size_t m_alloc_cnt = 0;
        size_t m_peak_alloc;
        size_t m_cur_alloc;
        size_t m_max_block_size;
        std::unordered_map<void*, size_t> m_allocated_info;  // ptr -> size of block
#endif

        MGB_MUTEX m_mutex;
        CompNode m_cn;

    public:
        XLAMemAllocHelperOfSingleCN(CompNode cn) : m_cn(cn) {
#ifdef XLA_MEM_DEBUG
            m_peak_alloc = 0;
            m_cur_alloc = 0;
            m_max_block_size = 0;
#endif
        }

        void* alloc(size_t sz) {
            mgb_assert(sz > 0);
            MGB_LOCK_GUARD(m_mutex);
            void* ret = m_cn.alloc_device(sz);
#ifdef XLA_MEM_DEBUG
            m_allocated_info[ret] = sz;

            m_cur_alloc += sz;
            m_peak_alloc = m_cur_alloc > m_peak_alloc ? m_cur_alloc : m_peak_alloc;
            m_max_block_size = sz > m_max_block_size ? sz : m_max_block_size;
            m_alloc_cnt += 1;
#endif
            return ret;
        }

        void dealloc(void* ptr) {
            mgb_assert(ptr != nullptr);
            MGB_LOCK_GUARD(m_mutex);

#ifdef XLA_MEM_DEBUG
            size_t sz = m_allocated_info[ptr];
            m_allocated_info.erase(ptr);
            m_cur_alloc -= sz;
#endif

            m_cn.free_device(ptr);
        }

        void log_states() {
#ifdef XLA_MEM_DEBUG
            MGB_LOCK_GUARD(m_mutex);
            printf("%s: peak alloc (%ld KB), max block size: (%ld KB), current alloc: "
                   "(%ld KB), alloc cnt: %zu\n",
                   m_cn.to_string().c_str(), (long)((long double)(m_peak_alloc) / 1024),
                   (long)((long double)(m_max_block_size) / 1024),
                   (long)((long double)(m_cur_alloc) / 1024), m_alloc_cnt);
#else
            printf("xla mem status: please compile megengine with XLA_MEM_DEBUG "
                   "enabled\n");
#endif
        }

        void reset_states() {
#ifdef XLA_MEM_DEBUG
            MGB_LOCK_GUARD(m_mutex);
            m_peak_alloc = 0;
            m_max_block_size = 0;
            m_cur_alloc = 0;
            m_alloc_cnt = 0;
#endif
        }
    };

    CompNode::UnorderedMap<std::unique_ptr<XLAMemAllocHelperOfSingleCN>> m_helpers;
    MGB_MUTEX m_mutex;
    XLAMemAllocHelper() = default;

public:
    static XLAMemAllocHelper* inst() {
        static XLAMemAllocHelper item;
        return &item;
    }

    void* alloc(CompNode cn, size_t sz) {
        if (mgb_unlikely(sz == 0)) {
            return nullptr;
        }

        MGB_LOCK_GUARD(m_mutex);
        if (mgb_unlikely(m_helpers.find(cn) == m_helpers.end())) {
            m_helpers[cn] = std::make_unique<XLAMemAllocHelperOfSingleCN>(cn);
        }
        return m_helpers[cn]->alloc(sz);
    }

    void dealloc(CompNode cn, void* ptr) {
        if (mgb_likely(ptr != nullptr)) {
            if (!is_finalize()) {
                MGB_LOCK_GUARD(m_mutex);
                m_helpers[cn]->dealloc(ptr);
            }
        }
    }

    bool is_xla_used() { return m_helpers.size() != 0; }

#define FUNC_DEF(func)                \
    void func() {                     \
        MGB_LOCK_GUARD(m_mutex);      \
        for (auto&& kv : m_helpers) { \
            kv.second->func();        \
        }                             \
    }

    FUNC_DEF(log_states)
    FUNC_DEF(reset_states)

#undef FUNC_DEF
};

void* alloc(mgbComputeNode_t device, size_t s) {
    auto cn = *reinterpret_cast<mgb::CompNode*>(device);
    mgb_assert(!is_finalize());
    return XLAMemAllocHelper::inst()->alloc(cn, s);
}

void dealloc(mgbComputeNode_t device, void* addr) {
    auto cn = *reinterpret_cast<mgb::CompNode*>(device);
    XLAMemAllocHelper::inst()->dealloc(cn, addr);
}

void log_xla_mem_states() {
    XLAMemAllocHelper::inst()->log_states();
}

void reset_xla_mem_states() {
    XLAMemAllocHelper::inst()->reset_states();
}

bool is_xla_used() {
    return XLAMemAllocHelper::inst()->is_xla_used();
}

}  // namespace pubapi
}  // namespace mgb
