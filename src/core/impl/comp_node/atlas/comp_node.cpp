#include "./comp_node.h"
#include "megbrain/comp_node_env.h"

#include <memory>
#include <string>

using namespace mgb;

#if MGB_ATLAS

#include "megbrain/common.h"
#include "megbrain/comp_node/alloc.h"
#include "megbrain/utils//timer.h"
#include "megcore_atlas.h"

#include <cctype>
#include <cstdio>

#include <acl/acl.h>
#include <limits>

using AtlasCompNodeImpl = AtlasCompNode::CompNodeImpl;

namespace {
size_t get_min_system_memory(size_t available) {
    // taken from src/core/impl/cuda/comp_node.cpp
    if (available < (1u << 31)) {
        // 225MiB
        return 225 * 1024 * 1024;
    } else {
        // max(300 MiB, 0.05 * available)
        return std::max<size_t>(300 * 1024 * 1024, available / 20);
    }
}

}  // anonymous namespace

/* ======================= AtlasRawAlloctor ======================*/
namespace mgb {
namespace mem_alloc {
class AtlasRawAllocator final : public RawAllocator {
public:
    void* alloc(size_t size) override {
        void* addr;
        aclError acl_error = aclrtMalloc(&addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (acl_error == ACL_SUCCESS) {
            mgb_assert(addr);
            return addr;
        }
        // TODO: add more error msg
        auto msg = mgb_ssprintf_log(
                "aclrtMalloc failed while requesting %zd bytes (%.3fMiB) of memory",
                size, size / (1024.0 * 1024));
        msg.append(AtlasError::get_atlas_extra_info());
        mgb_throw_raw(MemAllocError{msg});
    }

    void free(void* ptr) override {
        aclError acl_error = aclrtFree(ptr);
        if (acl_error == ACL_SUCCESS)
            return;
        auto msg = ssprintf("aclrtFree failed for %p", ptr);
        msg.append(AtlasError::get_atlas_extra_info());
        mgb_throw_raw(MemAllocError{msg});
    }

    void get_mem_info(size_t& free, size_t& tot) override {
        aclError acl_error = aclrtGetMemInfo(ACL_HBM_MEM, &free, &tot);
        if (acl_error == ACL_SUCCESS)
            return;
        auto msg = ssprintf("aclrtGetMemInfo failed");
        msg.append(AtlasError::get_atlas_extra_info());
        mgb_throw_raw(MegBrainError{msg});
    }
};

class AtlasHostAllocator : public RawAllocator {
public:
    void* alloc(size_t size) override {
        void* addr;
        aclError acl_error = aclrtMallocHost(&addr, size);
        if (acl_error == ACL_SUCCESS) {
            mgb_assert(addr);
            return addr;
        }
        auto msg = mgb_ssprintf_log(
                "aclrtMallocHost failed while requesting %zd bytes (%.3fMiB)"
                " of pinned host memory",
                size, size / (1024.0 * 1024));
        msg.append(AtlasError::get_atlas_extra_info());
        mgb_throw_raw(MemAllocError{msg});
    }

    void free(void* ptr) override {
        aclError acl_error = aclrtFreeHost(ptr);
        if (acl_error == ACL_SUCCESS)
            return;
        auto msg = ssprintf("aclrtFreeHost failed for %p", ptr);
        msg.append(AtlasError::get_atlas_extra_info());
        mgb_throw_raw(MemAllocError{msg});
    }

    void get_mem_info(size_t& free, size_t& tot) override {
        free = 0;
        tot = 0;
    }
};

class AtlasDeviceRuntimePolicy : public DeviceRuntimePolicy {
public:
    CompNode::DeviceType device_type() override { return CompNode::DeviceType::ATLAS; }
    void set_device(int device) override { MGB_ATLAS_CHECK(aclrtSetDevice(device)); }
    void device_synchronize(int device) override {
        MGB_ATLAS_CHECK(aclrtSetDevice(device));
        MGB_ATLAS_CHECK(aclrtSynchronizeDevice());
    }
};

/* ===================== DevMemAlloc  ===================== */
std::unique_ptr<DevMemAlloc> DevMemAlloc::make_atlas_alloc() {
    return std::make_unique<FwdDevMemAlloc>(std::make_shared<AtlasRawAllocator>());
}
}  // namespace mem_alloc
}  // namespace mgb

/* ===================== AtlasCompNodeImpl  ===================== */
class AtlasCompNode::CompNodeImpl final : public CompNode::Impl {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

    friend class EventImpl;
    friend class AtlasCompNode;

    struct DeviceInfo;
    struct StaticData;
    static StaticData* sd;
    static Spinlock sd_mtx;
#if !MGB_BUILD_SLIM_SERVING
    std::mutex m_update_mem;
#endif

    //! set to true when m_locator is assigned; set to false if async init
    //! failed
    bool m_initialized = false;
    Locator m_locator, m_locator_logical;
    mem_alloc::StreamMemAlloc* m_mem_alloc;
    DeviceInfo* m_device_info;

    std::unique_ptr<Event> m_sync_event;
    Spinlock m_sync_event_mtx;

    void activate() { m_env.atlas_env().activate(); }

    void init(const Locator& locator, const Locator& locator_logical);
    void fini();

    //! return whether global finalized, and print warning in such case
    static inline bool check_global_finalized();

    //! enable peer copy from dev0 to dev1
    static void enable_peer_access(int dev0, int dev1);

    static void static_free_device(ImplBase* self, void* ptr) {
        static_cast<CompNodeImpl*>(self)->free_device(ptr);
    }

    static void static_free_host(ImplBase* self, void* ptr) {
        static_cast<CompNodeImpl*>(self)->free_host(ptr);
    }

public:
    CompNodeImpl() : Impl(static_free_device, static_free_host) {}

    void* alloc_device(size_t size) override;

    void free_device(void* ptr);

    void* alloc_host(size_t size) override;

    void free_host(void* ptr);

    void copy_to_host(void* host_ptr, const void* device_ptr, size_t size) override {
        if (size == 0) {
            return;
        }
        activate();
#if MGB_USE_ATLAS_ASYNC_API
        if (reinterpret_cast<uintptr_t>(host_ptr) % 64 != 0 ||
            reinterpret_cast<uintptr_t>(device_ptr) % 64 != 0) {
            MGB_ATLAS_CHECK(aclrtSynchronizeStream(m_env.atlas_env().stream));
            MGB_ATLAS_CHECK(aclrtMemcpy(
                    host_ptr, size, device_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST));
        } else {
            MGB_ATLAS_CHECK(aclrtMemcpyAsync(
                    host_ptr, size, device_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST,
                    m_env.atlas_env().stream));
        }
#else
        // aclrtMemcpy is not synchronized, so we need sync mannually before copy
        MGB_ATLAS_CHECK(aclrtSynchronizeStream(m_env.atlas_env().stream));
        MGB_ATLAS_CHECK(aclrtMemcpy(
                host_ptr, size, device_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST));
#endif
    }

    void copy_to_device(void* device_ptr, const void* host_ptr, size_t size) override {
        if (size == 0) {
            return;
        }
        activate();
        // aclrtMemcpy is not synchronized, so we need sync mannually before copy
        MGB_ATLAS_CHECK(aclrtSynchronizeStream(m_env.atlas_env().stream));
        MGB_ATLAS_CHECK(aclrtMemcpy(
                device_ptr, size, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    void peer_copy_to(
            Impl* dest_impl, void* dest, const void* src, size_t size) override;

    size_t get_mem_addr_alignment() override { return m_env.property().mem_alignment; }

    std::unique_ptr<Event> create_event(size_t flags) override;

    void sync() override;

    MemNode mem_node() override;

    size_t get_mem_padding() override { return 32; }

    std::pair<size_t, size_t> get_mem_status_bytes() override {
        m_env.atlas_env().activate();
        size_t tot, free;
        // TODO: ensure the mem attr
        MGB_ATLAS_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &free, &tot));
        free += m_mem_alloc->get_free_memory_dev().tot;
        return {tot, free};
    }

#if !MGB_BUILD_SLIM_SERVING
    size_t get_used_memory() override;

    size_t get_max_used_memory() override;

    size_t get_reserved_memory() override;

    size_t get_max_reserved_memory() override;

    void reset_max_used_memory() override;
    void reset_max_reserved_memory() override;
#endif

    Locator locator() override { return m_locator; }

    Locator locator_logical() override { return m_locator_logical; }

    uint64_t get_uid() override { return m_uid; }

private:
    uint64_t m_uid;
#if !MGB_BUILD_SLIM_SERVING
    std::unordered_map<void*, size_t> ptr2size;
#endif
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(AtlasCompNode::CompNodeImpl);

struct AtlasCompNodeImpl::DeviceInfo {
    int dev_num = -1;
    std::atomic_size_t m_used_mem{0};
    std::atomic_size_t m_max_used_mem{0};
    std::unique_ptr<mem_alloc::DevMemAlloc> mem_alloc;

    bool init_done() const { return mem_alloc.get(); }

    void init(const CompNodeEnv& env);

    void fini() { mem_alloc.reset(); }

    size_t get_mem_reserve_size();
};

struct AtlasCompNodeImpl::StaticData {
    static constexpr int MAX_NR_COMP_NODE = 1024, MAX_NR_DEVICE = 64;

    std::recursive_mutex mtx;

    mem_alloc::DevMemAlloc::PreAllocConfig prealloc_config;

    std::unique_ptr<mem_alloc::SimpleCachingAlloc> host_alloc;
    AtlasCompNode::CompNodeImpl node[MAX_NR_COMP_NODE];
    DeviceInfo dev_info[MAX_NR_DEVICE];
    int nr_node = 0,          //!< number of loaded node[]
            nr_dev_used = 0;  //!< number of used dev_info[]

    StaticData()
            : host_alloc(mem_alloc::SimpleCachingAlloc::make(
                      std::make_unique<mem_alloc::AtlasHostAllocator>())) {
        prealloc_config.max_overhead = 0;
        // TODO: fix the alignment
        prealloc_config.alignment = 64;
        host_alloc->alignment(64);
        host_alloc->addr_alignment(64);
    }

    ~StaticData() {
        for (int i = 0; i < nr_node; ++i)
            node[i].fini();
        for (int i = 0; i < nr_dev_used; ++i)
            dev_info[i].fini();
    }
};
AtlasCompNodeImpl::StaticData* AtlasCompNodeImpl::sd = nullptr;
Spinlock AtlasCompNodeImpl::sd_mtx;

void AtlasCompNodeImpl::DeviceInfo::init(const CompNodeEnv& env) {
    mgb_assert(!mem_alloc);
#if 0
    // forward aclrtMalloc
    mem_alloc = mem_alloc::DevMemAlloc::make_atlas_alloc();
#else
    auto&& atlas_env = env.atlas_env();
    atlas_env.activate();
    dev_num = atlas_env.device;
    auto reserve_size = get_mem_reserve_size();
    mem_alloc = mem_alloc::DevMemAlloc::make(
            dev_num, reserve_size, std::make_shared<mem_alloc::AtlasRawAllocator>(),
            std::make_shared<mem_alloc::AtlasDeviceRuntimePolicy>());
    mem_alloc->prealloc_config(sd->prealloc_config);
    auto align = env.property().mem_alignment;
    mem_alloc->alignment(align);
    // TODO: get addr_alignment from env.
    mem_alloc->addr_alignment(64);
    mgb_log_debug(
            "atlas: card%d: name=`%s' dyn_mem_reserve=%.2fMiB alignment=0x%zx", dev_num,
            "no name", reserve_size / 1024.0 / 1024, align);
#endif
}

size_t AtlasCompNodeImpl::DeviceInfo::get_mem_reserve_size() {
    if (auto setting = MGB_GETENV("MGB_ATLAS_RESERVE_MEMORY")) {
        if (!strncmp(setting, "b:", 2)) {
            return std::stoull(setting + 2);
        }
        size_t tot, free;
        // TODO: ensure the mem attr
        MGB_ATLAS_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, &free, &tot));
        return free - get_min_system_memory(free);
    } else {
        return 0;
    }
}

void AtlasCompNodeImpl::init(const Locator& locator, const Locator& locator_logical) {
    m_locator = locator;
    m_locator_logical = locator_logical;
    m_initialized = true;

#if defined(__linux__) || defined(TARGET_OS_MAC)
    FILE* fp;
    fp = fopen("/dev/urandom", "r");
    mgb_assert(fread(&m_uid, sizeof(m_uid), 1, fp) == 1);
    fclose(fp);
#else
    m_uid = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count();
#endif

    auto on_succ = [this](aclrtStream stream) {
        auto locator = m_locator;
        log_comp_node_created(locator, m_locator_logical);

        MGB_LOCK_GUARD(sd->mtx);
        DeviceInfo* dev_info = nullptr;
        for (int i = 0; i < sd->nr_dev_used; ++i) {
            if (sd->dev_info[i].dev_num == locator.device) {
                dev_info = &sd->dev_info[i];
                break;
            }
        }

        if (!dev_info) {
            dev_info = &sd->dev_info[sd->nr_dev_used];
            dev_info->init(m_env);
            ++sd->nr_dev_used;
        }
        m_device_info = dev_info;
        m_mem_alloc = dev_info->mem_alloc->add_stream(static_cast<void*>(stream));
    };

    auto on_error = [this](std::exception&) {
        MGB_LOCK_GUARD(sd->mtx);
        m_initialized = false;
    };
    m_env.init_atlas(
            locator.device, make_comp_node_from_impl(this), {on_succ, on_error});
}

void AtlasCompNodeImpl::fini() {
    if (!m_initialized)
        return;

    m_sync_event.reset();
    m_env.fini();
    m_mem_alloc = nullptr;
    m_initialized = false;
    m_device_info = nullptr;
}

void* AtlasCompNodeImpl::alloc_device(size_t size) {
    activate();
#if MGB_BUILD_SLIM_SERVING
    return m_mem_alloc->alloc(size);
#else
    void* ptr = m_mem_alloc->alloc(size);
    {
        MGB_LOCK_GUARD(m_update_mem);
        ptr2size[ptr] = size;
        m_device_info->m_used_mem += size;
        if (m_device_info->m_used_mem > m_device_info->m_max_used_mem) {
            m_device_info->m_max_used_mem = m_device_info->m_used_mem.load();
        }
    }
    return ptr;
#endif
}

void AtlasCompNodeImpl::free_device(void* ptr) {
    if (check_global_finalized())
        return;

    activate();
#if !MGB_BUILD_SLIM_SERVING
    {
        MGB_LOCK_GUARD(m_update_mem);
        mgb_assert(ptr2size.find(ptr) != ptr2size.end(), "ptr %p not found!", ptr);
        m_device_info->m_used_mem -= ptr2size.at(ptr);
        ptr2size.erase(ptr);
    }
#endif
    m_mem_alloc->free(ptr);
}

#if !MGB_BUILD_SLIM_SERVING

size_t AtlasCompNodeImpl::get_used_memory() {
    return m_device_info->m_used_mem.load();
}

size_t AtlasCompNodeImpl::get_max_used_memory() {
    return m_device_info->m_max_used_mem.load();
}

void AtlasCompNodeImpl::reset_max_used_memory() {
    m_device_info->m_max_used_mem = 0;
}

size_t AtlasCompNodeImpl::get_reserved_memory() {
    return m_device_info->mem_alloc->get_used_memory();
}

size_t AtlasCompNodeImpl::get_max_reserved_memory() {
    return m_device_info->mem_alloc->get_max_used_memory();
}

void AtlasCompNodeImpl::reset_max_reserved_memory() {
    m_device_info->mem_alloc->reset_max_used_memory();
}
#endif

void* AtlasCompNodeImpl::alloc_host(size_t size) {
    activate();
    return sd->host_alloc->alloc(size);
}

void AtlasCompNodeImpl::free_host(void* ptr) {
    if (check_global_finalized())
        return;
    sd->host_alloc->free(ptr);
}

void AtlasCompNodeImpl::peer_copy_to(
        Impl* dest_impl, void* dest, const void* src, size_t size) {
    if (size == 0) {
        return;
    }
    if (dest_impl->same_type<AtlasCompNodeImpl>()) {
        auto&& dst_env = static_cast<AtlasCompNodeImpl*>(dest_impl)->m_env.atlas_env();
        auto&& src_env = m_env.atlas_env();
        activate();
        if (dst_env.device == src_env.device) {
            // async d2d use SDMA which is faster than sync ctrl cpu d2d
            if (reinterpret_cast<uintptr_t>(dest) % 64 != 0 ||
                reinterpret_cast<uintptr_t>(src) % 64 != 0) {
                // FIXME: fix the sync stream.
                MGB_ATLAS_CHECK(aclrtSynchronizeStream(src_env.stream));
                MGB_ATLAS_CHECK(aclrtMemcpy(
                        dest, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE));
            } else {
                MGB_ATLAS_CHECK(aclrtMemcpyAsync(
                        dest, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE,
                        dst_env.stream));
            }

        } else {
            mgb_throw(
                    MegBrainError,
                    "Atlas does not support peer copy between differents "
                    "device.");
        }
        return;
    }
    mgb_assert(
            dest_impl->env().property().type == DeviceType::CPU,
            "cuda peer_copy_to only implemented for CPU");
    auto copy = [this, dest, src, size]() {
        m_env.atlas_env().activate();

#if MGB_USE_ATLAS_ASYNC_API
        auto stream = m_env.atlas_env().stream;
        if (reinterpret_cast<uintptr_t>(dest) % 64 != 0 ||
            reinterpret_cast<uintptr_t>(src) % 64 != 0) {
            MGB_ATLAS_CHECK(aclrtSynchronizeStream(stream));
            MGB_ATLAS_CHECK(
                    aclrtMemcpy(dest, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST));
        } else {
            MGB_ATLAS_CHECK(aclrtMemcpyAsync(
                    dest, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST,
                    m_env.atlas_env().stream));
            MGB_ATLAS_CHECK(aclrtSynchronizeStream(stream));
        }
#else
        // aclrtMemcpy is not synchronized, so we need sync mannually before copy
        MGB_ATLAS_CHECK(aclrtSynchronizeStream(m_env.atlas_env().stream));
        MGB_ATLAS_CHECK(aclrtMemcpy(dest, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST));
#endif
    };
    dest_impl->env().cpu_env().dispatch(copy);
}

MemNode AtlasCompNodeImpl::mem_node() {
    // m_device_info would be null before async init finishes; so we just return
    // a private pointer related to device number here
    return MemNode{sd->dev_info + m_locator.device};
}

void AtlasCompNodeImpl::sync() {
    activate();

    Event* event;
    {
        MGB_LOCK_GUARD(m_sync_event_mtx);
        if (!m_sync_event)
            m_sync_event = create_event(0);
        event = m_sync_event.get();
    }
    event->record();
    event->host_wait();
}

void AtlasCompNodeImpl::enable_peer_access(int dev0, int dev1) {
    MGB_MARK_USED_VAR(dev0);
    MGB_MARK_USED_VAR(dev1);
    mgb_throw(
            MegBrainError,
            "Atlas does not support peer copy between differents "
            "device.");
}

bool AtlasCompNodeImpl::check_global_finalized() {
    if (!sd) {
        static std::atomic_flag warn_printed = ATOMIC_FLAG_INIT;
        if (!warn_printed.test_and_set()) {
            mgb_log_debug("atlas comp node method called after global finalize");
        }
        return true;
    }
    return false;
}

/* ===================== EventImpl  ===================== */
/**
 * \warning Current we just use cpu timer to do record, later when the api of
 * ddk is ready, we change to normal event.
 */
class AtlasCompNode::EventImpl final : public EventImplHelper {
    AtlasCompNodeImpl* const m_comp_node_impl;
    aclrtEvent m_atlas_event;
    bool m_init_finished = false;
    bool m_used_for_sync = false;

    void do_record() override {
        m_comp_node_impl->activate();
        auto&& env = m_comp_node_impl->m_env.atlas_env();
        MGB_ATLAS_CHECK(aclrtRecordEvent(m_atlas_event, env.stream));
    }

    bool do_finished() override {
        m_comp_node_impl->activate();
        aclrtEventStatus status;
        MGB_ATLAS_CHECK(aclrtQueryEvent(m_atlas_event, &status));
        if (status == ACL_EVENT_STATUS_COMPLETE)
            return true;
        if (status == ACL_EVENT_STATUS_NOT_READY)
            return false;
        mgb_throw(AtlasError, "invalid event status: %d", int(status));
    }

    void host_wait_cv() override {
        m_comp_node_impl->activate();
        MGB_ATLAS_CHECK(aclrtSynchronizeEvent(m_atlas_event));
        if (m_used_for_sync) {
            MGB_ATLAS_CHECK(aclrtResetEvent(
                    m_atlas_event, m_comp_node_impl->m_env.atlas_env().stream));
        }
    }

    double do_elapsed_time_until(EventImplHelper& end) override {
        m_comp_node_impl->activate();
        float ret = 0.0;
        MGB_ATLAS_CHECK(aclrtEventElapsedTime(
                &ret, m_atlas_event, static_cast<EventImpl&>(end).m_atlas_event));
        return static_cast<double>(ret) * 1e-3;
    }

    void do_device_wait_by(Impl* cn_impl) override;

public:
    EventImpl(AtlasCompNodeImpl* comp_node_impl, size_t create_flags)
            : EventImplHelper(comp_node_impl, create_flags),
              m_comp_node_impl{comp_node_impl} {
        m_used_for_sync = !(m_create_flags & NEED_TIMER);
        m_comp_node_impl->activate();
        if (m_used_for_sync) {
            MGB_ATLAS_CHECK(aclrtCreateEvent(&m_atlas_event));
        } else {
            MGB_ATLAS_CHECK(
                    aclrtCreateEventWithFlag(&m_atlas_event, ACL_EVENT_TIME_LINE));
        }
        m_init_finished = true;
    }
    ~EventImpl() {
        if (m_init_finished) {
            MGB_TRY { MGB_ATLAS_CHECK(aclrtDestroyEvent(m_atlas_event)); }
            MGB_CATCH(MegBrainError & exc, {
                mgb_log_error("failed to destroy cuda event: %s", exc.what());
            })
        }
    }
};

std::unique_ptr<CompNode::Event> AtlasCompNodeImpl::create_event(size_t flags) {
    return std::make_unique<EventImpl>(this, flags);
}

void AtlasCompNode::EventImpl::do_device_wait_by(Impl* cn_impl) {
    if (cn_impl->dyn_typeinfo() == AtlasCompNodeImpl::typeinfo()) {
        auto imp = static_cast<AtlasCompNodeImpl*>(cn_impl);
        imp->m_env.atlas_env().activate();
        auto stream = imp->m_env.atlas_env().stream;
        MGB_ATLAS_CHECK(aclrtStreamWaitEvent(stream, m_atlas_event));
        if (m_used_for_sync) {
            MGB_ATLAS_CHECK(aclrtResetEvent(
                    m_atlas_event, m_comp_node_impl->m_env.atlas_env().stream));
        }
        return;
    }
    if (cn_impl->env().property().type == DeviceType::CPU) {
        auto waiter = [this]() {
            m_comp_node_impl->m_env.atlas_env().activate();
            MGB_ATLAS_CHECK(aclrtSynchronizeEvent(m_atlas_event));
            if (m_used_for_sync) {
                MGB_ATLAS_CHECK(aclrtResetEvent(
                        m_atlas_event, m_comp_node_impl->m_env.atlas_env().stream));
            }
        };
        cn_impl->add_callback(std::move(waiter));
        return;
    }
    mgb_throw(MegBrainError, "unimplemented event device_wait_by config");
}

/* ===================== AtlasCompNode static methods ===================== */

bool AtlasCompNode::available() {
    return true;
}

void AtlasCompNode::finalize() {
    if (AtlasCompNodeImpl::sd) {
        sync_all();

        auto ptr = AtlasCompNodeImpl::sd;
        AtlasCompNodeImpl::sd = nullptr;
        ptr->~StaticData();
    }
}

CompNode::Impl* AtlasCompNode::load_atlas(
        const Locator& locator, const Locator& locator_logical) {
    auto&& sdptr = AtlasCompNodeImpl::sd;
    {
        MGB_LOCK_GUARD(AtlasCompNodeImpl::sd_mtx);
        if (!sdptr) {
            // use static storage so object can be safely accessed even after
            // global finalize
            using T = AtlasCompNodeImpl::StaticData;
            static std::aligned_storage_t<sizeof(T), alignof(T)> storage;
            sdptr = new (&storage) T;
        }
    }
    auto&& sd = *sdptr;
    MGB_LOCK_GUARD(sd.mtx);

    CompNodeImpl* available_node = nullptr;
    for (int i = 0; i < sd.nr_node; ++i) {
        auto&& cur = sd.node[i];
        if (cur.m_initialized) {
            if (cur.m_locator == locator && cur.m_locator_logical == locator_logical) {
                return &cur;
            }
        } else {
            available_node = &cur;
        }
    }

    if (!available_node) {
        mgb_assert(sd.nr_node < sd.MAX_NR_COMP_NODE, "too many CompNode allocated");
        mgb_assert(locator.device < sd.MAX_NR_COMP_NODE, "device number too large");
        available_node = &sd.node[sd.nr_node++];
    }

    mgb_assert(!available_node->m_initialized);
    available_node->init(locator, locator_logical);
    log_comp_node_created(locator, locator_logical);

    return available_node;
}

void AtlasCompNode::sync_all() {
    auto sd = AtlasCompNodeImpl::sd;
    if (!sd)
        return;

    for (int i = 0;; ++i) {
        // ensure async init finished
        CompNodeEnv* env;
        {
            MGB_LOCK_GUARD(sd->mtx);
            if (i >= sd->nr_node) {
                break;
            }
            env = &sd->node[i].env();
        }
        env->atlas_env();
    }

    MGB_LOCK_GUARD(sd->mtx);
    MGB_ATLAS_CHECK(aclrtSynchronizeDevice());
}

void AtlasCompNode::foreach (thin_function<void(CompNode)> callback) {
    auto sd = AtlasCompNodeImpl::sd;
    if (!sd)
        return;

    for (int i = 0;; ++i) {
        CompNode cur;
        {
            MGB_LOCK_GUARD(sd->mtx);
            if (i >= sd->nr_node)
                return;
            cur = make_comp_node_from_impl(&sd->node[i]);
        }
        callback(cur);
    }
}

size_t AtlasCompNode::get_device_count() {
    static uint32_t cnt = 0;
    static Spinlock mtx;
    MGB_LOCK_GUARD(mtx);
    if (cnt == 0) {
        uint32_t dev_cnt = 0;
        auto ret = aclrtGetDeviceCount(&dev_cnt);
        if (ret != ACL_ERROR_NONE) {
            mgb_log_error(
                    "aclrtGetDeviceCountfaild: %s (err %d)",
                    ::megcore::atlas::get_error_str(ret), static_cast<int>(ret));
            cnt = 0;
        }
        cnt = dev_cnt;
    }
    return cnt;
}

#else

bool AtlasCompNode::available() {
    return false;
}
void AtlasCompNode::foreach (thin_function<void(CompNode)>) {}
void AtlasCompNode::finalize() {}
size_t AtlasCompNode::get_device_count() {
    return 0;
}
AtlasCompNode::Impl* AtlasCompNode::load_atlas(const Locator&, const Locator&) {
    mgb_throw(MegBrainError, "atlas disabled at compile time");
}
void AtlasCompNode::sync_all() {}

#endif  // MGB_ATLAS

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
