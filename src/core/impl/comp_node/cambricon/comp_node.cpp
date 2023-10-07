#include "./comp_node.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/utils/thread.h"

#include <string>

using namespace mgb;

#if MGB_CAMBRICON

#include "megbrain/comp_node/alloc.h"

#include <cctype>
#include <cstdio>

#include <thread>

#include <cn_api.h>
#include <cndev.h>
#include <cnrt.h>

using CambriconCompNodeImpl = CambriconCompNode::CompNodeImpl;

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

/* ======================= CambriconRawAlloctor ======================*/
namespace mgb {
namespace mem_alloc {
class CambriconRawAlloctor final : public RawAllocator {
public:
    void* alloc(size_t size) override {
        void* addr;
        cnrtRet_t ret = cnrtMalloc(&addr, size);
        if (ret == CNRT_RET_SUCCESS) {
            mgb_assert(addr);
            return addr;
        }
        auto msg = mgb_ssprintf_log(
                "cnrtMalloc failed while requesting %zd bytes (%.3fMiB) of "
                "memory; error: %s",
                size, size / (1024.0 * 1024), cnrtGetErrorStr(ret));
        msg.append(CnrtError::get_cnrt_extra_info());
        mgb_throw_raw(MemAllocError{msg});
    }

    void free(void* ptr) override {
        cnrtRet_t ret = cnrtFree(ptr);
        if (ret == CNRT_RET_SUCCESS)
            return;
        auto msg = ssprintf("cnrtFree failed for %p: %s", ptr, cnrtGetErrorStr(ret));
        msg.append(CnrtError::get_cnrt_extra_info());
        mgb_throw_raw(MemAllocError{msg});
    }

    void get_mem_info(size_t& free, size_t& tot) override;
};

class CambriconDeviceRuntimePolicy : public DeviceRuntimePolicy {
public:
    CompNode::DeviceType device_type() override {
        return CompNode::DeviceType::CAMBRICON;
    }
    void set_device(int device) override { MGB_CNRT_CHECK(cnrtSetDevice(device)); }
    void device_synchronize(int device) override {
        MGB_CNRT_CHECK(cnrtSetDevice(device));
        MGB_CNRT_CHECK(cnrtSyncDevice());
    }
};

/* ====================== DevMemAlloc ================================*/
std::unique_ptr<DevMemAlloc> DevMemAlloc::make_cambricon_alloc() {
    return std::make_unique<FwdDevMemAlloc>(std::make_shared<CambriconRawAlloctor>());
}
}  // namespace mem_alloc
}  // namespace mgb

/* ====================== CambriconCompNodeImpl ======================*/
class CambriconCompNode::CompNodeImpl final : public CompNode::Impl {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

    friend class EventImpl;
    friend class CambriconCompNode;
    friend class mgb::mem_alloc::CambriconRawAlloctor;

    struct DeviceInfo;
    struct StaticData;
    static StaticData* sd;
    static Spinlock sd_mtx;
#if !MGB_BUILD_SLIM_SERVING
    std::mutex m_update_mem;
#endif
    //! set to true when m_locator is assigned; set to false if init
    //! failed
    bool m_initialized = false;
    Locator m_locator, m_locator_logical;
    mem_alloc::StreamMemAlloc* m_mem_alloc;
    DeviceInfo* m_device_info;
    int m_dev;

    void activate() { m_env.cnrt_env().activate(); }

    void init(const Locator& locator, const Locator& locator_logical);
    void fini();

    static inline bool check_global_finalized();

    //! enable peer copy from dev0 to dev1
    static bool enable_peer_access(int dev0, int dev1);

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

    void* alloc_host(size_t size) override {
        activate();
        void* ptr;
        MGB_CNRT_CHECK(cnrtHostMalloc(&ptr, size));
        return ptr;
    }

    void free_host(void* ptr) {
        if (!check_global_finalized()) {
            activate();
        }
        MGB_CNRT_CHECK(cnrtSetDevice(m_dev));
        MGB_CNRT_CHECK(cnrtFreeHost(ptr));
    }

    void copy_to_host(void* host_ptr, const void* device_ptr, size_t size) override {
        activate();
        MGB_CNRT_CHECK(cnrtMemcpyAsync(
                host_ptr, const_cast<void*>(device_ptr), size, m_env.cnrt_env().queue,
                CNRT_MEM_TRANS_DIR_DEV2HOST));
    }

    void copy_to_device(void* device_ptr, const void* host_ptr, size_t size) override {
        activate();
        MGB_CNRT_CHECK(cnrtMemcpyAsync(
                device_ptr, const_cast<void*>(host_ptr), size, m_env.cnrt_env().queue,
                CNRT_MEM_TRANS_DIR_HOST2DEV));
    }

    void peer_copy_to(
            Impl* dest_impl, void* dest, const void* src, size_t size) override;

    size_t get_mem_addr_alignment() override { return m_env.property().mem_alignment; }

    std::unique_ptr<Event> create_event(size_t flags) override;

    void sync() override;

    MemNode mem_node() override;

    std::pair<size_t, size_t> get_mem_status_bytes() override {
        m_env.cnrt_env().activate();
        cndevMemoryInfo_t mem_info;
#if CNRT_MAJOR_VERSION >= 5
        mem_info.version = CNDEV_VERSION_5;
#endif
        MGB_CNDEV_CHECK(cndevGetMemoryUsage(&mem_info, m_env.cnrt_env().device));
        size_t tot, used, free;
        constexpr size_t mb2size = 1024 * 1024;
#if CNRT_MAJOR_VERSION >= 5
        tot = static_cast<size_t>(mem_info.physicalMemoryTotal) * mb2size;
        used = static_cast<size_t>(mem_info.physicalMemoryUsed) * mb2size;
#else
        tot = static_cast<size_t>(mem_info.PhysicalMemoryTotal) * mb2size;
        used = static_cast<size_t>(mem_info.PhysicalMemoryUsed) * mb2size;
#endif
        free = tot - used + m_mem_alloc->get_free_memory_dev().tot;
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
#if !MGB_BUILD_SLIM_SERVING
    std::unordered_map<void*, size_t> ptr2size;
#endif

    uint64_t get_uid() override { return m_uid; }

private:
    uint64_t m_uid;
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(CambriconCompNode::CompNodeImpl);

struct CambriconCompNodeImpl::DeviceInfo {
    int dev_num = -1;
    int dev;
    std::atomic_size_t m_used_mem{0};
    std::atomic_size_t m_max_used_mem{0};
    std::unique_ptr<mem_alloc::DevMemAlloc> mem_alloc;

    bool init_done() const { return mem_alloc.get(); }

    void init(const CompNodeEnv& env);

    // unlike cuda, we have to set device first, then release device memory
    void fini() {
        cnrtSetDevice(dev);
        return mem_alloc.reset();
    }

    size_t get_mem_reserve_size();
};

struct CambriconCompNodeImpl::StaticData {
    static constexpr int MAX_NR_COMP_NODE = 4096, MAX_NR_DEVICE = 64;

    std::recursive_mutex mtx;

    mem_alloc::DevMemAlloc::PreAllocConfig prealloc_config;

    CambriconCompNode::CompNodeImpl node[MAX_NR_COMP_NODE];
    DeviceInfo dev_info[MAX_NR_DEVICE];
    int nr_node = 0, nr_dev_used = 0;

    StaticData() {
        prealloc_config.max_overhead = 0;
        prealloc_config.alignment = 1;
    }

    ~StaticData() {
        for (int i = 0; i < nr_node; ++i)
            node[i].fini();
        for (int i = 0; i < nr_dev_used; ++i)
            dev_info[i].fini();
    }
};
CambriconCompNodeImpl::StaticData* CambriconCompNodeImpl::sd = nullptr;
Spinlock CambriconCompNodeImpl::sd_mtx;

void CambriconCompNodeImpl::init(
        const Locator& locator, const Locator& locator_logical) {
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

    auto on_succ = [this](cnrtQueue_t queue) {
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
        m_mem_alloc = dev_info->mem_alloc->add_stream(static_cast<void*>(queue));
        m_dev = m_device_info->dev;
    };

    auto on_error = [this](std::exception&) {
        MGB_LOCK_GUARD(sd->mtx);
        m_initialized = false;
    };

    m_env.init_cnrt(
            locator.device, make_comp_node_from_impl(this), {on_succ, on_error});
}

void CambriconCompNodeImpl::fini() {
    if (!m_initialized)
        return;

    m_env.fini();
    m_mem_alloc = nullptr;
    m_device_info = nullptr;
    m_initialized = false;
}

void* CambriconCompNodeImpl::alloc_device(size_t size) {
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

void CambriconCompNodeImpl::free_device(void* ptr) {
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
size_t CambriconCompNodeImpl::get_used_memory() {
    return m_device_info->m_used_mem.load();
}

size_t CambriconCompNodeImpl::get_max_used_memory() {
    return m_device_info->m_max_used_mem.load();
}

void CambriconCompNodeImpl::reset_max_used_memory() {
    m_device_info->m_max_used_mem = 0;
}

size_t CambriconCompNodeImpl::get_reserved_memory() {
    return m_device_info->mem_alloc->get_used_memory();
}

size_t CambriconCompNodeImpl::get_max_reserved_memory() {
    return m_device_info->mem_alloc->get_max_used_memory();
}

void CambriconCompNodeImpl::reset_max_reserved_memory() {
    m_device_info->mem_alloc->reset_max_used_memory();
}
#endif

void CambriconCompNodeImpl::peer_copy_to(
        Impl* dest_impl, void* dest, const void* src, size_t size) {
    if (dest_impl->same_type<CambriconCompNodeImpl>()) {
        auto&& dst_env =
                static_cast<CambriconCompNodeImpl*>(dest_impl)->m_env.cnrt_env();
        auto&& src_env = m_env.cnrt_env();
        activate();
        if (dst_env.device == src_env.device) {
            // remark: transfering data from device to device does not
            // support async
            MGB_CNRT_CHECK(cnrtMemcpyAsync(
                    dest, const_cast<void*>(src), size, dst_env.queue,
                    CNRT_MEM_TRANS_DIR_DEV2DEV));
        } else {
            mgb_throw_if(
                    !enable_peer_access(src_env.device, dst_env.device) ||
                            !enable_peer_access(dst_env.device, src_env.device),
                    CnrtError,
                    "directly memory access is not available for "
                    "src=%d,dst=%d",
                    src_env.device, dst_env.device);
            sync();
            MGB_CNRT_CHECK(cnrtMemcpyPeerAsync(
                    dest, dst_env.device, const_cast<void*>(src), src_env.device, size,
                    dst_env.queue));
        }
        return;
    }
    mgb_assert(
            dest_impl->env().property().type == DeviceType::CPU,
            "cnrt peer_copy_to only implemented for CPU");
    auto copy = [this, dest, src, size]() {
        m_env.cnrt_env().activate();
        auto queue = m_env.cnrt_env().queue;
        MGB_CNRT_CHECK(cnrtMemcpyAsync(
                dest, const_cast<void*>(src), size, queue,
                CNRT_MEM_TRANS_DIR_DEV2HOST));
        MGB_CNRT_CHECK(cnrtQueueSync(queue));
    };
    dest_impl->env().cpu_env().dispatch(copy);
}

MemNode CambriconCompNodeImpl::mem_node() {
    return MemNode{sd->dev_info + m_locator.device};
}

void CambriconCompNodeImpl::sync() {
    activate();

    // remark: CNRT does not provide interface like cudaEventQuery to test
    // whether an event is finished. so we just call the cnrtQueueSync
    MGB_CNRT_CHECK(cnrtQueueSync(m_env.cnrt_env().queue));
}

bool CambriconCompNodeImpl::enable_peer_access(int dev0, int dev1) {
    static bool queried_enabled[StaticData::MAX_NR_DEVICE][StaticData::MAX_NR_DEVICE];
    if (queried_enabled[dev0][dev1])
        return queried_enabled[dev0][dev1];

    static std::mutex global_lock;
    MGB_LOCK_GUARD(global_lock);
    unsigned int can = 0;
    MGB_CNRT_CHECK(cnrtGetPeerAccessibility(&can, dev0, dev1));
    if (can)
        mgb_log("device(%d) can directly access memories on device(%d)", dev0, dev1);
    queried_enabled[dev0][dev1] = can;
    return can;
}

/* ================== CambriconCompNodeImpl::DeviceInfo ===============*/

void CambriconCompNodeImpl::DeviceInfo::init(const CompNodeEnv& env) {
    mgb_assert(!mem_alloc);
    auto&& cnenv = env.cnrt_env();
    cnenv.activate();
    dev_num = cnenv.device;
    // MGB_CNRT_CHECK(cnrtGetDeviceHandle(&dev, dev_num));
    dev = dev_num;
    // remark: Because free_device will be called after global finalize, so the
    // implementation of mem_alloc should handle the deallocation of memories
    // allocated by the mem_alloc. As a result, we should use the DevMemAlloc
    // instead of FwdDevMemAlloc.
#if 0
    // forward cnrtMalloc
    mem_alloc = mem_alloc::DevMemAlloc::make_cambricon_alloc();
#else
    auto reserve_size = get_mem_reserve_size();
    mem_alloc = mem_alloc::DevMemAlloc::make(
            dev_num, reserve_size, std::make_shared<mem_alloc::CambriconRawAlloctor>(),
            std::make_shared<mem_alloc::CambriconDeviceRuntimePolicy>());
    mem_alloc->prealloc_config(sd->prealloc_config);
    auto align = env.property().mem_alignment;
    mem_alloc->alignment(align);
    cnrtDeviceProp_t device_info;
    MGB_CNRT_CHECK(cnrtGetDeviceProperties(&device_info, dev_num));
    mgb_log("cambricon: card%d: name=`%s' dyn_mem_reserve=%.2fMiB "
            "alignment=0x%zx",
            dev_num, device_info.name, reserve_size / 1024.0 / 1024, align);
#endif
}

size_t CambriconCompNodeImpl::DeviceInfo::get_mem_reserve_size() {
    if (auto setting = MGB_GETENV("MGB_CAMBRICON_RESERVE_MEMORY")) {
        if (!strncmp(setting, "b:", 2)) {
            return std::stoull(setting + 2);
        }
        size_t tot, free;
        cndevMemoryInfo_t mem_info;
#if CNRT_MAJOR_VERSION >= 5
        mem_info.version = CNDEV_VERSION_5;
#endif
        MGB_CNDEV_CHECK(cndevGetMemoryUsage(&mem_info, dev_num));
        constexpr size_t mb2size = 1024 * 1024;
#if CNRT_MAJOR_VERSION >= 5
        tot = static_cast<size_t>(mem_info.physicalMemoryTotal) * mb2size;
        size_t used = static_cast<size_t>(mem_info.physicalMemoryUsed) * mb2size;
#else
        tot = static_cast<size_t>(mem_info.PhysicalMemoryTotal) * mb2size;
        size_t used = static_cast<size_t>(mem_info.PhysicalMemoryUsed) * mb2size;
#endif
        free = tot - used;
        return free - get_min_system_memory(free);
    } else {
        return 0;
    }
}

bool CambriconCompNodeImpl::check_global_finalized() {
    if (!sd) {
        static std::atomic_flag warn_printed = ATOMIC_FLAG_INIT;
        if (!warn_printed.test_and_set()) {
            mgb_log_warn("cambricon comp node method called after global finalize");
        }
        return true;
    }
    return false;
}

/* ================== CambriconCompNodeImpl::EventImpl ================*/

class CambriconCompNode::EventImpl final : public EventImplHelper {
    bool m_init_finished = false;
    CambriconCompNodeImpl* const m_comp_node_impl;
    cnrtNotifier_t m_cnrt_notifier;

    void do_record() override {
        m_comp_node_impl->activate();
        auto&& env = m_comp_node_impl->m_env.cnrt_env();
        MGB_CNRT_CHECK(cnrtPlaceNotifier(m_cnrt_notifier, env.queue));
    }

    bool do_finished() override {
        m_comp_node_impl->activate();
        cnrtRet_t err = cnrtQueryNotifier(m_cnrt_notifier);
        if (err == cnrtSuccess)
            return true;
        if (err == cnrtErrorNotReady)
            return false;
        mgb_throw(
                CnrtError, "failed to query event: %d: %s", int(err),
                cnrtGetErrorStr(err));
    }

    void host_wait_cv() override { MGB_CNRT_CHECK(cnrtWaitNotifier(m_cnrt_notifier)); }

    double do_elapsed_time_until(EventImplHelper& end) override {
        m_comp_node_impl->activate();
        float ret = 0.0;
        MGB_CNRT_CHECK(cnrtNotifierElapsedTime(
                m_cnrt_notifier, static_cast<EventImpl&>(end).m_cnrt_notifier, &ret));
        return static_cast<double>(ret) * 1e-3;
    }

    void do_device_wait_by(Impl* cn_impl) override;

public:
    EventImpl(CambriconCompNodeImpl* comp_node_impl, size_t create_flags)
            : EventImplHelper(comp_node_impl, create_flags),
              m_comp_node_impl{comp_node_impl} {
        m_comp_node_impl->activate();
        cnrtNotifierFlags_t flags = CNRT_NOTIFIER_DISABLE_TIMING_ALL;
        if (create_flags & NEED_TIMER) {
            flags = CNRT_NOTIFIER_DEFAULT;
        }
        MGB_CNRT_CHECK(cnrtNotifierCreateWithFlags(&m_cnrt_notifier, flags));
        m_init_finished = true;
    }

    ~EventImpl() {
        if (m_init_finished) {
            MGB_TRY { MGB_CNRT_CHECK(cnrtNotifierDestroy(m_cnrt_notifier)); }
            MGB_CATCH(MegBrainError & exc, {
                mgb_log_error("failed to destroy cuda event: %s", exc.what());
            })
        }
    }
};

std::unique_ptr<CompNode::Event> CambriconCompNodeImpl::create_event(size_t flags) {
    return std::make_unique<EventImpl>(this, flags);
}

void CambriconCompNode::EventImpl::do_device_wait_by(Impl* cn_impl) {
    if (cn_impl->dyn_typeinfo() == CambriconCompNodeImpl::typeinfo()) {
        auto imp = static_cast<CambriconCompNodeImpl*>(cn_impl);
        auto queue = imp->m_env.cnrt_env().queue;
        imp->activate();
        MGB_CNRT_CHECK(cnrtQueueWaitNotifier(m_cnrt_notifier, queue, 0));
        return;
    }
    if (cn_impl->env().property().type == DeviceType::CPU) {
        auto waiter = [this]() { MGB_CNRT_CHECK(cnrtWaitNotifier(m_cnrt_notifier)); };
        cn_impl->add_callback(std::move(waiter));
        return;
    }
    mgb_throw(MegBrainError, "unimplemented event device_wait_by config");
}

/* ================== CambriconCompNode static methods ================*/

namespace {

#ifndef __unix__
template <typename Func, typename Val>
CNresult call_cndrv_forksafe(Func func, Val* val, size_t len) {
    cnInit(0);
    return func();
}
#else
struct RAIICloseFD : NonCopyableObj {
    int m_fd = -1;

    RAIICloseFD(int fd) : m_fd(fd) {}
    ~RAIICloseFD() { close(); }
    void close() {
        if (m_fd != -1) {
            ::close(m_fd);
            m_fd = -1;
        }
    }
};
// an implementation that does not call cnInit
template <typename Func, typename Val>
CNresult call_cndrv_forksafe(Func func, Val* val, size_t len) {
    int count = 0;
    // use cnDeviceGetCount to detect cambricon initialization to avoid abnormal
    // behavior
    auto err = cnDeviceGetCount(&count);
    if (err != CN_ERROR_NOT_INITIALIZED)
        return func();
    // cnInit not called, call it in child process
    int fd[2];
    mgb_assert(pipe(fd) == 0, "pipe() failed");
    int fdr = fd[0], fdw = fd[1];
    RAIICloseFD fdr_guard(fdr);
    RAIICloseFD fdw_guard(fdw);
    auto cpid = fork();
    mgb_assert(cpid != -1, "fork() failed");
    if (cpid == 0) {
        fdr_guard.close();
        do {
            err = cnInit(0);
            if (err != CN_SUCCESS)
                break;
            err = func();
        } while (0);
        auto sz = write(fdw, &err, sizeof(err));
        if (sz == sizeof(err) && err == CN_SUCCESS) {
            sz = write(fdw, val, sizeof(*val) * len);
        }
        fdw_guard.close();
        std::quick_exit(0);
    }
    fdw_guard.close();
    auto sz = read(fdr, &err, sizeof(err));
    mgb_assert(sz == sizeof(err), "failed to read error code from child");
    if (err == CN_SUCCESS) {
        sz = read(fdr, val, sizeof(*val) * len);
        mgb_assert(
                static_cast<size_t>(sz) == sizeof(*val) * len,
                "failed to read value from child");
        return err;
    }
    // try again, maybe another thread called cnInit while we fork
    auto err2 = func();
    if (err2 == CN_SUCCESS)
        return err2;
    if (err2 == CN_ERROR_NOT_INITIALIZED)
        return err;
    return err2;
}
#endif

const char* cn_get_error_string(CNresult err) {
    const char* ret = nullptr;
    cnGetErrorString(err, &ret);
    if (!ret) {
        ret = "invalid_stub_call";
    }
    return ret;
}

#define MGB_CALL_CNDRV_FORKSAFE_NOASSERT(func, ptr, len, ...) \
    call_cndrv_forksafe([&]() { return func(ptr, ##__VA_ARGS__); }, ptr, len)

#define MGB_CALL_CNDRV_FORKSAFE(func, ptr, len, ...)                                \
    {                                                                               \
        auto err = MGB_CALL_CNDRV_FORKSAFE_NOASSERT(func, ptr, len, ##__VA_ARGS__); \
        if (err != CNDEV_SUCCESS) {                                                 \
            auto err_s = cn_get_error_string(err);                                  \
            mgb_log_error(#func " failed: %s (err %d)", err_s, int(err));           \
        }                                                                           \
    }
}  // namespace

bool CambriconCompNode::available() {
    static int result = -1;
    static Spinlock mtx;
    MGB_LOCK_GUARD(mtx);
    if (result == -1) {
        int count = 0;
        auto err = MGB_CALL_CNDRV_FORKSAFE_NOASSERT(cnDeviceGetCount, &count, 1);
        result = err == CN_SUCCESS && count >= 1;
        if (!result) {
            mgb_log_warn(
                    "cambricon unavailable: %d(%s) dev_num=%u", static_cast<int>(err),
                    cn_get_error_string(err), count);
        }
    }
    return result;
}

void CambriconCompNode::finalize() {
    if (CambriconCompNodeImpl::sd) {
        sync_all();

        auto ptr = CambriconCompNodeImpl::sd;
        CambriconCompNodeImpl::sd = nullptr;
        ptr->~StaticData();
    }
}

CompNode::Impl* CambriconCompNode::load_cambricon(
        const Locator& locator, const Locator& locator_logical) {
    int nr_devs = get_device_count();
    mgb_assert(
            locator.device >= 0 && locator.device < nr_devs,
            "request device%d out of range [0, %d)", locator.device, nr_devs);

    auto&& sdptr = CambriconCompNodeImpl::sd;
    {
        MGB_LOCK_GUARD(CambriconCompNodeImpl::sd_mtx);
        if (!sdptr) {
            using T = CambriconCompNodeImpl::StaticData;
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

    return available_node;
}

void CambriconCompNode::try_coalesce_all_free_memory() {
    auto sd = CambriconCompNodeImpl::sd;
    if (!sd)
        return;

    size_t size = 0;
    for (int i = 0; i < sd->nr_dev_used; ++i) {
        size += sd->dev_info[i].mem_alloc->gather_stream_free_blk_and_release_full();
    }
    if (size) {
        mgb_log_debug("%zu bytes freed by try_coalesce_all_free_memory()", size);
    }
}

void CambriconCompNode::sync_all() {
    auto sd = CambriconCompNodeImpl::sd;
    if (!sd)
        return;
    for (int i = 0;; ++i) {
        CompNodeEnv* env;
        {
            MGB_LOCK_GUARD(sd->mtx);
            if (i >= sd->nr_node) {
                break;
            }
            env = &sd->node[i].env();
        }
        env->cnrt_env();
    }

    MGB_LOCK_GUARD(sd->mtx);
    for (int i = 0; i < sd->nr_dev_used; ++i) {
        MGB_CNRT_CHECK(cnrtSetDevice(sd->dev_info[i].dev_num));
        MGB_CNRT_CHECK(cnrtSyncDevice());
    }
}

void CambriconCompNode::foreach (thin_function<void(CompNode)> callback) {
    auto sd = CambriconCompNodeImpl::sd;
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

size_t CambriconCompNode::get_device_count(bool warn) {
    static int cnt = -1;
    static Spinlock mtx;
    MGB_LOCK_GUARD(mtx);
    if (cnt == -1) {
        auto err = MGB_CALL_CNDRV_FORKSAFE_NOASSERT(cnDeviceGetCount, &cnt, 1);
        auto err_s = cn_get_error_string(err);
        if (err != CN_SUCCESS) {
            if (warn && (std::string(err_s) != "invalid_stub_call"))
                mgb_log_error("cuDeviceGetCount failed: %s (err %d)", err_s, int(err));
            cnt = 0;
        }
        mgb_assert(cnt >= 0);
    }
    return cnt;
}

void mgb::mem_alloc::CambriconRawAlloctor::get_mem_info(size_t& free, size_t& tot) {
    auto sd = CambriconCompNodeImpl::sd;
    int device = -1;
    {
        int dev;
        MGB_CNRT_CHECK(cnrtGetDevice(&dev));
        for (int i = 0; i < sd->nr_dev_used; ++i) {
            if (sd->dev_info[i].dev == dev) {
                device = sd->dev_info[i].dev_num;
                break;
            }
        }
    }
    mgb_assert(device >= 0, "current device has not been initialized in static data");
    cndevMemoryInfo_t mem_info;
#if CNRT_MAJOR_VERSION >= 5
    mem_info.version = CNDEV_VERSION_5;
#endif
    auto ret = cndevGetMemoryUsage(&mem_info, device);
    if (ret == CNDEV_SUCCESS) {
        constexpr size_t mb2size = 1024 * 1024;
#if CNRT_MAJOR_VERSION >= 5
        tot = static_cast<size_t>(mem_info.physicalMemoryTotal) * mb2size;
        size_t used = static_cast<size_t>(mem_info.physicalMemoryUsed) * mb2size;
#else
        tot = static_cast<size_t>(mem_info.PhysicalMemoryTotal) * mb2size;
        size_t used = static_cast<size_t>(mem_info.PhysicalMemoryUsed) * mb2size;
#endif
        free = tot - used;
        return;
    }
    auto msg = ssprintf("cndevGetMemoryUsage faild %s", cndevGetErrorString(ret));
    mgb_throw_raw(MemAllocError{msg});
}

#else

bool CambriconCompNode::available() {
    return false;
}
void CambriconCompNode::try_coalesce_all_free_memory() {}
void CambriconCompNode::foreach (thin_function<void(CompNode)>) {}
void CambriconCompNode::finalize() {}
size_t CambriconCompNode::get_device_count(bool warn) {
    return 0;
}
CambriconCompNode::Impl* CambriconCompNode::load_cambricon(
        const Locator&, const Locator&) {
    mgb_throw(MegBrainError, "cambricon disabled at compile time");
}
void CambriconCompNode::sync_all() {}

#undef err

#endif  // MGB_CAMBRICON

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
