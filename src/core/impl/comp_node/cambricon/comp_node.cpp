/**
 * \file src/core/impl/comp_node/cambricon/comp_node.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

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
        auto msg = ssprintf("cnrtFree failed for %p: %s", ptr,
                            cnrtGetErrorStr(ret));
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
    void set_device(int device) override {
        cnrtDev_t dev;
        MGB_CNRT_CHECK(cnrtGetDeviceHandle(&dev, device));
        MGB_CNRT_CHECK(cnrtSetCurrentDevice(dev));
    }
    void device_synchronize(int device) override {
        cnrtDev_t dev;
        MGB_CNRT_CHECK(cnrtGetDeviceHandle(&dev, device));
        MGB_CNRT_CHECK(cnrtSetCurrentDevice(dev));
        MGB_CNRT_CHECK(cnrtSyncDevice());
    }
};

/* ====================== DevMemAlloc ================================*/
std::unique_ptr<DevMemAlloc> DevMemAlloc::make_cambricon_alloc() {
    return std::make_unique<FwdDevMemAlloc>(
            std::make_shared<CambriconRawAlloctor>());
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

    //! set to true when m_locator is assigned; set to false if init
    //! failed
    bool m_initialized = false;
    Locator m_locator, m_locator_logical;
    mem_alloc::StreamMemAlloc* m_mem_alloc;
    DeviceInfo* m_device_info;
    cnrtDev_t m_dev;

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

    void* alloc_device(size_t size) override {
        activate();
        return m_mem_alloc->alloc(size);
    }

    void free_device(void* ptr);

    void* alloc_host(size_t size) override {
        activate();
        void* ptr;
        MGB_CNRT_CHECK(cnrtMallocHost(&ptr, size, CNRT_MEMTYPE_DEFAULT));
        return ptr;
    }

    void free_host(void* ptr) {
        if (!check_global_finalized()) {
            activate();
        }
        MGB_CNRT_CHECK(cnrtSetCurrentDevice(m_dev));
        MGB_CNRT_CHECK(cnrtFreeHost(ptr));
    }

    void copy_to_host(void* host_ptr, const void* device_ptr,
                      size_t size) override {
        activate();
        MGB_CNRT_CHECK(cnrtMemcpyAsync(host_ptr, const_cast<void*>(device_ptr),
                                       size, m_env.cnrt_env().queue,
                                       CNRT_MEM_TRANS_DIR_DEV2HOST));
    }

    void copy_to_device(void* device_ptr, const void* host_ptr,
                        size_t size) override {
        activate();
        MGB_CNRT_CHECK(cnrtMemcpyAsync(device_ptr, const_cast<void*>(host_ptr),
                                       size, m_env.cnrt_env().queue,
                                       CNRT_MEM_TRANS_DIR_HOST2DEV));
    }

    void peer_copy_to(Impl* dest_impl, void* dest, const void* src,
                      size_t size) override;

    size_t get_mem_addr_alignment() override {
        return m_env.property().mem_alignment;
    }

    std::unique_ptr<Event> create_event(size_t flags) override;

    void sync() override;

    MemNode mem_node() override;

    std::pair<size_t, size_t> get_mem_status_bytes() override {
        m_env.cnrt_env().activate();
        cndevMemoryInfo_t mem_info;
        MGB_CNDEV_CHECK(
                cndevGetMemoryUsage(&mem_info, m_env.cnrt_env().device));
        size_t tot, used, free;
        constexpr size_t mb2size = 1024 * 1024;
        tot = static_cast<size_t>(mem_info.PhysicalMemoryTotal) * mb2size;
        used = static_cast<size_t>(mem_info.PhysicalMemoryUsed) * mb2size;
        free = tot - used + m_mem_alloc->get_free_memory_dev().tot;
        return {tot, free};
    }

    Locator locator() override { return m_locator; }

    Locator locator_logical() override { return m_locator_logical; }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(CambriconCompNode::CompNodeImpl);

struct CambriconCompNodeImpl::DeviceInfo {
    int dev_num = -1;
    cnrtDev_t dev;
    std::unique_ptr<mem_alloc::DevMemAlloc> mem_alloc;

    bool init_done() const { return mem_alloc.get(); }

    void init(const CompNodeEnv& env);

    // unlike cuda, we have to set device first, then release device memory
    void fini() {
        cnrtSetCurrentDevice(dev);
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

void CambriconCompNodeImpl::init(const Locator& locator,
                                 const Locator& locator_logical) {
    m_locator = locator;
    m_locator_logical = locator_logical;
    m_initialized = true;

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
        m_mem_alloc =
                dev_info->mem_alloc->add_stream(static_cast<void*>(queue));
        m_dev = m_device_info->dev;
    };

    auto on_error = [this](std::exception&) {
        MGB_LOCK_GUARD(sd->mtx);
        m_initialized = false;
    };

    m_env.init_cnrt(locator.device, make_comp_node_from_impl(this),
                    {on_succ, on_error});
}

void CambriconCompNodeImpl::fini() {
    if (!m_initialized)
        return;

    m_env.fini();
    m_mem_alloc = nullptr;
    m_device_info = nullptr;
    m_initialized = false;
}

void CambriconCompNodeImpl::free_device(void* ptr) {
    if (check_global_finalized())
        return;

    activate();
    m_mem_alloc->free(ptr);
}

void CambriconCompNodeImpl::peer_copy_to(Impl* dest_impl, void* dest,
                                         const void* src, size_t size) {
    if (dest_impl->same_type<CambriconCompNodeImpl>()) {
        auto&& dst_env = static_cast<CambriconCompNodeImpl*>(dest_impl)
                                 ->m_env.cnrt_env();
        auto&& src_env = m_env.cnrt_env();
        activate();
        if (dst_env.device == src_env.device) {
            // remark: transfering data from device to device does not
            // support async
            sync();
            dest_impl->sync();
            MGB_CNRT_CHECK(cnrtMemcpy(dest, const_cast<void*>(src), size,
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
            dest_impl->sync();
            MGB_CNRT_CHECK(cnrtMemcpyPeer(dest, dst_env.device,
                                          const_cast<void*>(src),
                                          src_env.device, size));
        }
        return;
    }
    mgb_assert(dest_impl->env().property().type == DeviceType::CPU,
               "cnrt peer_copy_to only implemented for CPU");
    auto copy = [this, dest, src, size]() {
        m_env.cnrt_env().activate();
        auto queue = m_env.cnrt_env().queue;
        MGB_CNRT_CHECK(cnrtMemcpyAsync(dest, const_cast<void*>(src), size,
                                       queue, CNRT_MEM_TRANS_DIR_DEV2HOST));
        MGB_CNRT_CHECK(cnrtSyncQueue(queue));
    };
    dest_impl->env().cpu_env().dispatch(copy);
}

MemNode CambriconCompNodeImpl::mem_node() {
    return MemNode{sd->dev_info + m_locator.device};
}

void CambriconCompNodeImpl::sync() {
    activate();

    // remark: CNRT does not provide interface like cudaEventQuery to test
    // whether an event is finished. so we just call the cnrtSyncQueue
    MGB_CNRT_CHECK(cnrtSyncQueue(m_env.cnrt_env().queue));
}

bool CambriconCompNodeImpl::enable_peer_access(int dev0, int dev1) {
    static bool queried_enabled[StaticData::MAX_NR_DEVICE]
                               [StaticData::MAX_NR_DEVICE];
    if (queried_enabled[dev0][dev1])
        return queried_enabled[dev0][dev1];

    static std::mutex global_lock;
    MGB_LOCK_GUARD(global_lock);
    unsigned int can = 0;
    MGB_CNRT_CHECK(cnrtGetPeerAccessibility(&can, dev0, dev1));
    if (can)
        mgb_log("device(%d) can directly access memories on device(%d)", dev0,
                dev1);
    queried_enabled[dev0][dev1] = can;
    return can;
}

/* ================== CambriconCompNodeImpl::DeviceInfo ===============*/

void CambriconCompNodeImpl::DeviceInfo::init(const CompNodeEnv& env) {
    mgb_assert(!mem_alloc);
    auto&& cnenv = env.cnrt_env();
    cnenv.activate();
    dev_num = cnenv.device;
    MGB_CNRT_CHECK(cnrtGetDeviceHandle(&dev, dev_num));
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
            dev_num, reserve_size,
            std::make_shared<mem_alloc::CambriconRawAlloctor>(),
            std::make_shared<mem_alloc::CambriconDeviceRuntimePolicy>());
    mem_alloc->prealloc_config(sd->prealloc_config);
    auto align = env.property().mem_alignment;
    mem_alloc->alignment(align);
    cnrtDeviceInfo_t device_info;
    MGB_CNRT_CHECK(cnrtGetDeviceInfo(&device_info, dev_num));
    mgb_log("cambricon: card%d: name=`%s' dyn_mem_reserve=%.2fMiB "
            "alignment=0x%zx",
            dev_num, device_info.device_name, reserve_size / 1024.0 / 1024,
            align);
#endif
}

size_t CambriconCompNodeImpl::DeviceInfo::get_mem_reserve_size() {
    if (auto setting = MGB_GETENV("MGB_CAMBRICON_RESERVE_MEMORY")) {
        if (!strncmp(setting, "b:", 2)) {
            return std::stoull(setting + 2);
        }
        size_t tot, free;
        cndevMemoryInfo_t mem_info;
        MGB_CNDEV_CHECK(cndevGetMemoryUsage(&mem_info, dev_num));
        constexpr size_t mb2size = 1024 * 1024;
        tot = static_cast<size_t>(mem_info.PhysicalMemoryTotal) * mb2size;
        size_t used =
                static_cast<size_t>(mem_info.PhysicalMemoryUsed) * mb2size;
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
            mgb_log_warn(
                    "cambricon comp node method called after global finalize");
        }
        return true;
    }
    return false;
}

/* ================== CambriconCompNodeImpl::EventImpl ================*/

class CambriconCompNode::EventImpl final : public EventImplHelper {
    bool m_placed_notifier = false;
    bool m_sync_queue_called = false;
    bool m_init_finished = false;
    cnrtNotifier_t m_cnrt_notifier;

    CambriconCompNodeImpl* cambricon_comp_node_impl() const {
        return static_cast<CambriconCompNodeImpl*>(m_comp_node_impl);
    }

    void do_record() override {
        m_sync_queue_called = false;
        cambricon_comp_node_impl()->activate();
        auto&& env = cambricon_comp_node_impl()->m_env.cnrt_env();
        if (!m_placed_notifier) {
            MGB_CNRT_CHECK(cnrtPlaceNotifier(m_cnrt_notifier, env.queue));
            m_placed_notifier = true;
        }
    }

    void call_sync_queue() {
        mgb_assert(m_placed_notifier);
        if (!m_sync_queue_called) {
            cambricon_comp_node_impl()->activate();
            auto&& env = cambricon_comp_node_impl()->m_env.cnrt_env();
            MGB_CNRT_CHECK(cnrtSyncQueue(env.queue));
            m_sync_queue_called = true;
        }
    }

    bool do_finished() override {
        call_sync_queue();
        return true;
    }

    void host_wait_cv() override {
        mgb_assert(m_placed_notifier);
        cambricon_comp_node_impl()->activate();
        auto&& env = cambricon_comp_node_impl()->m_env.cnrt_env();
        MGB_CNRT_CHECK(cnrtSyncQueue(env.queue));
    }

    double do_elapsed_time_until(EventImplHelper& end) override {
        cambricon_comp_node_impl()->activate();
        auto&& env = cambricon_comp_node_impl()->m_env.cnrt_env();
        MGB_CNRT_CHECK(cnrtSyncQueue(env.queue));
        float ret = 0.f;
        MGB_CNRT_CHECK(cnrtNotifierDuration(
                m_cnrt_notifier, static_cast<EventImpl&>(end).m_cnrt_notifier,
                &ret));
        return static_cast<double>(ret) * 1e-3;
    }

    void do_device_wait_by(Impl* cn_impl) override;

public:
    EventImpl(CambriconCompNodeImpl* comp_node_impl, size_t create_flags)
            : EventImplHelper(comp_node_impl, create_flags) {
        cambricon_comp_node_impl()->activate();
        MGB_CNRT_CHECK(cnrtCreateNotifier(&m_cnrt_notifier));
        m_init_finished = true;
    }

    ~EventImpl() {
        if (m_init_finished) {
            MGB_TRY { MGB_CNRT_CHECK(cnrtDestroyNotifier(&m_cnrt_notifier)); }
            MGB_CATCH(MegBrainError & exc, {
                mgb_log_error("failed to destroy cnrt notifier: %s",
                              exc.what());
            })
        }
    }
};

std::unique_ptr<CompNode::Event> CambriconCompNodeImpl::create_event(
        size_t flags) {
    return std::make_unique<EventImpl>(this, flags);
}

void CambriconCompNode::EventImpl::do_device_wait_by(Impl* cn_impl) {
    if (cn_impl->env().property().type == DeviceType::CAMBRICON) {
        auto imp = static_cast<CambriconCompNodeImpl*>(cn_impl);
        auto queue = imp->m_env.cnrt_env().queue;
        imp->activate();
        MGB_CNRT_CHECK(cnrtSyncQueue(queue));
        return;
    }
    if (cn_impl->env().property().type == DeviceType::CPU) {
        auto waiter = [this]() {
            cambricon_comp_node_impl()->activate();
            auto queue = cambricon_comp_node_impl()->m_env.cnrt_env().queue;
            MGB_CNRT_CHECK(cnrtSyncQueue(queue));
        };
        cn_impl->add_callback(std::move(waiter));
        return;
    }
    mgb_throw(MegBrainError, "unimplemented event device_wait_by config");
}

/* ================== CambriconCompNode static methods ================*/

bool CambriconCompNode::available() {
    CompNodeEnv::CnrtEnv::init();
    static int result = -1;
    static Spinlock mtx;
    MGB_LOCK_GUARD(mtx);
    if (result == -1) {
        unsigned int dev_num = 0;
        auto err = cnrtGetDeviceCount(&dev_num);
        result = err == CNRT_RET_SUCCESS && dev_num >= 1;
        if (!result) {
            mgb_log_warn("cambricon unavailable: %d(%s) dev_num=%u",
                         static_cast<int>(err), cnrtGetErrorStr(err), dev_num);
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
    mgb_assert(locator.device >= 0 && locator.device < nr_devs,
               "request device%d out of range [0, %d)", locator.device,
               nr_devs);

    auto&& sdptr = CambriconCompNodeImpl::sd;
    {
        MGB_LOCK_GUARD(CambriconCompNodeImpl::sd_mtx);
        if (!sdptr) {
            using T = CambriconCompNodeImpl::StaticData;
            static std::aligned_storage_t<sizeof(T), alignof(T)> storage;
            sdptr = new(&storage)T;
        }
    }
    auto&& sd = *sdptr;
    MGB_LOCK_GUARD(sd.mtx);

    CompNodeImpl* available_node = nullptr;
    for (int i = 0; i < sd.nr_node; ++i) {
        auto&& cur = sd.node[i];
        if (cur.m_initialized) {
            if (cur.m_locator_logical == locator_logical) {
                return &cur;
            }
        } else {
            available_node = &cur;
        }
    }

    if (!available_node) {
        mgb_assert(sd.nr_node < sd.MAX_NR_COMP_NODE,
                   "too many CompNode allocated");
        mgb_assert(locator.device < sd.MAX_NR_COMP_NODE,
                   "device number too large");
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
        size += sd->dev_info[i]
                        .mem_alloc->gather_stream_free_blk_and_release_full();
    }
    if (size) {
        mgb_log_debug("%zu bytes freed by try_coalesce_all_free_memory()",
                      size);
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
        cnrtDev_t dev;
        MGB_CNRT_CHECK(cnrtGetDeviceHandle(&dev, sd->dev_info[i].dev_num));
        MGB_CNRT_CHECK(cnrtSetCurrentDevice(dev));
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

size_t CambriconCompNode::get_device_count() {
    CompNodeEnv::CnrtEnv::init();
    static int cnt = -1;
    static Spinlock mtx;
    MGB_LOCK_GUARD(mtx);
    if (cnt == -1) {
        unsigned int dev_cnt = 0;
        auto ret = cnrtGetDeviceCount(&dev_cnt);
        if (ret != CNRT_RET_SUCCESS) {
            mgb_log_error("cnrtGetDeviceCount faild: %s (err %d)",
                          cnrtGetErrorStr(ret), int(ret));
            cnt = 0;
        }
        cnt = dev_cnt;
        mgb_assert(cnt >= 0);
    }
    return cnt;
}

void mgb::mem_alloc::CambriconRawAlloctor::get_mem_info(size_t& free,
                                                        size_t& tot) {
    auto sd = CambriconCompNodeImpl::sd;
    int device = -1;
    {
        cnrtDev_t dev;
        MGB_CNRT_CHECK(cnrtGetCurrentDevice(&dev));
        for (int i = 0; i < sd->nr_dev_used; ++i) {
            if (sd->dev_info[i].dev == dev) {
                device = sd->dev_info[i].dev_num;
                break;
            }
        }
    }
    mgb_assert(device >= 0,
               "current device has not been initialized in static data");
    cndevMemoryInfo_t mem_info;
    auto ret = cndevGetMemoryUsage(&mem_info, device);
    if (ret == CNDEV_SUCCESS) {
        constexpr size_t mb2size = 1024 * 1024;
        tot = static_cast<size_t>(mem_info.PhysicalMemoryTotal) * mb2size;
        size_t used =
                static_cast<size_t>(mem_info.PhysicalMemoryUsed) * mb2size;
        free = tot - used;
        return;
    }
    auto msg =
            ssprintf("cndevGetMemoryUsage faild %s", cndevGetErrorString(ret));
    mgb_throw_raw(MemAllocError{msg});
}

#else

bool CambriconCompNode::available() {
    return false;
}
void CambriconCompNode::try_coalesce_all_free_memory() {}
void CambriconCompNode::foreach (thin_function<void(CompNode)>) {}
void CambriconCompNode::finalize() {}
size_t CambriconCompNode::get_device_count() {
    return 0;
}
CambriconCompNode::Impl* CambriconCompNode::load_cambricon(const Locator&,
                                                           const Locator&) {
    mgb_throw(MegBrainError, "cambricon disabled at compile time");
}
void CambriconCompNode::sync_all() {}

#undef err

#endif  // MGB_CAMBRICON

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

