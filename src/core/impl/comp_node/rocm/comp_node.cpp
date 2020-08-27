/**
 * \file src/core/impl/comp_node/rocm/comp_node.cpp
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

#if MGB_ROCM

#include "megbrain/comp_node/alloc.h"

#include <cctype>
#include <cstdio>

#include <thread>

#include "hip_header.h"

using ROCmCompNodeImpl = ROCmCompNode::CompNodeImpl;

namespace {
size_t get_min_system_memory(size_t available) {
    if (available < (1u << 31)) {
        // 225MiB
        return 225 * 1024 * 1024;
    } else {
        // max(300 MiB, 0.05 * available)
        return std::max<size_t>(300 * 1024 * 1024, available / 20);
    }
}
}  // anonymous namespace

namespace mgb {
namespace mem_alloc {
class ROCmRawAllocator final : public RawAllocator {
public:
    void* alloc(size_t size) override {
        void* addr;
        hipError_t hip_error = hipMalloc(&addr, size);
        if (hip_error == hipSuccess) {
            mgb_assert(addr);
            return addr;
        }
        auto msg = mgb_ssprintf_log(
                "hipMalloc failed while requesting %zd bytes (%.3fMiB)"
                " of memory; error: %s",
                size, size / (1024.0 * 1024), hipGetErrorString(hip_error));
        msg.append(ROCmError::get_rocm_extra_info());
        if (hip_error == hipErrorMemoryAllocation) {
            mgb_log_error("%s", msg.c_str());
            // clear hip error
            hipGetLastError();
            mgb_assert(hipGetLastError() == hipSuccess);
            return nullptr;
        }
        mgb_throw_raw(MemAllocError{msg});
    }

    void free(void* ptr) override {
        hipError_t hip_error = hipFree(ptr);
        if (hip_error == hipSuccess)
            return;
        auto msg = ssprintf("hipFree failed for %p: %s", ptr,
                            hipGetErrorString(hip_error));
        msg.append(ROCmError::get_rocm_extra_info());
        mgb_throw_raw(MemAllocError{msg});
    }

    void get_mem_info(size_t& free, size_t& tot) override {
        hipError_t hip_error = hipMemGetInfo(&free, &tot);
        if (hip_error == hipSuccess)
            return;
        auto msg = ssprintf("hipMemGetInfo failed %s",
                            hipGetErrorString(hip_error));
        msg.append(ROCmError::get_rocm_extra_info());
        mgb_throw_raw(MegBrainError{msg});
    }
};

class ROCmDeviceRuntimePolicy : public DeviceRuntimePolicy {
public:
    CompNode::DeviceType device_type() override {
        return CompNode::DeviceType::ROCM;
    }
    void set_device(int device) override {
        MGB_ROCM_CHECK(hipSetDevice(device));
    }
    void device_synchronize(int device) override {
        MGB_ROCM_CHECK(hipSetDevice(device));
        MGB_ROCM_CHECK(hipDeviceSynchronize());
    }
};

/* ===================== DevMemAlloc  ===================== */
std::unique_ptr<DevMemAlloc> DevMemAlloc::make_rocm_alloc() {
    return std::make_unique<FwdDevMemAlloc>(
            std::make_shared<ROCmRawAllocator>());
}
}  // namespace mem_alloc
}  // namespace mgb

/* ===================== ROCmCompNodeImpl  ===================== */
class ROCmCompNode::CompNodeImpl final : public CompNode::Impl {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

    friend class EventImpl;
    friend class ROCmCompNode;

    struct DeviceInfo;
    struct StaticData;
    static StaticData* sd;
    static Spinlock sd_mtx;

    //! set to true when m_locator is assigned; set to false if async init
    //! failed
    bool m_initialized = false;
    Locator m_locator, m_locator_logical;
    mem_alloc::StreamMemAlloc* m_mem_alloc;
    DeviceInfo* m_device_info;

    std::unique_ptr<Event> m_sync_event;
    Spinlock m_sync_event_mtx;

    void activate() { m_env.rocm_env().activate(); }

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
    CompNodeImpl() : Impl(static_free_device, static_free_host) { }

    void* alloc_device(size_t size) override {
        activate();
        return m_mem_alloc->alloc(size);
    }

    void free_device(void* ptr);

    //! hipMallocHost is deprecated, we cannot allocate cpu pinned memory
    // we can use hipHostAlloc to emulate hipMallocHost
    void* alloc_host(size_t size) override {
        activate();
        void* ptr;
        MGB_ROCM_CHECK(hipHostMalloc(&ptr, size, hipHostMallocDefault));
        return ptr;
    }

    void free_host(void* ptr) {
        if (!check_global_finalized()) {
            activate();
        }
        MGB_ROCM_CHECK(hipHostFree(ptr));
    }

    void copy_to_host(void* host_ptr, const void* device_ptr,
                      size_t size) override {
        MGB_ROCM_CHECK(hipMemcpyAsync(host_ptr, device_ptr, size,
                                      hipMemcpyDeviceToHost,
                                      m_env.rocm_env().stream));
    }

    void copy_to_device(void* device_ptr, const void* host_ptr,
                        size_t size) override {
        MGB_ROCM_CHECK(hipMemcpyAsync(device_ptr, host_ptr, size,
                                      hipMemcpyHostToDevice,
                                      m_env.rocm_env().stream));
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
        // explicitly call rocm_env() to ensure async init is finished
        m_env.rocm_env().activate();
        size_t tot, free;
        MGB_ROCM_CHECK(hipMemGetInfo(&free, &tot));
        free += m_mem_alloc->get_free_memory_dev().tot;
        return {tot, free};
    }

    Locator locator() override { return m_locator; }

    Locator locator_logical() override { return m_locator_logical; }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(ROCmCompNode::CompNodeImpl);

struct ROCmCompNodeImpl::DeviceInfo {
    int dev_num = -1;
    std::unique_ptr<mem_alloc::DevMemAlloc> mem_alloc;

    bool init_done() const { return mem_alloc.get(); }

    void init(const CompNodeEnv& env);

    void fini() { mem_alloc.reset(); }
};

struct ROCmCompNodeImpl::StaticData {
    static constexpr int MAX_NR_COMP_NODE = 1024, MAX_NR_DEVICE = 64;

    std::recursive_mutex mtx;

    mem_alloc::DevMemAlloc::PreAllocConfig prealloc_config;

    ROCmCompNode::CompNodeImpl node[MAX_NR_COMP_NODE];
    DeviceInfo dev_info[MAX_NR_DEVICE];
    int nr_node = 0,          //!< number of loaded node[]
            nr_dev_used = 0;  //!< number of used dev_info[]

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

    static size_t get_mem_reserve_size() {
        if (auto setting = MGB_GETENV("MGB_ROCM_RESERVE_MEMORY")) {
            if (!strncmp(setting, "b:", 2)) {
                return std::stoull(setting + 2);
            }
            size_t tot, free;
            MGB_ROCM_CHECK(hipFree(0));
            MGB_ROCM_CHECK(hipMemGetInfo(&free, &tot));
            return free - get_min_system_memory(free);
        } else {
            return 0;
        }
    }
};
ROCmCompNodeImpl::StaticData* ROCmCompNodeImpl::sd = nullptr;
Spinlock ROCmCompNodeImpl::sd_mtx;

void ROCmCompNodeImpl::init(const Locator& locator,
                            const Locator& locator_logical) {
    m_locator = locator;
    m_locator_logical = locator_logical;
    m_initialized = true;

    auto on_succ = [this](hipStream_t stream) {
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
            // note: add nr_dev_used only after init succeeds
            ++sd->nr_dev_used;
        }
        m_device_info = dev_info;
        m_mem_alloc = dev_info->mem_alloc->add_stream(stream);
    };

    auto on_error = [this](std::exception&) {
        MGB_LOCK_GUARD(sd->mtx);
        m_initialized = false;
    };

    m_env.init_rocm_async(locator.device, make_comp_node_from_impl(this),
                          {on_succ, on_error});
}

void ROCmCompNodeImpl::fini() {
    if (!m_initialized)
        return;

    m_sync_event.reset();
    m_env.fini();
    m_mem_alloc = nullptr;
    m_device_info = nullptr;
    m_initialized = false;
}

void ROCmCompNodeImpl::free_device(void* ptr) {
    if (check_global_finalized())
        return;

    activate();
    m_mem_alloc->free(ptr);
}

void ROCmCompNodeImpl::peer_copy_to(Impl* dest_impl, void* dest,
                                    const void* src, size_t size) {
    if (dest_impl->same_type<ROCmCompNodeImpl>()) {
        auto&& dst_env =
                static_cast<ROCmCompNodeImpl*>(dest_impl)->m_env.rocm_env();
        auto&& src_env = m_env.rocm_env();
        if (dst_env.device == src_env.device) {
            MGB_ROCM_CHECK(hipMemcpyAsync(
                    dest, src, size, hipMemcpyDeviceToDevice, dst_env.stream));
        } else {
            enable_peer_access(src_env.device, dst_env.device);
            enable_peer_access(dst_env.device, src_env.device);
            MGB_ROCM_CHECK(hipMemcpyPeerAsync(dest, dst_env.device, src,
                                              src_env.device, size,
                                              dst_env.stream));
        }
        return;
    }
    mgb_assert(dest_impl->env().property().type == DeviceType::CPU,
               "rocm peer_copy_to only implemented for CPU");
    auto copy = [this, dest, src, size]() {
        auto stream = m_env.rocm_env().stream;
        MGB_ROCM_CHECK(
                hipMemcpyAsync(dest, src, size, hipMemcpyDeviceToHost, stream));
        MGB_ROCM_CHECK(hipStreamSynchronize(stream));
    };
    dest_impl->env().cpu_env().dispatch(copy);
}

MemNode ROCmCompNodeImpl::mem_node() {
    // m_device_info would be null before async init finishes; so we just return
    // a prive pointer related to device number here
    return MemNode{sd->dev_info + m_locator.device};
}

void ROCmCompNodeImpl::sync() {
    activate();

    // same behavior as cuda
    // do not use MGB_ROCM_CHECK(hipStreamSynchronize(m_env->stream)) since
    // other threads may be adding operations into the stream, and we only care
    // about previous operations in current thread. However docs of
    // hipStreamSynchronize did not describe details of such condition, so we
    // use manual event implementation

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

void ROCmCompNodeImpl::enable_peer_access(int dev0, int dev1) {
    static bool already_enabled[StaticData::MAX_NR_DEVICE]
                               [StaticData::MAX_NR_DEVICE];
    if (already_enabled[dev0][dev1])
        return;

    static std::mutex global_lock;
    MGB_LOCK_GUARD(global_lock);
    if (already_enabled[dev0][dev1])
        return;

    int can;
    MGB_ROCM_CHECK(hipDeviceCanAccessPeer(&can, dev0, dev1));
    if (can) {
        mgb_log("enable peer access from GPU %d to GPU %d", dev0, dev1);
        MGB_ROCM_CHECK(hipSetDevice(dev0));
        auto err = hipDeviceEnablePeerAccess(dev1, 0);
        if (err != hipSuccess) {
            mgb_log_error("failed to enable peer access from %d to %d: %s(%d)",
                          dev0, dev1, hipGetErrorString(err),
                          static_cast<int>(err));
            hipGetLastError();
        }
    }

    // check for hipMemcpyPeer usable
    int v0 = 1, v1 = 2;

    int *dp0, *dp1;
    MGB_ROCM_CHECK(hipSetDevice(dev0));
    MGB_ROCM_CHECK(hipMalloc(&dp0, sizeof(int)));
    MGB_ROCM_CHECK(hipSetDevice(dev1));
    MGB_ROCM_CHECK(hipMalloc(&dp1, sizeof(int)));
    MGB_ROCM_CHECK(hipMemcpy(dp0, &v0, sizeof(int), hipMemcpyHostToDevice));
    MGB_ROCM_CHECK(hipMemcpy(dp1, &v1, sizeof(int), hipMemcpyHostToDevice));
    MGB_ROCM_CHECK(hipMemcpyPeer(dp1, dev1, dp0, dev0, sizeof(int)));
    int get = 0;
    MGB_ROCM_CHECK(hipMemcpy(&get, dp1, sizeof(int), hipMemcpyDeviceToHost));

    mgb_throw_if(get != 1, ROCmError,
                 "P2P copy (%d => %d) check failed; consider disabling "
                 "Access Control Services(ACS) for the PCI device",
                 dev0, dev1);

    already_enabled[dev0][dev1] = true;
}

/* ===================== ROCmCompNodeImpl::DeviceInfo  ===================== */

void ROCmCompNodeImpl::DeviceInfo::init(const CompNodeEnv& env) {
    mgb_assert(!mem_alloc);
#if 0
    // forward hipMalloc
    auto&& rocm_env = env.rocm_env();
    rocm_env.activate();
    dev_num = rocm_env.device;
    mem_alloc = mem_alloc::DevMemAlloc::make_rocm_alloc();
#else
    auto&& rocm_env = env.rocm_env();
    rocm_env.activate();
    dev_num = rocm_env.device;
    auto reserve_size = StaticData::get_mem_reserve_size();
    mem_alloc = mem_alloc::DevMemAlloc::make(
            dev_num, reserve_size,
            std::make_shared<mem_alloc::ROCmRawAllocator>(),
            std::make_shared<mem_alloc::ROCmDeviceRuntimePolicy>());
    mem_alloc->prealloc_config(sd->prealloc_config);
    auto align = env.property().mem_alignment;
    mem_alloc->alignment(align);
    mgb_log("rocm: gpu%d: name=`%s' dyn_mem_reserve=%.2fMiB alignment=0x%zx",
            dev_num, rocm_env.device_prop.name, reserve_size / 1024.0 / 1024,
            align);
#endif
}

bool ROCmCompNodeImpl::check_global_finalized() {
    if (!sd) {
        static std::atomic_flag warn_printed = ATOMIC_FLAG_INIT;
        if (!warn_printed.test_and_set()) {
            mgb_log_warn("rocm comp node method called after global finalize");
        }
        return true;
    }
    return false;
}

/* ===================== EventImpl  ===================== */

class ROCmCompNode::EventImpl final : public EventImplHelper {
    bool m_init_finished = false;
    ROCmCompNodeImpl* const m_comp_node_impl;
    hipEvent_t m_hip_event;

    void do_record() override {
        m_comp_node_impl->activate();
        auto&& env = m_comp_node_impl->m_env.rocm_env();
        MGB_ROCM_CHECK(hipEventRecord(m_hip_event, env.stream));
    }

    bool do_finished() override {
        m_comp_node_impl->activate();
        hipError_t err = hipEventQuery(m_hip_event);
        if (err == hipSuccess)
            return true;
        if (err == hipErrorNotReady)
            return false;
        mgb_throw(ROCmError, "failed to query event: %d: %s", int(err),
                  hipGetErrorString(err));
    }

    void host_wait_cv() override {
        MGB_ROCM_CHECK(hipEventSynchronize(m_hip_event));
    }

    double do_elapsed_time_until(EventImplHelper& end) override {
        m_comp_node_impl->activate();
        float ret = 0.0;
        MGB_ROCM_CHECK(hipEventElapsedTime(
                &ret, m_hip_event, static_cast<EventImpl&>(end).m_hip_event));
        return static_cast<double>(ret) * 1e-3;
    }

    void do_device_wait_by(Impl* cn_impl) override;

public:
    EventImpl(ROCmCompNodeImpl* comp_node_impl, size_t create_flags)
            : EventImplHelper(comp_node_impl, create_flags),
              m_comp_node_impl{comp_node_impl} {
        m_comp_node_impl->activate();
        size_t hip_flags = hipEventDisableTiming;
        if (create_flags & NEED_TIMER)
            hip_flags = 0;
        MGB_ROCM_CHECK(hipEventCreateWithFlags(&m_hip_event, hip_flags));
        m_init_finished = true;
    }

    ~EventImpl() {
        if (m_init_finished) {
            MGB_TRY { MGB_ROCM_CHECK(hipEventDestroy(m_hip_event)); }
            MGB_CATCH(MegBrainError & exc, {
                mgb_log_error("failed to destroy hip event: %s", exc.what());
            })
        }
    }
};

std::unique_ptr<CompNode::Event> ROCmCompNodeImpl::create_event(size_t flags) {
    return std::make_unique<EventImpl>(this, flags);
}

void ROCmCompNode::EventImpl::do_device_wait_by(Impl* cn_impl) {
    if (cn_impl->dyn_typeinfo() == ROCmCompNodeImpl::typeinfo()) {
        auto imp = static_cast<ROCmCompNodeImpl*>(cn_impl);
        auto stream = imp->m_env.rocm_env().stream;
        imp->activate();
        MGB_ROCM_CHECK(hipStreamWaitEvent(stream, m_hip_event, 0));
        return;
    }
    if (cn_impl->env().property().type == DeviceType::CPU) {
        auto waiter = [this]() {
            MGB_ROCM_CHECK(hipEventSynchronize(m_hip_event));
        };
        cn_impl->add_callback(std::move(waiter));
        return;
    }
    mgb_throw(MegBrainError, "unimplemented event device_wait_by config");
}

/* ===================== ROCmCompNode static methods ===================== */

bool ROCmCompNode::available() {
    static int result = -1;
    static Spinlock mtx;
    MGB_LOCK_GUARD(mtx);
    if (result == -1) {
        int ndev = -1;
        auto err = hipGetDeviceCount(&ndev);
        result = err == hipSuccess && ndev > 0;
        if (!result) {
            mgb_log_warn("rocm unavailable: %s(%d) ndev=%d",
                         hipGetErrorString(err), static_cast<int>(err), ndev);
        }
    }
    return result;
}

void ROCmCompNode::finalize() {
    if (ROCmCompNodeImpl::sd) {
        sync_all();

        auto ptr = ROCmCompNodeImpl::sd;
        ROCmCompNodeImpl::sd = nullptr;
        ptr->~StaticData();
    }
}

CompNode::Impl* ROCmCompNode::load_rocm(const Locator& locator,
                                        const Locator& locator_logical) {
    int nr_gpu = get_device_count();
    mgb_assert(locator.device >= 0 && locator.device < nr_gpu,
               "request gpu%d out of valid range [0, %d)", locator.device,
               nr_gpu);

    auto&& sdptr = ROCmCompNodeImpl::sd;
    {
        MGB_LOCK_GUARD(ROCmCompNodeImpl::sd_mtx);
        if (!sdptr) {
            // use static storage so object can be safely accessed even after
            // global finalize
            using T = ROCmCompNodeImpl::StaticData;
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

void ROCmCompNode::try_coalesce_all_free_memory() {
    // TODO: optimized implementation
    auto sd = ROCmCompNodeImpl::sd;
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

void ROCmCompNode::sync_all() {
    auto sd = ROCmCompNodeImpl::sd;
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
        env->rocm_env();
    }

    MGB_LOCK_GUARD(sd->mtx);
    for (int i = 0; i < sd->nr_dev_used; ++i) {
        MGB_ROCM_CHECK(hipSetDevice(sd->dev_info[i].dev_num));
        MGB_ROCM_CHECK(hipDeviceSynchronize());
    }
}

void ROCmCompNode::foreach (thin_function<void(CompNode)> callback) {
    auto sd = ROCmCompNodeImpl::sd;
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

size_t ROCmCompNode::get_device_count() {
    static int cnt = -1;
    static Spinlock mtx;
    MGB_LOCK_GUARD(mtx);
    if (cnt == -1) {
        auto err = hipGetDeviceCount(&cnt);
        if (err != hipSuccess) {
            mgb_log_error("hipGetDeviceCount failed: %s (err %d)",
                          hipGetErrorString(err), int(err));
            cnt = 0;
        }
        mgb_assert(cnt >= 0);
    }
    return cnt;
}

#else

bool ROCmCompNode::available() {
    return false;
}
void ROCmCompNode::try_coalesce_all_free_memory() {}
void ROCmCompNode::foreach (thin_function<void(CompNode)>) {}
void ROCmCompNode::finalize() {}
size_t ROCmCompNode::get_device_count() {
    return 0;
}
ROCmCompNode::Impl* ROCmCompNode::load_rocm(const Locator&, const Locator&) {
    mgb_throw(MegBrainError, "rocm disabled at compile time");
}
void ROCmCompNode::sync_all() {}

#undef err

#endif // MGB_ROCM

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
