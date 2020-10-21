/**
 * \file src/core/impl/comp_node/atlas/comp_node.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

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

/* ===================== AtlasCompNodeImpl  ===================== */
class AtlasCompNode::CompNodeImpl final : public CompNode::Impl {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

    friend class EventImpl;
    friend class AtlasCompNode;

    struct DeviceInfo;
    struct StaticData;
    static StaticData* sd;
    static Spinlock sd_mtx;

    //! set to true when m_locator is assigned; set to false if async init
    //! failed
    bool m_initialized = false;
    Locator m_locator, m_locator_logical;
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

    void* alloc_device(size_t size) override {
        activate();
        void* addr;
        MGB_ATLAS_CHECK(aclrtMalloc(&addr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        return addr;
    }

    void free_device(void* ptr) {
        if (check_global_finalized())
            return;

        activate();

        MGB_ATLAS_CHECK(aclrtFree(ptr));
    }

    void* alloc_host(size_t size) override {
        void* ptr;
        MGB_ATLAS_CHECK(aclrtMallocHost(&ptr, size));
        return ptr;
    }

    void free_host(void* ptr) { MGB_ATLAS_CHECK(aclrtFreeHost(ptr)); }

    void copy_to_host(void* host_ptr, const void* device_ptr,
                      size_t size) override {
        activate();
#if MGB_USE_ATLAS_ASYNC_API
        MGB_ATLAS_CHECK(aclrtMemcpyAsync(host_ptr, size, device_ptr, size,
                                         ACL_MEMCPY_DEVICE_TO_HOST,
                                         m_env.atlas_env().stream));
#else
        MGB_ATLAS_CHECK(aclrtMemcpy(host_ptr, size, device_ptr, size,
                                    ACL_MEMCPY_DEVICE_TO_HOST));
#endif
    }

    void copy_to_device(void* device_ptr, const void* host_ptr,
                        size_t size) override {
        activate();
        MGB_ATLAS_CHECK(aclrtMemcpy(device_ptr, size, host_ptr, size,
                                    ACL_MEMCPY_HOST_TO_DEVICE));
    }

    void peer_copy_to(Impl* dest_impl, void* dest, const void* src,
                      size_t size) override;

    size_t get_mem_addr_alignment() override {
        return m_env.property().mem_alignment;
    }

    std::unique_ptr<Event> create_event(size_t flags) override;

    void sync() override;

    MemNode mem_node() override;

    size_t get_mem_padding() override { return 32; }

    std::pair<size_t, size_t> get_mem_status_bytes() override {
        return {std::numeric_limits<size_t>::max(),
                std::numeric_limits<size_t>::max()};
    }

    Locator locator() override { return m_locator; }

    Locator locator_logical() override { return m_locator_logical; }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(AtlasCompNode::CompNodeImpl);

struct AtlasCompNodeImpl::DeviceInfo {
    int dev_num = -1;

    void init(const CompNodeEnv& env) {
        auto&& atlas_env = env.atlas_env();
        atlas_env.activate();
        dev_num = atlas_env.device;
    }

    void fini() {
        MGB_ATLAS_CHECK(aclrtResetDevice(dev_num));
    }
};


struct AtlasCompNodeImpl::StaticData {
    static constexpr int MAX_NR_COMP_NODE = 1024, MAX_NR_DEVICE = 64;

    std::recursive_mutex mtx;

    AtlasCompNode::CompNodeImpl node[MAX_NR_COMP_NODE];
    DeviceInfo dev_info[MAX_NR_DEVICE];
    int nr_node = 0,          //!< number of loaded node[]
            nr_dev_used = 0;  //!< number of used dev_info[]

    StaticData() {}

    ~StaticData() {
        for (int i = 0; i < nr_node; ++i)
            node[i].fini();
        for (int i = 0; i < nr_dev_used; ++i)
            dev_info[i].fini();
    }
};
AtlasCompNodeImpl::StaticData* AtlasCompNodeImpl::sd = nullptr;
Spinlock AtlasCompNodeImpl::sd_mtx;

void AtlasCompNodeImpl::init(const Locator& locator,
                             const Locator& locator_logical) {
    m_locator = locator;
    m_locator_logical = locator_logical;
    m_initialized = true;

    CompNodeEnv::AtlasEnv atlas_env;
    atlas_env.device = locator.device;
    m_env.init_atlas(make_comp_node_from_impl(this), atlas_env);

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

}

void AtlasCompNodeImpl::fini() {
    if (!m_initialized)
        return;

    m_sync_event.reset();
    m_env.fini();
    m_initialized = false;
    m_device_info = nullptr;
}

void AtlasCompNodeImpl::peer_copy_to(Impl* dest_impl, void* dest,
                                     const void* src, size_t size) {
    if (dest_impl->same_type<AtlasCompNodeImpl>()) {
        auto&& dst_env =
                static_cast<AtlasCompNodeImpl*>(dest_impl)->m_env.atlas_env();
        auto&& src_env = m_env.atlas_env();
        activate();
        if (dst_env.device == src_env.device) {
            // async d2d use SDMA which is faster than sync ctrl cpu d2d 
            MGB_ATLAS_CHECK(aclrtMemcpyAsync(dest, size, src, size,
                                             ACL_MEMCPY_DEVICE_TO_DEVICE,
                                             dst_env.stream));
        } else {
            mgb_throw(MegBrainError,
                      "Atlas does not support peer copy between differents "
                      "device.");
        }
        return;

    }
    mgb_assert(dest_impl->env().property().type == DeviceType::CPU,
               "cuda peer_copy_to only implemented for CPU");
    auto copy = [this, dest, src, size]() {
        m_env.atlas_env().activate();

#if MGB_USE_ATLAS_ASYNC_API
        auto stream = m_env.atlas_env().stream;
        MGB_ATLAS_CHECK(aclrtMemcpyAsync(dest, size, src, size,
                                         ACL_MEMCPY_DEVICE_TO_HOST,
                                         m_env.atlas_env().stream));
        MGB_ATLAS_CHECK(aclrtSynchronizeStream(stream));
#else
        MGB_ATLAS_CHECK(
                aclrtMemcpy(dest, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST));
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
    mgb_throw(MegBrainError,
              "Atlas does not support peer copy between differents "
              "device.");
}

bool AtlasCompNodeImpl::check_global_finalized() {
    if (!sd) {
        static std::atomic_flag warn_printed = ATOMIC_FLAG_INIT;
        if (!warn_printed.test_and_set()) {
            mgb_log_debug(
                    "atlas comp node method called after global finalize");
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

    void do_record() override {
        m_comp_node_impl->activate();
        auto &&env = m_comp_node_impl->m_env.atlas_env();
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
        MGB_ATLAS_CHECK(aclrtSynchronizeEvent(m_atlas_event));
    }

    double do_elapsed_time_until(EventImplHelper& end) override {
        m_comp_node_impl->activate();
        float ret = 0.0;
        MGB_ATLAS_CHECK(aclrtEventElapsedTime(&ret, m_atlas_event,
                    static_cast<EventImpl&>(end).m_atlas_event));
        return static_cast<double>(ret) * 1e-3;

    }

    void do_device_wait_by(Impl* cn_impl) override;

public:
    EventImpl(AtlasCompNodeImpl* comp_node_impl, size_t create_flags)
            : EventImplHelper(comp_node_impl, create_flags),
              m_comp_node_impl{comp_node_impl} {
        m_comp_node_impl->activate();
        MGB_ATLAS_CHECK(aclrtCreateEvent(&m_atlas_event));
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
        auto stream = imp->m_env.atlas_env().stream;
        imp->activate();
        MGB_ATLAS_CHECK(aclrtStreamWaitEvent(stream, m_atlas_event));
        return;
    }
    if (cn_impl->env().property().type == DeviceType::CPU) {
        auto waiter = [this]() {
            MGB_ATLAS_CHECK(aclrtSynchronizeEvent(m_atlas_event));
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

CompNode::Impl* AtlasCompNode::load_atlas(const Locator& locator,
                                          const Locator& locator_logical) {
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
            mgb_log_error("aclrtGetDeviceCountfaild: %s (err %d)",
                          ::megcore::atlas::get_error_str(ret),
                          static_cast<int>(ret));
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
