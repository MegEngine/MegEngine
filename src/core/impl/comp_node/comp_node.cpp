/**
 * \file src/core/impl/comp_node/comp_node.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/comp_node.h"
#include "megbrain/common.h"
#include "megbrain/comp_node/alloc.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/graph/exc_extra_info.h"

#include "./cuda/comp_node.h"
#include "./cpu/comp_node.h"
#include "./atlas/comp_node.h"
#include "./cambricon/comp_node.h"
#include "./rocm/comp_node.h"

#include <atomic>
#include <cstring>

using namespace mgb;

int CompNode::Event::sm_cpu_sync_level;

namespace {
std::atomic_flag g_default_cpu_initialized,
        g_exit_handler_registered[CompNode::NR_DEVICE_TYPE];
MGB_MUTEX g_device_map_mtx;
ThinHashMap<CompNode::DeviceType, ThinHashMap<int, int>> g_device_map;
CompNode::DeviceType g_unspec_locator_type;

const char* device_type2str(CompNode::DeviceType type) {
    using DT = CompNode::DeviceType;
    switch (type) {
        case DT::UNSPEC:
            return "xpu";
        case DT::CUDA:
            return "gpu";
        case DT::CPU:
            return "cpu";
        case DT::ATLAS:
            return "atlas";
        case DT::ROCM:
            return "rocm";
        case DT::CAMBRICON:
            return "cambricon";
        case DT::MULTITHREAD:
            return "multithread";
        default:
            mgb_throw(MegBrainError, "bad device type");
    }
}

std::string get_stream_str(int stream) {
    using S = CompNode::Stream;
    switch (stream) {
        case S::COPY:
            return "COPY";
        case S::REMOTE_SEND:
            return "REMOTE_SEND";
        case S::LOOP_SWAP:
            return "LOOP_SWAP";
        default:
            return std::to_string(stream);
    }
}

//! resolve to actual device type if type is unspec
CompNode::DeviceType resolve_device_type(CompNode::DeviceType type) {
    using DT = CompNode::DeviceType;
    if (type == DT::UNSPEC) {
        if (g_unspec_locator_type == DT::UNSPEC) {
            if (CudaCompNode::available()) {
                g_unspec_locator_type = DT::CUDA;
            } else if (ROCmCompNode::available()) {
                g_unspec_locator_type = DT::ROCM;
            } else {
                g_unspec_locator_type = DT::CPU;
            }
        }
        type = g_unspec_locator_type;
    }
    return type;
}
}  // namespace

/* ==================== EventPool ==================== */

CompNode::EventPool::EventPool(CompNode cn, size_t flags) : m_cn{cn}, m_flags{flags} {}

CompNode::EventPool::~EventPool() {
    assert_all_freed();
}

CompNode::Event* CompNode::EventPool::alloc() {
    MGB_LOCK_GUARD(m_lock);
    if (!m_free.empty()) {
        auto rst = m_free.back();
        m_free.pop_back();
        return rst;
    }
    m_allocated.push_back(m_cn.create_event(m_flags));
    return m_allocated.back().get();
}

void CompNode::EventPool::free(CompNode::Event* ev) {
    MGB_LOCK_GUARD(m_lock);
    m_free.push_back(ev);
}

void CompNode::EventPool::assert_all_freed() {
    mgb_assert(m_allocated.size() == m_free.size());
}

/* ==================== CompNodeImplHelper ==================== */
void CompNodeImplHelper::log_comp_node_created(
        const Locator& locator, const Locator& locator_logical) {
    mgb_log_debug(
            "create CompNode %s from logical %s", locator.to_string().c_str(),
            locator_logical.to_string().c_str());
}

/* ==================== Locator ==================== */

CompNode::Locator CompNode::Locator::parse(const std::string& id) {
    auto err = [&]() {
        mgb_throw(MegBrainError, "invalid comp node id: %s", id.c_str());
    };
    if (id.size() < 3)
        err();
    // current parsing location
    const char* ptr = id.data();
    if (id == "cpu:default") {
        return {DeviceType::CPU, DEVICE_CPU_DEFAULT, {0}};
    }
    if (!strncmp(ptr, "multithread:default", 19)) {
        //! the multithread default compnode string like "multithread:default:x"
        if (id.size() > 20) {
            ptr += 20;
            int nr_thread = std::stoi(ptr);
            return {DeviceType::MULTITHREAD, DEVICE_MULTITHREAD_DEFAULT, {nr_thread}};
        } else {
            err();
        }
    }

    DeviceType dev_type;

    // parse dev_type
            if (ptr[0] == 'a') {
        if (strncmp(ptr, "atlas", 5)) {
            err();
        }
        dev_type = DeviceType::ATLAS;
        ptr += 5;
    }
    else if (ptr[0] == 'r') {
        if (strncmp(ptr, "rocm", 4)) {
            err();
        }
        dev_type = DeviceType::ROCM;
        ptr += 4;
    } else if (ptr[2] == 'm') {
        if (strncmp(ptr, "cambricon", 9)) {
            err();
        }
        dev_type = DeviceType::CAMBRICON;
        ptr += 9;
    } else if (ptr[0] == 'm') {
            if (strncmp(ptr, "multithread", 11)) {
                err();
            }
            dev_type = DeviceType::MULTITHREAD;
            ptr += 11;
    }
    else {
        if (ptr[1] != 'p' || ptr[2] != 'u') {
            err();
        }
        if (ptr[0] == 'c') {
            dev_type = DeviceType::CPU;
        } else if (ptr[0] == 'g') {
            dev_type = DeviceType::CUDA;
        } else {
            dev_type = DeviceType::UNSPEC;
            if (ptr[0] != 'x')
                err();
        }

        ptr += 3;
    }

    int num_dev;
    auto parse_int = [&]() {
        int ret = 0;
        while (*ptr >= '0' && *ptr <= '9') {
            ret = ret * 10 + (*ptr) - '0';
            ++ptr;
        }
        return ret;
    };

    if (*ptr == 'x' || (dev_type == DeviceType::UNSPEC && !*ptr)) {
        num_dev = -1;
        if (*ptr)
            ++ptr;
    } else {
        if (!*ptr)
            err();
        num_dev = parse_int();
    }
    if (*ptr) {
        if (*ptr != ':')
            err();
        ++ptr;
        if (!*ptr)
            err();
    }
    int num_stream = parse_int();
    if (*ptr)
        err();
    //! multi thread with thread number(num_stream) being zero is illegal
    if (dev_type == DeviceType::MULTITHREAD) {
        if (num_dev == 0) {
            err();
        }
        //! num_steam store the nr_thread
        std::swap(num_dev, num_stream);
    }

    return {dev_type, num_dev, {num_stream}};
}

void CompNode::Locator::set_device_map(DeviceType type, int from, int to) {
    mgb_assert(to >= 0);

    MGB_LOCK_GUARD(g_device_map_mtx);
    g_device_map[type][from] = to;
}

void CompNode::Locator::set_unspec_device_type(DeviceType type) {
    mgb_assert(type != DeviceType::UNSPEC);
    if (type != DeviceType::CPU && type != DeviceType::CUDA) {
        mgb_log_warn(
                "to resolve unspec device type as one except "
                "CUDA and CPU may lead to unknown problems.");
    }
    g_unspec_locator_type = type;
}

CompNode::Locator CompNode::Locator::to_physical() const {
    mgb_assert(stream >= 0);
    DeviceType type_physical;
    int device_physical;
    int stream_physical;

    type_physical = resolve_device_type(type);
    device_physical = device;
    stream_physical = stream;

    if ((MGB_HAVE_THREAD) ||
        CompNode::contain_flag(type_physical, Flag::SUPPORT_NO_THREAD)) {
#if MGB_THREAD_SAFE
        MGB_LOCK_GUARD(g_device_map_mtx);
#endif
        auto&& cur_dmap = g_device_map[type_physical];
        auto iter = cur_dmap.find(device);
        if (iter != cur_dmap.end())
            device_physical = iter->second;

        if (device_physical == -1)
            device_physical = 0;
    } else {
        // we map all logical locators to cpu0:1023 except cpu:default,
        // when thread is disabled.
        type_physical = DeviceType::CPU;
        device_physical = DEVICE_CPU_DEFAULT;
        stream_physical = 0;

        if (device != DEVICE_CPU_DEFAULT) {
            device_physical = 0;
            stream_physical = 1023;
        }
    }
    return {type_physical, device_physical, {stream_physical}};
}

std::string CompNode::Locator::to_string() const {
    if (device == DEVICE_CPU_DEFAULT) {
        return "cpu:default";
    } else if (device == DEVICE_MULTITHREAD_DEFAULT) {
        std::string ret = "multithread:default:";
        ret.append(get_stream_str(stream));
        return ret;
    } else if (type == DeviceType::MULTITHREAD) {
        std::string ret("multithread");
        ret.append(get_stream_str(stream)).append(":").append(get_stream_str(device));
        return ret;
    }
    char numstr[32];
    if (device == -1) {
        numstr[0] = 'x';
        numstr[1] = 0;
    } else {
        mgb_assert(device >= 0);
        sprintf(numstr, "%d", device);
    }
    std::string ret(device_type2str(type));
    ret.append(numstr).append(":").append(get_stream_str(stream));
    return ret;
}

/* ==================== CompNodeDepedentObject ==================== */

//! alignas is not required, it does not affect the result and almost does not
//! affect performance, macro \c MGB_MAX_SECTION_ALIGNMENT is intended for
//! environments that do not provide large alignment support.
#if defined(MGB_MAX_SECTION_ALIGNMENT) && MGB_MAX_SECTION_ALIGNMENT < 64
struct comp_node_detail::DepedentObjList::StaticInfo {
#else
// use a large alignment to avoid cache line pollution
struct alignas(64) comp_node_detail::DepedentObjList::StaticInfo {
#endif
    Spinlock lock;
    DepedentObjList* head;
};
comp_node_detail::DepedentObjList::StaticInfo
        comp_node_detail::DepedentObjList::sm_info;

class comp_node_detail::DepedentObjList::Sentinel final
        : public comp_node_detail::DepedentObjList {
    std::shared_ptr<void> callback() override { return {}; }

public:
    Sentinel() { init_list(); }

    void init_list() {
        sm_info.head = this;
        m_next = m_prev = this;
    }

    static Sentinel* get() {
        // no need to delete; use static storage to avoid its dtor being invoked
        static std::aligned_storage_t<sizeof(Sentinel), alignof(Sentinel)> storage;
        static Sentinel* ptr = new (&storage) Sentinel{};
        return ptr;
    }
};

void comp_node_detail::DepedentObjList::add(DepedentObjList* ptr) {
    MGB_LOCK_GUARD(sm_info.lock);
    // if this becomes slow (which I do not think is likely to happen), we can
    // try a lock-free list implementation
    Sentinel::get();
    auto a = sm_info.head, b = a->m_next;
    // insert and delete from head, so items added last can be deleted first
    link(a, ptr);
    link(ptr, b);
}

void comp_node_detail::DepedentObjList::remove(DepedentObjList* ptr) {
    if (ptr->m_prev) {
        MGB_LOCK_GUARD(sm_info.lock);
        link(ptr->m_prev, ptr->m_next);
    }
}

void comp_node_detail::DepedentObjList::invoke_callback_and_clean() {
    SmallVector<std::shared_ptr<void>> refholds;
    {
        MGB_LOCK_GUARD(sm_info.lock);
        auto st = Sentinel::get();
        for (DepedentObjList *i = st->m_next, *inext; i != st; i = inext) {
            inext = i->m_next;
            i->m_prev = i->m_next = nullptr;
            auto ref = i->callback();
            if (ref.use_count() == 1) {
                // clear them later
                refholds.emplace_back(std::move(ref));
            }
        }
        st->init_list();
    }

    // call dtor without holding the lock
    refholds.clear();
}

void CompNodeDepedentObject::check_not_finalized() const {
    mgb_throw_if(
            m_state == 2, InternalError,
            "method called on CompNode-depdendent object after CompNode "
            "finalization");
}

std::shared_ptr<void> CompNodeDepedentObject::callback() {
    mgb_assert(!m_state);
    std::shared_ptr<void> ref;
    m_state = 1;
#if MGB_ENABLE_EXCEPTION
    std::exception_ptr ptr;
#endif
    MGB_TRY { ref = on_comp_node_finalize(); }
    MGB_CATCH_ALL_EXCEPTION("comp node finalize", ptr);
    m_state = 2;
    return ref;
}

/* ==================== CompNode ==================== */

void CompNode::activate() const {
    static_cast<Impl*>(m_impl)->env().activate();
}

void CompNode::set_prealloc_config(
        size_t alignment, size_t min_req, size_t max_overhead, double growth_factor,
        DeviceType device_type) {
    switch (device_type) {
        case DeviceType::CUDA:
            CudaCompNode::set_prealloc_config(
                    alignment, min_req, max_overhead, growth_factor);
            break;
        default:
            mgb_log_warn("unsupported device type for set_prealloc_config");
    };
}

size_t CompNode::get_compute_capability(int dev, DeviceType device_type) {
    switch (device_type) {
        case DeviceType::CUDA:
            return CudaCompNode::get_compute_capability(dev);
        default:
            mgb_log_warn("unsupport device type for get_compute_capability");
            return 0;
    };
}

void* CompNode::alloc_device(size_t size) const {
    auto ret = m_impl->alloc_device(size);
    static_cast<Impl*>(m_impl)->env().on_mem_event(size, true, ret);
    return ret;
}

void CompNode::free_device(void* ptr) const {
    static_cast<Impl*>(m_impl)->env().on_mem_event(0, true, ptr);
    return m_impl->free_device(m_impl, ptr);
}

void* CompNode::alloc_host(size_t size) const {
    auto ret = m_impl->alloc_host(size);
    static_cast<Impl*>(m_impl)->env().on_mem_event(size, false, ret);
    return ret;
}

void CompNode::free_host(void* ptr) const {
    static_cast<Impl*>(m_impl)->env().on_mem_event(0, false, ptr);
    return m_impl->free_host(m_impl, ptr);
}

std::unique_ptr<MegBrainError> CompNode::check_async_error() const {
#if MGB_NEED_MEGDNN_ASYNC_ERROR
    auto&& env = CompNodeEnv::from_comp_node(*this);
    if (!env.has_user_data<MegDNNHandle>()) {
        // comp nodes like fpga do not have megdnn handle
        return nullptr;
    }

    auto ptr = MegDNNHandle::get(env).async_error_info_devptr();
    if (!ptr) {
        // this device type does not need async error report
        return nullptr;
    }

    megcore::AsyncErrorInfo error_info;
    copy_to_host(&error_info, ptr, sizeof(error_info));
    sync();
    if (!error_info.nr_error)
        return nullptr;

    // clear previous error
    megcore::AsyncErrorInfo zero_info{0, nullptr, "", {0, 0, 0, 0}};
    copy_to_device(ptr, &zero_info, sizeof(zero_info));
    sync();

    // throw exception
    mgb_assert(error_info.tracker_ptr, "error tracker unavailable");
    return cg::OperatorNodeExcExtraInfo::ExcMaker{
            static_cast<cg::OperatorNodeBase*>(error_info.tracker_ptr)}
            .make_unique<MegBrainError>(
                    ssprintf(
                            "%u async error%s recorded; first msg: ",
                            error_info.nr_error, error_info.nr_error > 1 ? "s" : "") +
                    ssprintf(
                            error_info.msg, error_info.msg_args[0],
                            error_info.msg_args[1], error_info.msg_args[2],
                            error_info.msg_args[3]));
#else
    return nullptr;
#endif
}

CompNode::DeviceType CompNode::device_type() const {
    return static_cast<Impl*>(m_impl)->env().property().type;
}

CompNode CompNode::load(
        const Locator& locator_physical, const Locator& locator_logical) {
    auto phy_device_type_num = static_cast<size_t>(locator_physical.type);
    mgb_assert(
            phy_device_type_num < NR_DEVICE_TYPE,
            "bad device type; maybe new device type is added but "
            "NR_DEVICE_TYPE is not modified?");
    if (!g_default_cpu_initialized.test_and_set()) {
        // to ensure default_cpu comp node is initialized first, so destructed
        // after all other comp nodes
        default_cpu();
    }

    CompNode ret;
    switch (locator_physical.type) {
        case DeviceType::CUDA:
            ret = CudaCompNode::load_cuda(locator_physical, locator_logical);
            break;
        case DeviceType::MULTITHREAD:
        case DeviceType::CPU:
            ret = CpuCompNode::load_cpu(locator_physical, locator_logical);
            break;
        case DeviceType::ATLAS:
            ret = AtlasCompNode::load_atlas(locator_physical, locator_logical);
            break;
        case DeviceType::ROCM:
            ret = ROCmCompNode::load_rocm(locator_physical, locator_logical);
            break;
        case DeviceType::CAMBRICON:
            ret = CambriconCompNode::load_cambricon(locator_physical, locator_logical);
            break;
        default:
            mgb_throw(MegBrainError, "bad device type");
    }

    if (!g_exit_handler_registered[phy_device_type_num].test_and_set()) {
        // register atexit after comp node has been loaded; so ::finalze() can
        // be called before other libraries' exit handler
        auto err = atexit(&CompNode::finalize);
        mgb_assert(!err, "failed to register CompNode::finalize at exit");
    }

    return ret;
}

void CompNode::finalize() {
#if MGB_CUDA && defined(WIN32)
    //! FIXME: windows cuda driver shutdown before call atexit function even
    //! register atexit function after init cuda driver! as a workround recovery
    //! resource by OS temporarily, may need remove this after upgrade cuda
    //! runtime
    return;
#endif
    comp_node_detail::DepedentObjList::invoke_callback_and_clean();
    CudaCompNode::finalize();
    CpuCompNode::finalize();
    ROCmCompNode::finalize();
    CambriconCompNode::finalize();
    AtlasCompNode::finalize();
}

void CompNode::try_coalesce_all_free_memory() {
    CudaCompNode::try_coalesce_all_free_memory();
    ROCmCompNode::try_coalesce_all_free_memory();
    CambriconCompNode::try_coalesce_all_free_memory();
}

void CompNode::sync_all() {
    CudaCompNode::sync_all();
    CpuCompNode::sync_all();
    ROCmCompNode::sync_all();
    CambriconCompNode::sync_all();
    AtlasCompNode::sync_all();
}

void CompNode::foreach (thin_function<void(CompNode)> callback) {
    CudaCompNode::foreach (callback);
    CpuCompNode::foreach (callback);
    ROCmCompNode::foreach (callback);
    CambriconCompNode::foreach (callback);
    AtlasCompNode::foreach (callback);
}

size_t CompNode::get_device_count(DeviceType type, bool warn) {
    switch (resolve_device_type(type)) {
        case DeviceType::CUDA:
            return CudaCompNode::get_device_count(warn);
        case DeviceType::MULTITHREAD:
        case DeviceType::CPU:
            return CpuCompNode::get_device_count();
        case DeviceType::ROCM:
            return ROCmCompNode::get_device_count();
        case DeviceType::CAMBRICON:
            return CambriconCompNode::get_device_count();
        case DeviceType::ATLAS:
            return AtlasCompNode::get_device_count();
        default:
            mgb_throw(MegBrainError, "bad device type");
    }
}

bool CompNode::contain_flag(DeviceType device_type, Flag flag) {
    Flag cn_flag{};
    switch (resolve_device_type(device_type)) {
        case DeviceType::CUDA:
            cn_flag = CudaCompNode::sm_flag;
            break;
        case DeviceType::MULTITHREAD:
        case DeviceType::CPU:
            cn_flag = CpuCompNode::sm_flag;
            break;
        case DeviceType::ROCM:
            cn_flag = ROCmCompNode::sm_flag;
            break;
        case DeviceType::CAMBRICON:
            cn_flag = CambriconCompNode::sm_flag;
            break;
        case DeviceType::ATLAS:
            cn_flag = AtlasCompNode::sm_flag;
            break;
        default:
            mgb_throw(MegBrainError, "unexpected device type");
    }
    return static_cast<bool>(cn_flag & flag);
}

CompNode CompNode::change_stream(int dest_stream) const {
    mgb_assert(m_impl);
    auto loc = m_impl->locator(), loc_logical = m_impl->locator_logical();
    loc.stream = loc_logical.stream = dest_stream;
    return load(loc, loc_logical);
}

std::unique_ptr<CompNodeSeqRecorder> CompNode::ImplBase::create_seq_recorder(
        cg::ComputingGraph*) {
    return {};
}

size_t CompNode::ImplBase::get_mem_padding() {
    return 0;
}

void CompNode::ImplBase::add_callback(megdnn::thin_function<void()>&&) {
    mgb_throw(
            MegBrainError,
            "Unsupported add callback to "
            "comp node %s",
            locator().to_string().c_str());
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
