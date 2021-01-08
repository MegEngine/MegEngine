/**
 * \file src/core/impl/comp_node/cpu/comp_node.cpp
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
#include "megbrain/system.h"
#include "megbrain/utils/arith_helper.h"
#include "megbrain/utils/thread.h"
#include "megbrain/utils/timer.h"
#include "megbrain/utils/thread_pool.h"
#include "megbrain/common.h"

#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <atomic>

#include <stdlib.h>
#ifndef __APPLE__
#include <malloc.h>
#endif

using namespace mgb;

namespace {
bool enable_affinity = false;
using Task = CompNodeEnv::CpuEnv::Task;
using MultiThreadingTask = megcore::CPUDispatcher::MultiThreadingTask;

struct TaskElem {
    //! the task to be execute
    MultiThreadingTask task;
    //! number of the parallelism
    size_t nr_parallelism;
};
}  // anonymous namespace

using CpuCompNodeImpl = CpuCompNode::CompNodeImpl;

void CpuCompNode::CpuDispatchableBase::add_callback(Task&& task) {
    dispatch(std::move(task));
}

class CpuCompNode::WorkerQueue final
        : public AsyncQueueSC<TaskElem, WorkerQueue> {
    const Locator m_locator;
    ThreadPool* m_thread_pool = nullptr;

    void on_async_queue_worker_thread_start() override {
        mgb_assert(m_locator.device >= 0);
        if (enable_affinity) {
#if !defined(ANDROID) && !defined(__ANDROID__)
            sys::set_cpu_affinity({m_locator.device});
#endif
        }
        sys::set_thread_name(m_locator.to_string());
    }

    void on_sync_all_task_finish() override {
        if (m_thread_pool) {
            m_thread_pool->deactive();
        }
    }

public:
    class DispatcherImpl;

    explicit WorkerQueue(Locator locator) : m_locator(locator) {}

    void attach_thread_pool(ThreadPool* thread_pool) {
        m_thread_pool = thread_pool;
    }

    void process_one_task(const TaskElem& task_elem) {
        if (m_thread_pool) {
            m_thread_pool->add_task(task_elem);
        } else {
            for (size_t i = 0; i < task_elem.nr_parallelism; i++) {
                task_elem.task(i, 0);
            }
        }
    }

    int nr_threads() {
        return m_thread_pool ? m_thread_pool->nr_threads() : 1_z;
    }

    ThreadPool* get_thread_pool() { return m_thread_pool; }
};

class CpuCompNode::SeqRecorderImpl final : public CompNodeSeqRecorder {
    using CpuEnv = CompNodeEnv::CpuEnv;
    bool m_fake_exec = false, m_synchronized = false, m_stopped = false,
         m_first_replay = true;
    SeqRecorderImpl** const m_self_pointer;

    std::vector<TaskElem> m_tasks;
    ThreadPool* m_thread_pool = nullptr;
    const CompNode m_record_compnode;
    /*!
     * \brief use to check the all ther recording tasks are its self CompNode
     * related task, void hook other CompNode related task to the recorder.
     */
    void check_the_same_comp_node(const CompNode& comp_node) const;
public:
    SeqRecorderImpl(SeqRecorderImpl** self_pointer, ThreadPool* thread_pool,
                    const CompNode& comp_node)
            : m_self_pointer{self_pointer},
              m_thread_pool{thread_pool},
              m_record_compnode{comp_node} {
        mgb_assert(!*m_self_pointer);
        *m_self_pointer = this;
    }

    ~SeqRecorderImpl() {
        if (*m_self_pointer) {
            stop();
        }
    }

    void enter_fake_exec(const CompNode&  comp_node) override {
        check_the_same_comp_node(comp_node);
        mgb_assert(!m_stopped && !m_fake_exec);
        m_fake_exec = true;
    }

    void exit_fake_exec(const CompNode&  comp_node) override {
        check_the_same_comp_node(comp_node);
        mgb_assert(!m_stopped && m_fake_exec);
        mgb_assert(m_tasks.empty());
        m_fake_exec = false;
        m_synchronized = false;
    }

    void stop(const CompNode& comp_node = {}) override {
        check_the_same_comp_node(comp_node);
        mgb_assert(*m_self_pointer == this);
        mgb_assert(!m_fake_exec);
        *m_self_pointer = nullptr;
        m_stopped = true;
    }

    void replay() override {
        mgb_assert(m_stopped, "not stopped yet");
        if (m_first_replay) {
            // check that dispatch is not called from tasks
            mgb_assert(!*m_self_pointer,
                       "no other seq recorder should be created before first "
                       "replay");
            *m_self_pointer = this;
        }
        MGB_TRY {
            if (m_thread_pool) {
                m_thread_pool->active();
                for (auto&& i : m_tasks) {
                    m_thread_pool->add_task(i);
                }
                m_thread_pool->deactive();
            }else{
                for (auto&& task : m_tasks) {
                    for(size_t i=0; i<task.nr_parallelism;i++){
                        task.task(i, 0);
                    }
                }
            }
        }
        MGB_FINALLY({
            if (m_first_replay) {
                stop();
                m_first_replay = false;
            }
        });
    }

    void on_alloc(const CompNode& comp_node) {
        check_the_same_comp_node(comp_node);
        mgb_assert(m_fake_exec,
                   "alloc is disallowed during comp node seq recording");
    }

    void on_free(const CompNode& comp_node) {
        check_the_same_comp_node(comp_node);
        mgb_assert(m_fake_exec,
                   "free is disallowed during comp node seq recording");
    }

    void on_sync(const CompNode& comp_node) {
        check_the_same_comp_node(comp_node);
        m_synchronized = true;
    }

    void dispatch(Task&& task, const CompNode& comp_node) {
        mgb_assert(!m_synchronized,
                   "no more tasks should be dispatched after synchronization");
        auto kern = [task](size_t, size_t) { task(); };
        dispatch_allow_after_sync({std::move(kern), static_cast<size_t>(1_z)},
                                  comp_node);
    }
    void dispatch_allow_after_sync(Task&& task, const CompNode& comp_node) {
        check_the_same_comp_node(comp_node);
        mgb_assert(!m_stopped,
                   "dispatch should not be called after recording is stopped");
        if (!m_fake_exec) {
            auto kern = [task](size_t, size_t) { task(); };
            m_tasks.push_back({std::move(kern), static_cast<size_t>(1_z)});
        }
    }
    void dispatch(TaskElem&& task_elem, const CompNode& comp_node) {
        mgb_assert(!m_synchronized,
                   "no more tasks should be dispatched after synchronization");
        dispatch_allow_after_sync(std::move(task_elem), comp_node);
    }
    void dispatch_allow_after_sync(TaskElem&& task_elem,
                                   const CompNode& comp_node) {
        check_the_same_comp_node(comp_node);
        mgb_assert(!m_stopped,
                   "dispatch should not be called after recording is stopped");
        if (!m_fake_exec) {
            m_tasks.push_back(task_elem);
        }
    }
    size_t nr_threads(const CompNode& comp_node) {
        check_the_same_comp_node(comp_node);
        return m_thread_pool ? m_thread_pool->nr_threads() : 1_z;
    }

    ThreadPool* get_thread_pool() { return m_thread_pool; }
};

class CpuCompNode::CompNodeImpl final: public CpuDispatchableBase {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

    //! used during comp node seq rec
    class CompSeqRecEventImpl;
    class CpuEventImpl;

//! TODO: because the x-code bug, see
//! https://github.com/tensorflow/tensorflow/issues/18356
//! thread local is no support on IOS,
//! When update x-xode, this code should be deleted
#ifndef IOS
    static thread_local SeqRecorderImpl* sm_cur_recorder;
#else
    SeqRecorderImpl* sm_cur_recorder = nullptr;
#endif

    std::shared_ptr<WorkerQueue> m_worker_queue;
    Locator m_locator, m_locator_logical;
    std::unique_ptr<ThreadPool> m_thread_pool;

    //! ptr to default cpu, only used by check_global_finalized
    static CpuCompNodeImpl *sm_default_cpu_comp_node_ptr;

    //! return whether global finalized, and print warning in such case
    inline bool check_global_finalized(const char* reason);

    static void static_free_device(ImplBase* self, void* ptr) {
        static_cast<CompNodeImpl*>(self)->free_device(ptr);
    }

    static void static_free_host(ImplBase* self, void* ptr) {
        static_cast<CompNodeImpl*>(self)->free_host(ptr);
    }

    public:
        CompNodeImpl(const Locator& locator, const Locator& locator_logical,
                     const std::shared_ptr<WorkerQueue>& worker_queue);
        ~CompNodeImpl() {
            if (sm_cur_recorder) {
                sm_cur_recorder->stop();
            }
            if (m_worker_queue) {
                // synchronize before fini
                m_worker_queue->wait_all_task_finish();
            }
            m_env.fini();
            if (m_worker_queue) {
                // wait for new kernels dispatched in fini() (like free_device())
                m_worker_queue->wait_all_task_finish();
            }
            if (this == sm_default_cpu_comp_node_ptr) {
                // This should only happen in global library .fini. We clear
                // sm_default_cpu_comp_node_ptr so check_global_finalized() can
                // work correctly
                sm_default_cpu_comp_node_ptr = nullptr;
            }
        }

        ThreadPool* get_thread_pool() const { return m_thread_pool.get(); }

        void* mgb_aligned_alloc(size_t size) {
            auto alignment = get_mem_addr_alignment();
#ifdef WIN32
            return _aligned_malloc(size, alignment);
#elif defined(__ANDROID__) || defined(ANDROID)
            return memalign(alignment, size);
#else
            void *ptr = nullptr;
            auto err = posix_memalign(&ptr, alignment, size);
            mgb_assert(!err, "failed to malloc %zubytes with align %zu",
                    size, alignment);
            return ptr;
#endif
        }

        static void mgb_aligned_free(void* ptr) {
#ifdef WIN32
                _aligned_free(ptr);
#else
                ::free(ptr);
#endif
        }

        void* alloc_device(size_t size) override {
            if (sm_cur_recorder) {
                sm_cur_recorder->on_alloc(this);
            }
            return mgb_aligned_alloc(size);
        }

        void free_device(void *ptr) {
            if (sm_cur_recorder || check_global_finalized("free_device()")) {
                mgb_aligned_free(ptr);
                if (sm_cur_recorder) {
                    sm_cur_recorder->on_free(this);
                }
                return;
            } else {
                auto do_free = [ptr]() {
                    mgb_aligned_free(ptr);
                };
                m_env.cpu_env().dispatch(do_free);
            }
        }

        void *alloc_host(size_t size) override {
            if (m_worker_queue) {
                m_worker_queue->check_exception();
            }
            return mgb_aligned_alloc(size);
        }

        void free_host(void *ptr) {
            if (check_global_finalized("free_host()")) {
                mgb_aligned_free(ptr);
                return;
            }
            if (m_worker_queue) {
                m_worker_queue->check_exception();
            }
            return mgb_aligned_free(ptr);
        }

        void copy_to_host(void *host_ptr,
                const void *device_ptr, size_t size) override {
            if (m_worker_queue) {
                m_worker_queue->check_exception();
            }
            // use lambda capture to avoid memory allocation in std::bind
            auto do_copy = [host_ptr, device_ptr, size]() {
                std::memcpy(host_ptr, device_ptr, size);
            };
            m_env.cpu_env().dispatch(do_copy);
        }

        void copy_to_device(void *device_ptr,
                const void *host_ptr, size_t size) override {
            if (m_worker_queue) {
                m_worker_queue->check_exception();
            }
            // use lambda capture to avoid memory allocation in std::bind
            auto do_copy = [device_ptr, host_ptr, size]() {
                std::memcpy(device_ptr, host_ptr, size);
            };
            m_env.cpu_env().dispatch(do_copy);
        }

        void peer_copy_to(
                Impl *dest_impl, void *dest,
                const void *src, size_t size) override {
            if (!dest_impl->same_type<CpuCompNode::CompNodeImpl>()) {
                if (dest_impl->env().property().type == DeviceType::ATLAS) {
#if MGB_ATLAS
                    dest_impl->copy_to_device(dest, src, size);
                    return;
#else
                    mgb_throw(MegBrainError,
                              "Atlas comp_node used but "
                              "MGB_ATLAS not enabled");
#endif
                } else if (dest_impl->env().property().type ==
                           DeviceType::CAMBRICON) {
#if MGB_CAMBRICON
                    dest_impl->copy_to_device(dest, src, size);
                    return;
#else
                    mgb_throw(MegBrainError,
                              "Cambricon comp_node used but "
                              "MGB_CAMBRICON not enabled");
#endif

                } else {
                    mgb_assert(locator().device == Locator::DEVICE_CPU_DEFAULT,
                               "currently only peer copy from default cpu comp "
                               "nodes "
                               "is implemented");
                }
            }
            dest_impl->copy_to_device(dest, src, size);
        }

        size_t get_mem_addr_alignment() override {
            return m_env.property().mem_alignment;
        }

        std::unique_ptr<Event> create_event(size_t flags) override;

        void sync() override {
            if (sm_cur_recorder) {
                sm_cur_recorder->on_sync(this);
            } else if (m_worker_queue) {
                m_worker_queue->wait_all_task_finish();
            }
            if (m_thread_pool) {
                m_thread_pool->deactive();
            }
        }

        void dispatch(Task &&task) override {
            m_env.cpu_env().dispatch(std::move(task));
        }

        MemNode mem_node() override {
            // TODO: numa nodes
            return get_host_cpu_mem_node();
        }

        std::pair<size_t, size_t> get_mem_status_bytes() override {
            return sys::get_ram_status_bytes();
        }

        Locator locator() override {
            return m_locator;
        }

        Locator locator_logical() override {
            return m_locator_logical;
        }

        std::unique_ptr<CompNodeSeqRecorder> create_seq_recorder(
                cg::ComputingGraph*) override {
            return std::make_unique<SeqRecorderImpl>(&sm_cur_recorder,
                                                     m_thread_pool.get(), this);
        }

        //! current sequence recorder of this thread
#ifndef IOS
        static SeqRecorderImpl* cur_recorder() { return sm_cur_recorder; }
#else
        SeqRecorderImpl* cur_recorder() { return sm_cur_recorder; }
#endif

        void add_callback(Task &&task) override {
            if (!check_global_finalized("add_callback()")) {
                CpuDispatchableBase::add_callback(std::move(task));
            } else {
                task();
            }
        }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(CpuCompNodeImpl);
CpuCompNodeImpl* CpuCompNodeImpl::sm_default_cpu_comp_node_ptr;
#ifndef IOS
thread_local CpuCompNode::SeqRecorderImpl* CpuCompNodeImpl::sm_cur_recorder =
        nullptr;
#endif

void CpuCompNode::SeqRecorderImpl::check_the_same_comp_node(
        const CompNode& comp_node) const {
    if (mgb_unlikely(comp_node.valid())) {
        mgb_assert(m_record_compnode == comp_node,
                   "CompNode %s can't hook in CompNode %s when recording\n",
                   comp_node.locator().to_string().c_str(),
                   m_record_compnode.locator().to_string().c_str());
    }
}

//! implementation of CPUDispatcher that is passed to megdnn via megcore
class CpuCompNode::WorkerQueue::DispatcherImpl final: public CPUDispatcher {
    std::atomic_size_t m_nr_task{0};
    std::shared_ptr<WorkerQueue> m_queue;
    CpuCompNode::CompNodeImpl* const m_comp_node;

public:
    DispatcherImpl(const std::shared_ptr<WorkerQueue>& queue,
                   CpuCompNode::CompNodeImpl* comp_node)
            : m_queue{queue}, m_comp_node{comp_node} {}

    void dispatch(Task&& task) override {
        if (auto recorder = m_comp_node->cur_recorder()) {
            recorder->dispatch(std::move(task), m_comp_node);
        } else {
            m_nr_task.fetch_add(1, std::memory_order_relaxed);
            auto kern = [task](size_t, size_t) { task(); };
            m_queue->add_task({kern, static_cast<size_t>(1_z)});
        }
    }

    void dispatch(MultiThreadingTask&& task, size_t parallelism) override {
        if (auto recorder = m_comp_node->cur_recorder()) {
            recorder->dispatch({std::move(task), parallelism}, m_comp_node);
        } else {
            m_nr_task.fetch_add(1, std::memory_order_relaxed);
            m_queue->add_task({std::move(task), parallelism});
        }
    }

    void sync() override {
        if (auto recorder = m_comp_node->cur_recorder()) {
            recorder->on_sync(m_comp_node);
        } else {
            m_queue->wait_all_task_finish();
        }
    }

    size_t nr_threads() override {
        if (auto recorder = m_comp_node->cur_recorder()) {
            return recorder->nr_threads(m_comp_node);
        } else {
            return m_queue->nr_threads();
        }
    }

    size_t get_nr_dispatched_tasks() const override { return m_nr_task; }

    void set_affinity(AffinityCallBack&& affinity_cb) override {
        auto thread_pool = m_queue->get_thread_pool();
        if (thread_pool) {
            thread_pool->set_affinity(affinity_cb);
        } else {
            auto affinity_run = [affinity_cb](size_t, size_t) {
                affinity_cb(0);
            };
            m_queue->add_task({affinity_run, 1_z});
        }
    }
};

//! implementation of InplaceCPUDispatcher
class InplaceCPUDispatcher final : public CPUDispatcher {
    std::atomic_size_t m_nr_task{0};
    ThreadPool* m_thread_pool = nullptr;
    CpuCompNode::CompNodeImpl* const m_comp_node;

public:
    InplaceCPUDispatcher(CpuCompNode::CompNodeImpl* comp_node,
                         ThreadPool* thread_pool = nullptr)
            : m_thread_pool(thread_pool), m_comp_node(comp_node) {}

    void dispatch(Task&& task) override {
        if (auto recorder = m_comp_node->cur_recorder()) {
            recorder->dispatch(std::move(task), m_comp_node);
        } else if (m_thread_pool) {
            m_nr_task.fetch_add(1, std::memory_order_relaxed);
            auto kern = [task](size_t, size_t) { task(); };
            m_thread_pool->add_task({kern, static_cast<size_t>(1_z)});
        } else {
            m_nr_task.fetch_add(1, std::memory_order_relaxed);
            task();
        }
    }

    void dispatch(MultiThreadingTask&& task, size_t parallelism) override {
        if (auto recorder = m_comp_node->cur_recorder()) {
            recorder->dispatch({std::move(task), parallelism}, m_comp_node);
        } else if (m_thread_pool) {
            m_nr_task.fetch_add(1, std::memory_order_relaxed);
            m_thread_pool->add_task({task, parallelism});
        }else{
            m_nr_task.fetch_add(1, std::memory_order_relaxed);
            for(size_t i=0; i<parallelism;i++){
                task(i, 0);
            }
        }
    }

    size_t nr_threads() override {
        return m_thread_pool ? m_thread_pool->nr_threads() : 1_z;
    }

    void sync() override {
        if (auto recorder = m_comp_node->cur_recorder()) {
            recorder->on_sync(m_comp_node);
        } else if (m_thread_pool) {
            m_thread_pool->deactive();
        }
    }

    size_t get_nr_dispatched_tasks() const override { return m_nr_task; }

    void set_affinity(AffinityCallBack&& affinity_cb) override {
        if (auto recorder = m_comp_node->cur_recorder()) {
            recorder->get_thread_pool()->set_affinity(affinity_cb);
        } else if (m_thread_pool) {
            m_thread_pool->set_affinity(affinity_cb);
        }else{
            affinity_cb(0);
        }
    }
};

CpuCompNode::CompNodeImpl::CompNodeImpl(
        const Locator& locator, const Locator& locator_logical,
        const std::shared_ptr<WorkerQueue>& worker_queue)
        : CpuDispatchableBase(static_free_device, static_free_host),
          m_worker_queue{worker_queue},
          m_locator(locator),
          m_locator_logical(locator_logical) {
    auto cn = make_comp_node_from_impl(this);
    if (locator.type == DeviceType::MULTITHREAD) {
        m_thread_pool = std::unique_ptr<ThreadPool>(
                new ThreadPool(static_cast<size_t>(locator.nr_threads)));
        mgb_assert(m_thread_pool, "ThradPool create failed");
    }

    if (locator.type == DeviceType::CPU) {
        if (locator.device == Locator::DEVICE_CPU_DEFAULT) {
            sm_default_cpu_comp_node_ptr = this;
            m_env.init_cpu({std::make_shared<InplaceCPUDispatcher>(this)}, cn);
        } else {
            m_env.init_cpu({std::make_shared<WorkerQueue::DispatcherImpl>(
                                   m_worker_queue, this)},
                           cn);
        }
    } else if (locator.type == DeviceType::MULTITHREAD) {
        if (locator.device == Locator::DEVICE_MULTITHREAD_DEFAULT) {
            m_env.init_cpu({std::make_shared<InplaceCPUDispatcher>(
                                   this, m_thread_pool.get())},
                           cn);
        } else {
            m_worker_queue->attach_thread_pool(m_thread_pool.get());
            m_env.init_cpu({std::make_shared<WorkerQueue::DispatcherImpl>(
                                   m_worker_queue, this)},
                           cn);
        }
    }
}

class CpuCompNodeImpl::CompSeqRecEventImpl final
        : public CpuDispatchableBase::EventImpl {
    void do_record() override {
        auto impl = static_cast<CpuCompNodeImpl*>(m_comp_node_impl);
        if (auto rec = impl->cur_recorder()) {
            auto callback = [this]() {
                incr_nr_req();
                on_finish();
            };
            rec->dispatch_allow_after_sync(callback, m_comp_node_impl);
        } else {
            EventImpl::do_record();
        }
    }

    void do_device_wait_by(Impl*) override {
        mgb_throw(MegBrainError,
                  "device_wait() should not be called on events created during "
                  "comp node seq recording");
    }

public:
    using EventImpl::EventImpl;
};

class CpuCompNodeImpl::CpuEventImpl final
        : public CpuDispatchableBase::EventImpl {
#if MGB_HAVE_THREAD
    void host_wait_cv() override {
        CpuDispatchableBase::EventImpl::host_wait_cv();
        auto thread_pool = static_cast<CpuCompNodeImpl*>(m_comp_node_impl)
                                   ->get_thread_pool();
        if (thread_pool) {
            thread_pool->deactive();
        }
    }
#endif
public:
    using EventImpl::EventImpl;
};

std::unique_ptr<CompNode::Event> CpuCompNodeImpl::create_event(size_t flags) {
    if (m_worker_queue) {
        m_worker_queue->check_exception();
    }
    if (sm_cur_recorder) {
        return std::make_unique<CompSeqRecEventImpl>(this, flags);
    } else {
        return std::make_unique<CpuEventImpl>(this, flags);
    }
}

/* ======================== CpuCompNode ======================== */
struct CpuCompNode::Pool {
    static constexpr int MAX_NR_COMP_NODE = 1024;
    struct CpuCompNodeImplDeleter {
        void operator ()(CpuCompNodeImpl *p) {
            p->~CpuCompNodeImpl();
        }
    };

    std::recursive_mutex mtx;
    // use global memory pool to ensuare object memory accessible even after
    // global finalize
    std::aligned_storage_t<sizeof(CpuCompNodeImpl), alignof(CpuCompNodeImpl)>
        impl_storage[MAX_NR_COMP_NODE];
    size_t nr_used_impl_storage = 0;

    ThinHashMap<std::pair<int, int>,
        std::unique_ptr<CpuCompNodeImpl, CpuCompNodeImplDeleter>> logical2impl;
    ThinHashMap<std::pair<int, int>, std::weak_ptr<WorkerQueue>> physical2queue;
    ThinHashMap<std::pair<int, int>,
                std::unique_ptr<CpuCompNodeImpl, CpuCompNodeImplDeleter>>
            logical2impl_multi_thread;
    ThinHashMap<std::pair<int, int>, std::weak_ptr<WorkerQueue>>
            physical2queue_multithead;
};
CpuCompNode::Pool* CpuCompNode::sm_pool;
Spinlock CpuCompNode::sm_pool_mtx;

void CpuCompNode::foreach(thin_function<void(CompNode)> callback) {
    if (!sm_pool)
        return;

    for (size_t i = 0; ; ++ i) {
        CompNode cur;
        {
            MGB_LOCK_GUARD(sm_pool->mtx);
            if (i >= sm_pool->nr_used_impl_storage)
                return;
            cur = make_comp_node_from_impl(
                    reinterpret_cast<CpuCompNodeImpl*>(
                        &sm_pool->impl_storage[i]));
        }
        callback(cur);
    }
}

void CpuCompNode::finalize() {
    if (sm_pool) {
        sync_all();

        sm_pool->~Pool();
        sm_pool = nullptr;
    }
}

size_t CpuCompNode::get_device_count() {
    return sys::get_cpu_count();
}

CpuCompNode::Impl* CpuCompNode::load_cpu(Locator locator,
                                         Locator locator_logical) {
#if !MGB_HAVE_THREAD
    // use only cpu:default and cpu0:1023 comp node when threading is disabled
    mgb_assert(locator.device == Locator::DEVICE_CPU_DEFAULT ||
               (locator.device == 0 && locator.stream == 1023));
    locator_logical = {locator_logical.type, locator.device, locator.stream};
#endif
    {
        MGB_LOCK_GUARD(sm_pool_mtx);
        if (!sm_pool) {
            // use static storage so object can be safely accessed even after
            // global finalize
            static std::aligned_storage_t<sizeof(Pool), alignof(Pool)> storage;
            sm_pool = new(&storage) Pool;
        }
    }
    mgb_assert(locator.device >= 0 ||
                       (locator.device == Locator::DEVICE_CPU_DEFAULT &&
                        locator.stream == 0) ||
                       locator.device == Locator::DEVICE_MULTITHREAD_DEFAULT,
               "failed to load cpu for device:%d stream:%d", locator.device,
               locator.stream);
    MGB_LOCK_GUARD(sm_pool->mtx);

    // encode both device ID and type into a int
    int compact_logical_device = locator_logical.device;
    mgb_assert(compact_logical_device >= -1 ||
               compact_logical_device <= Locator::DEVICE_CPU_DEFAULT);
    if (locator_logical.type == CompNode::DeviceType::UNSPEC) {
        compact_logical_device += std::numeric_limits<int>::min() + 1;
        mgb_assert(compact_logical_device <
                   Locator::DEVICE_MULTITHREAD_DEFAULT);
    } else {
        mgb_assert(locator_logical.type == CompNode::DeviceType::CPU ||
                   locator_logical.type == CompNode::DeviceType::MULTITHREAD);
    }
    if (locator.type == DeviceType::CPU) {
        auto &&pqueue_weak =
        sm_pool->physical2queue[{locator.device, locator.stream}];
        auto pqueue = pqueue_weak.lock();
        if (!pqueue) {
            pqueue = std::make_shared<WorkerQueue>(locator);
            pqueue_weak = pqueue;
        }
        auto&& pimpl = sm_pool->logical2impl[{compact_logical_device,
                                              locator_logical.stream}];
        if (!pimpl) {
            mgb_assert(sm_pool->nr_used_impl_storage < Pool::MAX_NR_COMP_NODE,
                       "too many cpu comp nodes; max %d allowed",
                       Pool::MAX_NR_COMP_NODE);
            pimpl.reset(new (
                    &sm_pool->impl_storage[sm_pool->nr_used_impl_storage++])
                                CpuCompNodeImpl{locator, locator_logical,
                                                pqueue});
        }
        log_comp_node_created(locator, locator_logical);
        return pimpl.get();
    } else {
        mgb_assert(locator.type == DeviceType::MULTITHREAD);
        auto&& pqueue_weak = sm_pool->physical2queue_multithead[{
                locator.device, locator.nr_threads}];
        auto pqueue = pqueue_weak.lock();
        if (!pqueue) {
            pqueue = std::make_shared<WorkerQueue>(locator);
            pqueue_weak = pqueue;
        }
        auto&& pimpl = sm_pool->logical2impl_multi_thread[{
                compact_logical_device, locator_logical.nr_threads}];
        if (!pimpl) {
            mgb_assert(sm_pool->nr_used_impl_storage < Pool::MAX_NR_COMP_NODE,
                       "too many cpu multithread comp nodes; max %d allowed",
                       Pool::MAX_NR_COMP_NODE);
            pimpl.reset(new (
                    &sm_pool->impl_storage[sm_pool->nr_used_impl_storage++])
                                CpuCompNodeImpl{locator, locator_logical,
                                                pqueue});
        }
        log_comp_node_created(locator, locator_logical);
        return pimpl.get();
    }
}

void CpuCompNode::sync_all() {
    if (!sm_pool)
        return;

    MGB_LOCK_GUARD(sm_pool->mtx);
    for (auto &&i: sm_pool->logical2impl)
        i.second->sync();
    for (auto&& i : sm_pool->logical2impl_multi_thread)
        i.second->sync();
}

bool CpuCompNode::CompNodeImpl::check_global_finalized(const char* reason) {
    MGB_MARK_USED_VAR(reason);
    if (this != sm_default_cpu_comp_node_ptr && !sm_pool) {
        static std::atomic_flag warn_printed = ATOMIC_FLAG_INIT;
        if (!warn_printed.test_and_set()) {
            mgb_log_debug("cpu comp node method called after global finalize: "
                    "reason=%s", reason);
        }
        return true;
    }
    return false;
}

/* ======================== CompNode methods ========================  */
// CompNode get by default_cpu() is different from the CompNode which is
// produced by CompNode::load("cpu:default")
// default_cpu() is used for static infer and it is not allowed to send up the
// compute kernel
// CompNode::load("cpu:default") is "inplace cpu" which is in the
// CpuCompNode::Pool
CompNode CompNode::default_cpu() {
    static Locator locator{DeviceType::CPU, Locator::DEVICE_CPU_DEFAULT, {-1}};
    static auto empty_queue =
        std::make_shared<CpuCompNode::WorkerQueue>(locator);
    static CpuCompNodeImpl impl{locator, locator, empty_queue};
    return &impl;
}

bool CompNode::enable_affinity_for_cpu(bool flag) {
    bool old = enable_affinity;
    enable_affinity = flag;
    return old;
}


/* ======================== EventImpl ========================  */

double CpuCompNode::CpuDispatchableBase::EventImpl::do_elapsed_time_until(
        EventImplHelper &end) {
    auto &&f1 = static_cast<EventImpl&>(end).m_prev_finish_time;
    return m_prev_finish_time.time_until_secs(f1);
}

#if MGB_HAVE_THREAD
void CpuCompNode::CpuDispatchableBase::EventImpl::do_device_wait_by(
        Impl *cn_impl) {
    {
        auto locator = m_comp_node_impl->locator();
        if (locator.device == Locator::DEVICE_CPU_DEFAULT &&
            !static_cast<CpuCompNode::CompNodeImpl*>(m_comp_node_impl)
                     ->cur_recorder()) {
            auto v0 = m_record_nr_req.load(std::memory_order_relaxed),
                 v1 = m_record_nr_finish.load(std::memory_order_relaxed);
            mgb_assert(v0 && v0 == v1,
                       "event on cpu:default hasn't been recorded inplace.");
            return;
        }
    }

    {
        auto type = cn_impl->env().property().type;
        mgb_throw_if(
                type != CompNode::DeviceType::CPU &&
                        type != CompNode::DeviceType::CUDA
                        && type != CompNode::DeviceType::ATLAS &&
                        type != CompNode::DeviceType::CAMBRICON,
                MegBrainError,
                "currently CPU can only wait for CPU, CUDA, ATLAS, CAMBRICON"
        );
    }

    if (cn_impl->env().property().type == CompNode::DeviceType::ATLAS) {
#if MGB_ATLAS
        return m_comp_node_impl->sync();
#else
        mgb_throw(MegBrainError,
                  "Atlas comp_node used but MGB_ATLAS not enabled");
#endif
    } else if (cn_impl->env().property().type == CompNode::DeviceType::CAMBRICON) {
#if MGB_CAMBRICON
        return m_comp_node_impl->sync();
#else
        mgb_throw(MegBrainError,
                  "Cambricon comp_node used but MGB_CAMBRICON not enabled");
#endif

    }

    auto version = m_record_nr_req.load(std::memory_order_relaxed);
    mgb_assert(version, "device wait on non-recorded event");

    auto waiter = [this, version]() {
        while (m_record_nr_finish.load(std::memory_order_acquire) < version) {
            std::unique_lock<std::mutex> lk{m_dev_wait_mtx};
            if (m_record_nr_finish.load(std::memory_order_acquire) >= version) {
                break;
            }
            m_dev_wait_cv.wait(lk);
        }
        m_dev_wait_nr_waiter.fetch_sub(1, std::memory_order_release);
    };
    m_dev_wait_nr_waiter.fetch_add(1, std::memory_order_release);
    cn_impl->add_callback(waiter);
}

void CpuCompNode::CpuDispatchableBase::EventImpl::do_record() {
    incr_nr_req();
    auto call_on_finish = [this]() { on_finish(); };
    static_cast<CpuDispatchableBase*>(m_comp_node_impl)
            ->dispatch(call_on_finish);
}

void CpuCompNode::CpuDispatchableBase::EventImpl::on_finish() {
    if (m_create_flags & Flags::NEED_TIMER) {
        auto v0 = m_record_nr_finish.load(std::memory_order_relaxed) + 1,
             v1 = m_record_nr_req.load(std::memory_order_relaxed);
        if (v0 == v1) {
            m_prev_finish_time = RealTimer::get_time();
        }
    }

    m_record_nr_finish.fetch_add(1, std::memory_order_release);
    if (m_dev_wait_nr_waiter.load(std::memory_order_acquire)) {
        MGB_LOCK_GUARD(m_dev_wait_mtx);
        m_dev_wait_cv.notify_all();
    }
}

bool CpuCompNode::CpuDispatchableBase::EventImpl::do_finished() {
    auto v0 = m_record_nr_req.load(std::memory_order_relaxed);
    auto v1 = m_record_nr_finish.load(std::memory_order_acquire);
    return v0 == v1;
}

void CpuCompNode::CpuDispatchableBase::EventImpl::host_wait_cv() {
    for (size_t i = 0, it = SCQueueSynchronizer::get_default_max_spin() / 20; i < it; ++i) {
        if (finished()) {
            return;
        }
    }

    m_dev_wait_nr_waiter.fetch_add(1, std::memory_order_release);
    for (; ; ) {
        std::unique_lock<std::mutex> lock{m_dev_wait_mtx};
        if (finished()) {
            break;
        }
        m_dev_wait_cv.wait(lock);
    }
    m_dev_wait_nr_waiter.fetch_sub(1, std::memory_order_release);
}

CpuCompNode::CpuDispatchableBase::EventImpl::~EventImpl() noexcept {
    auto check_all_finished = [this]() {
        return do_finished() &&
            !m_dev_wait_nr_waiter.load(std::memory_order_acquire);
    };
    if (!check_all_finished()) {
        mgb_log_debug("event %p has unfinished callbacks when destructed; "
                "waiting ...", this);
        while (!check_all_finished()) {
            std::this_thread::yield();
        }
    }
}
#else   // MGB_HAVE_THREAD

void CpuCompNode::CpuDispatchableBase::EventImpl::host_wait_cv() {
}

void CpuCompNode::CpuDispatchableBase::EventImpl::do_device_wait_by(Impl*) {
}

void CpuCompNode::CpuDispatchableBase::EventImpl::do_record() {
    if (m_create_flags & Flags::NEED_TIMER) {
        m_prev_finish_time = RealTimer::get_time();
    }
}

void CpuCompNode::CpuDispatchableBase::EventImpl::on_finish() {
}

bool CpuCompNode::CpuDispatchableBase::EventImpl::do_finished() {
    return true;
}

CpuCompNode::CpuDispatchableBase::EventImpl::~EventImpl() noexcept = default;

#endif  // MGB_HAVE_THREAD


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
