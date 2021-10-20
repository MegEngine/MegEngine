/**
 * \file src/core/include/megbrain/comp_node.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/utils/hash.h"
#include "megbrain/utils/metahelper.h"
#include "megbrain/utils/thin/function.h"
#include "megbrain/utils/thin/hash_table.h"
#include "megbrain/utils/thread.h"
#include "megdnn/thin/function.h"

#include <cstddef>
#include <memory>
#include <string>

namespace mgb {

// forward declaration; defined in comp_node_env.h
class CompNodeEnv;

namespace cg {
class ComputingGraph;
}

class CompNodeSeqRecorder;

/*!
 * \brief identifier for a memory node
 *
 * MemNode is comparable. CompNodes with the same MemNode can access memory of
 * each other directly
 */
class MemNode {
    const void* m_id = nullptr;

public:
    MemNode() = default;

    explicit MemNode(const void* id) : m_id{id} {}

    bool operator==(const MemNode& rhs) const { return m_id == rhs.m_id; }

    bool operator!=(const MemNode& rhs) const { return m_id != rhs.m_id; }

    operator bool() const { return m_id != nullptr; }
};

/*!
 * \brief abstraction of a streaming computing resource on localhost (a
 *      thread on CPU, a cuda stream, etc.)
 *
 * Note that most of the operations are asynchronous with respect to the caller
 * thread
 */
class CompNode {
public:
    //! computing device type
    enum class DeviceType {
        //! for "xpu" comp node that would mapped to available cn on
        //! current system
        UNSPEC = 0,

        CUDA = 1,
        CPU = 2,
        CAMBRICON = 3,
        ROCM = 8,
        ATLAS = 9,
        MULTITHREAD = 11,
        MAX_DEVICE_ID,
    };
    static constexpr size_t NR_DEVICE_TYPE =
            static_cast<size_t>(DeviceType::MAX_DEVICE_ID);

    /*!
     * \brief an identifier to specify a computing node
     *
     * Note: logical locator is directly parsed from a string identifier
     * given by user; it should be translated to physical locator by calling
     * to_physical() before actual use.
     *
     * Unless explicitly specified otherwise, all locators are physical
     * locators.
     */
    struct Locator {
        /*!
         * \brief special device number for the "cpu default" comp node,
         *      which dispatches all tasks in the caller thread
         */
        static constexpr int DEVICE_CPU_DEFAULT = -1024;
        /*!
         * \brief special device number for the "multithread_default"
         * comp node, which dispatches all tasks to thread pool and the
         * caller thread is the main thread of thread pool
         */
        static constexpr int DEVICE_MULTITHREAD_DEFAULT = -1025;

        DeviceType type = DeviceType::UNSPEC;

        /*!
         * corresponding to a physical computing device; memories between
         * different devices are not shared.
         *
         * device == -1 means logical default device (maps to 0 by default,
         * and can be changed by set_device_map)
         *
         */
        int device = -1;

        //! multiple streams can execute on one computing device and share
        //! memory, when compnode type is multithread the field also stand
        //! for nr_threads
        union {
            int stream = 0;
            int nr_threads;
        };

        /*!
         * \brief parse a string identifier
         *
         * currently supported ID format: (gpu|cpu)<n>[:m] where n is the
         * device number, possibly with m as the stream id.
         */
        static Locator parse(const std::string& id);

        /*!
         * \brief set mapping between device numbers of a device type
         */
        static void set_device_map(DeviceType type, int from, int to);

        /*!
         * \brief set the actual device type to be used for
         *      DeviceType::UNSPEC
         */
        static void set_unspec_device_type(DeviceType type);

        /*!
         * \brief get corresponding physical Locator
         *
         * DeviceType::UNSPEC would be resolved, and device map would be
         * applied on device number
         */
        Locator to_physical() const;

        /*!
         * \brief get string description of this locator that can be parsed
         *      again
         */
        std::string to_string() const;

        bool operator==(const Locator& rhs) const {
            return type == rhs.type && device == rhs.device && stream == rhs.stream;
        }
    };

    struct LocatorPairHashKey {
        Locator locator, locator_logical;

        bool operator==(const LocatorPairHashKey& rhs) const {
            return locator == rhs.locator && locator_logical == rhs.locator_logical;
        }

        struct Hash {
            size_t operator()(const LocatorPairHashKey& k) const {
                return hash_pair_combine(
                        mgb::hash(k.locator), mgb::hash(k.locator_logical));
            }
        };
    };

    //! predefined special streams
    struct Stream {
        static constexpr int COPY = -1, REMOTE_SEND = -2, LOOP_SWAP = -3;
    };

    CompNode() = default;

    /*!
     * \brief manually destroy all comp node resources
     */
    static void finalize();

    /*!
     * \brief load a computing node from logical locator ID;
     * \see Locator::parse
     */
    static CompNode load(const std::string& id) { return load(Locator::parse(id)); }

    /*!
     * \brief create a CompNode object from **logical** locator
     */
    static CompNode load(const Locator& locator) {
        return load(locator.to_physical(), locator);
    }

    static CompNode load(
            const Locator& locator_physical, const Locator& locator_logical);

    /* =================== memory management ======================== */

    /*!
     * \brief allocate memory on this computing node
     *
     * Note: allocation of device memory is synchronous with the host,
     * meaning that the memory can be used immediately; however deallocation
     * is asynchronous to ensure that the memory can be used by
     * already-launched kernels on the computing node.
     *
     * Exception should be raised if allocation fails.
     */
    void* alloc_device(size_t size) const;

    //! deallocate device buffer; see alloc_device() for more details
    void free_device(void* ptr) const;

    /*!
     * \brief allocate memory on host that is associated with the device,
     *      which may accelerate I/O
     *
     * Both allocation and deallocation on host are synchronous.
     */
    void* alloc_host(size_t size) const;

    void free_host(void* ptr) const;

    //! copy from underlying device to host
    void copy_to_host(void* host_ptr, const void* device_ptr, size_t size) const {
        return m_impl->copy_to_host(host_ptr, device_ptr, size);
    }

    //! copy from host to underlying device
    void copy_to_device(void* device_ptr, const void* host_ptr, size_t size) const {
        return m_impl->copy_to_device(device_ptr, host_ptr, size);
    }

    /*!
     * \brief copy from this device to another device; would use the
     *      computing resource on dest_node
     * \param src source memory that must be allocated on this device
     */
    void peer_copy_to(
            CompNode dest_node, void* dest, const void* src, size_t size) const {
        return m_impl->peer_copy_to(
                reinterpret_cast<Impl*>(dest_node.m_impl), dest, src, size);
    }

    //! get alignment requiement in bytes; guaranteed to be power of 2
    size_t get_mem_addr_alignment() const { return m_impl->get_mem_addr_alignment(); }

    /*!
     * \brief get the size of the paddings which must be reserved at the
     * end of memory chunk; guaranteed to be power of 2
     */
    size_t get_mem_padding() const {
        size_t padding = m_impl->get_mem_padding();
        mgb_assert(!(padding & (padding - 1)), "mem padding should be power of 2");
        return padding;
    }

    /*!
     * \brief release consecutive free chunks on all devices to defragment;
     *      see DevMemAlloc::try_coalesce_free
     */
    static void try_coalesce_all_free_memory();

    /*
     * \brief specifies how to pre-allocate from raw dev allocator
     *
     */
    static void set_prealloc_config(
            size_t alignment, size_t min_req, size_t max_overhead, double growth_factor,
            DeviceType device_type);
    /*!
     * \brief get compute capability of the specified device
     */
    static size_t get_compute_capability(int dev, DeviceType device_type);

    /* =================== synchronization ======================== */

    class Event;
    class EventPool;

    std::unique_ptr<Event> create_event(size_t flags = 0) const {
        return m_impl->create_event(flags);
    }

    //! wait for an event created on another CompNode
    inline void device_wait_event(Event& event) const;

    /*!
     * \brief block host thread to wait for all previous operations on this
     *      computing node to finish
     */
    void sync() const { return m_impl->sync(); }

    /*!
     * \brief synchronize all computing nodes
     */
    static void sync_all();

    /* =================== misc ======================== */

    /*!
     * \brief get id of underlying memory node; comp nodes that share the
     *      same mem node can access memory allocated by each other.
     */
    MemNode mem_node() const { return m_impl->mem_node(); }

    bool operator==(const CompNode& rhs) const { return m_impl == rhs.m_impl; }

    bool operator!=(const CompNode& rhs) const { return !this->operator==(rhs); }

    bool valid() const { return m_impl; }

    //! get total and free memory on the computing device in bytes
    std::pair<size_t, size_t> get_mem_status_bytes() const {
        return m_impl->get_mem_status_bytes();
    }

#if !MGB_BUILD_SLIM_SERVING
    std::pair<size_t, size_t> get_free_left_and_right(
            size_t begin_ptr, size_t end_ptr) {
        return m_impl->get_free_left_and_right(begin_ptr, end_ptr);
    }

    size_t get_used_memory() const { return m_impl->get_used_memory(); }

    size_t get_max_block_size_available() const {
        return m_impl->get_max_block_size_available();
    }
#endif

    //! change to another stream on the same memory node
    CompNode change_stream(int dest_stream) const;

    //! get string representation
    std::string to_string() const {
        return m_impl ? mgb::ssprintf(
                                "CompNode(\"%s\" from \"%s\")",
                                to_string_physical().c_str(),
                                to_string_logical().c_str())
                      : "invalid";
    }

    //! get string representation of physical device
    std::string to_string_physical() const {
        return m_impl ? m_impl->locator().to_string() : "invalid";
    }

    //! get string representation of logical device
    std::string to_string_logical() const {
        return m_impl ? m_impl->locator_logical().to_string() : "invalid";
    }

    uint64_t get_uid() { return m_impl->get_uid(); }

    //! get the physical locator that created this comp node
    Locator locator() const { return m_impl->locator(); }

    //! get the logical locator that created this comp node
    Locator locator_logical() const { return m_impl->locator_logical(); }

    //! see CompNodeEnv::activate
    void activate() const;

    //! get device type of this comp node
    DeviceType device_type() const;

    /*!
     * \brief check for error on the asynchronous computing stream
     *
     * This is used for devices with limited error handling such as CUDA.
     *
     * It will return MegBrainError with error messages rather than
     * directly throw exception; return nullptr if no error.
     */
    MGB_WARN_UNUSED_RESULT
    std::unique_ptr<MegBrainError> check_async_error() const;

    /*!
     * \brief create a CompNodeSeqRecorder associated with this computing
     * node
     *
     * Note: the implementation must be thread safe: simultaneous calls to
     * create_seq_recorder() must block until existing CompNodeSeqRecorder
     * objects are either destructed or stopped.
     *
     * \return the recorder object; nullptr is returned if recording is not
     *      supported
     */
    std::unique_ptr<CompNodeSeqRecorder> create_seq_recorder(cg::ComputingGraph* cg) {
        return m_impl->create_seq_recorder(cg);
    }

    /*!
     *  insert callback into current compute stream.
     *  The callack is to be called after all currently enqueued
     *  iterms in the stream have completed. And the later tasks
     *  in the stream must wait for the callback to finish.
     */
    void add_callback(megdnn::thin_function<void()>&& cb) {
        return m_impl->add_callback(std::move(cb));
    }

    enum class Flag : uint32_t {
        //! Whether computing recorder is supported on this comp node (i.e.
        //! whether non-zero comp_node_seq_record_level is allowed)
        SUPPORT_RECORDER = 1 << 0,

        //! Whether dynamic memory allocation is supported in seq recorder.
        //! If this flag is not setted, ComputingSequence::do_execute()
        //! would skip the warm up and allow seq recorder to start
        //! immediately
        RECORDER_SUPPORT_DYNAMIC_ALLOC = 1 << 1,

        //! Whether the capacity of the asynchronous execution queue on this
        //! comp node is limited.
        //! If this flag is set, tasks on multiple comp nodes would be
        //! dispatched from multiple cpu threads.
        //! \see ComputingGraph::Options::async_exec_level
        QUEUE_LIMITED = 1 << 2,

        //! Whether this comp node supports copy stream, so computation and
        //! I/O can be parallelized
        HAS_COPY_STREAM = 1 << 3,

        //! Destructing an event is unsafe if the comp node is not
        //! synchronized; setting this flag would cause computing sequence
        //! to sync the comp node in its dtor.
        EVENT_DTOR_UNSAFE = 1 << 4,

        //! CompNode is available even there is no thread support, i.e.
        //! MGB_HAVE_THREAD=0. Usually this means that execution on the
        //! CompNode is synchronous, i.e. behaves like cpu:default
        SUPPORT_NO_THREAD = 1 << 5,

        //! Whether this comp node supports unified address. i.e. CPU and
        //! CUDA supports unified address.
        SUPPORT_UNIFIED_ADDRESS = 1 << 6,
    };

    bool contain_flag(Flag flag) { return contain_flag(device_type(), flag); }

    static bool contain_flag(DeviceType device_type, Flag flag);

    using UnorderedSet = ThinHashSet<CompNode>;

    template <typename T>
    using UnorderedMap = ThinHashMap<CompNode, T>;

    //! apply function to each initialized comp node
    static void foreach (thin_function<void(CompNode)> callback);

    //! get total number of specific devices on this system
    static size_t get_device_count(DeviceType type, bool warn = true);

    /* =================== specialized ======================== */

    //! get default CPU comp node
    // implemented in comp_node/cpu/comp_node.cpp
    static CompNode default_cpu();

    /*!
     * \brief set whether to enable affinity setting for CPU comp nodes
     *
     * If enabled, computation on cpux would be bound to the x'th CPU.
     *
     * This is disabled by default.
     *
     * (implemented in comp_node/cpu/comp_node.cpp)
     *
     * \return original setting
     */
    static bool enable_affinity_for_cpu(bool flag);

protected:
    //! ImplBase with env(); defined in CompNodeEnv
    class Impl;

    class ImplBase : public NonCopyableObj, public DynTypeObj {
    public:
        typedef void (*free_func_t)(ImplBase* self, void* ptr);
        //! memory free might be called after finalize(); so we should
        //! not rely on virtual function for this
        const free_func_t free_device;
        const free_func_t free_host;

        virtual void* alloc_device(size_t size) = 0;
        virtual void* alloc_host(size_t size) = 0;

        virtual void copy_to_host(
                void* host_ptr, const void* device_ptr, size_t size) = 0;
        virtual void copy_to_device(
                void* device_ptr, const void* host_ptr, size_t size) = 0;
        virtual void peer_copy_to(
                Impl* dest_impl, void* dest, const void* src, size_t size) = 0;

        virtual size_t get_mem_addr_alignment() = 0;
        virtual size_t get_mem_padding();

        virtual std::unique_ptr<Event> create_event(size_t flags) = 0;

        virtual void sync() = 0;

        virtual MemNode mem_node() = 0;
        virtual std::pair<size_t, size_t> get_mem_status_bytes() = 0;

#if !MGB_BUILD_SLIM_SERVING
        virtual std::pair<size_t, size_t> get_free_left_and_right(size_t x, size_t y) {
            return {x - x, y - y};
        }
        virtual size_t get_used_memory() { return 0; }
        virtual size_t get_max_block_size_available() { return 0; }
#endif

        virtual Locator locator() = 0;
        virtual Locator locator_logical() = 0;

        virtual std::unique_ptr<CompNodeSeqRecorder> create_seq_recorder(
                cg::ComputingGraph* cg);

        virtual void add_callback(megdnn::thin_function<void()>&&);

        virtual uint64_t get_uid() {
            mgb_throw(MegBrainError, "get_uid is not impl yet");
        };

    protected:
        ImplBase(free_func_t fd, free_func_t fh) : free_device{fd}, free_host{fh} {}

        ~ImplBase() = default;
    };

    //! implementations are allocated statically, so no memory management
    //! is needed
    ImplBase* m_impl = nullptr;

    friend class CompNodeEnv;
    friend struct HashTrait<CompNode>;
    friend struct HashTrait<CompNode::Locator>;
    friend class CompNodeImplHelper;

public:
    CompNode(ImplBase* impl) : m_impl{impl} {}
};

MGB_DEF_ENUM_CLASS_BIT_OPR(CompNode::Flag)

/*!
 * \brief record computation operations on a computing node
 *
 * This is used for fast execution of an identical computation sequence where
 * only input/output data differ.
 *
 * When this object is created from a comp node, recording starts immediately.
 * Call stop() when computation finishes, and call replay() when it needs to be
 * re-executed.
 *
 * Implementations should consider thread safe in comp_node, in order to support
 * multi threads reording in the same comp_node simultaneously, using thread
 * local recorder in comp_node.
 *
 * Note. When recording is over, the recorder is independent with comp_node, so
 * the task dispatched into recorder should not related to the comp_node
 * methord, and the thread of recorder replay is the user thread.
 */
class CompNodeSeqRecorder {
public:
    virtual ~CompNodeSeqRecorder() noexcept = default;

    /*!
     * \brief Enter fake-exec mode
     *
     * Memory allocation/free is only allowed in fake-exec mode, and kernels
     * should not be actually recorded in this mode.
     *
     * This should be paired with exit_fake_exec()
     */
    virtual void enter_fake_exec(const CompNode& comp_node) = 0;

    //! Exit fake-exec mode
    virtual void exit_fake_exec(const CompNode& comp_node) = 0;

    virtual void stop(const CompNode& comp_node) = 0;

    virtual void replay() = 0;
};

/*!
 * \brief event associated with a CompNode node, used for cross-device
 *      synchronization
 */
class CompNode::Event : public NonCopyableObj {
protected:
    static int sm_cpu_sync_level;

    //! flags when this event is created
    size_t const m_create_flags;

    Event(size_t create_flags) : m_create_flags{create_flags} {}

public:
    enum Flags { NEED_TIMER = 1 };

    virtual ~Event() = default;

    /*!
     * \brief record this event on the comp node that creates it
     *
     * Note that if a comp node is recorded multiple times, then subsequent
     * calls would overwrite its internal state and other methods that
     * examine the status would only examine the completion of the most
     * recent call to record().
     */
    virtual void record() = 0;

    //! whether this event has finished; it must has been recorded
    virtual bool finished() = 0;

    //! block the host thread (caller thread) to wait for this event
    virtual void host_wait() = 0;

    //! get elapsed time in seconds from this to another event; the events
    //! must be finished
    virtual double elapsed_time_until(Event& end) = 0;

    //! record an action on another comp node so it would wait for this
    //! event
    virtual void device_wait_by(CompNode cn) = 0;

    //! get the comp node to which this event is associated
    virtual CompNode comp_node() const = 0;

    //! flags when this event is created
    size_t create_flags() const { return m_create_flags; }

    /*!
     * \brief set CPU resource usage level when performing synchronization
     * \param level CPU waiting level:
     *      0. condition var (the default)
     *      1. busy wait with yield
     *      2. busy wait
     */
    static void set_cpu_sync_level(int level) { sm_cpu_sync_level = level; }
};

/*!
 * \brief pool of events that can be reused
 */
class CompNode::EventPool {
    CompNode m_cn;
    std::vector<std::unique_ptr<CompNode::Event>> m_allocated;
    std::vector<CompNode::Event*> m_free;
    Spinlock m_lock;
    size_t m_flags;

public:
    explicit EventPool(CompNode cn, size_t flags = 0);
    ~EventPool();

    CompNode::Event* alloc();

    void free(CompNode::Event* ev);

    //! assert that all allocated events have been freed
    void assert_all_freed();
};

void CompNode::device_wait_event(Event& event) const {
    event.device_wait_by(*this);
}

template <>
struct HashTrait<CompNode> {
    static size_t eval(const CompNode& val) {
        static_assert(sizeof(size_t) == sizeof(void*), "bad hash type");
        return reinterpret_cast<size_t>(static_cast<void*>(val.m_impl));
    }
};

template <>
struct HashTrait<CompNode::Locator> {
    static size_t eval(const CompNode::Locator& val) {
        return static_cast<size_t>(val.device) + (static_cast<size_t>(val.type) << 4) +
               (static_cast<size_t>(val.stream) << 8);
    }
};

namespace comp_node_detail {

/*!
 * \brief an inplace doubly linked list for efficient inserting/deleting
 *
 * Note: do not use this directly; it is only for CompNodeDepedentObject
 */
class DepedentObjList {
    class Sentinel;

    struct StaticInfo;
    static StaticInfo sm_info;

    DepedentObjList *m_prev = nullptr, *m_next = nullptr;

    static void link(DepedentObjList* a, DepedentObjList* b) {
        a->m_next = b;
        b->m_prev = a;
    }

protected:
    virtual std::shared_ptr<void> callback() = 0;
    ~DepedentObjList() = default;

    static void add(DepedentObjList* ptr);
    static void remove(DepedentObjList* ptr);

public:
    static void invoke_callback_and_clean();
};

}  // namespace comp_node_detail

/*!
 * \brief base class for objects that depend on CompNode
 *
 * There is a CompNode::finalize() method that destorys all global comp nodes.
 * Therefore objects that depend on CompNode should all be marked as invalid at
 * that time.
 *
 * CompNode::finalize() is called in atexit() because some external libraries
 * that CompNode depends on seems to be registering exit handlers. It is also
 * impractical to require a correct destruction order because, for example, in
 * python atexit() handlers are invoked before global python objects get
 * reclaimed.
 *
 * As a result we give up enforcing a correct destruction order, but rather
 * require all CompNode-dependent objects to derive from this class so they can
 * get notified possibly do most of the cleanup when CompNode is finalized.
 */
class CompNodeDepedentObject : private comp_node_detail::DepedentObjList {
    //! 1: in on_comp_node_finalize(); 2: after on_comp_node_finalize()
    int m_state = 0;
    std::shared_ptr<void> callback() override final;

protected:
    CompNodeDepedentObject() { add(this); }
    ~CompNodeDepedentObject() { remove(this); }

    /*!
     * \brief overwritten by subclasses to perform clean up jobs
     *
     * Note: in case the object has nested objects which hold a reference to the
     * object itself, a reference to this object must be kept so it would not be
     * released during the call of on_comp_node_finalize().
     */
    virtual std::shared_ptr<void> on_comp_node_finalize() = 0;

    //! exception would thrown if on_comp_node_finalize() has been called (do
    //! not raise if invoked from on_comp_node_finalize())
    void check_not_finalized() const;

    //! whether on_comp_node_finalize() has been called (true when invoked
    //! from on_comp_node_finalize())
    bool is_finalized() const { return m_state; }
};

}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
