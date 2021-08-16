/**
 * \file imperative/src/impl/interpreter/interpreter_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <deque>
#include <future>
#include <list>
#include <stack>
#include <thread>
#include <unordered_set>
#include <variant>
#include "megbrain/comp_node.h"
#include "megbrain/utils/mempool.h"
#include "megbrain/imperative/interpreter.h"
#include "megbrain/imperative/profiler.h"

#include "./commands.h"
#include "./tensor_info.h"
#include "./option_manager.h"

#include "../profiler/events.h"

namespace mgb::imperative::interpreter::intl {

using Handle = Interpreter::Handle;

struct InterpreterImpl : Interpreter {
    std::unique_ptr<Channel> create_channel() override;
};

struct ChannelImpl : Interpreter::Channel {
    ChannelImpl();
    ~ChannelImpl() override;

    Handle put(const HostTensorND& value, bool no_cache) override;
    Handle put(const DeviceTensorND& value, const HostTensorND& hvalue) override;

    void del(Handle) override;
    void swap_in(Handle) override;
    void swap_out(Handle) override;
    void drop(Handle) override;

    SmallVector<Handle> apply_op(
            std::shared_ptr<OpDef> op,
            const SmallVector<Handle>& inputs) override;

    HostTensorND get_value(Handle) override;
    TensorShape get_shape(Handle) override;
    DType get_dtype(Handle) override;
    CompNode get_device(Handle) override;

    DeviceTensorND get_dev_tensor(Handle) override;

    bool check_available() override;
    void sync() override;
    void close() override;

    size_t get_option(std::string name) override;
    void set_option(std::string name, size_t value) override;

    void start_profile() override;
    void stop_profile() override;

    void push_scope(std::string) override;
    void pop_scope(std::string) override;
private:
    struct WorkQueue;
    struct State;

    TensorInfo* alloc();
    void init(TensorInfo*, LogicalTensorDesc desc);
    void free(TensorInfo*);
    void real_free(TensorInfo*);
    void recursive_free(TensorInfo*);
    void do_drop(TensorInfo*, bool);
    void detach_users(TensorInfo*);

    TensorInfo* put_impl(const HostTensorND& value, bool no_cache);
    TensorInfo* put_impl(const DeviceTensorND& value, const HostTensorND& hvalue);
    void del_impl(Handle);
    void sync_impl();
    SmallVector<Handle> apply_op_impl(
            std::shared_ptr<OpDef> op,
            const SmallVector<Handle>& inputs);
    TensorPtr wait_tensor(TensorInfo* info, profiler::TensorProp prop);
    void notify_tensor_unsafe(TensorInfo* info);

    void process_one_task(IdentifiedCommand&);

    void check_worker_exc_unsafe();

    void produce_tensor(TensorInfo* dest, TensorPtr ptr);

    void release_tensor(TensorInfo* dest);

    void regenerate(TensorInfo* dest);
    void do_apply_op(const ApplyOp& cmd);
    void flush_apply_stack();
    
    std::tuple<SmallVector<MemoryDesc>, SmallVector<TensorPtr>, SmallVector<TensorPtr>> init_output_and_workspace(
        const OpDef& def,
        SmallVector<TensorPtr> inputs,
        SmallVector<MemoryDesc> inputs_mem_desc);

    void dispatch_default_cpu(
        std::shared_ptr<OpDef> op,
        const SmallVector<TensorInfo*>& input_infos,
        const SmallVector<LogicalTensorDesc>& input_descs,
        SmallVector<Handle>* outputs);
    void dispatch_kernel(
        std::shared_ptr<OpDef> op,
        const SmallVector<TensorInfo*>& input_infos,
        const SmallVector<LogicalTensorDesc>& input_descs,
        SmallVector<Handle>* outputs);

    void push_scope(std::string, State&);
    void pop_scope(std::string, State&);

    void assert_in_channel();
    void assert_in_worker();
    std::thread::id get_worker_tid();

    template <typename TCommand>
    void enqueue_command(TCommand&& cmd) {
        m_buffer.enqueue(Command{std::forward<TCommand>(cmd)});
    }

    void sample_on_device(CompNode device, bool force);

    // valid => status != Deleted
    std::unordered_set<TensorInfo*> collect_valid_tensors();

    std::mutex m_mutex;
    Spinlock m_spin;
    std::condition_variable m_cv;
    MemPool<TensorInfo> m_pool;
    std::unordered_set<Handle> m_valid_handle;
    TensorInfo* m_waitee = nullptr;
    uint64_t m_waitee_id = 0;
    std::exception_ptr m_worker_exc;
    std::function<void(std::string, std::string)> m_profile_dump_callback;
    size_t m_storage_id = 0;
    std::stack<std::tuple<ApplyOp, size_t, TensorInfo*>> m_apply_stack;
    bool m_applying = false;
    bool m_closed = false;

    struct WorkQueue : AsyncQueueSC<IdentifiedCommand, WorkQueue> {
        // set max_spin=0 to prevent Queue fetch task in busy wait manner.
        // this won't affect throughput when python interpreter is sending enough task,
        // but will significantly save CPU time when waiting for task, e.g. wait for data input
        // limit pending tasks to 1000000
        WorkQueue(ChannelImpl* owner)
                : AsyncQueueSC<IdentifiedCommand, WorkQueue>(0, 1000000), m_owner(owner) {
            sys::set_thread_name("interpreter");
        }
        void process_one_task(IdentifiedCommand& icmd) {
            m_owner->process_one_task(icmd);
        }
        void on_async_queue_worker_thread_start() override;
    private:
        ChannelImpl* m_owner;
    } m_worker;

    /**
     * Buf a command window for following fuse
     * example:
     *     ---------------------------------------------------------------------
     *     | ..., Apply{in: (i0, i1), out: (o0, o1)}, ... + Del{i0} + Del{i1}  |
     *     ---------------------------------------------------------------------
     *     | ..., Apply{in: (i0, i1), out: (o0, o1), del: (i0)}, ... + Del{i1} |
     *     ---------------------------------------------------------------------
     *     | ..., Apply{in: (i0, i1), out: (o0, o1), del: (i0, i1)}, ...       |
     *     ---------------------------------------------------------------------
     *     Then the fused Apply may be invoked inplace. see: ChannelImpl::process_one_task
     */
    struct CommandBuffer {
        CommandBuffer(ChannelImpl* owner) : m_owner(owner) {}
        void enqueue(Command cmd);
        bool empty() const {
            return m_commands.empty();
        }
        void flush();
    private:
        ChannelImpl* m_owner;
        std::deque<Command> m_commands;

        using Handle = decltype(m_commands)::iterator;
        // [begin, end)
        using Range = std::array<Handle, 2>;

        // Launch commands in range [m_commands.begin(), pos)
        void flush(Handle pos);
        // Select flush position for incoming cmd
        Handle flush_pos_for(const Command& cmd);
        // Fuse del command into suitable ApplyOp
        bool fuse_del(const Del& cmd);
        // Returns the last handle that dest is used within range. If dest is not used, returns range[1]
        Handle find_last_usage(TensorInfo* dest, Range range);
        // Returns the produce position of dest. If not found, returns range[1]
        Handle find_produce(TensorInfo* dest, Range range);
    } m_buffer;

    //! config whether raise error exactly when invoking op.
    //! level 2: both device and user side errors are async;
    //! level 1: user side errors are sync;
    //! level 0: both sync.
    int m_async_level = 2;

    struct Scope {
        std::string name;
        std::unordered_map<std::string, std::unique_ptr<Scope>> children;
        size_t version = 0;
        size_t parent_version = 0;
        size_t tensor_count = 0;
        Scope* active_child = nullptr;
        Scope* parent = nullptr;

        Scope* enter(std::string name) {
            auto& child = children[name];
            if (!child) {
                child = std::make_unique<Scope>();
                child->name = name;
                child->parent = this;
            }
            if (version != child->parent_version) {
                child->version = 0;
                child->parent_version = version;
            } else {
                child->version++;
            }
            child->tensor_count = 0;
            return active_child = child.get();
        }

        Scope* exit(std::string name) {
            mgb_assert(this->name == name, "scope name mismatch");
            parent->active_child = nullptr;
            return parent;
        }
    };

    class ScopeManager {
    private:
        Scope m_root;
        Scope* m_current_scope = &m_root;
    public:
        class ScopeGuard{
        private:
            ScopeManager* m_manager;
            std::string m_name;
        public:
            ScopeGuard(ScopeManager* manager, std::string name): m_manager{manager}, m_name{name} {
                m_manager->push(m_name);
            }
            ~ScopeGuard() {
                m_manager->pop(m_name);
            }
        };
        void push(std::string name) {
            m_current_scope = m_current_scope->enter(name);
        }
        void pop(std::string name) {
            m_current_scope = m_current_scope->exit(name);
        }
        std::string next_tensor_name() {
            std::string builder;
            Scope* scope = &m_root;
            while (true) {
                builder.append(scope->name);
                if (scope->version != 0) {
                    builder.append(ssprintf("(%ld)", scope->version));
                }
                if (scope != &m_root) {
                    builder.append(".");
                }
                if (scope->active_child == nullptr) {
                    builder.append(ssprintf(":%%%ld", scope->tensor_count++));
                    break;
                } else {
                    scope = scope->active_child;
                }
            }
            return builder;
        }
    };

    struct State {
        std::thread::id tid;
        OptionManager options;
    };

    struct ChannelState: State {
        ScopeManager scopes;
    };

    struct WorkerState: State {};

    ChannelState m_channel_state;
    WorkerState m_worker_state;

    /*!
     * \brief A framework of dynamic sublienar memory optimization
     *
     * Note: The main idea is that during the training process, if the memory
     * usage exceeds the threshold, select some tensors to evict until the
     * memory usage is below the threshold.
     */
    struct DynamicSublinear {
        /*!
         * \brief find an available tensor with the largest evaluation function
         *
         * Note: An available tensor must satisfy: (1) has computing path,
         * (2) is in memory, (3) is not pinned. Evaluation function refers to:
         * @see: TensorInfo::eval_func.
         *
         * \return the pointer of the best tensor; nullptr is returned if no 
         * available tensor is found
         */
        TensorInfo* find_best_tensor(bool);

        /*!
         * \brief estimate the cost of recomputing tensor ptr
         *
         * Note: We define the cost as the sum of the costs of each evicted
         * components where all the neighbors of ptr are located.
         */
        double estimate_neighbor_cost(TensorInfo* ptr);

        /*!
         * \brief update the last used time of the tensor ptr
         */
        void update_used_time(TensorInfo* ptr);

        /*!
         * \brief merge the two specified sets (the set in which the element x
         * is located, and the set in which the element y is located)
         */
        void merge(std::shared_ptr<DsuNode> &x, std::shared_ptr<DsuNode> &y);

        /*!
         * \brief return the representative of the set that contains the
         * element x
         */
        std::shared_ptr<DsuNode> find_father(std::shared_ptr<DsuNode> &x);

        /*!
         * \brief update DSU after recomputing tensor ptr
         *
         * Delete ptr from the set where ptr is located. Since DSU does not
         * support this operation, instead, we reset the DSU father of ptr, and
         * subtract the recomputation cost of ptr from the cost of the original
         * set.
         */
        void update_dsu_after_recompute(TensorInfo* ptr);

        /*!
         * \brief update DSU after evicting tensor ptr
         *
         * Check the neighbors of x, that is, the input and output tensors, and
         * if they are evicted, merge their respective sets.
         */
        void update_dsu_after_evict(TensorInfo* ptr);

        /*!
         * \brief pin the tensors in vec
         */
        void pin(const SmallVector<TensorInfo*>& vec);

        /*!
         * \brief unpin the tensors in vec
         */
        void unpin(const SmallVector<TensorInfo*>& vec);

        /*!
         * \brief add the tensor to the candidate set
         *
         * If the size of the tensor does not exceed the minimum threshold,
         * it will do nothing.
         */
        void insert_candidate(TensorInfo* ptr);

        /*!
         * \brief erase the tensor from the candidate set
         *
         * If the size of the tensor does not exceed the minimum threshold,
         * it will do nothing.
         */
        void erase_candidate(TensorInfo* ptr);

        //! estimate the current time, in order to reduce the overhead of timer
        double estimate_timestamp = 0;

        //! the comp node where dynamic sublinear memory optimization works
        CompNode comp_node;

        //! store all tensors that may be evicted
        std::unordered_set<TensorInfo*> candidates;

        bool is_bad_op(std::string op_name) {
            return std::find(op_blacklist.begin(), op_blacklist.end(), op_name) != op_blacklist.end();
        }

        std::vector<std::string> op_blacklist = {"CollectiveComm", "InplaceAdd",
                                "ParamPackSplit", "ParamPackConcat", "GaussianRNG", "UniformRNG",
                                "GammaRNG", "PermutationRNG", "PoissonRNG", "BetaRNG"};
    } m_dtr;

    //! automatically evict an optimal tensor
    bool auto_evict(size_t);

    void alloc_tensor_with_evict(Blob*);

    // assert thread id when call get_xxx_state to avoid misuse
    ChannelState& get_channel_state();
    WorkerState& get_worker_state();
};

} // namespace mgb::imperative::interpreter::intl
