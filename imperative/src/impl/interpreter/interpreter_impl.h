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
#include "megbrain/imperative/interpreter.h"
#include "megbrain/imperative/profiler.h"
#include "megbrain/utils/mempool.h"

#include "./commands.h"
#include "./option_manager.h"
#include "./stack_manager.h"
#include "./tensor_info.h"

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
    void drop(Handle) override;

    SmallVector<Handle> apply_op(
            std::shared_ptr<OpDef> op, const SmallVector<Handle>& inputs) override;

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
    void clear_candidates() override;

    void start_profile() override;
    void stop_profile() override;

    void push_scope(std::string) override;
    void pop_scope(std::string) override;

private:
    struct WorkQueue;
    struct State;

    TensorInfo* alloc();
    void init(TensorInfo*, LogicalTensorDesc&& desc);
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
            std::shared_ptr<OpDef> op, const SmallVector<Handle>& inputs);
    TensorPtr wait_tensor(TensorInfo* info, profiler::TensorProp prop);
    void notify_tensor_unsafe(TensorInfo* info);

    void process_one_task(Command&);

    void check_worker_exc_unsafe();

    void produce_tensor(TensorInfo* dest, TensorPtr ptr);

    void release_tensor(TensorInfo* dest);

    void regenerate(TensorInfo* dest);
    void flush_apply_stack();
    void do_apply_op(const ApplyOp& cmd, std::string reason);

    void dispatch_default_cpu(
            std::shared_ptr<OpDef> op, const SmallVector<TensorInfo*>& input_infos,
            const SmallVector<LogicalTensorDesc>& input_descs,
            SmallVector<Handle>* outputs);
    void dispatch_kernel(
            std::shared_ptr<OpDef> op, const SmallVector<TensorInfo*>& input_infos,
            const SmallVector<LogicalTensorDesc>& input_descs,
            SmallVector<Handle>* outputs);

    void push_scope(std::string, State&);
    void pop_scope(std::string, State&);

    void assert_in_channel();
    void assert_in_worker();
    std::thread::id get_worker_tid();

    void sample_on_device(CompNode device, bool force);

    // valid => status != Deleted
    std::unordered_set<TensorInfo*> collect_valid_tensors();

    std::mutex m_mutex;
    Spinlock m_spin;
    std::condition_variable m_cv;
    MemPool<TensorInfo> m_pool;
    std::unordered_set<Handle> m_valid_handle;
    TensorInfo* m_waitee = nullptr;
    Spinlock m_pool_spin;
    Spinlock m_info_spin;
    uint64_t m_waitee_id = 0;
    std::exception_ptr m_worker_exc;
    std::function<void(std::string, std::string)> m_profile_dump_callback;
    size_t m_storage_id = 0;
    // TODO: use explicit struct
    std::stack<std::tuple<ApplyOp, size_t, TensorInfo*, std::string>> m_apply_stack;
    bool m_applying = false;
    bool m_closed = false;

    struct WorkQueue : AsyncQueueSC<Command, WorkQueue> {
        // set max_spin=0 to prevent Queue fetch task in busy wait manner.
        // this won't affect throughput when python interpreter is sending enough task,
        // but will significantly save CPU time when waiting for task, e.g. wait for
        // data input limit pending tasks to 10000
        WorkQueue(ChannelImpl* owner)
                : AsyncQueueSC<Command, WorkQueue>(0, 10000), m_owner(owner) {
            sys::set_thread_name("interpreter");
            if (const char* env_val = MGB_GETENV("MEGENGINE_ASYNC_QUEUE_SIZE")) {
                int len = strlen(env_val);
                for (int i = 0; i < len; i++) {
                    mgb_assert(
                            env_val[i] >= '0' && env_val[i] <= '9',
                            "async queue size should be an integer");
                }
                size_t val;
                sscanf(env_val, "%zu", &val);
                update_max_items(val);
            }
        }
        void process_one_task(Command& icmd) { m_owner->process_one_task(icmd); }
        void on_async_queue_worker_thread_start() override;

    private:
        ChannelImpl* m_owner;
    } m_worker;

    //! config whether raise error exactly when invoking op.
    //! level 2: both device and user side errors are async;
    //! level 1: user side errors are sync;
    //! level 0: both sync.
    int m_async_level = 2;

    struct State {
        std::thread::id tid;
        OptionManager options;
    };

    struct ChannelState : State {
        StackManager stack_manager;
    };

    struct WorkerState : State {};

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
        void merge(std::shared_ptr<DsuNode>& x, std::shared_ptr<DsuNode>& y);

        /*!
         * \brief return the representative of the set that contains the
         * element x
         */
        std::shared_ptr<DsuNode> find_father(std::shared_ptr<DsuNode>& x);

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
        void unpin(const SmallVector<TensorInfo*>& vec, WorkerState& state);

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
        SmallVector<TensorInfo*> candidates;

        bool is_bad_op(std::string op_name) {
            return std::find(op_blacklist.begin(), op_blacklist.end(), op_name) !=
                   op_blacklist.end();
        }

        // operators that cannot be re-computed, including :
        // distributed operators, inplace operator, random generator operators
        std::vector<std::string> op_blacklist = {
                "CollectiveComm", "InplaceAdd", "ParamPackSplit", "ParamPackConcat",
                "GaussianRNG",    "UniformRNG", "GammaRNG",       "PermutationRNG",
                "PoissonRNG",     "BetaRNG"};
    } m_dtr;

    //! automatically evict an optimal tensor
    bool auto_evict(size_t);

    void alloc_tensor_with_evict(Blob*);

    // assert thread id when call get_xxx_state to avoid misuse
    ChannelState& get_channel_state();
    WorkerState& get_worker_state();
};

}  // namespace mgb::imperative::interpreter::intl
