/**
 * \file imperative/src/impl/interpreter_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <deque>
#include <future>
#include <list>
#include <unordered_set>
#include <variant>

#include "megbrain/utils/mempool.h"
#include "megbrain/imperative/interpreter.h"

namespace mgb::imperative::interpreter::intl {

using Handle = Interpreter::Handle;

struct InterpreterImpl : Interpreter {
    std::unique_ptr<Channel> create_channel() override;
};

enum EvictType {
    NONE = 0,
    SWAP = 1,
    DROP = 2,
};

struct TensorInfo;
using TensorInfoPtr = std::shared_ptr<TensorInfo>;

struct TensorInfo {
    TensorPtr ptr;
    LogicalTensorDesc desc;
    bool value_fetched = false;
    bool invalid = false;
    bool allow_delete = false;

    EvictType evict_type = NONE;

    HostTensorND h_value;
    size_t locked = 0;
    size_t recompute_times = 0;

    struct ComputePath {
        std::shared_ptr<OpDef> op;
        SmallVector<TensorInfoPtr> inputs;
        SmallVector<std::weak_ptr<TensorInfo>> outputs;
        SmallVector<std::weak_ptr<TensorInfo>> dep_outputs;
    } path;
};

struct Put {
    TensorInfo* dest;
    HostTensorND value;
    bool no_cache = false;

    std::string to_string() const { return ssprintf("Command: Put %p", dest); }
};
struct ApplyOp {
    std::shared_ptr<OpDef> op;
    SmallVector<TensorInfo*> inputs;
    SmallVector<TensorInfo*> outputs;
    SmallVector<TensorInfo*> dels;

    std::string to_string() const {
        std::string builder{"Command: ApplyOp {"};
        builder += "inputs [";
        for (auto* input : inputs) {
            builder += ssprintf("%p, ", input);
        }
        builder += "], outputs [";
        for (auto* output : outputs) {
            builder += ssprintf("%p, ", output);
        }
        builder += "], dels [";
        for (auto* del : dels) {
            builder += ssprintf("%p, ", del);
        }
        builder += "]";
        return builder;
    }
};
struct Del {
    TensorInfo* dest;

    std::string to_string() const { return ssprintf("Command: Del %p", dest); }
};
struct GetValue {
    TensorInfo* dest;

    std::string to_string() const {
        return ssprintf("Command: GetValue %p", dest);
    }
};
struct SwapIn {
    TensorInfo* dest;

    std::string to_string() const {
        return ssprintf("Command: SwapIn %p", dest);
    }
};
struct SwapOut {
    TensorInfo* dest;

    std::string to_string() const {
        return ssprintf("Command: SwapOut %p", dest);
    }
};
struct Drop {
    TensorInfo* dest;

    std::string to_string() const {
        return ssprintf("Command: Drop %p", dest);
    }
};
struct Move {
    TensorInfo* src;
    TensorInfo* dest;

    std::string to_string() const {
        return ssprintf("Command: Move %s to %s",
                        src->desc.layout.to_string().c_str(),
                        dest->desc.layout.to_string().c_str());
    }
};
struct Flush {
    TensorInfo* dest = nullptr;

    std::string to_string() const {
        return ssprintf("Command: Flush %p", dest);
    }
};
struct Nop {
    std::string to_string() const { return "Command: Nop"; }
};
using Command = std::variant<Put,
                             ApplyOp,
                             Del,
                             GetValue,
                             SwapIn,
                             SwapOut,
                             Drop,
                             Move,
                             Flush,
                             Nop>;

struct ChannelImpl : Interpreter::Channel {
    ChannelImpl() : m_worker(this), m_buffer(this) {}
    ~ChannelImpl() override;

    Handle put(const HostTensorND& value, bool no_cache) override;
    Handle put(const DeviceTensorND& value) override;

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

    void sync() override;
    void close() override;
    void set_swap_flag(bool) override;
    void set_drop_flag(bool) override;
    void set_buffer_length(int) override;

    void config_async_level(int level) override;
    int get_async_level() override;

private:
    TensorInfo* alloc();
    void free(TensorInfo*);
    void remove_dep(TensorInfo*);

    void process_one_task(Command&);

    void check_worker_exc_unsafe();

    void produce_tensor(TensorInfo* dest, TensorPtr ptr, bool notice);
    void do_swap_out(TensorInfo* dest);
    void do_swap_in(TensorInfo* dest);
    void do_drop(TensorInfo* dest);
    void regenerate(TensorInfo* dest, bool must_drop);

    std::mutex m_mutex;
    std::condition_variable m_cv;
    MemPool<TensorInfo> m_pool;
    std::unordered_set<Handle> m_valid_handle;
    TensorInfo* m_waitee = nullptr;
    std::exception_ptr m_worker_exc;
    size_t m_enable_evict = 0;

    struct WorkQueue : AsyncQueueSC<Command, WorkQueue> {
        // set max_spin=0 to prevent Queue fetch task in busy wait manner.
        // this won't affect throughput when python interpreter is sending enough task,
        // but will significantly save CPU time when waiting for task, e.g. wait for data input
        WorkQueue(ChannelImpl* owner)
                : AsyncQueueSC<Command, WorkQueue>(0), m_owner(owner) {
            sys::set_thread_name("interpreter");
        }
        void process_one_task(Command& cmd) {
            m_owner->process_one_task(cmd);
        }
        void on_async_queue_worker_thread_start() override {
               sys::set_thread_name("worker");
        }
    private:
        ChannelImpl* m_owner;
    } m_worker;

    struct SharedTensorInfoMap {
        void insert(TensorInfo* info) {
            MGB_LOCK_GUARD(mtx);
            tmap.emplace(info, TensorInfoPtr{info, [](TensorInfo* ptr){ ptr->allow_delete = true;}});
        }
        void erase(TensorInfo* info) {
            MGB_LOCK_GUARD(mtx);
            tmap.erase(info);
        }
        TensorInfoPtr at(TensorInfo* info) {
            MGB_LOCK_GUARD(mtx);
            return tmap.at(info);
        }
    private:
        std::mutex mtx;
        std::unordered_map<TensorInfo*, TensorInfoPtr> tmap;
    }m_st;

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
        CommandBuffer(ChannelImpl* owner) : m_owner(owner) {
            int capacity = 3;
            if(const char* capacity_str = MGB_GETENV("MEGENGINE_COMMAND_BUFFER_LENGTH")) {
                capacity = atoi(capacity_str);
            }
            set_capacity(capacity);
        }
        void enqueue(Command cmd);
        bool empty() const {
            return m_commands.empty();
        }
        void set_capacity(int capacity) {
            mgb_assert(capacity >= 0 && capacity < 100, "invalid command buffer length");
            m_capacity = capacity;
        }
    private:
        ChannelImpl* m_owner;
        size_t m_capacity;
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
    int m_max_recompute_time = 1;
};

} // namespace mgb::imperative::interpreter::intl
