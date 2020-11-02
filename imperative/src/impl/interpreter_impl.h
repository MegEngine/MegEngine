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

#include <variant>
#include <future>

#include "megbrain/utils/mempool.h"
#include "megbrain/imperative/interpreter.h"


namespace mgb::imperative::interpreter::intl {

using Handle = Interpreter::Handle;

struct InterpreterImpl : Interpreter {
    std::unique_ptr<Channel> create_channel() override;
};

struct TensorInfo {
    TensorPtr ptr;
    LogicalTensorDesc desc;
    bool value_fetched = false;
};

struct Put {
    TensorInfo* dest;
    HostTensorND value;
};
struct ApplyOp {
    std::shared_ptr<OpDef> op;
    SmallVector<TensorInfo*> inputs;
    SmallVector<TensorInfo*> outputs;
};
struct Del {
    TensorInfo* dest;
};
struct GetValue {
    TensorInfo* dest;
};
using Command = std::variant<Put,
                             ApplyOp,
                             Del,
                             GetValue>;

struct ChannelImpl : Interpreter::Channel {
    ChannelImpl() : m_worker(this) {}
    ~ChannelImpl() override;

    Handle put(const HostTensorND& value) override;
    Handle put(const DeviceTensorND& value) override;

    void del(Handle) override;

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

    void config_async_level(int level) override;
    int get_async_level() override;

private:
    TensorInfo* alloc();
    void free(TensorInfo*);

    void process_one_task(Command&);

    void check_worker_exc_unsafe();

    void produce_tensor(TensorInfo* dest, TensorPtr ptr);

    std::mutex m_mutex;
    std::condition_variable m_cv;
    MemPool<TensorInfo> m_pool;
    std::unordered_set<Handle> m_valid_handle;
    TensorInfo* m_waitee = nullptr;
    std::exception_ptr m_worker_exc;

    struct WorkQueue : AsyncQueueSC<Command, WorkQueue> {
        WorkQueue(ChannelImpl* owner) : m_owner(owner) {}
        void process_one_task(Command& cmd) {
            m_owner->process_one_task(cmd);
        }
    private:
        ChannelImpl* m_owner;
    } m_worker;

    //! config whether raise error exactly when invoking op.
    //! level 2: both device and user side errors are async;
    //! level 1: user side errors are sync;
    //! level 0: both sync.
    int m_async_level = 1;
};

} // namespace mgb::imperative::interpreter::intl
