/**
 * \file imperative/src/include/megbrain/imperative/interpreter.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <atomic>

#include "megbrain/imperative/op_def.h"

namespace mgb::imperative::interpreter {

struct Interpreter {
    using Handle = void*;

    struct Channel {
        virtual ~Channel() = default;

        virtual Handle put(const HostTensorND& value, bool no_cache) = 0;
        virtual Handle put(const DeviceTensorND& value) = 0;

        virtual void del(Handle) = 0;
        virtual void swap_in(Handle) = 0;
        virtual void swap_out(Handle) = 0;
        virtual void drop(Handle) = 0;

        virtual SmallVector<Handle> apply_op(
                std::shared_ptr<OpDef> op,
                const SmallVector<Handle>& inputs) = 0;

        virtual HostTensorND get_value(Handle) = 0;
        virtual TensorShape get_shape(Handle) = 0;
        virtual DType get_dtype(Handle) = 0;
        virtual CompNode get_device(Handle) = 0;

        virtual DeviceTensorND get_dev_tensor(Handle) = 0;

        virtual void sync() = 0;
        virtual void close() = 0;
        virtual void set_swap_flag(bool) = 0;
        virtual void set_drop_flag(bool) = 0;
        virtual void set_buffer_length(int) = 0;

        virtual void config_async_level(int level) = 0;
        virtual int get_async_level() = 0;
    };

    virtual std::unique_ptr<Channel> create_channel() = 0;

    static Interpreter& inst();
};

} // namespace mgb::imperative::interpreter
