/**
 * \file imperative/src/impl/interpreter_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./interpreter_impl.h"


using namespace mgb;
using namespace imperative;
using namespace interpreter;
using namespace interpreter::intl;


std::unique_ptr<Interpreter::Channel> InterpreterImpl::create_channel() {
    return std::make_unique<ChannelImpl>();
}

Interpreter& Interpreter::inst() {
    static InterpreterImpl inst_;
    return inst_;
}

void* ChannelImpl::put(const HostTensorND& value) {
    auto info = alloc();
    info->desc.layout = value.layout();
    info->desc.comp_node = value.comp_node();
    info->desc.value = value.proxy_to_default_cpu();
    m_valid_handle.insert(info);
    m_worker.add_task(Put{info, value});
    return info;
}

void* ChannelImpl::put(const DeviceTensorND& data) {
    auto info = alloc();
    info->desc.layout = data.layout();
    info->desc.comp_node = data.comp_node();
    info->ptr = Tensor::make(data);
    m_valid_handle.insert(info);
    return info;
}

void ChannelImpl::del(void* handle) {
    mgb_assert(m_valid_handle.erase(handle), "invalid handle: %p", handle);
    m_worker.add_task(Del{reinterpret_cast<TensorInfo*>(handle)});
}

SmallVector<void*> ChannelImpl::apply_op(
        std::shared_ptr<OpDef> op,
        const SmallVector<void*>& inputs) {
    SmallVector<TensorInfo*> input_infos;
    input_infos.reserve(inputs.size());
    SmallVector<LogicalTensorDesc> input_descs;
    input_descs.reserve(inputs.size());
    for (auto i : inputs) {
        auto info = reinterpret_cast<TensorInfo*>(i);
        input_infos.push_back(info);
        input_descs.push_back(info->desc);
    }
    auto output_descs = OpDef::infer_output_attrs_fallible(*op, input_descs);
    ApplyOp cmd{std::move(op)};
    cmd.inputs = std::move(input_infos);
    cmd.outputs.reserve(output_descs.size());
    SmallVector<void*> outputs;
    bool is_fallible = false;
    for (auto&& desc : output_descs) {
        if (desc.layout.ndim == 0) {
            is_fallible = true;
        }
        auto info = alloc();
        info->desc = desc;
        m_valid_handle.insert(info);
        cmd.outputs.push_back(info);
        outputs.push_back(info);
    }
    m_worker.add_task(std::move(cmd));
    if (is_fallible && m_async_level <= 1) {
        sync();
    }
    return outputs;
}

HostTensorND ChannelImpl::get_value(void* handle) {
    mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
               "invalid handle: %p", handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);
    mgb_assert(!m_waitee);
    if (!info->value_fetched) {
        m_waitee = info;
        m_worker.add_task(GetValue{info});
        m_cv.wait(lock, [&]() {
            check_worker_exc_unsafe();
            return info->value_fetched;
        });
        m_waitee = nullptr;
    }
    mgb_assert(info->ptr->value_fetched());
    return info->ptr->get_value();
}

TensorShape ChannelImpl::get_shape(void* handle) {
    mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
               "invalid handle: %p", handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    if (info->desc.layout.ndim != 0) {
        return info->desc.layout;
    }
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);
    mgb_assert(!m_waitee);
    m_waitee = info;
    m_cv.wait(lock, [&]() {
        check_worker_exc_unsafe();
        return bool(info->ptr);
    });
    m_waitee = nullptr;
    TensorShape ret = info->ptr->layout();
    mgb_assert(ret.ndim != 0);
    return ret;
}

DType ChannelImpl::get_dtype(void* handle) {
    mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
               "invalid handle: %p", handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    auto ret = info->desc.layout.dtype;
    mgb_assert(ret.valid());
    return ret;
}

CompNode ChannelImpl::get_device(void* handle) {
    mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
               "invalid handle: %p", handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    auto ret = info->desc.comp_node;
    mgb_assert(ret.valid());
    return ret;
}

DeviceTensorND ChannelImpl::get_dev_tensor(void* handle) {
    mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
               "invalid handle: %p", handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);
    mgb_assert(!m_waitee);
    m_waitee = info;
    m_cv.wait(lock, [&]() {
        check_worker_exc_unsafe();
        return bool(info->ptr);
    });
    m_waitee = nullptr;
    return info->ptr->dev_tensor();
}

void ChannelImpl::sync() {
    m_worker.wait_all_task_finish();
    MGB_LOCK_GUARD(m_mutex);
    check_worker_exc_unsafe();
}

void ChannelImpl::close() {
    sync();
}

void ChannelImpl::config_async_level(int level) {
    mgb_assert(level <= 2 and level >= 0, "async_level should be 0, 1 or 2");
    m_async_level = level;
}

int ChannelImpl::get_async_level() {
    return m_async_level;
}

TensorInfo* ChannelImpl::alloc() {
    MGB_LOCK_GUARD(m_mutex);
    return m_pool.alloc();
}

void ChannelImpl::free(TensorInfo* ptr) {
    MGB_LOCK_GUARD(m_mutex);
    m_pool.free(ptr);
}

ChannelImpl::~ChannelImpl() {
    close();
}

void ChannelImpl::produce_tensor(TensorInfo* dest, TensorPtr ptr) {
    MGB_LOCK_GUARD(m_mutex);
    dest->value_fetched = ptr->value_fetched();
    dest->ptr = std::move(ptr);
    if (m_waitee == dest) {
        m_cv.notify_all();
    }
}

void ChannelImpl::process_one_task(Command& cmd) {
    //TODO: remove std::visit for support osx 10.12
    std::visit([this](auto& cmd) {
        using T = std::remove_reference_t<decltype(cmd)>;
        try {
            if constexpr (std::is_same_v<T, Put>) {
                produce_tensor(cmd.dest, Tensor::make(cmd.value));
            } else if constexpr (std::is_same_v<T, ApplyOp>) {
                SmallVector<TensorPtr> tensor_inputs;
                tensor_inputs.reserve(cmd.inputs.size());
                for (auto i : cmd.inputs) {
                    tensor_inputs.push_back(i->ptr);
                }
                auto tensor_outputs = OpDef::apply_on_physical_tensor(*cmd.op, tensor_inputs);
                mgb_assert(tensor_outputs.size() == cmd.outputs.size());
                for (size_t i = 0; i < tensor_outputs.size(); ++i) {
                    produce_tensor(cmd.outputs[i], std::move(tensor_outputs[i]));
                }
            } else if constexpr (std::is_same_v<T, Del>) {
                free(cmd.dest);
            } else if constexpr (std::is_same_v<T, GetValue>) {
                cmd.dest->ptr->fetch_value();
                MGB_LOCK_GUARD(m_mutex);
                cmd.dest->value_fetched = true;
                if (m_waitee == cmd.dest) {
                    m_cv.notify_all();
                }
            } else {
                static_assert(!std::is_same_v<T, T>);
            }
        } catch (...) {
            MGB_LOCK_GUARD(m_mutex);
            m_worker_exc = std::current_exception();
            m_cv.notify_all();
        }
    }, cmd);
}


void ChannelImpl::check_worker_exc_unsafe() {
    if (m_worker_exc) {
        std::exception_ptr exc;
        std::swap(exc, m_worker_exc);
        std::rethrow_exception(exc);
    }
}
