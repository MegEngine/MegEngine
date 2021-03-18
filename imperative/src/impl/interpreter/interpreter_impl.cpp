/**
 * \file imperative/src/impl/interpreter/interpreter_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./interpreter_impl.h"

#include "megbrain/common.h"
#include "megbrain/imperative/opr_utility.h"
#include "megbrain/imperative/ops/backward_graph.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/utils/to_string.h"

#include "../op_trait.h"

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

Handle ChannelImpl::put(const HostTensorND& value, bool no_cache) {
    auto info = alloc();
    info->desc.layout = value.layout();
    info->desc.comp_node = value.comp_node();
    info->desc.value = value.proxy_to_default_cpu();
    info->h_value = value;
    m_buffer.enqueue(Put{info, value, no_cache});
    if (m_async_level == 0) {
        sync();
        info->desc.comp_node.sync();
    }
    return info;
}

Handle ChannelImpl::put(const DeviceTensorND& data) {
    auto info = alloc();
    info->desc.layout = data.layout();
    info->desc.comp_node = data.comp_node();
    info->ptr = Tensor::make(data);
    if (m_channel_state.profiler->is_profiling()) {
        m_channel_state.profiler->record_host<TensorProduceEvent>(info->id, info->desc.layout, info->desc.comp_node);
    }
    return info;
}

void ChannelImpl::del(Handle handle) {
    mgb_assert(m_valid_handle.count(handle), "invalid handle: %p", handle);
    auto* info = reinterpret_cast<TensorInfo*>(handle);
    detach_users(info);
    info->detach_producer();
    m_valid_handle.erase(handle);
    m_buffer.enqueue(Del{info});
}

void ChannelImpl::swap_in(Handle handle) {
    if (m_worker_state.options.enable_swap) {
        mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
                "invalid handle: %p", handle);
        auto* info = reinterpret_cast<TensorInfo*>(handle);
        m_buffer.enqueue(SwapIn{info});
        info->evict_type = NONE;
    }
}

void ChannelImpl::swap_out(Handle handle) {
    if (m_worker_state.options.enable_swap) {
        mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
                "invalid handle: %p", handle);
        auto* info = reinterpret_cast<TensorInfo*>(handle);
        m_buffer.enqueue(SwapOut{info});
        info->evict_type = SWAP;
    }
}

void ChannelImpl::drop(Handle handle) {
    if (m_worker_state.options.enable_drop) {
        mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
                "invalid handle: %p", handle);
        auto* info = reinterpret_cast<TensorInfo*>(handle);
        if (!info->producer) {
            mgb_log_warn("the input that produced tensor %p has been deleted, this drop operation will be ignored", info);
            return;
        }
        info->evict_type = DROP;
        m_buffer.enqueue(Drop{info});
    }
}

void ChannelImpl::dispatch_default_cpu(
        std::shared_ptr<OpDef> op,
        const SmallVector<TensorInfo*>& input_infos,
        const SmallVector<LogicalTensorDesc>& input_descs,
        SmallVector<Handle>* outputs) {
    auto [output_descs, validated] = OpDef::infer_output_attrs_fallible(*op, input_descs);
    MGB_MARK_USED_VAR(validated);

    SmallVector<DeviceTensorND> input_tensornds;
    input_tensornds.reserve(input_descs.size());
    CompNode output_cn;
    {
        MGB_LOCK_GUARD(m_mutex);
        for (auto&& info : input_infos) {
            auto input_cn = info->desc.comp_node;
            if (!output_cn.valid()) {
                output_cn = input_cn;
            } else {
                mgb_assert(output_cn == input_cn, "cannot decide output comp node");
            }

            if (info->ptr && info->ptr->try_get_value()) {
                input_tensornds.emplace_back(info->ptr->get_value().proxy_to_default_cpu());
            } else {
                mgb_assert(!info->h_value.empty(), "inp->h_value is empty!");
                input_tensornds.emplace_back(info->h_value.proxy_to_default_cpu());
            }
        }
    }

    outputs->reserve(output_descs.size());
    SmallVector<DeviceTensorND> output_tensornds;
    output_tensornds.reserve(output_descs.size());
    for (auto&& desc : output_descs) {
        // TODO: may conflict with condtake, which need alloc inside
        mgb_assert(!desc.layout.is_empty());
        // use HostTensorND alloc_host for cuda pinned memory
        output_tensornds.emplace_back(HostTensorND(output_cn, desc.layout).proxy_to_default_cpu());
    }

    auto tinfo_to_tid = [&](SmallVector<TensorInfo*> tinfo) {
        SmallVector<uint64_t> tid;
        for (auto* ptinfo: tinfo) {
            tid.push_back(ptinfo->id);
        }
        return tid;
    };
    OpEvent event_data = {++m_last_id, op, tinfo_to_tid(input_infos), {}};
    if (m_channel_state.profiler->is_profiling()) {
        m_channel_state.profiler->record_host<HostOpExecuteEvent>(event_data);
    }

    OpDef::apply_on_device_tensornd(*op, input_tensornds, &output_tensornds);

    SmallVector<TensorInfo*> output_infos;
    output_infos.reserve(output_descs.size());
    for (auto&& tensornd : output_tensornds) {
        HostTensorND host_tensornd = HostTensorND::make_proxy(tensornd)
            .proxy_to_comp_node(output_cn);
        // use `put` for consistency
        auto info = reinterpret_cast<TensorInfo*>(put(host_tensornd, false));
        mgb_assert(info->desc.layout.ndim != 0);
        output_infos.push_back(info);
        outputs->push_back(info);
    }

    if (m_channel_state.options.enable_drop) {
        TensorInfo::ComputePath::make(op, input_infos, output_infos);
    }

    event_data.outputs = tinfo_to_tid(output_infos);
    if (m_channel_state.profiler->is_profiling()) {
        m_channel_state.profiler->record_host<HostOpFinishEvent>(event_data);
    }
}

void ChannelImpl::dispatch_kernel(
        std::shared_ptr<OpDef> op,
        const SmallVector<TensorInfo*>& input_infos,
        const SmallVector<LogicalTensorDesc>& input_descs,
        SmallVector<Handle>* outputs) {
    auto [output_descs, validated] = OpDef::infer_output_attrs_fallible(*op, input_descs);

    ApplyOp cmd{std::move(op)};
    cmd.inputs = std::move(input_infos);
    cmd.outputs.reserve(output_descs.size());
    outputs->reserve(output_descs.size());
    for (auto&& desc : output_descs) {
        auto info = alloc();
        info->desc = desc;
        // make sure desc's value is consistent with h_value
        if (!info->desc.value.empty()) {
            info->h_value = HostTensorND::make_proxy(desc.value)
                .proxy_to_comp_node(desc.comp_node);
        }
        cmd.outputs.push_back(info);
        outputs->push_back(info);
    }
    if (m_channel_state.options.enable_drop) {
        TensorInfo::ComputePath::make(cmd.op, cmd.inputs, cmd.outputs);
    }
    m_buffer.enqueue(std::move(cmd));
    if (!validated && m_channel_state.options.async_level == 1) {
        sync();
    } else if (m_channel_state.options.async_level == 0) {
        sync();
        // check device error
        for (auto&& oup : *outputs) {
            auto info = reinterpret_cast<TensorInfo*>(oup);
            info->ptr->comp_node().sync();
        }
    }
}

SmallVector<Handle> ChannelImpl::apply_op(
        std::shared_ptr<OpDef> op,
        const SmallVector<Handle>& inputs) {
    for (auto i : inputs) {
        mgb_assert(m_valid_handle.find(i) != m_valid_handle.end(),
                "invalid handle: %p", i);
    }
    SmallVector<TensorInfo*> input_infos;
    input_infos.reserve(inputs.size());
    SmallVector<LogicalTensorDesc> input_descs;
    input_descs.reserve(inputs.size());
    {
        MGB_LOCK_GUARD(m_mutex);
        for (auto i : inputs) {
            auto info = reinterpret_cast<TensorInfo*>(i);
            mgb_assert(!info->invalid, "Invalid tensor, unable to apply_op!");
            input_infos.push_back(info);
            input_descs.push_back(info->desc);
            regenerate(info);
        }
    }

    SmallVector<Handle> outputs;
    DispatchMode dispatch_mode = m_channel_state.options.enable_host_compute
            ? OpDef::decide_dispatch_mode(*op, input_descs)
            : DispatchMode::KERNEL;
    switch (dispatch_mode) {
        case DEFAULT_CPU: {
            dispatch_default_cpu(op, input_infos, input_descs, &outputs);
            break;
        }
        case KERNEL: {
            dispatch_kernel(op, input_infos, input_descs, &outputs);
            break;
        }
    }
    return outputs;
}

HostTensorND ChannelImpl::get_value(Handle handle) {
    // TODO: maybe get_value should be done on host. i.e. delete GetValue
    mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
               "invalid handle: %p", handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    mgb_assert(!m_waitee);
    // donnot use info->value_fetched, it's unsafe
    mgb_assert(!info->invalid, "Invalid tensor, unable to get_value!");
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);
    TensorPtr tensor_ptr = info->ptr;
    auto value_fetched = [&]() {
        return tensor_ptr && tensor_ptr->value_fetched();
    };
    if (!value_fetched()) {
        m_waitee = info;
        regenerate(info);
        m_buffer.enqueue(GetValue{info});
        if (m_channel_state.profiler->is_profiling()) {
            m_channel_state.profiler->record_host<TensorWaitPropEvent>(info->id, TensorInfo::HostValue);
        }
        m_cv.wait(lock, [&]() {
            check_worker_exc_unsafe();
            tensor_ptr = info->ptr;
            return value_fetched();
        });
        if (m_channel_state.profiler->is_profiling()) {
            m_channel_state.profiler->record_host<TensorWaitPropFinishEvent>(info->id, TensorInfo::HostValue);
        }
        m_waitee = nullptr;
    }
    return tensor_ptr->get_value();
}

TensorShape ChannelImpl::get_shape(Handle handle) {
    mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
               "invalid handle: %p", handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    if (info->desc.layout.ndim != 0) {
        return info->desc.layout;
    }
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);
    mgb_assert(!m_waitee);
    m_waitee = info;
    m_buffer.flush();
    if (m_channel_state.profiler->is_profiling()) {
        m_channel_state.profiler->record_host<TensorWaitPropEvent>(info->id, TensorInfo::Shape);
    }
    m_cv.wait(lock, [&]() {
        check_worker_exc_unsafe();
        return static_cast<bool>(info->ptr);
    });
    if (m_channel_state.profiler->is_profiling()) {
        m_channel_state.profiler->record_host<TensorWaitPropFinishEvent>(info->id, TensorInfo::Shape);
    }
    m_waitee = nullptr;
    TensorShape ret = info->ptr->layout();
    mgb_assert(ret.ndim != 0);
    return ret;
}

DType ChannelImpl::get_dtype(Handle handle) {
    mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
               "invalid handle: %p", handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    if (m_channel_state.profiler->is_profiling()) {
        m_channel_state.profiler->record_host<TensorGetPropEvent>(info->id, TensorInfo::DType);
    }
    auto ret = info->desc.layout.dtype;
    mgb_assert(ret.valid());
    return ret;
}

CompNode ChannelImpl::get_device(Handle handle) {
    mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
               "invalid handle: %p", handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    if (m_channel_state.profiler->is_profiling()) {
        m_channel_state.profiler->record_host<TensorGetPropEvent>(info->id, TensorInfo::Device);
    }
    auto ret = info->desc.comp_node;
    mgb_assert(ret.valid());
    return ret;
}

DeviceTensorND ChannelImpl::get_dev_tensor(Handle handle) {
    mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
               "invalid handle: %p", handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);
    mgb_assert(!m_waitee);
    m_waitee = info;
    regenerate(info);
    m_buffer.flush();
    if (m_channel_state.profiler->is_profiling()) {
        m_channel_state.profiler->record_host<TensorWaitPropEvent>(info->id, TensorInfo::DevValue);
    }
    m_cv.wait(lock, [&]() {
        check_worker_exc_unsafe();
        return static_cast<bool>(info->ptr);
    });
    if (m_channel_state.profiler->is_profiling()) {
        m_channel_state.profiler->record_host<TensorWaitPropFinishEvent>(info->id, TensorInfo::DevValue);
    }
    m_waitee = nullptr;
    return info->ptr->dev_tensor();
}

void ChannelImpl::sync() {
    m_buffer.flush();
    if (m_channel_state.profiler->is_profiling()) {
        m_channel_state.profiler->record_host<SyncStartEvent>();
    }
    m_worker.wait_all_task_finish();
    CompNode::sync_all();
    if (m_channel_state.profiler->is_profiling()) {
        m_channel_state.profiler->record_host<SyncFinishEvent>();
    }
    MGB_LOCK_GUARD(m_mutex);
    check_worker_exc_unsafe();
}

void ChannelImpl::close() {
    sync();
}

int ChannelImpl::get_option(std::string name) {
    return m_channel_state.options.get_option(name);
}

void ChannelImpl::set_option(std::string name, int value) {
    m_channel_state.options.set_option(name, value);
    m_buffer.enqueue(SetOption{name, value});
}

TensorInfo* ChannelImpl::alloc() {
    MGB_LOCK_GUARD(m_mutex);
    auto info = m_pool.alloc();
    m_valid_handle.insert(info);
    info->id = m_last_id++;
    if (m_channel_state.profiler->is_profiling()) {
        m_channel_state.profiler->record_host<TensorDeclareEvent>(info->id);
    }
    return info;
}

void ChannelImpl::free(TensorInfo* ptr) {
    MGB_LOCK_GUARD(m_mutex);
    if (m_channel_state.profiler->is_profiling()) {
        m_channel_state.profiler->record_host<TensorEraseEvent>(ptr->id);
    }
    m_pool.free(ptr);
}

ChannelImpl::ChannelImpl() : m_worker(this), m_buffer(this){
    m_channel_state.tid = std::this_thread::get_id();
}

ChannelImpl::~ChannelImpl() {
    close();
}

void ChannelImpl::produce_tensor(TensorInfo* dest, TensorPtr ptr) {
    MGB_LOCK_GUARD(m_mutex);
    if (m_worker_state.profiler->is_profiling()) {
        m_worker_state.profiler->record_host<TensorProduceEvent>(dest->id, ptr->layout(), ptr->comp_node());
    }
    dest->value_fetched = ptr->value_fetched();
    // update tensor desc for static infer
    dest->desc.layout = ptr->layout();
    dest->desc.comp_node = ptr->comp_node();
    dest->ptr = std::move(ptr);
    if (m_waitee == dest) {
        m_cv.notify_all();
    }
}

void ChannelImpl::release_tensor(TensorInfo* dest) {
    MGB_LOCK_GUARD(m_mutex);
    dest->ptr.reset();
}

void ChannelImpl::regenerate(TensorInfo* dest) {
    if (dest->evict_type == DROP) {
        recompute(dest->producer);
    } else if (dest->evict_type == SWAP) {
        swap_in(dest);
    }
    mgb_assert(dest->evict_type == NONE);
}

void ChannelImpl::recompute(TensorInfo::ComputePath* path) {
    SmallVector<TensorInfo*> workspaces(path->outputs.size(), nullptr);
    for (auto&& input: path->inputs) {
        regenerate(input);
    }
    for (auto&& output: path->outputs) {
        if(output == nullptr) {
            continue;
        }
        output->evict_type = NONE;
    }
    m_buffer.enqueue(ApplyOp{path->op, path->inputs, path->outputs});
}

void ChannelImpl::detach_users(TensorInfo* dest) {
    SmallVector<TensorInfo::ComputePath*> users = dest->users;
    for (auto* user: users) {
        for (auto* output: user->outputs) {
            if (output == nullptr) {
                continue;
            }
            regenerate(output);
            output->detach_producer();
        }
    }
    mgb_assert(dest->users.size() == 0);
    //dest->users.clear();
}

void ChannelImpl::sync_device_scope(CompNode device) {
    auto& prev = m_worker_state.device_scope_map[device];
    auto& current = m_worker_state.scopes;
    auto push_scope = [&](std::string name) {
        m_worker_state.profiler->record_device<DeviceBeginScope>(device, name);
    };
    auto pop_scope = [&](std::string name) {
        m_worker_state.profiler->record_device<DeviceEndScope>(device, name);
    };
    size_t similarity = 0;
    for (size_t i = 0; i < prev.size() && i < current.size(); i++) {
        if (prev[i] == current[i]) {
            similarity++;
        } else {
            break;
        }
    }
    while (prev.size() > similarity) {
        pop_scope(prev.back());
        prev.pop_back();
    }
    while (prev.size() < current.size()) {
        prev.push_back(current[prev.size()]);
        push_scope(prev.back());
    }
}

void ChannelImpl::process_one_task(IdentifiedCommand& icmd) {
    if (m_worker_state.profiler->is_profiling()) {
        m_worker_state.profiler->record_host<CommandExecuteEvent>(icmd);
    }
    bool finished = false;
    auto do_finish_command = [&]{
        if (finished) {
            return;
        }
        if (m_worker_state.profiler->is_profiling()) {
            m_worker_state.profiler->record_host<CommandFinishEvent>(icmd);
        }
        finished = true;
    };
    //TODO: remove std::visit for support osx 10.12
    auto cmd_visitor = [&](const auto& cmd) {
            using T = std::decay_t<decltype(cmd)>;
            if constexpr (std::is_same_v<T, Put>) {
                auto value = cmd.no_cache ? std::make_shared<Tensor>(cmd.value) : Tensor::make(cmd.value);
                produce_tensor(cmd.dest, std::move(value));
            } else if constexpr (std::is_same_v<T, ApplyOp>) {
                uint64_t apply_id = ++m_last_id;
                SmallVector<TensorPtr> tensor_inputs;
                SmallVector<CompNode> devices;
                tensor_inputs.reserve(cmd.inputs.size());
                // refcnt == 1, owners: [TensorInfo::ptr]
                for (auto i : cmd.inputs) {
                    mgb_assert(i->ptr, "Invalid input tensor ptr!");
                    // refcnt ++, owners: [i->ptr, tensor_inputs]
                    tensor_inputs.push_back(i->ptr);
                }
                // Begin profiling operator
                OpEvent event_data;
                if (m_worker_state.profiler->is_profiling()) {
                    auto tinfo_to_tid = [&](SmallVector<TensorInfo*> tinfo) {
                        SmallVector<uint64_t> tid;
                        for (auto* ptinfo: tinfo) {
                            tid.push_back(ptinfo->id);
                        }
                        return tid;
                    };
                    event_data = {apply_id, cmd.op, tinfo_to_tid(cmd.inputs), tinfo_to_tid(cmd.outputs)};
                    // Collecting devices
                    for (auto i : cmd.inputs) {
                        devices.push_back(i->desc.comp_node);
                    }
                    for (auto i : cmd.outputs) {
                        devices.push_back(i->desc.comp_node);
                    }
                    devices.erase(std::unique(devices.begin(), devices.end()), devices.end());
                }
                // Fused by command buffer. @see: CommandBuffer::fuse_del
                // Now if dest is inplacable, it's refcnt would be decreased to 1 and owned by tensor_inputs after Del.
                // Note for exprs like 'y = x op x', inplace is unsupported yet but Del would be also fused.
                for (auto* del : cmd.dels) {
                    // refcnt --, owners: [tensor_inputs]
                    // if it's decreased to 1, would be detected at @see: proxy_graph_detail::apply_on_physical_tensor
                    free(del);
                }
                // Before wait
                //TODO: split operator wait and execute so that OpWait could be corrected recorded.
                // Before execute
                if (m_worker_state.profiler->is_profiling()) {
                    m_worker_state.profiler->record_host<HostOpExecuteEvent>(event_data);
                    for (auto&& device: devices) {
                        sync_device_scope(device);
                        m_worker_state.profiler->record_device<DeviceOpExecuteEvent>(device, event_data);
                    }
                }
                // Apply op
                // Here std::move is REQUIRED for removing duplicated references.
                auto tensor_outputs = OpDef::apply_on_physical_tensor(
                    *cmd.op, std::move(tensor_inputs));
                // After execute
                if (m_worker_state.profiler->is_profiling()) {
                    m_worker_state.profiler->record_host<HostOpFinishEvent>(event_data);
                    for (auto&& device: devices) {
                        m_worker_state.profiler->record_device<DeviceOpFinishEvent>(device, event_data);
                    }
                }
                // End profiling operator
                mgb_assert(tensor_outputs.size() == cmd.outputs.size());
                for (size_t i = 0; i < tensor_outputs.size(); ++i) {
                    if (cmd.outputs[i] == nullptr) {
                        continue;
                    }
                    produce_tensor(cmd.outputs[i], std::move(tensor_outputs[i]));
                }
            } else if constexpr (std::is_same_v<T, Del>) {
                free(cmd.dest);
            } else if constexpr (std::is_same_v<T, GetValue>) {
                mgb_assert(cmd.dest->ptr, "Invalid tensor ptr!");
                cmd.dest->ptr->fetch_value();
                MGB_LOCK_GUARD(m_mutex);
                cmd.dest->value_fetched = true;
                if (m_waitee == cmd.dest) {
                    m_cv.notify_all();
                }
            } else if constexpr (std::is_same_v<T, SwapIn>) {
                produce_tensor(cmd.dest, Tensor::make(cmd.dest->h_value));
            } else if constexpr (std::is_same_v<T, SwapOut>) {
                cmd.dest->h_value = cmd.dest->ptr->get_value();
                release_tensor(cmd.dest);
            } else if constexpr (std::is_same_v<T, Drop>) {
                release_tensor(cmd.dest);
            } else if constexpr (std::is_same_v<T, SetOption>) {
                m_worker_state.options.set_option(cmd.key, cmd.value);
            } else if constexpr (std::is_same_v<T, StartProfile>) {
                CompNode::sync_all();
                m_worker_state.profiler.reset(cmd.profiler);
            } else if constexpr (std::is_same_v<T, StopProfile>) {
                for (auto&& [device, scopes]: m_worker_state.device_scope_map) {
                    MGB_MARK_USED_VAR(scopes);
                    sync_device_scope(device);
                }
                do_finish_command();
                auto profiler = std::make_unique<InterpreterProfiler>();
                std::swap(profiler, m_worker_state.profiler);
                auto records = profiler->stop();
                auto host_map = [this](std::thread::id tid) {
                    if (tid == m_channel_state.tid) {
                        return "channel";
                    } else if (tid == m_worker_state.tid) {
                        return "worker";
                    } else {
                        return "unknown";
                    }
                };
                InterpreterProfiler::dump_data(cmd.basename, cmd.format, records, profiler->get_option(), host_map);
            } else if constexpr (std::is_same_v<T, PushScope>) {
                m_worker_state.scopes.push_back(cmd.scope_name);
                do_finish_command();
                m_worker_state.profiler->record_host<WorkerBeginScope>(cmd.scope_name);
            } else if constexpr (std::is_same_v<T, PopScope>) {
                mgb_assert(m_worker_state.scopes.back() == cmd.scope_name, "scope name mismatch");
                m_worker_state.scopes.pop_back();
                do_finish_command();
                m_worker_state.profiler->record_host<WorkerEndScope>(cmd.scope_name);
            } else {
                static_assert(!std::is_same_v<T, T>);
            }
    };
    std::visit([&](const auto& cmd){
        using T = std::decay_t<decltype(cmd)>;
        if (!m_worker_state.options.catch_worker_execption) {
            cmd_visitor(cmd);
            return;
        }
        try {
            cmd_visitor(cmd);
        } catch (...) {
            MGB_LOCK_GUARD(m_mutex);
            if constexpr (std::is_same_v<T, ApplyOp>) {
                for (auto oup : cmd.outputs) {
                    oup->invalid = true;
                }
            } else if constexpr (std::is_same_v<T, Put>) {
                cmd.dest->invalid = true;
            }
            m_worker_exc = std::current_exception();
            m_cv.notify_all();
        }
    }, icmd.second);
    do_finish_command();
}

void ChannelImpl::check_worker_exc_unsafe() {
    if (m_worker_exc) {
        // for reuse interpreter_for_py after some exception tests
        m_waitee = nullptr;
        std::exception_ptr exc;
        std::swap(exc, m_worker_exc);
        std::rethrow_exception(exc);
    }
}

void ChannelImpl::CommandBuffer::enqueue(Command cmd) {
    if (std::get_if<Del>(&cmd) && fuse_del(std::get<Del>(cmd))) {
        return;
    }
    // mgb_log_debug("%s Enqueued", to_string(cmd).c_str());
    m_commands.push_back(std::move(cmd));
    auto flush_pos = flush_pos_for(m_commands.back());
    flush(flush_pos);
}

void ChannelImpl::CommandBuffer::flush() {
    flush(m_commands.end());
}

void ChannelImpl::CommandBuffer::flush(Handle pos) {
    for (auto iter = m_commands.begin(); iter != pos; ++iter) {
        // mgb_log_debug("%s Flushed", to_string(*iter).c_str());
        IdentifiedCommand icmd{++m_owner->m_last_id, std::move(*iter)};
        if (m_owner->m_channel_state.profiler->is_profiling()) {
            m_owner->m_channel_state.profiler->record_host<CommandEnqueueEvent>(icmd);
        }
        m_owner->m_worker.add_task(std::move(icmd));
    }
    m_commands.erase(m_commands.begin(), pos);
}

auto ChannelImpl::CommandBuffer::flush_pos_for(const Command& cmd) -> Handle {
    return std::visit([this](const auto& cmd) {
        using T = std::decay_t<decltype(cmd)>;
        if constexpr (std::is_same_v<T, ApplyOp>) {
            auto* op_type = cmd.op->dyn_typeinfo();
            if (op_type == RemoteRecv::typeinfo() ||
                op_type == RemoteSend::typeinfo() ||
                op_type == CollectiveComm::typeinfo() ||
                op_type == opr::InputCallback::typeinfo() ||
                op_type == opr::OutputCallback::typeinfo() ||
                op_type == BackwardGraph::typeinfo()) {
                return m_commands.end();
            }
        } else if constexpr (std::is_same_v<T, GetValue>) {
            return m_commands.end();
        }
        size_t buffer_length = m_owner->m_channel_state.options.buffer_length;
        if (m_commands.size() > buffer_length) {
            return m_commands.begin() + (m_commands.size() - buffer_length);
        }
        return m_commands.begin();
    }, cmd);
}

/**
 * 1. Find ApplyOp(dest) in buffered commands
 * 2. Check if there are other usages between ApplyOp and Del, return false if not
 * 3. Fuse Del into ApplyOp, return true
 */
bool ChannelImpl::CommandBuffer::fuse_del(const Del& cmd) {
    auto* dest = cmd.dest;
    // TODO: eliminate Puts
    auto begin = m_commands.begin(), end = m_commands.end();
    auto apply_iter = std::find_if(begin, end, [dest](const Command& cmd){
        if (auto* apply = std::get_if<ApplyOp>(&cmd)) {
            return std::count(apply->inputs.begin(), apply->inputs.end(), dest) > 0;
        }
        return false;
    });
    if (apply_iter == end || find_last_usage(dest, {apply_iter+1, end}) != end) {
        return false;
    }
    // mgb_log_debug("%s Fused", to_string(Command{cmd}).c_str());
    std::get<ApplyOp>(*apply_iter).dels.push_back(dest);
    return true;
}

auto ChannelImpl::CommandBuffer::find_last_usage(TensorInfo* dest, Range range)
        -> Handle {
    auto found = range[1];
    for (auto iter = range[0]; iter != range[1]; ++iter) {
        std::visit([&](const auto& cmd) {
            using T = std::decay_t<decltype(cmd)>;
            if constexpr (std::is_same_v<T, ApplyOp>) {
                if (std::count(cmd.inputs.begin(), cmd.inputs.end(),
                               dest) > 0) {
                    found = iter;
                }
            } else if constexpr (std::is_same_v<T, GetValue>) {
                if (cmd.dest == dest) {
                    found = iter;
                }
            } else if constexpr (std::is_same_v<T, SwapIn> ||
                    std::is_same_v<T, SwapOut> ||
                    std::is_same_v<T, Drop>) {
                //TODO: ignore swap-like commands, just remove them from buffer
                if (cmd.dest == dest) {
                    found = iter;
                }
            }
        }, *iter);
    };
    return found;
}

auto ChannelImpl::CommandBuffer::find_produce(TensorInfo* dest, Range range)
        -> Handle {
    return std::find_if(range[0], range[1], [dest](auto& cmd) {
        return std::visit([dest](const auto& cmd){
            using T = std::decay_t<decltype(cmd)>;
            if constexpr (std::is_same_v<T, ApplyOp>) {
                return std::count(cmd.outputs.begin(), cmd.outputs.end(), dest) > 0;
            } else if constexpr (std::is_same_v<T, Put>) {
                return cmd.dest == dest;
            }
            return false;
        }, cmd);
    });
}

void ChannelImpl::start_profile(std::unordered_map<std::string, int> option) {
    auto profiler_option = InterpreterProfiler::Option::from_dict(option);
    auto profiler = std::make_unique<InterpreterProfiler>();
    profiler->set_option(profiler_option);
    profiler->start(InterpreterProfiler::topic_to_mask(profiler_option.topic));
    std::swap(profiler, m_channel_state.profiler);
    m_buffer.enqueue(StartProfile{m_channel_state.profiler.get()});
}

void ChannelImpl::stop_profile(std::string basename, std::string format) {
    m_buffer.flush();
    auto profiler = std::make_unique<InterpreterProfiler>();
    std::swap(profiler, m_channel_state.profiler);
    profiler.release();
    m_buffer.enqueue(StopProfile{basename, format});
}

void ChannelImpl::push_scope(std::string name) {
    if (m_channel_state.profiler->is_profiling()) {
        m_channel_state.profiler->record_host<ChannelBeginScope>(name);
        m_channel_state.scopes.push_back(name);
        m_buffer.enqueue(PushScope{name});
    }
}

void ChannelImpl::pop_scope(std::string name) {
    if (m_channel_state.profiler->is_profiling()) {
        mgb_assert((!m_channel_state.scopes.empty()) && m_channel_state.scopes.back() == name, "scope name mismatch");
        m_channel_state.scopes.pop_back();
        m_channel_state.profiler->record_host<ChannelEndScope>(name);
        m_buffer.enqueue(PopScope{name});
    }
}

void ChannelImpl::assert_in_channel() {
    mgb_assert(m_channel_state.tid != std::this_thread::get_id());
}

void ChannelImpl::assert_in_worker() {
    mgb_assert(m_worker_state.tid == std::this_thread::get_id());
}
