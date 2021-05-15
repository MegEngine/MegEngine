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
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/backward_graph.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/utils/to_string.h"

using namespace mgb;
using namespace imperative;
using namespace interpreter;
using namespace interpreter::intl;

#define RECORD_EVENT(type, ...) \
    if (state.profiler->is_profiling()) { \
        state.profiler->record_host<type>(type{__VA_ARGS__}); \
    } \

#define RECORD_DEVICE_EVENT(type, device, ...) \
    if (state.profiler->is_profiling()) { \
        state.profiler->record_device<type>((device), type{__VA_ARGS__}); \
    } \


std::thread::id ChannelImpl::get_worker_tid() {
    return m_worker_state.tid;
}

ChannelImpl::ChannelState& ChannelImpl::get_channel_state() {
    assert_in_channel();
    return m_channel_state;
}

ChannelImpl::WorkerState& ChannelImpl::get_worker_state() {
    assert_in_worker();
    return m_worker_state;
}

#define m_channel_state
#define m_worker_state

std::unique_ptr<Interpreter::Channel> InterpreterImpl::create_channel() {
    return std::make_unique<ChannelImpl>();
}

Interpreter& Interpreter::inst() {
    static InterpreterImpl inst_;
    return inst_;
}

Handle ChannelImpl::put(const HostTensorND& value, bool no_cache) {
    mgb_assert(check_available(), "Channel already closed");
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
    auto& state = get_channel_state();
    mgb_assert(check_available(), "Channel already closed");
    auto info = alloc();
    info->desc.layout = data.layout();
    info->desc.comp_node = data.comp_node();
    info->ptr = Tensor::make(data);
    RECORD_EVENT(TensorProduceEvent, info->id, info->desc.layout, info->desc.comp_node);
    return info;
}

void ChannelImpl::del(Handle handle) {
    if (!check_available()){
        return;
    }
    mgb_assert(m_valid_handle.count(handle), "invalid handle: %p", handle);
    auto* info = reinterpret_cast<TensorInfo*>(handle);
    m_valid_handle.erase(handle);
    m_buffer.enqueue(Del{info});
}

void ChannelImpl::swap_in(Handle handle) {
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    if (state.options.enable_swap) {
        mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
                "invalid handle: %p", handle);
        auto* info = reinterpret_cast<TensorInfo*>(handle);
        m_buffer.enqueue(SwapIn{info});
    }
}

void ChannelImpl::swap_out(Handle handle) {
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    if (state.options.enable_swap) {
        mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
                "invalid handle: %p", handle);
        auto* info = reinterpret_cast<TensorInfo*>(handle);
        m_buffer.enqueue(SwapOut{info});
    }
}

void ChannelImpl::drop(Handle handle) {
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    if (state.options.enable_drop) {
        mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
                "invalid handle: %p", handle);
        auto* info = reinterpret_cast<TensorInfo*>(handle);
        m_buffer.enqueue(Drop{info});
    }
}

void ChannelImpl::dispatch_default_cpu(
        std::shared_ptr<OpDef> op,
        const SmallVector<TensorInfo*>& input_infos,
        const SmallVector<LogicalTensorDesc>& input_descs,
        SmallVector<Handle>* outputs) {
    auto& state = get_channel_state();
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
    auto apply_id = ++m_last_id;
    RECORD_EVENT(OpExecuteEvent, apply_id, op, tinfo_to_tid(input_infos), {});

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

    RECORD_EVENT(OpExecuteFinishEvent, apply_id, op, 
            tinfo_to_tid(input_infos), tinfo_to_tid(output_infos));
}

void ChannelImpl::dispatch_kernel(
        std::shared_ptr<OpDef> op,
        const SmallVector<TensorInfo*>& input_infos,
        const SmallVector<LogicalTensorDesc>& input_descs,
        SmallVector<Handle>* outputs) {
    auto& state = get_channel_state();
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
    m_buffer.enqueue(std::move(cmd));
    if (!validated && state.options.async_level == 1) {
        sync();
    } else if (state.options.async_level == 0) {
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
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
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
        }
    }

    SmallVector<Handle> outputs;
    DispatchMode dispatch_mode = state.options.enable_host_compute
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
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
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
        m_buffer.enqueue(GetValue{info});
        RECORD_EVENT(TensorWaitPropEvent, info->id, TensorInfo::HostValue);
        m_cv.wait(lock, [&]() {
            check_worker_exc_unsafe();
            tensor_ptr = info->ptr;
            return value_fetched();
        });
        RECORD_EVENT(TensorWaitPropFinishEvent, info->id, TensorInfo::HostValue);
        m_waitee = nullptr;
    }
    return tensor_ptr->get_value();
}

TensorShape ChannelImpl::get_shape(Handle handle) {
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
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
    RECORD_EVENT(TensorWaitPropEvent, info->id, TensorInfo::Shape);
    m_cv.wait(lock, [&]() {
        check_worker_exc_unsafe();
        return static_cast<bool>(info->ptr);
    });
    RECORD_EVENT(TensorWaitPropFinishEvent, info->id, TensorInfo::Shape);
    m_waitee = nullptr;
    TensorShape ret = info->ptr->layout();
    mgb_assert(ret.ndim != 0);
    return ret;
}

DType ChannelImpl::get_dtype(Handle handle) {
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
               "invalid handle: %p", handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    RECORD_EVENT(TensorGetPropEvent, info->id, TensorInfo::DType);
    auto ret = info->desc.layout.dtype;
    mgb_assert(ret.valid());
    return ret;
}

CompNode ChannelImpl::get_device(Handle handle) {
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
               "invalid handle: %p", handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    RECORD_EVENT(TensorGetPropEvent, info->id, TensorInfo::Device);
    auto ret = info->desc.comp_node;
    mgb_assert(ret.valid());
    return ret;
}

DeviceTensorND ChannelImpl::get_dev_tensor(Handle handle) {
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
               "invalid handle: %p", handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);
    mgb_assert(!m_waitee);
    m_waitee = info;
    m_buffer.flush();
    RECORD_EVENT(TensorWaitPropEvent, info->id, TensorInfo::DevValue);
    m_cv.wait(lock, [&]() {
        check_worker_exc_unsafe();
        return static_cast<bool>(info->ptr);
    });
    RECORD_EVENT(TensorWaitPropFinishEvent, info->id, TensorInfo::DevValue);
    m_waitee = nullptr;
    return info->ptr->dev_tensor();
}

void ChannelImpl::sync() {
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    m_buffer.flush();
    RECORD_EVENT(SyncEvent);
    m_worker.wait_all_task_finish();
    CompNode::sync_all();
    RECORD_EVENT(SyncFinishEvent);
    MGB_LOCK_GUARD(m_mutex);
    check_worker_exc_unsafe();
}

void ChannelImpl::close() {
    if (!check_available()) {
        return;
    }
    std::vector<Handle> valid_handles(m_valid_handle.begin(), m_valid_handle.end());
    for (auto* handle: valid_handles) {
        del(handle);
    }
    mgb_assert(m_valid_handle.empty());
    mgb_log_debug("%ld tensor exists before channel close", (long)valid_handles.size());
    sync();
    m_closed = true;
}

size_t ChannelImpl::get_option(std::string name) {
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    return state.options.get_option(name);
}

void ChannelImpl::set_option(std::string name, size_t value) {
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    state.options.set_option(name, value);
    m_buffer.enqueue(SetOption{name, value});
}

TensorInfo* ChannelImpl::alloc() {
    auto& state = get_channel_state();
    MGB_LOCK_GUARD(m_mutex);
    auto info = m_pool.alloc();
    m_valid_handle.insert(info);
    info->id = m_last_id++;
    RECORD_EVENT(TensorDeclareEvent, info->id);
    return info;
}


void ChannelImpl::do_drop(TensorInfo* ptr, bool user=false) {
    if (!ptr->producer) {
        if (user) {
            mgb_log_warn("the input that produced tensor %p has been deleted, this drop operation will be ignored", ptr);
        }
        return;
    }
    if (ptr->evict_type != EvictType::NONE) {
        return;
    }
    ptr->evict_type = EvictType::DROP;
    release_tensor(ptr);
}

void ChannelImpl::free(TensorInfo* ptr) {
    auto& state = get_worker_state();
    if (state.options.enable_dtr_auto_drop) {
        // Evicting a tensor, rather than freeing it, can avoid pinning
        // potentially exploding amounts of memory and allow us to save
        // more memory.
        ptr->allow_delete = true;
        if (!ptr->ref_cnt) {
            recursive_free(ptr);
        } else {
            do_drop(ptr);
        }
    } else {
        real_free(ptr);
    }
}

void ChannelImpl::recursive_free(TensorInfo* ptr) {
    SmallVector<TensorInfo*> inps(0);
    if (ptr->producer) {
        for (auto i : ptr->producer->inputs) {
            if (i && --i->ref_cnt == 0) {
                inps.push_back(i);
            }
        }
    }
    real_free(ptr);
    for (auto i : inps) {
        if (i->allow_delete) {
            recursive_free(i);
        }
    }
}

void ChannelImpl::real_free(TensorInfo* ptr) {
    auto& state = get_worker_state();
    MGB_LOCK_GUARD(m_mutex);
    RECORD_EVENT(TensorEraseEvent, ptr->id);
    if (ptr->size_exceeds_thd(state.options.dtr_evictee_minimum_size)) {
        m_dtr.erase_candidate(ptr);
    }
    detach_users(ptr);
    ptr->detach_producer();
    m_pool.free(ptr);
}

ChannelImpl::ChannelImpl() : m_worker(this), m_buffer(this){}

ChannelImpl::~ChannelImpl() {
    close();
}

void ChannelImpl::produce_tensor(TensorInfo* dest, TensorPtr ptr, bool notice=true) {
    auto& state = get_worker_state();
    auto lock = std::unique_lock<std::mutex>(m_mutex, std::defer_lock);
    if (notice) {
        lock.lock();
    }
    m_dtr.update_used_time(dest);
    if (notice) {
        RECORD_EVENT(TensorProduceEvent, dest->id, ptr->layout(), ptr->comp_node());
    }
    dest->value_fetched = ptr->value_fetched();
    // update tensor desc for static infer
    dest->desc.layout = ptr->layout();
    dest->desc.comp_node = ptr->comp_node();
    dest->memory = ptr->blob()->size();
    dest->ptr = std::move(ptr);
    dest->evict_type = EvictType::NONE;
    if (notice && dest->size_exceeds_thd(state.options.dtr_evictee_minimum_size)) {
        m_dtr.insert_candidate(dest);
    }
    if (notice && m_waitee == dest) {
        m_cv.notify_all();
    }
}

void ChannelImpl::release_tensor(TensorInfo* dest) {
    MGB_LOCK_GUARD(m_mutex);
    dest->ptr.reset();
}

void ChannelImpl::regenerate(TensorInfo* dest) {
    if (dest->evict_type == EvictType::DROP) {
        recompute(dest->producer);
    } else if (dest->evict_type == EvictType::SWAP) {
        produce_tensor(dest, Tensor::make(dest->h_value));
    }
}

void ChannelImpl::recompute(TensorInfo::ComputePath* path) {
    auto& state = get_worker_state();
    SmallVector<TensorPtr> inputs;
    inputs.reserve(path->inputs.size());
    m_dtr.pin(path->inputs);
    for (auto i : path->inputs) {
        if (!i->ptr) {
            regenerate(i);
        }
        inputs.push_back(i->ptr);
        m_dtr.update_used_time(i);
    }
    if (state.options.enable_dtr_auto_drop && state.options.dtr_eviction_threshold > 0) {
        auto_evict();
    }
    auto outputs = OpDef::apply_on_physical_tensor(*path->op, inputs);
    m_dtr.estimate_timestamp += path->compute_time / 1e8;
    m_dtr.unpin(path->inputs);
    for (size_t i = 0;i < outputs.size();i ++) {
        auto&& o = path->outputs[i];
        if (o) {
            o->recompute_times ++;
            if (!o->ptr) {
                produce_tensor(o, std::move(outputs[i]), false);
                if (state.options.enable_dtr_auto_drop) {
                    m_dtr.update_dsu_after_recompute(o);
                }
            }
        }
    }
}

void ChannelImpl::auto_evict() {
    auto& state = get_worker_state();
    if (!m_dtr.comp_node.valid()) {
        return;
    }
    size_t current_memory = m_dtr.comp_node.get_used_memory();
    while (current_memory > state.options.dtr_eviction_threshold) {
        auto best = m_dtr.find_best_tensor();
        if (!best) {
            if (!m_dtr.warn_printed) {
                m_dtr.warn_printed = true;
                mgb_log_warn("No tensors on %s can be evicted automatically "
                             "when memory usage is %.0lfMB. Maybe memory "
                             "budget is too small.",
                              m_dtr.comp_node.to_string().c_str(),
                              current_memory / 1024.0 / 1024.0);
            }
            break;
        }
        if (best->ptr.unique() && best->ptr->blob().unique()) {
            current_memory -= best->memory;
        }
        do_drop(best);
        if (best->evict_type == EvictType::DROP) {
            m_dtr.update_dsu_after_evict(best);
        }
    }
}

void ChannelImpl::detach_users(TensorInfo* dest) {
    SmallVector<TensorInfo::ComputePath*> users = dest->users;
    for (auto* user: users) {
        SmallVector<TensorInfo*> outputs = user->outputs;
        SmallVector<TensorInfo*> inputs = user->inputs;
        for (auto* output: outputs) {
            if (output == nullptr) {
                continue;
            }
            regenerate(output);
            output->detach_producer();
            for (auto* input: inputs) {
                input->ref_cnt --;
            }
        }
    }
    mgb_assert(dest->users.size() == 0);
    //dest->users.clear();
}

bool ChannelImpl::check_available() {
    return !m_closed;
}

void ChannelImpl::sync_device_scope(CompNode device) {
    auto& state = get_worker_state();
    auto& prev = state.device_scope_map[device];
    auto& current = state.scopes;
    auto push_scope = [&](std::string name) {
        RECORD_DEVICE_EVENT(DeviceScopeEvent, device, name);
    };
    auto pop_scope = [&](std::string name) {
        RECORD_DEVICE_EVENT(DeviceScopeFinishEvent, device, name);
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
    auto& state = get_worker_state();
    RECORD_EVENT(CommandExecuteEvent, icmd);
    bool finished = false;
    auto do_finish_command = [&]{
        if (finished) {
            return;
        }
        RECORD_EVENT(CommandFinishEvent, icmd);
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
                if (state.options.enable_dtr_auto_drop) {
                    m_dtr.pin(cmd.inputs);
                }
                for (auto i : cmd.inputs) {
                    if (!i->ptr && i->evict_type != EvictType::NONE) {
                        regenerate(i);
                    }
                    m_dtr.update_used_time(i);
                }
                tensor_inputs.reserve(cmd.inputs.size());
                // refcnt == 1, owners: [TensorInfo::ptr]
                for (auto i : cmd.inputs) {
                    mgb_assert(i->ptr, "Invalid input tensor ptr!");
                    // refcnt ++, owners: [i->ptr, tensor_inputs]
                    tensor_inputs.push_back(i->ptr);
                }
                // Begin profiling operator
                auto tinfo_to_tid = [&](SmallVector<TensorInfo*> tinfo) {
                    SmallVector<uint64_t> tid;
                    for (auto* ptinfo: tinfo) {
                        tid.push_back(ptinfo->id);
                    }
                    return tid;
                };
                if (state.profiler->is_profiling()) {
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
                RECORD_EVENT(OpExecuteEvent, apply_id, cmd.op, tinfo_to_tid(cmd.inputs), tinfo_to_tid(cmd.outputs));
                if (state.profiler->is_profiling()) {
                    for (auto&& device: devices) {
                        sync_device_scope(device);
                        RECORD_DEVICE_EVENT(KernelExecuteEvent, device, apply_id, cmd.op,
                                tinfo_to_tid(cmd.inputs), tinfo_to_tid(cmd.outputs));
                    }
                }
                if (state.options.enable_dtr_auto_drop && state.options.dtr_eviction_threshold > 0) {
                    auto_evict();
                }
                // Apply op
                // Here std::move is REQUIRED for removing duplicated references.
                auto tensor_outputs = OpDef::apply_on_physical_tensor(
                    *cmd.op, std::move(tensor_inputs));
                // After execute
                RECORD_EVENT(OpExecuteFinishEvent, apply_id, cmd.op, tinfo_to_tid(cmd.inputs), tinfo_to_tid(cmd.outputs));
                if (state.profiler->is_profiling()) {
                    for (auto&& device: devices) {
                        RECORD_DEVICE_EVENT(KernelExecuteFinishEvent, device, apply_id, cmd.op, tinfo_to_tid(cmd.inputs), tinfo_to_tid(cmd.outputs));
                    }
                }
                // End profiling operator
                double estimate_compute_time = 0;
                if (state.options.enable_dtr_auto_drop) {
                    for (auto i : cmd.inputs) {
                        estimate_compute_time += i->memory;
                    }
                    for (auto i : tensor_outputs) {
                        estimate_compute_time += i->blob()->size();
                    }
                    m_dtr.estimate_timestamp += estimate_compute_time / 1e8;
                    for (auto i : cmd.outputs) {
                        i->compute_time = estimate_compute_time;
                        m_dtr.update_used_time(i);
                    }
                    if (cmd.outputs[0]->producer) {
                        cmd.outputs[0]->producer->compute_time = estimate_compute_time;
                    }
                    m_dtr.unpin(cmd.inputs);
                }
                mgb_assert(tensor_outputs.size() == cmd.outputs.size());
                for (size_t i = 0; i < tensor_outputs.size(); ++i) {
                    if (cmd.outputs[i] == nullptr) {
                        continue;
                    }
                    produce_tensor(cmd.outputs[i], std::move(tensor_outputs[i]));
                    if (state.options.enable_dtr_auto_drop) {
                        cmd.outputs[i]->dsu_ptr = std::make_shared<DsuNode>(estimate_compute_time);
                    }
                }
                if (state.options.enable_drop == 1
                    && state.options.record_computing_path == 1){
                    bool is_inplace = false;
                    bool cross_cn = false;
                    for (auto input : cmd.inputs) {
                        for (auto output : cmd.outputs) {
                            if (input->ptr->blob()->storage() == output->ptr->blob()->storage()) {
                                is_inplace = true;
                                break;
                            }
                        }
                    }
                    for (auto input : cmd.inputs) {
                        if (input->ptr->comp_node() != m_dtr.comp_node) {
                            cross_cn = true;
                            break;
                        }
                    }
                    for (auto output : cmd.outputs) {
                        if (output->ptr->comp_node() != m_dtr.comp_node) {
                            cross_cn = true;
                            break;
                        }
                    }
                    // FIXME: do not use opname as identifier
                    auto get_name = [](const OpDef& opdef) {
                        if (auto attr = opdef.try_cast_final<OprAttr>()) {
                            return attr->type.c_str();
                        }
                        return opdef.dyn_typeinfo()->name;
                    };
                    if (!is_inplace && !cross_cn && !m_dtr.is_bad_op(get_name(*cmd.op))) {
                        TensorInfo::ComputePath::make(cmd.op, cmd.inputs, cmd.outputs);
                        size_t detach_cnt = 0;
                        for (auto output : cmd.outputs) {
                            if (!output->size_exceeds_thd(state.options.dtr_evictee_minimum_size)) {
                                output->detach_producer();
                                detach_cnt ++;
                            }
                        }
                        for (auto input : cmd.inputs) {
                            input->ref_cnt -= detach_cnt;
                        }
                    }
                }
            } else if constexpr (std::is_same_v<T, Del>) {
                free(cmd.dest);
            } else if constexpr (std::is_same_v<T, GetValue>) {
                if (!cmd.dest->ptr && cmd.dest->evict_type != EvictType::NONE) {
                    regenerate(cmd.dest);
                }
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
                if (cmd.dest->evict_type == EvictType::NONE) {
                    release_tensor(cmd.dest);
                    cmd.dest->evict_type = EvictType::SWAP;
                }
            } else if constexpr (std::is_same_v<T, Drop>) {
                do_drop(cmd.dest, true);
            } else if constexpr (std::is_same_v<T, SetOption>) {
                state.options.set_option(cmd.key, cmd.value);
            } else if constexpr (std::is_same_v<T, StartProfile>) {
                CompNode::sync_all();
                state.profiler.reset(cmd.profiler);
            } else if constexpr (std::is_same_v<T, StopProfile>) {
                for (auto&& [device, scopes]: state.device_scope_map) {
                    MGB_MARK_USED_VAR(scopes);
                    sync_device_scope(device);
                }
                do_finish_command();
                auto profiler = std::make_unique<InterpreterProfiler>();
                std::swap(profiler, state.profiler);
                auto records = profiler->stop();
                auto worker_tid = get_worker_tid();
                auto host_map = [worker_tid](std::thread::id tid) {
                    if (tid == worker_tid) {
                        return "worker";
                    } else {
                        return "unknown";
                    }
                };
                InterpreterProfiler::dump_data(cmd.basename, cmd.format, records, profiler->get_option(), host_map);
            } else if constexpr (std::is_same_v<T, PushScope>) {
                state.scopes.push_back(cmd.scope_name);
                do_finish_command();
                RECORD_EVENT(ScopeEvent, cmd.scope_name);
            } else if constexpr (std::is_same_v<T, PopScope>) {
                mgb_assert(state.scopes.back() == cmd.scope_name, "scope name mismatch");
                state.scopes.pop_back();
                do_finish_command();
                RECORD_EVENT(ScopeFinishEvent, cmd.scope_name);
            } else {
                static_assert(!std::is_same_v<T, T>);
            }
    };
    std::visit([&](const auto& cmd){
        using T = std::decay_t<decltype(cmd)>;
        if (!state.options.catch_worker_execption) {
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
    auto& state = m_owner->get_channel_state();
    for (auto iter = m_commands.begin(); iter != pos; ++iter) {
        // mgb_log_debug("%s Flushed", to_string(*iter).c_str());
        IdentifiedCommand icmd{++m_owner->m_last_id, std::move(*iter)};
        RECORD_EVENT(CommandEnqueueEvent, icmd);
        m_owner->m_worker.add_task(std::move(icmd));
    }
    m_commands.erase(m_commands.begin(), pos);
}

auto ChannelImpl::CommandBuffer::flush_pos_for(const Command& cmd) -> Handle {
    auto& state = m_owner->get_channel_state();
    return std::visit([&, this](const auto& cmd) {
        using T = std::decay_t<decltype(cmd)>;
        if constexpr (std::is_same_v<T, ApplyOp>) {
            auto* op_type = cmd.op->dyn_typeinfo();
            if (op_type == RemoteRecv::typeinfo() ||
                op_type == RemoteSend::typeinfo() ||
                op_type == CollectiveComm::typeinfo() ||
                op_type == opr::InputCallback::typeinfo() ||
                op_type == opr::OutputCallback::typeinfo()) {
                return m_commands.end();
            }
        } else if constexpr (std::is_same_v<T, GetValue>) {
            return m_commands.end();
        }
        size_t buffer_length = state.options.buffer_length;
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
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    auto profiler_option = InterpreterProfiler::Option::from_dict(option);
    auto profiler = std::make_unique<InterpreterProfiler>();
    profiler->set_option(profiler_option);
    profiler->start(InterpreterProfiler::topic_to_mask(profiler_option.topic));
    std::swap(profiler, state.profiler);
    m_buffer.enqueue(StartProfile{state.profiler.get()});
}

void ChannelImpl::stop_profile(std::string basename, std::string format) {
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    m_buffer.flush();
    auto profiler = std::make_unique<InterpreterProfiler>();
    std::swap(profiler, state.profiler);
    profiler.release();
    m_buffer.enqueue(StopProfile{basename, format});
}

void ChannelImpl::push_scope(std::string name) {
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    RECORD_EVENT(ScopeEvent, name);
    if (state.profiler->is_profiling()) {
        state.scopes.push_back(name);
        m_buffer.enqueue(PushScope{name});
    }
}

void ChannelImpl::pop_scope(std::string name) {
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    RECORD_EVENT(ScopeFinishEvent, name);
    if (state.profiler->is_profiling()) {
        mgb_assert((!state.scopes.empty()) && state.scopes.back() == name, "scope name mismatch");
        state.scopes.pop_back();
        m_buffer.enqueue(PopScope{name});
    }
}

void ChannelImpl::assert_in_channel() {
    mgb_assert(get_worker_tid() != std::this_thread::get_id(), "this method cannot be called in worker thread");
}

void ChannelImpl::assert_in_worker() {
    mgb_assert(get_worker_tid() == std::this_thread::get_id(), "this method can only be called in worker thread");
}

void ChannelImpl::DynamicSublinear::pin(const SmallVector<TensorInfo*>& vec) {
    for (auto i : vec) {
        i->pin();
    }
}

void ChannelImpl::DynamicSublinear::unpin(const SmallVector<TensorInfo*>& vec) {
    for (auto i : vec) {
        i->unpin();
    }
}

void ChannelImpl::DynamicSublinear::update_dsu_after_recompute(TensorInfo* ptr) {
    auto&& dsu_fa = find_father(ptr->dsu_ptr);
    dsu_fa->t -= ptr->compute_time;
    ptr->dsu_ptr->parent.reset();
    ptr->dsu_ptr->t = ptr->compute_time;
}

void ChannelImpl::DynamicSublinear::update_dsu_after_evict(TensorInfo* ptr) {
    for (auto i : ptr->producer->inputs) {
        if (i->evict_type == EvictType::DROP) {
            merge(i->dsu_ptr, ptr->dsu_ptr);
        }
    }
    for (auto i : ptr->producer->outputs) {
        if (i && i->evict_type == EvictType::DROP) {
            merge(ptr->dsu_ptr, i->dsu_ptr);
        }
    }
}

double ChannelImpl::DynamicSublinear::estimate_neighbor_cost(TensorInfo* ptr) {
    double cost = 0;
    for (auto i : ptr->producer->inputs) {
        if (i->evict_type == EvictType::DROP) {
            double t = find_father(i->dsu_ptr)->t;
            if (t < i->compute_time) {
                t = i->compute_time;
            }
            cost += t;
        }
    }
    for (auto i : ptr->producer->outputs) {
        if (i && i->evict_type == EvictType::DROP) {
            double t = find_father(i->dsu_ptr)->t;
            if (t < i->compute_time) {
                t = i->compute_time;
            }
            cost += t;
        }
    }
    return cost;
}

TensorInfo* ChannelImpl::DynamicSublinear::find_best_tensor() {
    double min_msps = -1;
    TensorInfo* best = nullptr;
    for (auto i : candidates) {
        if (i->producer && i->ptr && !i->pinned && i->evict_type == EvictType::NONE) {
            double neighbor_cost = estimate_neighbor_cost(i);
            size_t begin_ptr = reinterpret_cast<size_t>(i->ptr->blob()->storage().get());
            auto side_info = i->ptr->comp_node().get_free_left_and_right(begin_ptr, begin_ptr + i->ptr->blob()->size());
            double free_mem = side_info.first + side_info.second;
            double msps = i->eval_func(neighbor_cost, free_mem, estimate_timestamp, 1.0, 1.0, 1.0, 1.0001);
            if (min_msps < 0 || msps < min_msps) {
                min_msps = msps;
                best = i;
            }
        }
    }
    return best;
}

void ChannelImpl::DynamicSublinear::merge(std::shared_ptr<DsuNode> &x, std::shared_ptr<DsuNode> &y) {
    auto&& f_x = find_father(x);
    auto&& f_y = find_father(y);
    if (f_x.get() == f_y.get()) {
        return;
    }
    f_y->t += f_x->t;
    f_x->parent = f_y;
}

std::shared_ptr<DsuNode> ChannelImpl::DynamicSublinear::find_father(std::shared_ptr<DsuNode>& x) {
    if (x->is_root()) {
        return x;
    } else {
        auto&& fa = find_father(x->parent);
        return x->parent = fa;
    }
}

void ChannelImpl::DynamicSublinear::insert_candidate(TensorInfo* ptr) {
    candidates.insert(ptr);
    if (!comp_node.valid()) {
        comp_node = ptr->ptr->comp_node();
    }
}

void ChannelImpl::DynamicSublinear::erase_candidate(TensorInfo* ptr) {
    candidates.erase(ptr);
}

void ChannelImpl::DynamicSublinear::update_used_time(TensorInfo* ptr) {
    ptr->last_used_time = estimate_timestamp;
}
