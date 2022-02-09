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

#include "range/v3/all.hpp"

#include "megbrain/common.h"
#include "megbrain/imperative/opr_utility.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/backward_graph.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/ops/utility.h"
#include "megbrain/imperative/utils/stats.h"
#include "megbrain/imperative/utils/to_string.h"

#include "../blob_manager_impl.h"
#include "../event_pool.h"
#include "../op_trait.h"

using namespace mgb;
using namespace imperative;
using namespace interpreter;
using namespace interpreter::intl;

namespace {
auto tinfo_to_tid(SmallVector<TensorInfo*> tinfo) {
    SmallVector<uint64_t> tid;
    for (auto* ptinfo : tinfo) {
        tid.push_back(ptinfo->id);
    }
    return tid;
};
}  // namespace

namespace mgb {
using namespace profiler;
}

#if defined(_WIN32) || defined(_WIN64)
#define SYMBOL_EXPORT __declspec(dllexport)
#else
#define SYMBOL_EXPORT __attribute__((visibility("default")))
#endif

namespace mgb {

/**
 * USAGE
 *
 *   header:
 *     namespace mgb { void imperative_log_profile(const char* message); }
 *
 *   code:
 *     mgb::imperative_log_profile("MY MESSAGE");
 *
 **/
SYMBOL_EXPORT
void imperative_log_profile_begin(const char* message) {
    MGB_RECORD_EVENT(CustomEvent, std::string{message});
}

SYMBOL_EXPORT
void imperative_log_profile_end(const char* message) {
    MGB_RECORD_EVENT(CustomFinishEvent, std::string{message});
}

SYMBOL_EXPORT
void imperative_log_profile(const char* message) {
    imperative_log_profile_begin(message);
    imperative_log_profile_end(message);
}

SYMBOL_EXPORT
void imperative_log_profile_begin(const char* message, const char* device) {
    auto comp_node = CompNode::load(device);
    MGB_RECORD_EVENT(CustomEvent, std::string{message}, {}, comp_node);
    MGB_RECORD_EVENT(
            RecordDeviceEvent, EventPool::with_timer().alloc_shared(comp_node));
}

SYMBOL_EXPORT
void imperative_log_profile_end(const char* message, const char* device) {
    auto comp_node = CompNode::load(device);
    MGB_RECORD_EVENT(
            RecordDeviceEvent, EventPool::with_timer().alloc_shared(comp_node));
    MGB_RECORD_EVENT(CustomFinishEvent, std::string{message}, {}, comp_node);
}

}  // namespace mgb

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

void ChannelImpl::WorkQueue::on_async_queue_worker_thread_start() {
    sys::set_thread_name("worker");
    m_owner->m_worker_state.tid = std::this_thread::get_id();
    OpDef::set_allocator([&](CompNode device, size_t size) {
        auto blob = Blob::make(device, size);
        m_owner->alloc_tensor_with_evict(blob.get());
        return blob->storage();
    });
}

// Do not use m_xxx_state directly
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
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    auto _ = StackManager::Guard{"Put", &state.stack_manager};
    auto info = put_impl(value, no_cache);
    return reinterpret_cast<Handle>(info);
}

TensorInfo* ChannelImpl::put_impl(const HostTensorND& value, bool no_cache) {
    if (value.empty()) {
        auto layout = value.layout();
        layout.init_contiguous_stride();
        const_cast<HostTensorND&>(value).reset(value.storage(), layout);
    }
    auto info = alloc();
    constexpr int size_threshold = TensorShape::MAX_NDIM;
    init(info, {value.layout(), value.comp_node()});
    if (value.layout().total_nr_elems() <= size_threshold) {
        info->h_value = value;
        info->desc.value = value.proxy_to_default_cpu();
    }
    m_worker.add_task(
            {Profiler::next_id(), Put{info, value, no_cache},
             get_channel_state().stack_manager.dump()});
    if (m_async_level == 0) {
        sync_impl();
        info->desc.comp_node.sync();
        auto err = info->desc.comp_node.check_async_error();
        mgb_assert(!err, "%s", err->what());
    }
    return info;
}

Handle ChannelImpl::put(const DeviceTensorND& data, const HostTensorND& hvalue) {
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    return reinterpret_cast<Handle>(put_impl(data, hvalue));
}
TensorInfo* ChannelImpl::put_impl(
        const DeviceTensorND& data, const HostTensorND& hvalue) {
    auto& state = get_channel_state();
    auto _ = StackManager::Guard{"Put", &state.stack_manager};
    auto info = alloc();
    MGB_RECORD_EVENT(TensorCommandEvent, info->id, TensorCommandKind::Put);
    constexpr int size_threshold = TensorShape::MAX_NDIM;
    init(info, {data.layout(), data.comp_node()});
    if ((!hvalue.empty()) && info->desc.layout.total_nr_elems() <= size_threshold) {
        info->desc.value = hvalue.proxy_to_default_cpu();
    }
    info->ptr = Tensor::make(data, hvalue);
    MGB_RECORD_EVENT(
            TensorProduceEvent, info->id, info->desc.layout, info->desc.comp_node,
            data.raw_ptr());
    info->status = TensorInfo::Produced;
    MGB_RECORD_EVENT(TensorCommandFinishEvent, info->id, TensorCommandKind::Put);
    return info;
}

void ChannelImpl::del(Handle handle) {
    MGB_LOCK_GUARD(m_spin);
    if (!check_available()) {
        return;
    }
    del_impl(handle);
}

void ChannelImpl::del_impl(Handle handle) {
    mgb_assert(m_valid_handle.count(handle), "invalid handle: %p", handle);
    auto* info = reinterpret_cast<TensorInfo*>(handle);
    m_valid_handle.erase(handle);
    m_worker.add_task(
            {Profiler::next_id(), Del{info}, get_channel_state().stack_manager.dump()});
}

void ChannelImpl::drop(Handle handle) {
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    if (state.options.enable_drop) {
        mgb_assert(
                m_valid_handle.find(handle) != m_valid_handle.end(),
                "invalid handle: %p", handle);
        auto* info = reinterpret_cast<TensorInfo*>(handle);
        m_worker.add_task(
                {Profiler::next_id(), Drop{info},
                 get_channel_state().stack_manager.dump()});
    }
}

void ChannelImpl::dispatch_default_cpu(
        std::shared_ptr<OpDef> op, const SmallVector<TensorInfo*>& input_infos,
        const SmallVector<LogicalTensorDesc>& input_descs,
        SmallVector<Handle>* outputs) {
    auto& state = get_channel_state();

    auto name = op->trait()->make_name(*op);
    auto _ = StackManager::Guard(name, &state.stack_manager);

    auto [output_descs, validated] =
            OpDef::infer_output_attrs_fallible(*op, input_descs);
    MGB_RECORD_EVENT(ShapeInferEvent, validated);

    SmallVector<DeviceTensorND> input_tensornds;
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
                input_tensornds.emplace_back(
                        info->ptr->get_value().proxy_to_default_cpu());
            } else {
                // We assign h_value before drop ptr
                mgb_assert(!info->h_value.empty(), "inp->h_value is empty!");
                input_tensornds.emplace_back(info->h_value.proxy_to_default_cpu());
            }
        }
    }

    SmallVector<DeviceTensorND> output_tensornds;
    for (auto&& desc : output_descs) {
        // TODO: may conflict with condtake, which need alloc inside
        mgb_assert(!desc.layout.is_empty());
        // use HostTensorND alloc_host for cuda pinned memory
        output_tensornds.emplace_back(
                HostTensorND(output_cn, desc.layout).proxy_to_default_cpu());
    }

    uint64_t op_id = Profiler::next_id();

    if (op->trait()->apply_on_device_tensornd) {
        OpDef::apply_on_device_tensornd(*op, input_tensornds, &output_tensornds);
    } else {
        // proxy to apply_on_physical_tensor
        SmallVector<TensorPtr> input_tensors;
        for (auto&& input_tensornd : input_tensornds) {
            input_tensors.push_back(Tensor::make(
                    input_tensornd, HostTensorND::make_proxy(input_tensornd)));
        }
        auto output_tensors = OpDef::apply_on_physical_tensor(
                *op, input_tensors, output_descs, validated);
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            output_tensornds[i].copy_from_fixlayout(output_tensors[i]->dev_tensor());
        }
    }

    SmallVector<TensorInfo*> output_infos;
    for (auto&& tensornd : output_tensornds) {
        HostTensorND host_tensornd =
                HostTensorND::make_proxy(tensornd).proxy_to_comp_node(output_cn);
        // use `put` for consistency
        auto info = reinterpret_cast<TensorInfo*>(put_impl(host_tensornd, false));
        mgb_assert(info->desc.layout.ndim != 0);
        output_infos.push_back(info);
        outputs->push_back(reinterpret_cast<Handle>(info));
    }
    auto op_info_getter = [op] {
        std::unordered_map<std::string, std::string> op_info;
        auto props = OpDef::props(*op);
        for (auto&& [key, value] : props) {
            op_info[key] = value;
        }
        return op_info;
    };
    MGB_RECORD_EVENT(
            OpDispatchEvent, op_id, name, op_info_getter, tinfo_to_tid(input_infos),
            tinfo_to_tid(output_infos), state.stack_manager.dump());
}

void ChannelImpl::dispatch_kernel(
        std::shared_ptr<OpDef> op, const SmallVector<TensorInfo*>& input_infos,
        const SmallVector<LogicalTensorDesc>& input_descs,
        SmallVector<Handle>* outputs) {
    auto& state = get_channel_state();
    auto& options = state.options;

    auto name = op->trait()->make_name(*op);
    auto _ = StackManager::Guard{name, &state.stack_manager};

    auto [output_descs, validated] =
            OpDef::infer_output_attrs_fallible(*op, input_descs);
    MGB_RECORD_EVENT(ShapeInferEvent, validated);

    ApplyOp cmd{Profiler::next_id(), std::move(op)};
    cmd.validated = validated;
    cmd.inputs = std::move(input_infos);
    for (int i = 0; i < output_descs.size(); ++i) {
        auto&& desc = output_descs[i];
        auto info = alloc();
        init(info, desc);
        // make sure desc's value is consistent with h_value
        if (!info->desc.value.empty()) {
            info->h_value = HostTensorND::make_proxy(desc.value)
                                    .proxy_to_comp_node(desc.comp_node);
        }
        cmd.outputs.push_back(info);
        outputs->push_back(reinterpret_cast<Handle>(info));
    }
    auto op_info_getter = [op = cmd.op] {
        std::unordered_map<std::string, std::string> op_info;
        auto props = OpDef::props(*op);
        for (auto&& [key, value] : props) {
            op_info[key] = value;
        }
        return op_info;
    };
    MGB_RECORD_EVENT(
            OpDispatchEvent, cmd.id, name, op_info_getter, tinfo_to_tid(cmd.inputs),
            tinfo_to_tid(cmd.outputs), state.stack_manager.dump());
    m_worker.add_task(
            {Profiler::next_id(), std::move(cmd),
             get_channel_state().stack_manager.dump()});
    if (!validated && options.async_level == 1) {
        sync_impl();
    } else if (options.async_level == 0) {
        sync_impl();
        // check device error
        for (auto&& oup : *outputs) {
            auto info = reinterpret_cast<TensorInfo*>(oup);
            info->ptr->comp_node().sync();
            auto err = info->ptr->comp_node().check_async_error();
            mgb_assert(!err, "%s", err->what());
        }
    }
}

SmallVector<Handle> ChannelImpl::apply_op(
        std::shared_ptr<OpDef> op, const SmallVector<Handle>& inputs) {
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    auto* input = reinterpret_cast<TensorInfo*>(inputs[0]);
    if (op->same_type<GetVarShape>() && input->desc.layout.ndim) {
        size_t ndim = input->desc.layout.ndim;
        auto& gvs = op->cast_final_safe<GetVarShape>();
        if (gvs.axis == MEGDNN_MAX_NDIM) {
            HostTensorND shape_tensor{input->desc.comp_node, {ndim}, dtype::Int32()};
            DeviceTensorND shape_tensor_device = shape_tensor.proxy_to_default_cpu();
            cg::copy_shape_to_tensor_value(shape_tensor_device, input->desc.layout);
            return {reinterpret_cast<Handle>(put_impl(shape_tensor, false))};
        }
    }
    return apply_op_impl(std::move(op), inputs);
}

SmallVector<Handle> ChannelImpl::apply_op_impl(
        std::shared_ptr<OpDef> op, const SmallVector<Handle>& inputs) {
    auto& state = get_channel_state();
    for (auto i : inputs) {
        mgb_assert(
                m_valid_handle.find(i) != m_valid_handle.end(), "invalid handle: %p",
                i);
    }
    SmallVector<TensorInfo*> input_infos;
    SmallVector<LogicalTensorDesc> input_descs;
    {
        MGB_LOCK_GUARD(m_mutex);
        for (auto i : inputs) {
            auto info = reinterpret_cast<TensorInfo*>(i);
            mgb_assert(
                    !info->invalid,
                    "an input tensor is unusable due to previous error");
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
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    mgb_assert(
            m_valid_handle.find(handle) != m_valid_handle.end(), "invalid handle: %p",
            handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    // donnot use info->value_fetched, it's unsafe
    mgb_assert(!info->invalid, "tensor is unusable due to previous error");
    return wait_tensor(info, TensorProp::HostValue)->get_value();
}

TensorShape ChannelImpl::get_shape(Handle handle) {
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    mgb_assert(
            m_valid_handle.find(handle) != m_valid_handle.end(), "invalid handle: %p",
            handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    if (info->desc.layout.ndim != 0) {
        return info->desc.layout;
    }
    TensorShape ret = wait_tensor(info, TensorProp::Shape)->layout();
    mgb_assert(ret.ndim != 0);
    return ret;
}

DType ChannelImpl::get_dtype(Handle handle) {
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    mgb_assert(
            m_valid_handle.find(handle) != m_valid_handle.end(), "invalid handle: %p",
            handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    MGB_RECORD_EVENT(TensorGetPropEvent, info->id, TensorProp::DType);
    auto ret = info->desc.layout.dtype;
    mgb_assert(ret.valid());
    return ret;
}

CompNode ChannelImpl::get_device(Handle handle) {
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    mgb_assert(
            m_valid_handle.find(handle) != m_valid_handle.end(), "invalid handle: %p",
            handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    MGB_RECORD_EVENT(TensorGetPropEvent, info->id, TensorProp::Device);
    auto ret = info->desc.comp_node;
    mgb_assert(ret.valid());
    return ret;
}

DeviceTensorND ChannelImpl::get_dev_tensor(Handle handle) {
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    mgb_assert(
            m_valid_handle.find(handle) != m_valid_handle.end(), "invalid handle: %p",
            handle);
    auto info = reinterpret_cast<TensorInfo*>(handle);
    return wait_tensor(info, TensorProp::DevValue)->dev_tensor();
}

void ChannelImpl::sync() {
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    sync_impl();
}

void ChannelImpl::sync_impl() {
    m_worker.wait_all_task_finish();
    MGB_LOCK_GUARD(m_mutex);
    check_worker_exc_unsafe();
}

void ChannelImpl::close() {
    MGB_LOCK_GUARD(m_spin);
    if (!check_available()) {
        return;
    }
    std::vector<Handle> valid_handles(m_valid_handle.begin(), m_valid_handle.end());
    for (auto* handle : valid_handles) {
        del_impl(handle);
    }
    mgb_assert(m_valid_handle.empty());
    mgb_log_debug("%ld tensor exists before channel close", (long)valid_handles.size());
    sync_impl();
    m_closed = true;
}

size_t ChannelImpl::get_option(std::string name) {
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    return state.options.get_option(name);
}

void ChannelImpl::set_option(std::string name, size_t value) {
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    state.options.set_option(name, value);
    m_worker.add_task(
            {Profiler::next_id(), SetOption{name, value},
             get_channel_state().stack_manager.dump()});
}

void ChannelImpl::clear_candidates() {
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    m_dtr.candidates.clear();
}

TensorInfo* ChannelImpl::alloc() {
    auto& state = get_channel_state();
    auto info = [this] {
        MGB_LOCK_GUARD(m_mutex);
        return m_pool.alloc();
    }();
    info->id = Profiler::next_id();
    if (Profiler::is_profiling()) {
        size_t tensor_id = state.stack_manager.current()->next_id("tensor");
        info->name =
                state.stack_manager.dump().to_string() + ssprintf(":%zu", tensor_id);
    }
    return info;
}

void ChannelImpl::init(TensorInfo* info, LogicalTensorDesc desc) {
    m_valid_handle.insert(reinterpret_cast<Handle>(info));
    MGB_RECORD_EVENT(TensorDeclareEvent, info->id, info->name);
    info->status = TensorInfo::Allocated;
    info->desc = std::move(desc);
}

void ChannelImpl::do_drop(TensorInfo* ptr, bool user = false) {
    if (!ptr->producer) {
        if (user) {
            mgb_log_warn(
                    "the input that produced tensor %p has been deleted, this drop "
                    "operation will be ignored",
                    ptr);
        }
        return;
    }
    if (ptr->evict_type != EvictType::NONE) {
        return;
    }
    ptr->evict_type = EvictType::DROP;
    ptr->status = TensorInfo::Dropped;
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
    MGB_RECORD_EVENT(TensorCommandEvent, ptr->id, TensorCommandKind::RecFree);
    SmallVector<TensorInfo*> inps;
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
    MGB_RECORD_EVENT(TensorCommandFinishEvent, ptr->id, TensorCommandKind::RecFree);
}

void ChannelImpl::real_free(TensorInfo* ptr) {
    auto& state = get_worker_state();
    if (ptr->size_exceeds_thd(state.options.dtr_evictee_minimum_size)) {
        m_dtr.erase_candidate(ptr);
    }
    detach_users(ptr);
    ptr->detach_producer();
    bool has_value = ptr->ptr != nullptr;
    if (has_value) {
        MGB_RECORD_EVENT(TensorReleaseEvent, ptr->id);
    }
    MGB_RECORD_EVENT(TensorEraseEvent, ptr->id, ptr->ptr_use_count);
    ptr->status = TensorInfo::Deleted;
    MGB_LOCK_GUARD(m_mutex);
    m_pool.free(ptr);
}

ChannelImpl::ChannelImpl() : m_worker(this) {}

ChannelImpl::~ChannelImpl() {
    close();
}

void ChannelImpl::produce_tensor(TensorInfo* dest, TensorPtr ptr) {
    auto& state = get_worker_state();
    MGB_LOCK_GUARD(m_mutex);
    m_dtr.update_used_time(dest);
    MGB_RECORD_EVENT(
            TensorProduceEvent, dest->id, ptr->layout(), ptr->comp_node(),
            ptr->dev_tensor(false).raw_ptr());
    // update tensor desc for static infer
    if (dest->desc.layout.ndim) {
        mgb_assert(
                dest->desc.layout.eq_shape(ptr->layout()),
                "shape infer error, %s vs %s", dest->desc.layout.to_string().c_str(),
                ptr->layout().to_string().c_str());
    }
    dest->desc.layout = ptr->layout();
    dest->desc.comp_node = ptr->comp_node();
    dest->memory = ptr->blob()->size();
    dest->ptr = std::move(ptr);
    dest->evict_type = EvictType::NONE;
    dest->status = TensorInfo::Produced;
    if (dest->pinned == 0 &&
        dest->size_exceeds_thd(state.options.dtr_evictee_minimum_size)) {
        m_dtr.insert_candidate(dest);
    }
    notify_tensor_unsafe(dest);
}

void ChannelImpl::release_tensor(TensorInfo* dest) {
    MGB_RECORD_EVENT(TensorReleaseEvent, dest->id);
    MGB_LOCK_GUARD(m_mutex);
    dest->ptr.reset();
    auto& state = get_worker_state();
    if (dest->size_exceeds_thd(state.options.dtr_evictee_minimum_size)) {
        m_dtr.erase_candidate(dest);
    }
}

void ChannelImpl::regenerate(TensorInfo* dest) {
    if (dest->evict_type == EvictType::DROP) {
        auto&& path = dest->producer;
        m_apply_stack.push(
                {ApplyOp{path->id, path->op, path->inputs, path->outputs}, 0, dest,
                 "dtr"});
        if (!m_applying)
            flush_apply_stack();
    }
}

void ChannelImpl::do_apply_op(const ApplyOp& cmd, std::string reason) {
    using namespace ranges;
    using namespace ranges::views;
    auto& state = get_worker_state();
    bool profiling_device =
            Profiler::is_profiling() && Profiler::get_option("profile_device", 0);
    uint64_t apply_id = cmd.id;
    SmallVector<TensorPtr> inputs;
    inputs.reserve(cmd.inputs.size());
    // refcnt == 1, owners: [TensorInfo::ptr]
    for (auto i : cmd.inputs) {
        mgb_assert(i->ptr, "Invalid input tensor ptr!");
        // refcnt ++, owners: [i->ptr, tensor_inputs]
        // tensor_inputs.push_back(i->ptr);
        inputs.push_back(i->ptr);
    }
    if (state.options.enable_dtr_auto_drop &&
        state.options.dtr_eviction_threshold > 0) {
        auto_evict(0);
    }
    auto apply_on_physical_tensor =
            [&](auto&& self, const OpDef& def, SmallVector<TensorPtr> inputs,
                SmallVector<LogicalTensorDesc>& output_descs,
                const bool& validated) -> SmallVector<TensorPtr> {
        auto apply_functor = [&](std::shared_ptr<OpDef> op,
                                 SmallVector<TensorPtr> inputs,
                                 size_t nr_outputs) -> SmallVector<TensorPtr> {
            auto opname = op->trait()->make_name(*op);
            imperative_log_profile_begin(opname.c_str());
            // do not use infered output_desc in subgraph
            auto outputs = self(self, *op, inputs, output_descs, false);
            imperative_log_profile_end(opname.c_str());
            return outputs;
        };
        auto const_functor = [&](TensorPtr value) -> TensorPtr { return value; };
        if (def.trait()->make_forward_graph) {
            // apply recursivily
            SmallVector<LogicalTensorDesc> input_descs;
            for (auto&& input : inputs) {
                input_descs.push_back({{{}, input->dtype()}, input->comp_node()});
            }
            auto forward_graph = OpDef::make_forward_graph(def, input_descs);
            auto outputs = forward_graph.apply<TensorPtr>(
                    inputs, apply_functor, const_functor);
            return outputs;
        }
        // Check Input Layout
        // Get the input layout constraints, and if the constraint is not satisfied
        // inplace update the layout and blob to make the tensor contiguous
        auto&& constraints = OpDef::get_input_layout_constraint(def, inputs);
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
            auto&& layout_checker = constraints[idx];
            if (layout_checker) {
                inputs[idx]->to_contiguous_inplace(layout_checker);
            }
        }
        return OpDef::apply_on_physical_tensor(
                def, std::move(inputs), output_descs, validated);
    };
    MGB_RECORD_EVENT(OpExecuteEvent, apply_id, {}, reason);
    SmallVector<std::pair<CompNode, uint64_t>> kernels;
    if (profiling_device) {
        // Collecting devices
        SmallVector<CompNode> devices;
        for (auto&& i : concat(cmd.inputs, cmd.outputs)) {
            if (i != nullptr && count(devices, i->desc.comp_node) == 0) {
                devices.push_back(i->desc.comp_node);
                kernels.push_back({i->desc.comp_node, Profiler::next_id()});
            }
        }
    }
    for (auto* input : cmd.inputs) {
        auto input_id = input->id;
        MGB_RECORD_EVENT(OpInputEvent, input_id);
        MGB_RECORD_EVENT(TensorUsageEvent, input_id);
        MGB_RECORD_EVENT(OpInputFinishEvent, input_id);
    }
    // Before wait
    // TODO: split operator wait and execute so that OpWait could be corrected recorded.
    // Before execute
    for (auto&& [device, kernel_id] : kernels) {
        MGB_RECORD_EVENT(KernelLaunchEvent, apply_id, kernel_id, device);
        MGB_RECORD_EVENT_IF(
                (Profiler::get_option("profile_device", 0)), RecordDeviceEvent,
                Timer::record_device(device));
    }
    // Apply op
    SmallVector<LogicalTensorDesc> output_descs;
    for (auto i : cmd.outputs) {
        output_descs.push_back(i->desc);
    }
    // Here std::move is REQUIRED for removing duplicated references.
    auto outputs = apply_on_physical_tensor(
            apply_on_physical_tensor, *cmd.op, inputs, output_descs, cmd.validated);
    // After execute
    for (auto&& [device, kernel_id] : kernels) {
        MGB_RECORD_EVENT_IF(
                (Profiler::get_option("profile_device", 0)), RecordDeviceEvent,
                Timer::record_device(device));
        MGB_RECORD_EVENT(KernelLaunchFinishEvent, apply_id, kernel_id, device);
    }
    // End profiling operator
    mgb_assert(outputs.size() == cmd.outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        auto output = cmd.outputs[i];
        if (output == nullptr) {
            MGB_RECORD_EVENT(OpOutputEvent, 0);
            MGB_RECORD_EVENT(OpOutputFinishEvent, 0);
        } else if (output->ptr != nullptr) {
            MGB_RECORD_EVENT(OpOutputEvent, output->id);
            MGB_RECORD_EVENT(OpOutputFinishEvent, output->id);
        } else {
            MGB_RECORD_EVENT(OpOutputEvent, output->id);
            produce_tensor(output, outputs[i]);
            MGB_RECORD_EVENT(OpOutputFinishEvent, output->id);
            sample_on_device(output->desc.comp_node, false);
        }
    }

    if (state.options.enable_dtr_auto_drop) {
        double estimate_compute_time = 0;
        for (auto i : cmd.inputs) {
            estimate_compute_time += i->memory;
        }
        for (auto i : outputs) {
            estimate_compute_time += i->blob()->size();
        }
        m_dtr.estimate_timestamp += estimate_compute_time / 1e8;
        for (auto i : cmd.outputs) {
            if (i != nullptr) {
                i->compute_time = estimate_compute_time;
            }
        }
        m_dtr.unpin(cmd.inputs, state);
    }
    MGB_RECORD_EVENT(OpExecuteFinishEvent, apply_id, {}, reason);
    // End profiling operator
}

void ChannelImpl::flush_apply_stack() {
    m_applying = true;
    auto& state = get_worker_state();
    while (!m_apply_stack.empty()) {
        auto& [cmd, idx, recomp, reason] =
                m_apply_stack.top();  // cmd.inputs[0~idx-1] is in memory
        if (idx == 0) {
            if (state.options.enable_dtr_auto_drop) {
                m_dtr.pin(cmd.inputs);
            }
            if (recomp) {
                MGB_RECORD_EVENT(
                        TensorCommandEvent, recomp->id, TensorCommandKind::ReGen);
            }
        }
        bool regen = false;
        for (size_t i = idx; i < cmd.inputs.size(); i++) {
            auto&& p = cmd.inputs[i];
            if (state.options.enable_dtr_auto_drop) {
                m_dtr.update_used_time(p);
            }
            if (!p->ptr && p->evict_type != EvictType::NONE) {
                idx = i + 1;
                regenerate(p);  // add ApplyOp to the stack
                regen = true;
                break;
            }
        }
        if (regen)
            continue;
        // the required input tensors are already in memory
        auto [cmd_backup, recomp_backup, reason_backup] =
                std::make_tuple(cmd, recomp, reason);
        m_apply_stack.pop();
        do_apply_op(cmd_backup, reason_backup);
        if (recomp_backup) {
            MGB_RECORD_EVENT(
                    TensorCommandFinishEvent, recomp_backup->id,
                    TensorCommandKind::ReGen);
            for (auto o : cmd_backup.outputs) {
                if (o) {
                    m_dtr.update_dsu_after_recompute(o);
                }
            }
        }
    }
    m_applying = false;
}

bool ChannelImpl::auto_evict(size_t force_num) {
    auto& state = get_worker_state();
    if (!m_dtr.comp_node.valid()) {
        return false;
    }
    size_t current_memory = m_dtr.comp_node.get_used_memory();
    size_t flag = false;
    while ((state.options.dtr_eviction_threshold > 0 &&
            current_memory > state.options.dtr_eviction_threshold) ||
           force_num > 0) {
        MGB_RECORD_EVENT(AutoEvictEvent);
        sample_on_device(m_dtr.comp_node, false);
        auto best = m_dtr.find_best_tensor(state.options.enable_dtr_sqrt_sampling);
        if (!best) {
            MGB_RECORD_EVENT(AutoEvictFinishEvent);
            break;
        }
        if (best->ptr.unique() && best->ptr->blob().unique()) {
            current_memory -= best->memory;
            if (force_num > 0) {
                force_num--;
            }
            flag = true;
        }
        do_drop(best);
        if (best->evict_type == EvictType::DROP) {
            m_dtr.update_dsu_after_evict(best);
        }
        sample_on_device(m_dtr.comp_node, false);
        MGB_RECORD_EVENT(AutoEvictFinishEvent);
    }
    return flag;
}

void ChannelImpl::detach_users(TensorInfo* dest) {
    SmallVector<TensorInfo::ComputePath*> users = dest->users;
    for (auto* user : users) {
        SmallVector<TensorInfo*> outputs = user->outputs;
        SmallVector<TensorInfo*> inputs = user->inputs;
        for (auto* output : outputs) {
            // When a `ComputePath` is detach from it's input,
            // there is no need to reserve it,
            // so we detach all output of this path
            // to decrease it's `ref_cnt` to zero.
            if (output == nullptr) {
                continue;
            }
            regenerate(output);
            output->detach_producer();
            for (auto* input : inputs) {
                input->ref_cnt--;
            }
        }
        // now user is dead
    }
    mgb_assert(dest->users.empty(), "ComputePath leaking");
}

bool ChannelImpl::check_available() {
    return !m_closed;
}

TensorPtr ChannelImpl::wait_tensor(TensorInfo* info, TensorProp prop) {
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);
    mgb_assert(!m_waitee, "duplicate waitee");
    m_waitee = info;
    m_waitee_id = Profiler::next_id();
    MGB_RECORD_EVENT(TensorWaitPropEvent, info->id, m_waitee_id, prop);
    bool require_host = prop == TensorProp::HostValue;
    auto host_available = [&] { return info->ptr && info->ptr->value_fetched(); };
    bool wait_host = false;
    if (require_host && !host_available()) {
        // avoid dead lock
        lock.unlock();
        m_worker.add_task(
                {Profiler::next_id(), GetValue{info},
                 get_channel_state().stack_manager.dump()});
        lock.lock();
        wait_host = true;
    }
    m_cv.wait(lock, [&]() {
        check_worker_exc_unsafe();
        return require_host ? host_available() : static_cast<bool>(info->ptr);
    });
    MGB_RECORD_EVENT(TensorWaitPropFinishEvent, info->id, m_waitee_id, prop);
    m_waitee = nullptr;
    if (wait_host) {
        auto err = info->ptr->comp_node().check_async_error();
        mgb_assert(!err, "%s", err->what());
    }
    return info->ptr;
}

void ChannelImpl::notify_tensor_unsafe(TensorInfo* info) {
    if (info == m_waitee) {
        MGB_RECORD_EVENT(TensorNotifyPropEvent, info->id);
        m_cv.notify_all();
    }
}

std::unordered_set<TensorInfo*> ChannelImpl::collect_valid_tensors() {
    std::unordered_set<TensorInfo*> valid_tensors;
    for (auto* handle : m_valid_handle) {
        auto* info = reinterpret_cast<TensorInfo*>(handle);
        valid_tensors.insert(info);
    }
    return valid_tensors;
}

void ChannelImpl::alloc_tensor_with_evict(Blob* x) {
    auto reserve_size = [&](size_t size) {
        if (!m_dtr.comp_node.valid()) {
            return false;
        }
        while (size > m_dtr.comp_node.get_max_block_size_available()) {
            bool evict_suc = auto_evict(1);
            if (!evict_suc)
                return false;
        }
        return true;
    };
    auto pre_level = set_log_level(LogLevel::NO_LOG);
    reserve_size(x->size());
    MGB_TRY { BlobManager::inst()->alloc_direct(x, x->size()); }
    MGB_CATCH(MemAllocError&, {
        bool suc = false;
        while (!suc) {
            if (!auto_evict(1)) {
                break;
            }
            MGB_TRY { BlobManager::inst()->alloc_direct(x, x->size()); }
            MGB_CATCH(MemAllocError&, { continue; });
            suc = true;
        }
        if (!suc) {
            set_log_level(pre_level);
            mgb_log_warn(
                    "reallocating all cuda memory to alleviate fragmentation, the "
                    "performance may be affected");
            set_log_level(LogLevel::NO_LOG);
            imperative_log_profile_begin("defrag");
            BlobManager::inst()->defrag(x->comp_node());
            imperative_log_profile_end("defrag");
            BlobManager::inst()->alloc_direct(x, x->size());
        }
    });
    set_log_level(pre_level);
}

void ChannelImpl::process_one_task(Command& icmd) {
    using namespace ranges;
    using namespace ranges::views;
    auto& state = get_worker_state();
    auto& options = state.options;
    // TODO: remove std::visit for support osx 10.12
    auto cmd_visitor = [&](const auto& cmd) {
        using T = std::decay_t<decltype(cmd)>;
        if constexpr (std::is_same_v<T, Put>) {
            MGB_RECORD_EVENT(TensorCommandEvent, cmd.dest->id, TensorCommandKind::Put);
            MGB_RECORD_EVENT_IF(
                    (Profiler::get_option("profile_device", 0)), RecordDeviceEvent,
                    Timer::record_device(cmd.value.comp_node()));
            auto value = cmd.no_cache ? std::make_shared<Tensor>(cmd.value)
                                      : Tensor::make(cmd.value);
            MGB_RECORD_EVENT_IF(
                    (Profiler::get_option("profile_device", 0)), RecordDeviceEvent,
                    Timer::record_device(cmd.value.comp_node()));
            produce_tensor(cmd.dest, std::move(value));
            MGB_RECORD_EVENT(
                    TensorCommandFinishEvent, cmd.dest->id, TensorCommandKind::Put);
            sample_on_device(cmd.dest->desc.comp_node, false);
        } else if constexpr (std::is_same_v<T, ApplyOp>) {
            for (auto& i : cmd.inputs) {
                if (i->invalid) {
                    MGB_LOCK_GUARD(m_mutex);
                    for (auto& i : cmd.outputs) {
                        i->invalid = true;
                    }
                    return;
                }
            }
            m_apply_stack.push({cmd, 0, nullptr, "cmd"});
            flush_apply_stack();
            for (size_t i = 0; i < cmd.outputs.size(); ++i) {
                auto output = cmd.outputs[i];
                if (output == nullptr) {
                    continue;
                }
                if (state.options.enable_dtr_auto_drop) {
                    output->dsu_ptr = std::make_shared<DsuNode>(output->compute_time);
                }
            }
            if (state.options.enable_drop && state.options.record_computing_path) {
                auto is_inplace = [](std::tuple<TensorInfo*, TensorInfo*> tuple2) {
                    auto& input = std::get<0>(tuple2);
                    auto& output = std::get<1>(tuple2);
                    if (!input->ptr || !output->ptr) {
                        return false;
                    }
                    return input->ptr->blob()->storage() ==
                           output->ptr->blob()->storage();
                };
                // FIXME: do not use opname as identifier
                auto get_name = [](const OpDef& opdef) {
                    if (auto attr = opdef.try_cast_final<OprAttr>()) {
                        return attr->type.c_str();
                    }
                    return opdef.dyn_typeinfo()->name;
                };

                auto is_cross_cn = [comp_node = m_dtr.comp_node](TensorInfo* info) {
                    return info->desc.comp_node != comp_node;
                };

                bool cross_cn = any_of(concat(cmd.inputs, cmd.outputs), is_cross_cn);
                bool inplace =
                        any_of(cartesian_product(cmd.inputs, cmd.outputs), is_inplace);

                if (!inplace && !cross_cn && !m_dtr.is_bad_op(get_name(*cmd.op))) {
                    TensorInfo::ComputePath::make(
                            cmd.id, cmd.op, cmd.inputs, cmd.outputs);
                    size_t detach_cnt = 0;
                    if (!strcmp(get_name(*cmd.op), "BatchNorm") &&
                        cmd.outputs.size() == 5) {
                        cmd.outputs[0]->detach_producer();  // detach running_mean
                        cmd.outputs[1]->detach_producer();  // detach running_var
                        for (auto input : cmd.inputs) {
                            input->ref_cnt -= 2;
                        }
                    }
                    for (auto output : cmd.outputs) {
                        if (output->producer &&
                            !output->size_exceeds_thd(
                                    state.options.dtr_evictee_minimum_size)) {
                            output->detach_producer();
                            detach_cnt++;
                        }
                    }
                    for (auto input : cmd.inputs) {
                        input->ref_cnt -= detach_cnt;
                    }
                }
            }
        } else if constexpr (std::is_same_v<T, Del>) {
            MGB_RECORD_EVENT(TensorCommandEvent, cmd.dest->id, TensorCommandKind::Del);
            CompNode device = cmd.dest->desc.comp_node;
            uint64_t tensor_id = cmd.dest->id;
            free(cmd.dest);
            MGB_RECORD_EVENT(
                    TensorCommandFinishEvent, tensor_id, TensorCommandKind::Del);
            sample_on_device(device, false);
        } else if constexpr (std::is_same_v<T, GetValue>) {
            if (cmd.dest->invalid)
                return;
            imperative_log_profile_begin("GetValue");
            if (!cmd.dest->ptr && cmd.dest->evict_type != EvictType::NONE) {
                regenerate(cmd.dest);
            }
            cmd.dest->ptr->fetch_value();
            MGB_LOCK_GUARD(m_mutex);
            notify_tensor_unsafe(cmd.dest);
            imperative_log_profile_end("GetValue");
        } else if constexpr (std::is_same_v<T, Drop>) {
            if (cmd.dest->invalid)
                return;
            MGB_RECORD_EVENT(TensorCommandEvent, cmd.dest->id, TensorCommandKind::Drop);
            do_drop(cmd.dest, true);
            MGB_RECORD_EVENT(
                    TensorCommandFinishEvent, cmd.dest->id, TensorCommandKind::Drop);
        } else if constexpr (std::is_same_v<T, SetOption>) {
            options.set_option(cmd.key, cmd.value);
        } else if constexpr (std::is_same_v<T, StartProfile>) {
            MGB_RECORD_EVENT(StartProfileEvent);
            CompNode::sync_all();
            for (auto* info : cmd.capture_tensors) {
                MGB_RECORD_EVENT(TensorDeclareEvent, info->id, info->name);
                if (info->status == TensorInfo::Produced) {
                    // TODO: handle drop
                    MGB_RECORD_EVENT(
                            TensorProduceEvent, info->id, info->desc.layout,
                            info->desc.comp_node, info->ptr->dev_tensor().raw_ptr());
                }
            }
            CompNode::foreach ([&](CompNode device) {
                sample_on_device(device, true);
                MGB_RECORD_EVENT_IF(
                        (Profiler::get_option("profile_device", 0)), RecordDeviceEvent,
                        Timer::record_device(device));
            });
            MGB_RECORD_EVENT(StartProfileFinishEvent);
        } else if constexpr (std::is_same_v<T, StopProfile>) {
            MGB_RECORD_EVENT(StopProfileEvent);
            for (auto* info : cmd.escape_tensors) {
                bool has_value = info->status == TensorInfo::Produced;
                if (has_value) {
                    MGB_RECORD_EVENT(TensorReleaseEvent, info->id);
                }
                MGB_RECORD_EVENT(TensorEraseEvent, info->id);
            }
            CompNode::foreach (
                    [&](CompNode device) { sample_on_device(device, true); });
            MGB_RECORD_EVENT(StopProfileFinishEvent);
        } else if constexpr (std::is_same_v<T, PushScope>) {
            MGB_RECORD_EVENT(ScopeEvent, cmd.scope_name);
        } else if constexpr (std::is_same_v<T, PopScope>) {
            MGB_RECORD_EVENT(ScopeFinishEvent, cmd.scope_name);
        } else {
            static_assert(!std::is_same_v<T, T>);
        }
    };
    std::visit(
            [&](const auto& cmd) {
                using T = std::decay_t<decltype(cmd)>;
                if (!options.catch_worker_execption) {
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
                    MGB_RECORD_EVENT(WorkerExceptionEvent);
                    if (m_waitee) {
                        notify_tensor_unsafe(m_waitee);
                    }
                }
            },
            icmd.data);
}

void ChannelImpl::check_worker_exc_unsafe() {
    if (m_worker_exc) {
        // for reuse interpreter_for_py after some exception tests
        m_waitee = nullptr;
        std::exception_ptr exc;
        std::swap(exc, m_worker_exc);
        try {
            std::rethrow_exception(exc);
        } catch (...) {
            throw AsyncError();
        }
    }
}

void ChannelImpl::start_profile() {
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    auto capture_tensors = collect_valid_tensors();
    if (capture_tensors.size() > 0) {
        m_worker.add_task(
                {Profiler::next_id(), StartProfile{std::move(capture_tensors)},
                 get_channel_state().stack_manager.dump()});
    }
}

void ChannelImpl::stop_profile() {
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    auto escape_tensors = collect_valid_tensors();
    if (escape_tensors.size() > 0) {
        m_worker.add_task(
                {Profiler::next_id(), StopProfile{std::move(escape_tensors)},
                 get_channel_state().stack_manager.dump()});
    }
}

void ChannelImpl::push_scope(std::string name) {
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    state.stack_manager.enter(name);
    MGB_RECORD_EVENT(ScopeEvent, name);
    m_worker.add_task(
            {Profiler::next_id(), PushScope{name},
             get_channel_state().stack_manager.dump()});
}

void ChannelImpl::pop_scope(std::string name) {
    MGB_LOCK_GUARD(m_spin);
    mgb_assert(check_available(), "Channel already closed");
    auto& state = get_channel_state();
    state.stack_manager.exit(name);
    MGB_RECORD_EVENT(ScopeFinishEvent, name);
    m_worker.add_task(
            {Profiler::next_id(), PopScope{name},
             get_channel_state().stack_manager.dump()});
}

void ChannelImpl::assert_in_channel() {
    mgb_assert(
            get_worker_tid() != std::this_thread::get_id(),
            "this method cannot be called in worker thread");
}

void ChannelImpl::assert_in_worker() {
    mgb_assert(
            get_worker_tid() == std::this_thread::get_id(),
            "this method can only be called in worker thread");
}

void ChannelImpl::sample_on_device(CompNode device, bool force) {
    if (!force) {
        thread_local int last_sample_id = 0;
        int sample_rate =
                Profiler::is_profiling() ? Profiler::get_option("sample_rate", 0) : 0;
        if (!sample_rate || ((++last_sample_id) % sample_rate != 0)) {
            return;
        }
    }
    MGB_RECORD_EVENT(SampleDeviceEvent, device);
    auto [total, free] = device.get_mem_status_bytes();
    MGB_RECORD_EVENT(SampleDeviceFinishEvent, device, total, free);
}

void ChannelImpl::DynamicSublinear::pin(const SmallVector<TensorInfo*>& vec) {
    for (auto i : vec) {
        i->pin();
        erase_candidate(i);
    }
}

void ChannelImpl::DynamicSublinear::unpin(
        const SmallVector<TensorInfo*>& vec, WorkerState& state) {
    for (auto i : vec) {
        i->unpin();
        if (i->pinned == 0 &&
            i->size_exceeds_thd(state.options.dtr_evictee_minimum_size) &&
            i->cand_index == UINT_MAX) {
            insert_candidate(i);
        }
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

TensorInfo* ChannelImpl::DynamicSublinear::find_best_tensor(
        bool enable_dtr_sqrt_sampling = false) {
    if (candidates.empty())
        return nullptr;

    double min_msps = -1;
    TensorInfo* best = nullptr;
    size_t sz = 1;
    if (enable_dtr_sqrt_sampling) {
        while (sz * sz <= candidates.size())
            sz++;
        sz--;
    } else {
        sz = candidates.size();
    }

    size_t ti = rand() % sz;
    for (size_t vi = 0; vi < sz; vi++) {
        if (!enable_dtr_sqrt_sampling) {
            ti = vi;
        }
        auto i = candidates[ti];
        if (i->producer && i->ptr && i->evict_type == EvictType::NONE) {
            double neighbor_cost = estimate_neighbor_cost(i);
            size_t begin_ptr =
                    reinterpret_cast<size_t>(i->ptr->blob()->storage().get());
            auto side_info = i->ptr->comp_node().get_free_left_and_right(
                    begin_ptr, begin_ptr + i->ptr->blob()->size());
            double free_mem = side_info.first + side_info.second;
            double msps = i->eval_func(
                    neighbor_cost, free_mem, estimate_timestamp, 1.0, 1.0, 1.0, 1.0001);
            if (min_msps < 0 || msps < min_msps) {
                min_msps = msps;
                best = i;
            }
        }
        if (enable_dtr_sqrt_sampling) {
            ti += rand() % sz;
            if (ti > candidates.size())
                break;
        }
    }
    return best;
}

void ChannelImpl::DynamicSublinear::merge(
        std::shared_ptr<DsuNode>& x, std::shared_ptr<DsuNode>& y) {
    auto&& f_x = find_father(x);
    auto&& f_y = find_father(y);
    if (f_x.get() == f_y.get()) {
        return;
    }
    f_y->t += f_x->t;
    f_x->parent = f_y;
}

std::shared_ptr<DsuNode> ChannelImpl::DynamicSublinear::find_father(
        std::shared_ptr<DsuNode>& x) {
    if (x->is_root()) {
        return x;
    } else {
        auto&& fa = find_father(x->parent);
        return x->parent = fa;
    }
}

void ChannelImpl::DynamicSublinear::insert_candidate(TensorInfo* ptr) {
    // tensor to be inserted must be brand new
    mgb_assert(
            ptr->cand_index == UINT_MAX, "got wrong candidate index : %lu",
            ptr->cand_index);
    ptr->cand_index = candidates.size();
    candidates.push_back(ptr);
    if (!comp_node.valid()) {
        comp_node = ptr->ptr->comp_node();
    }
}

void ChannelImpl::DynamicSublinear::erase_candidate(TensorInfo* ptr) {
    // close dtr will just clear candidates, so nothing to erase
    if (candidates.empty()) {
        ptr->cand_index = UINT_MAX;
        return;
    }
    // some tensors may be erased already, just skip them
    if (ptr->cand_index != UINT_MAX) {
        std::swap(candidates[ptr->cand_index], candidates.back());
        candidates[ptr->cand_index]->cand_index = ptr->cand_index;
        candidates.pop_back();
        ptr->cand_index = UINT_MAX;
    }
}

void ChannelImpl::DynamicSublinear::update_used_time(TensorInfo* ptr) {
    ptr->last_used_time = estimate_timestamp;
}
