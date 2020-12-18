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
#include "megbrain/common.h"


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

void ChannelImpl::swap_in(void* handle) {
    if (m_enable_evict & SWAP) {
        mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
                "invalid handle: %p", handle);
        m_worker.add_task(SwapIn{reinterpret_cast<TensorInfo*>(handle)});
    }
}

void ChannelImpl::swap_out(void* handle) {
    if (m_enable_evict & SWAP) {
        mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
                "invalid handle: %p", handle);
        m_worker.add_task(SwapOut{reinterpret_cast<TensorInfo*>(handle)});
    }
}

void ChannelImpl::drop(void* handle) {
    if (m_enable_evict & DROP) {
        mgb_assert(m_valid_handle.find(handle) != m_valid_handle.end(),
                "invalid handle: %p", handle);
        m_worker.add_task(Drop{reinterpret_cast<TensorInfo*>(handle)});
    }
}

SmallVector<void*> ChannelImpl::apply_op(
        std::shared_ptr<OpDef> op,
        const SmallVector<void*>& inputs) {
    for (auto i : inputs) {
        mgb_assert(m_valid_handle.find(i) != m_valid_handle.end(),
                "invalid handle: %p", i);
    }
    SmallVector<TensorInfo*> input_infos;
    input_infos.reserve(inputs.size());
    SmallVector<LogicalTensorDesc> input_descs;
    input_descs.reserve(inputs.size());
    std::unique_lock<decltype(m_mutex)> lock(m_mutex);
    for (auto i : inputs) {
        auto info = reinterpret_cast<TensorInfo*>(i);
        mgb_assert(!info->invalid, "Invalid tensor, unable to apply_op!");
        input_infos.push_back(info);
        input_descs.push_back(info->desc);
    }
    lock.unlock();

    auto [output_descs, validated] = OpDef::infer_output_attrs_fallible(*op, input_descs);
    ApplyOp cmd{std::move(op)};
    cmd.inputs = std::move(input_infos);
    cmd.outputs.reserve(output_descs.size());
    SmallVector<void*> outputs;
    // FIXME: remove this check when op check is correct
    bool validated_bkp = true;
    for (size_t i = 0;i < output_descs.size();i ++) {
        auto&& desc = output_descs[i];
        if (desc.layout.ndim == 0) {
            validated_bkp = false;
        }
        auto info = alloc();
        info->desc = desc;
        m_valid_handle.insert(info);
        cmd.outputs.push_back(info);
        outputs.push_back(info);
    }
    if (m_enable_evict & DROP) {
        for (auto out : cmd.outputs) {
            out->path.op = cmd.op;
            for (auto out_ : cmd.outputs) {
                out->path.outputs.push_back(m_st.at(out_));
            }
            for (auto inp : cmd.inputs) {
                out->path.inputs.push_back(m_st.at(inp));
                inp->path.dep_outputs.push_back(m_st.at(out));
            }
        }
    }
    m_worker.add_task(std::move(cmd));
    if (!(validated && validated_bkp) && m_async_level == 1) {
        sync();
    } else if (m_async_level == 0) {
        sync();
        // check device error
        for (auto&& oup : outputs) {
            auto info = reinterpret_cast<TensorInfo*>(oup);
            info->ptr->comp_node().sync();
        }
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
        mgb_assert(!info->invalid, "Invalid tensor, unable to get_value!");
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
    auto info = m_pool.alloc();
    m_st.insert(info);
    return info;
}

void ChannelImpl::free(TensorInfo* ptr) {
    MGB_LOCK_GUARD(m_mutex);
    if (ptr->path.dep_outputs.size() > 0) {
        remove_dep(ptr);
    }
    m_st.erase(ptr);
    mgb_assert(ptr->allow_delete, "delete before ref_cnt = 0");
    m_pool.free(ptr);
}

ChannelImpl::~ChannelImpl() {
    close();
}

void ChannelImpl::produce_tensor(TensorInfo* dest, TensorPtr ptr, bool notice = true) {
    if (notice) {
        MGB_LOCK_GUARD(m_mutex);
        dest->value_fetched = ptr->value_fetched();
        // update tensor desc for static infer
        // if (dest->desc.layout.ndim) {
        //     mgb_assert(dest->desc.layout.eq_shape(ptr->layout()));
        // }
        dest->desc.layout = ptr->layout();
        dest->desc.comp_node = ptr->comp_node();
        dest->ptr = std::move(ptr);
        if (m_waitee == dest) {
            m_cv.notify_all();
        }
    } else {
        dest->value_fetched = ptr->value_fetched();
        // update tensor desc for static infer
        dest->desc.layout = ptr->layout();
        dest->desc.comp_node = ptr->comp_node();
        dest->ptr = std::move(ptr);
    }
}

void ChannelImpl::do_swap_out(TensorInfo* dest) {
    if (dest->evict_type == DROP) {
        mgb_log_warn("the evict type of tensor %p was set to DROP, this SWAP operation will be ignored", dest);
        return;
    }
    if (!dest->ptr) {
        return;
    }
    dest->evict_type = SWAP;
    dest->value_fetched = false;
    // TODO: swap in parallel
    dest->h_value.copy_from(dest->ptr->dev_tensor()).sync();
    dest->ptr.reset();
}

void ChannelImpl::do_swap_in(TensorInfo* dest) {
    if (dest->ptr) {
        return;
    }
    if (dest->h_value.empty()) {
        mgb_log_error("backup of the tensor %p not found", dest);
        return;
    }
    produce_tensor(dest, Tensor::make(dest->h_value), false);
    dest->evict_type = NONE;
}

void ChannelImpl::remove_dep(TensorInfo* dest) {
    for (auto i : dest->path.dep_outputs) {
        auto out_ptr = i.lock();
        if (out_ptr) {
            regenerate(out_ptr.get(), true);
        }
    }
}

void ChannelImpl::do_drop(TensorInfo* dest) {
    if (dest->evict_type == SWAP) {
        mgb_log_warn("the evict type of tensor %p was set to SWAP, this DROP operation will be ignored", dest);
        return;
    }
    if (!dest->path.op) {
        mgb_log_warn("the input that produced tensor %p has been deleted, this drop operation will be ignored", dest);
        return;
    }
    if (dest->recompute_times >= m_max_recompute_time) {
        mgb_log_warn("the recomputation time for tensor %p exceeds the limit, this drop operation will be ignored", dest);
        return;
    }
    if (!dest->ptr) {
        return;
    }
    dest->evict_type = DROP;
    dest->value_fetched = false;
    dest->ptr.reset();
}

void ChannelImpl::set_swap_flag(bool flag) {
    if (flag) {
        m_enable_evict |= SWAP;
    } else {
        m_enable_evict &= ~SWAP;
    }
}

void ChannelImpl::set_drop_flag(bool flag) {
    if (flag) {
        m_enable_evict |= DROP;
    } else {
        m_enable_evict &= ~DROP;
    }
}

void ChannelImpl::regenerate(TensorInfo* info, bool must_drop = false) {
    if (!info->ptr && info->evict_type != NONE) {
        if (info->evict_type == SWAP) {
            do_swap_in(info);
        } else {
            mgb_assert(info->evict_type == DROP);
            mgb_assert(info->path.op, "recomputation path not found");
            auto path = info->path;
            SmallVector<TensorPtr> inputs;
            inputs.reserve(path.inputs.size());
            for (auto i : path.inputs) {
                mgb_assert(i, "invalid history input");
                if (!i->ptr) {
                    regenerate(i.get(), must_drop);
                }
                inputs.push_back(i->ptr);
            }
            auto outputs = OpDef::apply_on_physical_tensor(*path.op, inputs);
            for (size_t i = 0; i < outputs.size(); i ++) {
                auto out_ptr = path.outputs[i].lock();
                if (out_ptr) {
                    out_ptr->recompute_times ++;
                    if (!out_ptr->ptr && out_ptr->evict_type == DROP) {
                        produce_tensor(out_ptr.get(), std::move(outputs[i]), false);
                    }
                }
            }
        }
    }
    if (must_drop) {
        if (info->path.op) {
            info->path.op.reset();
            info->path.inputs.clear();
            if (info->evict_type == DROP) {
                info->evict_type = NONE;
            }
        }
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
                    if (m_enable_evict && i->evict_type != NONE) {
                        if (!i->ptr) {
                            regenerate(i);
                        }
                    }
                    mgb_assert(i->ptr, "Invalid input tensor ptr!");
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
                if (m_enable_evict && cmd.dest->evict_type != NONE) {
                    if (!cmd.dest->ptr) {
                        regenerate(cmd.dest);
                    }
                }
                mgb_assert(cmd.dest->ptr, "Invalid tensor ptr!");
                cmd.dest->ptr->fetch_value();
                MGB_LOCK_GUARD(m_mutex);
                cmd.dest->value_fetched = true;
                if (m_waitee == cmd.dest) {
                    m_cv.notify_all();
                }
            } else if constexpr (std::is_same_v<T, SwapIn>) {
                do_swap_in(cmd.dest);
            } else if constexpr (std::is_same_v<T, SwapOut>) {
                do_swap_out(cmd.dest);
            } else if constexpr (std::is_same_v<T, Drop>) {
                do_drop(cmd.dest);
            } else {
                static_assert(!std::is_same_v<T, T>);
            }
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
    }, cmd);
}


void ChannelImpl::check_worker_exc_unsafe() {
    if (m_worker_exc) {
        std::exception_ptr exc;
        std::swap(exc, m_worker_exc);
        std::rethrow_exception(exc);
    }
}
