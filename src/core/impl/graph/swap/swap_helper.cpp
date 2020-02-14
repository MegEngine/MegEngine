/**
 * \file src/core/impl/graph/swap/swap_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./swap_helper.h"
#include "megbrain/comp_node_env.h"

#if MGB_ENABLE_MEMORY_SWAP

using namespace mgb;
using namespace swap;

/* ===================== SwapCopyThreadPool ===================== */

SwapCopyThreadPool& SwapCopyThreadPool::inst(CompNode cn) {
    auto maker = [cn]() { return std::make_shared<SwapCopyThreadPool>(cn); };
    return CompNodeEnv::from_comp_node(cn).get_user_data<SwapCopyThreadPool>(
            maker);
}

void SwapCopyThreadPool::start() {
    if ((++m_nr_start) == 1) {
        m_pool.start(1);
    }
}

void SwapCopyThreadPool::stop() {
    auto pre = m_nr_start--;
    mgb_assert(pre);
    if (pre == 1) {
        m_pool.stop();
    }
}

template <typename Func, typename... Args>
FutureThreadPool<void>::Future SwapCopyThreadPool::launch(Func&& func,
                                                          Args&&... args) {
    return m_pool.launch(std::forward<Func>(func), std::forward<Args>(args)...);
}

/* ===================== SwapVarRecorder ===================== */

void SwapVarRecorder::copy_host_to_bucket(size_t id, Bucket& dest) {
    mgb_assert(!dest.copy_task_need_wait);
    {
        auto p = dest.copy_task_running.exchange(true);
        mgb_assert(!p);
    }
    mgb_assert(m_saved_buckets.find(id) != m_saved_buckets.end(),
               " Opr %zu not executed", id);
    dest.h2d_copy_refhold = (m_saved_buckets[id]);
    auto do_copy = [&dest]() {
        dest.associate_tensor->copy_from_fixlayout(
                *(dest.h2d_copy_refhold.get()));
        auto p = dest.copy_task_running.exchange(false);
        mgb_assert(p);
        dest.associate_tensor = nullptr;
    };
    dest.copy_task = m_copy_threadpool.launch(do_copy);
    dest.copy_task_need_wait = true;
}

void SwapVarRecorder::copy_bucket_to_host(size_t id, Bucket& src,
                                          CompNode comp_node) {
    mgb_assert(!src.copy_task_need_wait);
    {
        auto p = src.copy_task_running.exchange(true);
        mgb_assert(!p);
    }
    bool flag = false;
    if (m_saved_buckets.find(id) == m_saved_buckets.end()) {
        auto ptr = new HostTensorND;
        m_saved_buckets[id].reset(ptr);
        flag = true;
    }
    auto do_copy = [&src, this, id, flag]() {
        src.buf_on_copy_stream.comp_node().device_wait_event(
                src.ev_comp2copy());
        src.ev_hd().record();
        src.ev_hd().host_wait();
        if (flag)
            (m_saved_buckets[id].get())->copy_from(src.buf_on_copy_stream);
        else {
            (m_saved_buckets[id].get())
                    ->copy_from_fixlayout(src.buf_on_copy_stream);
        }
        auto p = src.copy_task_running.exchange(false);
        mgb_assert(p);
    };
    src.copy_task = m_copy_threadpool.launch(do_copy);
    src.copy_task_need_wait = true;
}

SwapVarRecorder::SwapVarRecorder(SwapVarInfo* swap_var_info, size_t ensure_size)
        : m_copy_threadpool{SwapCopyThreadPool::inst(
                  swap_var_info->var->comp_node())},
          m_swap_var_info{swap_var_info},
          m_ensure_size{ensure_size} {
    m_copy_threadpool.start();
}

void SwapVarRecorder::pop_value(size_t swap_out_id, const DeviceTensorND& od) {
    auto&& bucket = m_buckets_out[m_cur_bucket_out];
    m_cur_bucket_out ^= 1;
    bucket.wait_copy();
    bucket.set_associate_tensor(&od);
    copy_host_to_bucket(swap_out_id, bucket);
}

SwapVarRecorder::~SwapVarRecorder() {
    m_copy_threadpool.stop();
}

void SwapVarRecorder::on_val_produced(size_t swap_out_id,
                                      const DeviceTensorND& val) {
    auto&& bucket = m_buckets_in[m_cur_bucket_in];
    m_cur_bucket_in ^= 1;
    bucket.wait_copy();
    bucket.init(val.comp_node(), val.dtype(), val.shape(), m_ensure_size);
    bucket.buf.copy_from(val);
    copy_bucket_to_host(swap_out_id, bucket, val.comp_node());
    bucket.ev_comp2copy().record();
}

void SwapVarRecorder::wait_mission_finish(
        const DeviceTensorND* waiting_dev_tensor) {
    for (size_t i = 0; i < Bucket::nr_buckets_out; ++i)
        if (m_buckets_out[i].associate_tensor != nullptr &&
            m_buckets_out[i].associate_tensor->raw_ptr() ==
                    waiting_dev_tensor->raw_ptr()) {
            m_buckets_out[i].wait_copy();
            return;
        }
}

#endif  // MGB_ENABLE_MEMORY_SWAP

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
