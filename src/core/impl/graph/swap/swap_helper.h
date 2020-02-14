/**
 * \file src/core/impl/graph/swap/swap_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/comp_node.h"
#include "megbrain/graph.h"
#include "megbrain/utils/async_worker.h"

#if MGB_ENABLE_MEMORY_SWAP
namespace mgb {
namespace swap {

struct SwapVarInfo {
    VarNode* var = nullptr;
};

/* ===================== SwapCopyThreadPool ===================== */
class SwapCopyThreadPool final : public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;

    std::atomic_size_t m_nr_start{0};
    FutureThreadPool<void> m_pool;

public:
    SwapCopyThreadPool(CompNode cn) : m_pool{"SwapCopy" + cn.to_string()} {}

    ~SwapCopyThreadPool() {
        if (m_nr_start.load())
            m_pool.stop();
    }

    static SwapCopyThreadPool& inst(CompNode cn);

    void start();

    void stop();

    template <typename Func, typename... Args>
    FutureThreadPool<void>::Future launch(Func&& func, Args&&... args);
};

/* ===================== SwapVarRecorder ===================== */
class SwapVarRecorder final : public NonCopyableObj {
private:
    struct Bucket {
        struct EventGroup {
            std::unique_ptr<CompNode::Event> comp2copy, hd;
        };

        std::atomic_bool copy_task_running{false};

        DeviceTensorND buf, buf_on_copy_stream;

        const DeviceTensorND* associate_tensor = nullptr;

        bool ev_d2h_has_prev = false;

        static constexpr int nr_buckets_in = 2;
        static constexpr int nr_buckets_out = 2;
        EventGroup ev_grp[2];
        int ev_grp_cur = 0;

        std::shared_ptr<HostTensorND> h2d_copy_refhold;

        FutureThreadPool<void>::Future copy_task;
        bool copy_task_need_wait = false;

        bool h2d_wait_copy_in_next_overwrite = false;

        void init(CompNode comp_node, DType dtype, TensorShape shape,
                  size_t es) {
            mgb_assert(shape.ndim < TensorShape::MAX_NDIM,
                       "tensor shape ndim too large");

            auto cn_copy = comp_node.change_stream(CompNode::Stream::LOOP_SWAP);

            DeviceTensorStorage dts{cn_copy};
            dts.ensure_size(es);
            buf.storage(dts);

            buf.comp_node(comp_node).dtype(dtype).resize(shape);

            buf_on_copy_stream = buf;
            buf_on_copy_stream.comp_node(cn_copy);

            mgb_assert(buf_on_copy_stream.raw_ptr() == buf.raw_ptr());

            if (!ev_grp[0].comp2copy) {
                for (int i = 0; i < 2; ++i) {
                    ev_grp[i].comp2copy = comp_node.create_event();
                    ev_grp[i].hd = cn_copy.create_event();
                }
            } else {
                mgb_assert(ev_grp[0].comp2copy->comp_node() == comp_node);
                mgb_assert(ev_grp[0].hd->comp_node() == cn_copy);
            }
        }

        CompNode::Event& ev_comp2copy() {
            return *ev_grp[ev_grp_cur].comp2copy;
        }

        CompNode::Event& ev_hd() { return *ev_grp[ev_grp_cur].hd; }

        void set_associate_tensor(const DeviceTensorND* x) { associate_tensor = x; }

        void wait_copy() {
            if (copy_task_need_wait) {
                copy_task.get();
                copy_task_need_wait = false;
                mgb_assert(!copy_task_running);
                associate_tensor = nullptr;
            }
        }

    };  // Bucket

    SwapCopyThreadPool& m_copy_threadpool;
    SwapVarInfo* const m_swap_var_info;
    Bucket m_buckets_in[Bucket::nr_buckets_in];
    Bucket m_buckets_out[Bucket::nr_buckets_out];
    int m_cur_bucket_in = 0, m_cur_bucket_out = 0;
    size_t m_ensure_size = 0;

    std::unordered_map<size_t, std::shared_ptr<HostTensorND>> m_saved_buckets;

    std::mutex m_saved_buckets_mtx;

    bool m_enabled = false;

    /*! pop m_saved_buckets and copy to target bucket
     * It is assume that its receiver's ptr would not be released or occupied by
     * other tensor, so buffer is not neccessary
     */
    void copy_host_to_bucket(size_t id, Bucket& dest);

    void copy_bucket_to_host(size_t id, Bucket& src, CompNode comp_node);

public:
    SwapVarRecorder(SwapVarInfo* swap_var_info, size_t ensure_size);

    void pop_value(size_t swap_out_id, const DeviceTensorND& od);

    ~SwapVarRecorder();

    bool enabled() const { return m_enabled; }

    void enable(bool flag) { m_enabled = flag; }

    void on_val_produced(size_t swap_out_id, const DeviceTensorND& val);

    /*!
     * Sync before the tensor is consumed
     */
    void wait_mission_finish(const DeviceTensorND* waiting_dev_tensor);

    SwapVarInfo* swap_var_info() const { return m_swap_var_info; }
};

}  // namespace swap
}  // namespace mgb

#endif  // MGB_ENABLE_MEMORY_SWAP

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
