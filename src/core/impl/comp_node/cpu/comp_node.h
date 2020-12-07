/**
 * \file src/core/impl/comp_node/cpu/comp_node.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "../impl_helper.h"
#include "megbrain/utils/timer.h"

#include <atomic>

namespace mgb {
    class CpuCompNode final: public CompNodeImplHelper {
        struct Pool;
        static Pool *sm_pool;
        static Spinlock sm_pool_mtx;

        public:
            class WorkerQueue;
            class SeqRecorderImpl;

            // to implement CompNode::default_cpu
            friend class CompNode;

            // see the impl of EventImpl::host_wait_cv(); it's hard to achieve
            // all the the following goals without requiring sync at dtor, so we
            // have EVENT_DTOR_UNSAFE.
            //  1. Only one writing in wait
            //  2. Thread safe
            //  3. Memory safe
            static constexpr Flag sm_flag =
                    Flag::SUPPORT_RECORDER |
                    Flag::RECORDER_SUPPORT_DYNAMIC_ALLOC |
                    Flag::EVENT_DTOR_UNSAFE |
                    Flag::SUPPORT_UNIFIED_ADDRESS;

            //! base class for comp nodes that can be dispatched on CPU.
            //! This is currently used by CPU, FPGA and CADENCE
            class CpuDispatchableBase: public CompNode::Impl {
                protected:
                    using Impl::Impl;
                    ~CpuDispatchableBase() = default;
                public:
                    class EventImpl;
                    using Task = megdnn::thin_function<void()>;
                    virtual void dispatch(Task &&task) = 0;
                    void add_callback(Task&& task) override;
            };

            class CompNodeImpl;

            static void foreach(thin_function<void(CompNode)> callback);
            static void finalize();
            static size_t get_device_count();
            static Impl* load_cpu(Locator locator, Locator locator_logical);
            static void sync_all();
    };

    //! implement Event on CpuDispatchableBase comp nodes
    class CpuCompNode::CpuDispatchableBase::EventImpl: public EventImplHelper {
    protected:
        TimeSpec m_prev_finish_time;
#if MGB_HAVE_THREAD
        std::atomic_size_t
            m_record_nr_req{0}, m_record_nr_finish{0},
            m_dev_wait_nr_waiter{0};
        std::mutex m_dev_wait_mtx;
        std::condition_variable m_dev_wait_cv;
#endif

        bool do_finished() override;

        double do_elapsed_time_until(EventImplHelper &end) override;

        void do_device_wait_by(Impl *cn_impl) override;

        void host_wait_cv() override;

        void do_record() override;

        //! incr m_record_nr_req; this is used in do_record()
        void incr_nr_req() {
#if MGB_HAVE_THREAD
            m_record_nr_req.fetch_add(1, std::memory_order_relaxed);
#endif
        }

        //! callback to be dispatched to comp node
        void on_finish();

    public:
        using EventImplHelper::EventImplHelper;
        ~EventImpl() noexcept;
    };
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

