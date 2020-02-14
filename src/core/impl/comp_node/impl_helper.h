/**
 * \file src/core/impl/comp_node/impl_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/common.h"
#include "megbrain/comp_node.h"
#include "megbrain/comp_node_env.h"

#include <thread>

namespace mgb {

    class CompNodeImplHelper: public CompNode {
        protected:
            class EventImplHelper;

            static inline CompNode make_comp_node_from_impl(Impl *imp) {
                return {imp};
            }

            static void log_comp_node_created(
                    const Locator &locator, const Locator &locator_logical);

            //! get a MemNode that represents the host CPU memory
            static MemNode get_host_cpu_mem_node() {
                static int data;
                return MemNode{&data};
            }

        public:
            static CompNode::ImplBase* impl_from_comp_node(CompNode cn) {
                return cn.m_impl;
            }
    };

    /*!
     * \brief helper for implementing Event
     *
     * Each do_* method is called with a lock, and necessary input checks have
     * been performed.
     */
    class CompNodeImplHelper::EventImplHelper: public Event {
        std::mutex m_mtx;

        bool m_recorded = false, m_finished = false;

        protected:
            CompNode::Impl * const m_comp_node_impl;

            virtual void do_record() = 0;

            //! only called when m_finished is false
            virtual bool do_finished() = 0;

            //! end and this are finished, and m_comp_node_impl are the same
            virtual double do_elapsed_time_until(EventImplHelper &end) = 0;

            virtual void do_device_wait_by(Impl *cn_impl) = 0;

            //! implement host_wait() using a conditional var; the default impl
            //! still busily waits on finished()
            virtual void host_wait_cv();

        public:
            EventImplHelper(
                    CompNode::Impl *comp_node_impl, size_t create_flags):
                Event(create_flags),
                m_comp_node_impl{comp_node_impl}
            {
            }

            void record() override final;

            bool finished() override final;

            //! the impl checks sm_cpu_sync_level and calls host_wait_cv() if
            //! it equals zero
            void host_wait() override;

            double elapsed_time_until(Event &end_) override final;

            void device_wait_by(CompNode cn) override final;

            CompNode comp_node() const override final;
    };

} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

