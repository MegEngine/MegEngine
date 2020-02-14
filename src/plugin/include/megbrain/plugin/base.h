/**
 * \file src/plugin/include/megbrain/plugin/base.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/utils/metahelper.h"
#include "megbrain/utils/event.h"
#include "megbrain/graph/cg.h"

namespace mgb {

    /*!
     * \brief base class for plugin
     *
     * A plugin is associated with a computing graph, and works by adding
     * handlers to event listeners
     */
    class PluginBase: public NonCopyableObj {
        std::vector<SyncEventConnecter::ReceiverHandler> m_event_handlers;

        protected:
            cg::ComputingGraph * const m_owner_graph;

            template<class Sub, class Event>
            void add_member_func_as_event_handler(
                    void (Sub::*hdl)(const Event&)) {
                static_assert(std::is_base_of<PluginBase, Sub>::value,
                        "not base class");
                using namespace std::placeholders;
                m_event_handlers.emplace_back(m_owner_graph->event(
                            ).register_receiver<Event>(
                                std::bind(hdl, static_cast<Sub*>(this), _1)));
            }

            void add_event_handler(
                    SyncEventConnecter::ReceiverHandler &&hdl) {
                m_event_handlers.emplace_back(std::move(hdl));
            }

            PluginBase(cg::ComputingGraph *owner_graph):
                m_owner_graph{owner_graph}
            {
            }

        public:
            virtual ~PluginBase() = default;

            auto owner_graph() {
                return m_owner_graph;
            }
    };

}  // mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

