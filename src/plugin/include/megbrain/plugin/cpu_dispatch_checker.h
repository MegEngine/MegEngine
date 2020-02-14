/**
 * \file src/plugin/include/megbrain/plugin/cpu_dispatch_checker.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/plugin/base.h"

namespace mgb {

    /*!
     * \brief print warning if an operator does not call dispatch on cpu comp
     *      nodes
     *
     * This is intended to find potential bugs in megdnn.
     */
    class CPUDispatchChecker final: public PluginBase {
        std::mutex m_cn2nr_task_mtx,
            m_failed_oprs_mtx_storage,
            *m_failed_oprs_mtx = &m_failed_oprs_mtx_storage;
        CompNode::UnorderedMap<size_t> m_cn2nr_task;
        std::unordered_set<cg::OperatorNodeBase*>
            m_failed_oprs_storage, *m_failed_oprs = &m_failed_oprs_storage;
        std::vector<std::unique_ptr<CPUDispatchChecker>> m_sub_graph_checkers;

        void record(CompNode cn);
        void check(CompNode cn, cg::OperatorNodeBase *opr);

        public:
            CPUDispatchChecker(cg::ComputingGraph *graph);

            //! get oprs that did not call cpu dispatch
            auto&& failed_oprs() const {
                return *m_failed_oprs;
            }
    };
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

