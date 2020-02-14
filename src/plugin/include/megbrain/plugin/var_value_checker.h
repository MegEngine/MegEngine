/**
 * \file src/plugin/include/megbrain/plugin/var_value_checker.h
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
#include "megbrain/graph.h"
#include "megbrain/graph/event.h"
#include "megbrain/opr/utility.h"

namespace mgb {

    /*!
     * \brief check values of all vars in a graph
     *
     * This graph should be executed multiple times. On the first execution, all
     * var values would be saved as ground truth. Then each time the graph is
     * executed, the value of a single var would be checked. The var to be
     * checked would start at \p init_var_idx, and changed to next var in
     * topological order after every \p var_switch_interval graph executions.
     */
    class VarValueChecker final: public PluginBase {
        class Checker {
            std::shared_ptr<DeviceTensorND> m_inp;
            std::unique_ptr<cg::AsyncExecutable> m_func;

            void setup_inp(VarNode *var);

            public:
                bool valid() const {
                    return m_func.get();
                }

                void reset();
                void init(VarNode *var,
                        const std::shared_ptr<DeviceTensorND> &expected);
                void check(VarNode *var);
        };

        bool m_init_val_dumped;
        const size_t m_init_var_idx, m_var_switch_interval;
        size_t m_cur_var_idx, m_nr_exec;

        VarNodeArray m_vars;
        std::mutex m_var2val_mtx;
        ThinHashMap<VarNode*, std::shared_ptr<DeviceTensorND>> m_var2val;
        Checker m_checker;

        void on_comp_seq_order_determined(
                const cg::event::CompSeqOrderDetermined &event);
        void on_opr_kern_end(const cg::event::OprExecKernelEnd &event);
        void on_comp_seq_exec_finished(
                const cg::event::CompSeqExecFinished &event);

        void on_var_computed(VarNode *var);

        public:
            using Error = opr::AssertEqual::UnequalError;

            VarValueChecker(
                    ComputingGraph *graph,
                    size_t var_switch_interval = 1, size_t init_var_idx = 0);
    };
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

