/**
 * \file src/plugin/include/megbrain/plugin/infkern_finder.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph.h"
#include "megbrain/graph/event.h"
#include "megbrain/plugin/base.h"

#include <atomic>
#include <unordered_map>

namespace mgb {

    /*!
     * \brief Find which operator in a computing sequence is currently running
     *      and support dump input values.
     */
    class InfkernFinder final: public PluginBase {
        struct GlobalState;
        class OprState;

        std::atomic_flag m_cg_start_log_printed = ATOMIC_FLAG_INIT;
        std::vector<OprState> m_opr_seq;
        std::unordered_map<cg::OperatorNodeBase*, OprState*> m_opr2state;
        cg::AsyncExecutable *m_current_comp_seq = nullptr;
        size_t m_prev_succ_comp_seq_run_id = 0;

        const std::unique_ptr<GlobalState> m_global_state_storage;
        GlobalState *m_global_state;

        std::vector<std::unique_ptr<InfkernFinder>> m_sub_graph_finders;

        void init();

        void on_comp_seq_determined(
                const cg::event::CompSeqOrderDetermined &ev);
        void on_comp_seq_finished(const cg::event::CompSeqExecFinished &ev);
        void on_opr_start(const cg::event::OprExecStart &ev);
        void on_waiting_finished(const cg::event::AfterWait &ev);
        void on_opr_kern_finish(const cg::event::OprExecKernelEnd &ev);
        void on_opr_finish(const cg::event::OprExecFinished &ev);
        void on_subgraph_associated(const cg::event::SubgraphAssociated &ev);

        cg::OperatorNodeBase* write_to_file_opronly(FILE *fout);

        public:
            //! copy of var values for helping opr debug
            struct InputValueRecord {
                size_t run_id;
                HostTensorND val;

                using FullRecord = std::vector<std::pair<
                    VarNode*, InputValueRecord>>;
            };

            InfkernFinder(cg::ComputingGraph *graph, bool record_input_value);
            ~InfkernFinder() noexcept;

            //! this constructor should not be called by user
            InfkernFinder(cg::ComputingGraph *graph, GlobalState *global_state);

            /*!
             * \brief write execution status to file
             * \return the first operator whose output is not finished; or
             *      nullptr if all finished
             */
            cg::OperatorNodeBase* write_to_file(const char *fpath);

            /*!
             * \brief get previous input values for dumped operators
             */
            InputValueRecord::FullRecord get_input_values(size_t opr_id);
    };

}  // mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
