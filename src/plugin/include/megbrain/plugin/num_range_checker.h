/**
 * \file src/plugin/include/megbrain/plugin/num_range_checker.h
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
#include "megbrain/plugin/base.h"
#include "megbrain/graph/event.h"
#include "megbrain/utils/thin/hash_table.h"

namespace mgb {

    /*!
     * \brief check that the absolute values of all numbers in a computing graph
     *      do not exceed some threshold
     */
    class NumRangeChecker final: public PluginBase {
        class Checker {
            std::shared_ptr<DeviceTensorND> m_inp;
            std::unique_ptr<HostTensorND> m_out;
            std::unique_ptr<cg::AsyncExecutable> m_func;

            public:
                void init(VarNode *var, float range);
                bool check(VarNode *var);
        };

        const float m_range;
        CompNode::UnorderedMap<ThinHashMap<megdnn::DTypeEnum, Checker>> \
                m_cn2dt2checker;
        std::vector<std::unique_ptr<NumRangeChecker>> m_sub_graph_checkers;

        void on_kern_end(const cg::event::OprExecKernelEnd &event);
        void on_subgraph_associated(const cg::event::SubgraphAssociated &event);

        void on_var_computed(VarNode *var);

        template<typename ctype>
        std::string format_msg(const HostTensorND &hv, float range);

        public:
            class Error final: public MegBrainError {
                public:
                    using MegBrainError::MegBrainError;
            };

            NumRangeChecker(cg::ComputingGraph *graph, float range);
    };
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
