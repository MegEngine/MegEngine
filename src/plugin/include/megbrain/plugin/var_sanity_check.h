/**
 * \file src/plugin/include/megbrain/plugin/var_sanity_check.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/exception.h"
#include "megbrain/graph.h"
#include "megbrain/plugin/base.h"
#include "megdnn/oprs.h"

#include <atomic>
#include <cstdint>
#include <thread>

namespace mgb {

/*!
 * \brief check that content of a variable does not change between when it
 *      is produced and when it is used
 */
class VarSanityCheck final : public PluginBase {
    using ChecksumResult = megdnn::opr_result::Checksum;

    class DebugLog {
        VarSanityCheck* m_checker;
        bool m_enable = false;
        int m_var_id = -1;
        VarNode* m_var = nullptr;
        size_t m_readcnt_init = 0;
        //! number of unfinished readers for the var being traced
        std::atomic_size_t m_readcnt{0};

    public:
        DebugLog(VarSanityCheck* checker);

        void add_producer(VarNode* var);
        void add_receiver(VarNode* var);
        void on_var_produced(VarSanityCheck* checker, VarNode* var,
                             ChecksumResult checksum);
        void on_var_received(VarNode* var);
    };

    struct WorkspaceCache {
        //! var comp node to workspace
        CompNode::UnorderedMap<DeviceTensorStorage> storage;
    };

    DebugLog m_debug_log{this};

    //! map from caller thread to workspace map
    ThinHashMap<std::thread::id, WorkspaceCache> m_workspace;
    std::mutex m_workspace_mtx;

    ThinHashMap<VarNode*, ChecksumResult> m_var2chksum;
    /*! the ids of varnodes that have been modified by recv_opr
     * (eg AddUpate) with flag
     * cg::OperatorNodeBase::NodeProp::Flag:: FORCE_UPDATE_INPUT_VAR.
     */
    ThinHashSet<VarNode*> m_modified_vars;
    std::mutex m_id2chksum_mtx;

    typedef void (VarSanityCheck::*input_checker_fn)(cg::OperatorNodeBase*,
                                                     VarNode*);

    void on_var_produced(VarNode* var);
    void on_var_received(cg::OperatorNodeBase* recv_opr, VarNode* var);
    //! check after opr exec that input is not modified
    void check_input_unmodified(cg::OperatorNodeBase* recv_opr, VarNode* var);
    void check_single_input(bool add_debug_log, cg::OperatorNodeBase* recv_opr,
                            VarNode* var);
    void setup_input_checker(bool add_debug_log, cg::OperatorNodeBase* opr,
                             cg::GraphExecutable::ExecEnv& env,
                             input_checker_fn checker);

    static std::string str(const ChecksumResult& result);

    ChecksumResult calc_checksum(VarNode* var);

public:
    VarSanityCheck(cg::ComputingGraph* graph);

    class Error : public MegBrainError {
    public:
        using MegBrainError::MegBrainError;
    };

    /*!
     * \brief perform basic sanity check after opr exec
     *
     * This checks var ptr and empty shape.
     * It should be called for non-statically allocated vars.
     */
    static void check_var_after_exec(
            VarNode* var, const ComputingGraph::VarReceiverInfo& recv);
};
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
