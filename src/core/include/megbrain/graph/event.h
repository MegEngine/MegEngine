/**
 * \file src/core/include/megbrain/graph/event.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph/operator_node.h"

namespace mgb {
namespace cg {

class AsyncExecutable;

namespace event {

/*!
 * \brief signaled when an operator is inserted
 */
struct OprInserted {
    //! true if this operator has been inserted before
    bool is_dedup;

    //! newly inserted operator
    OperatorNodeBase* opr;

    //! associated exception if insertion fails; nullptr if no error
    MegBrainError* exc;

    MGB_TYPEINFO_OBJ_DECL;
};

/*!
 * \brief signaled immediately after starting executing an operator, before
 *      waiting on other computing nodes
 */
struct OprExecStart {
    OperatorNodeBase* opr;
    GraphExecutable::ExecEnv* env;

    MGB_TYPEINFO_OBJ_DECL;
};

/*!
 * \brief signaled after all waiting commands on other computing nodea for an
 *      operator are dispatched
 */
struct AfterWait {
    CompNode comp_node;
    OperatorNodeBase* opr;

    MGB_TYPEINFO_OBJ_DECL;
};

/*!
 * \brief signaled after all waiting commands are issued, just before calling
 *      do_execute
 */
struct OprExecKernelStart {
    OperatorNodeBase* opr;
    GraphExecutable::ExecEnv* env;

    MGB_TYPEINFO_OBJ_DECL;
};

/*!
 * \brief signaled just after do_execute, before marking output vars ready
 */
struct OprExecKernelEnd {
    OperatorNodeBase* opr;
    GraphExecutable::ExecEnv* env;

    MGB_TYPEINFO_OBJ_DECL;
};

/*!
 * \brief signaled after execution of an operator is finished
 */
struct OprExecFinished {
    OperatorNodeBase* opr;
    GraphExecutable::ExecEnv* env;

    MGB_TYPEINFO_OBJ_DECL;
};

/*!
 * \brief before a kernel or a groups of kernels on the same CompNode executed,
 *  signaled by do_execute implementations, on the same thread of kernel
 *  dispatcher
 */
struct BeforeKernel {
    OperatorNodeBase* opr;
    CompNode comp_node;

    MGB_TYPEINFO_OBJ_DECL;
};

/*!
 * \brief after a kernel or a groups of kernels on the same CompNode executed,
 *  signaled by do_execute implementations, on the same thread of kernel
 *  dispatcher
 */
struct AfterKernel {
    OperatorNodeBase* opr;
    CompNode comp_node;

    MGB_TYPEINFO_OBJ_DECL;
};

/*!
 * \brief after static memory allocation strategy is determined on a computing
 *  node; the subscribers can set *need_realloc to indicate that static memory
 *  allocator should be re-run
 *
 * This event would be issued for static memory allocation on each comp node,
 * and after static memory alloc finished, it would be issued with need_realloc
 * == nullptr, comp_node being invalid and alloc_size == 0 to indicate
 * allocation has finished.
 */
struct StaticMemAlloc {
    bool* need_realloc;
    CompNode comp_node;
    size_t alloc_size;

    MGB_TYPEINFO_OBJ_DECL;
};

/*!
 * \brief signaled after the order of oprs and comp nodes in a computing
 * sequence is determined
 */
struct CompSeqOrderDetermined {
    ComputingGraph* graph;
    AsyncExecutable* exec;

    MGB_TYPEINFO_OBJ_DECL;
};

/*!
 * \brief signaled before executing a computing sequence
 *
 * Note: this event may not match CompSeqExecFinished in this case: when fake
 * exec is enabled, CompSeqExecBeforeStart is signaled but CompSeqExecFinished
 * would not be signaled.
 */
struct CompSeqExecBeforeStart {
    ComputingGraph* graph;
    AsyncExecutable* exec;

    //! callbacks to be invoked after the kernels have been dispatched
    CleanupCallback* after_kern_dispatch;

    //! computing nodes used by this sequence
    const CompNode::UnorderedSet* used_comp_node;

    //! sequence version (the version is determined by graph event listener
    //! configuration)
    size_t seq_version;

    MGB_TYPEINFO_OBJ_DECL;
};

/*!
 * \brief signaled when execution of a computing sequence is totally finished
 *      (i.e. being waited on host)
 *
 * Note: CompSeqExecBeforeStart and CompSeqExecFinished are not necessarily
 * matched. If .wait() is not called after an exec, then CompSeqExecBeforeStart
 * for the next .execute() would be signaled before CompSeqExecFinished for this
 * execution.
 *
 * This event would not be signaled if there is an error (see CompSeqExecError).
 */
struct CompSeqExecFinished {
    /*!
     * whether wait is issued explicitly by user (true), or due to consecutive
     * graph exec causing waiting for previous opr
     */
    bool explicit_user_wait;
    /*!
     * Whether device exec has actually finished; being false means that only
     * operators have been issued to exec queue, and this can be false only
     * when the graph is a subgraph executed for multiple times (see
     * ComputingGraphImpl::ComputingSequence::execute() implementation for
     * details).
     *
     * When device_actually_finished is false, explicit_user_wait must also be
     * false.
     */
    bool device_actually_finished;
    ComputingGraph* graph;
    AsyncExecutable* exec;

    MGB_TYPEINFO_OBJ_DECL;
};

/*!
 * \brief signaled when execution of a computing sequence is aborted due to
 *      error
 *
 * has_uncaught_exception() would be true when this event is signaled, so be
 * careful in the handlers to not introduce any new exceptions.
 */
struct CompSeqExecError {
    ComputingGraph* grah;
    AsyncExecutable* exec;

    MGB_TYPEINFO_OBJ_DECL;
};

/*!
 * \brief invoked when a graph is registered as subgraph of another
 */
struct SubgraphAssociated {
    ComputingGraph* par_graph;
    ComputingGraph* sub_graph;

    MGB_TYPEINFO_OBJ_DECL;
};

}  // namespace event
}  // namespace cg
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
