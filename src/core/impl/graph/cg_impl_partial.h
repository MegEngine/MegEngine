/**
 * \file src/core/impl/graph/cg_impl_partial.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./cg_impl_seq.h"

#if MGB_ENABLE_PARTIAL_EXECUTION

namespace mgb {
namespace cg {

/*!
 * \brief graph compiler for partial execution
 *
 * Note that a MultiPartCompiler instance should only be used once
 */
class ComputingGraphImpl::MultiPartCompiler {
public:
    using OutputSpecArr = SmallVector<OutputSpec>;
    explicit MultiPartCompiler(ComputingGraphImpl* owner) : m_owner{owner} {}

    SmallVector<std::unique_ptr<AsyncExecutable>> compile(
            const OutputSpecArr& out_specs);

    //! used in testcase to get the types of internal operators
    static SmallVector<Typeinfo*> test_get_internal_opr_types();

private:
    using NodeProp = cg::OperatorNodeBase::NodeProp;
    using DepType = NodeProp::DepType;

    //! check if compiled funcs are called in given order and clean up on abort
    class ExecOrderChecker;

    //! opr that provides only shape
    class ShapeProvider;

    //! opr that provides device data from given device tensor
    class DeviceDataProvider;

    //! an opr that does nothing, to act as a source for DEV_COMP_ORDER
    class EmptyExecuteOpr;

    //! the opr to read vars needed by other parts
    class VarSinkOpr;

    //! extra info about a single opr
    struct OprTrait {
        //! final priority derived from step number in the sorted sequence; it
        //! would be 0 for oprs without DEV_VALUE receivers
        int priority = 0;
        //! the part in the OutputSpecArr that this opr belongs to
        int part = -1;
    };

    //! input/output info of a part
    struct PartIOInfo;

    ComputingGraphImpl* const m_owner;
    //! output vars and callbacks (modified by concat_and_prepare())
    OutputSpecArr m_out_specs;
    //! var => (recv part => merged receiver type in this part)
    ThinHashMap<VarNode*, ThinHashMap<size_t, DepType>> m_var_receiver_type;
    ThinHashMap<OperatorNodeBase*, OprTrait> m_opr_trait;

    //! refhold for the computing graphs of the parts
    SmallVector<std::shared_ptr<ComputingGraph>> m_sub_graphs;

    //! update m_out_specs so each part is a standalone graph
    void update_out_specs();

    //! concat oprs in m_out_specs and compile into a whole sequence such that
    //! oprs in the same part are consecutive
    const OprNodeArray* concat_and_prepare();

    void init_opr_trait_and_var_reader_type(const OprNodeArray& opr_seq);

    //! make part I/O info from m_var_receiver_type
    SmallVector<PartIOInfo> make_part_io_info() const;

    /*!
     * \brief whether an operator should be duplicated (rather than get
     *      forwarded by VarSinkOpr between parts)
     */
    static bool should_dup_between_part(VarNode* var);

    //! set graph options based on existing option in the owner graph
    static void assign_graph_opt(Options& dst, const Options& src);

    //! check if an opr has one output not on given comp node
    static bool has_different_comp_node(OperatorNodeBase* opr, CompNode cn);
};

}  // namespace cg
}  // namespace mgb

#endif  // MGB_ENABLE_PARTIAL_EXECUTION

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
