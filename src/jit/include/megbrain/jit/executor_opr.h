/**
 * \file src/jit/include/megbrain/jit/executor_opr.h
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
#include "megbrain/jit/internal_graph.h"
#include "megbrain/opr/internal/identical_fwd.h"

#if MGB_JIT

namespace mgb {
namespace jit {

class Executable;
class Compiler;

/*!
 * \brief JITExecutor opr
 *
 * This operator represents a subgraph to be computed by JIT-compiled kernel.
 *
 * Each pair of (internal graph, inputs) would correspond to a JITExecutor opr.
 * JITExecutor generates runtime Args for this specific inputs, and calls
 * methods in Compiler to get the Executable object for actual computing.
 */
MGB_DEFINE_OPR_CLASS(JITExecutor, cg::SingleCNOperatorNodeBase,
        opr::mixin::FwdIn2OutWritableHelper) // {
    using ModeTrait = megdnn::Elemwise::ModeTrait;

    InternalGraphPtr m_internal_graph;
    using DimshuffleParam = std::pair<std::vector<int>, uint32_t>;

public:
    using Mode = opr::Elemwise::Mode;

    void scn_do_execute() override;

    void init_output_static_infer_desc() override;

    JITExecutor(const InternalGraphPtr& internal_graph,
                const VarNodeArray& inputs, const OperatorNodeConfig& config);

    static SymbolVar make(const InternalGraphPtr& internal_graph,
                          const VarNodeArray& inputs,
                          const OperatorNodeConfig& config = {});

    struct LoadDumpImpl;

    void add_input_layout_constraint() override;

    void init_output_mem_plan(bool dynamic) override;

    void mem_plan_fwd_in2out_writable() override;

    const InternalGraph& internal_graph() const { return *m_internal_graph; }

    const InternalGraphPtr internal_graph_ptr() const {
        return m_internal_graph;
    }

    auto&& input_broadcastable() const { return m_input_broadcastable; }

    //! runtime args for the executable (i.e. the actual value of
    //! inputs/outputs)
    struct Args {
        struct Data {
            VarNode* from;
            //! for HOST_VALUE_FOR_SHAPE input, this would contain only shape;
            //! dtype would be invalid and stride would be zero. If
            //! \p need_input_collapse is set, this layout would be collapsed.
            TensorLayout layout;
            int idx;  //!< index in the input array; -1 for output
        };

        bool need_update = true;
        std::vector<Data> inputs, outputs;

        size_t hash;

        //! version from a global counter for fast equality test;
        //! Args objects with identical version are always equal
        mutable uint64_t version;

        JITExecutor* const owner = nullptr;

        explicit Args(JITExecutor* owner_) : owner{owner_} {}

        Args() = default;

        bool operator==(const Args& rhs) const;

        struct HashEq {
            static size_t hash(const Args& x) { return x.hash; }
            static bool eq(const Args& x, const Args& y) { return x == y; }
        };
    };

    const Args& args() const;

    //! get the underlying executable; only used for test purpose
    Executable* executable() const { return m_executable; }

    bool has_reduce() const {
        return static_cast<bool>(m_feature_bits & JITFeatureBits::REDUCE);
    }

    bool has_dimshuffle() const {
        return static_cast<bool>(m_feature_bits & JITFeatureBits::DIMSHUFFLE);
    }

    const ThinHashMap<jit::JITPlaceholder*, DimshuffleParam>&
    dimshuffle_params() const {
        return m_jitph2dimshuffle;
    }

    //! get broadcasted shape of inputs
    megdnn::TensorShape broadcasted_input_shape() const;

    //! the Compiler associated with this JIT subgraph
    Compiler* compiler() const { return m_compiler; }

private:
    Args m_args{this};
    JITFeatureBits m_feature_bits{JITFeatureBits::NONE};
    Compiler* const m_compiler = nullptr;
    Executable* m_executable = nullptr;
    std::vector<bool> m_input_broadcastable;
    // JITPlaceHolder -> pair of (dimshuffle pattern, ndim)
    // do DFS on internal graph only once in prepare_dimshuffle(), so we can
    // easily get the dimshuffle param which should be applied on given
    // JITPlaceholder
    ThinHashMap<jit::JITPlaceholder*, DimshuffleParam> m_jitph2dimshuffle;
    void update_args();
    void do_dimshuffle();
    void prepare_dimshuffle();

    NodeProp* do_make_node_prop() const override;
};

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
