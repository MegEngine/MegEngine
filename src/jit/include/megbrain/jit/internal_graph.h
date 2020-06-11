/**
 * \file src/jit/include/megbrain/jit/internal_graph.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/gopt/gtrans.h"
#include "megbrain/graph.h"
#include "megbrain/jit/fusion_pass.h"
#include "megbrain/jit/placeholder_opr.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/serialization/serializer.h"

#if MGB_JIT

namespace mgb {
namespace jit {

using JITFeatureBits = gopt::JITFeatureBits;

class InternalGraph;
using InternalGraphPtr = std::shared_ptr<InternalGraph>;

/*!
 * \brief internal graph in the JIT sub-graph
 *
 * This represents the computation of the JIT operator.
 */
class InternalGraph {
public:
    InternalGraph(VarNode* output, VarNode* shape_infer, VarNode* value_infer,
                  PlaceholderArray placeholders)
            : m_output{output},
              m_shape_infer{shape_infer},
              m_value_infer{value_infer},
              m_placeholders{std::move(placeholders)} {}

    struct PtrHash {
        size_t operator()(const InternalGraph* graph) const {
            return reinterpret_cast<size_t>(graph->m_output);
        }
    };

    struct PtrEqual {
        bool operator()(const InternalGraph* lhs,
                        const InternalGraph* rhs) const {
            return lhs->m_output == rhs->m_output;
        }
    };

    VarNode* output() const { return m_output; }
    VarNode* shape_infer() const { return m_shape_infer; }
    VarNode* value_infer() const { return m_value_infer; }

    const PlaceholderArray& placeholders() const { return m_placeholders; }

private:
    // For compilation cache, if the output_for_cache is same means the
    // expression tree is same.
    VarNode *m_output, *m_shape_infer, *m_value_infer;
    PlaceholderArray m_placeholders;
};

/*!
 * \brief helper object used in the fusion pass to generate InternalGraph
 *
 * This object stores intermediate state during visiting the computing graph in
 * JITFusionPass.
 *
 * The graph is iterated in reverse topological order. InternalGraphGenerator
 * starts with a single operator (i.e. the output node of the fused opr), and
 * new oprs are gradually added into it. Thus the process is expanding a tree
 * rooted at the output node.
 */
class InternalGraphGenerator {
    //! replace oprs in the graph of m_output and populate m_orig_inps,
    //! m_placeholders
    VarNode* replace_graph_by_placeholder();

    // TODO: relax constraints and change the algo
    //! find oprs which depend on Reduce
    void find_reduce_opr_deps(cg::OperatorNodeBase* opr);

    //! find oprs that depended by dimshuffle or JITExecutor(with Dimshuffle)
    //! in one branch
    void find_oprs_depended_by_dimshuffle(cg::OperatorNodeBase* opr);

public:
    explicit InternalGraphGenerator(cg::OperatorNodeBase* opr);

    //! generate the graph; this method can be called multiple times
    InternalGraphPtr generate();

    /*!
     * \brief needed input vars in the original (i.e. outer) graph
     *
     * This is accessible only after calling generate()
     */
    const VarNodeArray& orig_inps() const {
        mgb_assert(!m_orig_inps.empty());
        return m_orig_inps;
    }

    /*!
     * \brief JITPlaceholder vars in the internal graph, corresponding to
     *      orig_inps()
     *
     * This is accessible only after calling generate().
     */
    const VarNodeArray& placeholder_inps() const {
        mgb_assert(!m_placeholders.empty());
        return m_placeholders;
    }

    //! currently added operators
    const ThinHashSet<cg::OperatorNodeBase*>& opr_set() { return m_opr_set; }

    //! input vars (i.e. tree leaves) of currently added operators
    const ThinHashSet<VarNode*>& graph_input_set() { return m_graph_input_set; }

    const megdnn::TensorShape& before_reduce_shape() {
        return m_before_reduce_shape;
    }

    //! get number of inputs of this internal graph after adding a new operator
    size_t get_cnt_input_if_add(cg::OperatorNodeBase* opr) const;

    //! add an operator into this graph; its outputs must have been added
    void add_opr(cg::OperatorNodeBase* opr);

    //! output var in the outer graph (i.e. the root node)
    VarNode* output() const { return m_output; }

    //! attained features due to existing oprs
    JITFeatureBits feature_bits() const { return m_feature_bits; }

    //! shorthand for checking JITFeatureBits::REDUCE
    bool has_reduce() const {
        return static_cast<bool>(feature_bits() & JITFeatureBits::REDUCE);
    }

    //! shorthand for checking JITFeatureBits::DIMSHUFFLE
    bool has_dimshuffle() const {
        return static_cast<bool>(feature_bits() & JITFeatureBits::DIMSHUFFLE);
    }

    const ThinHashMap<VarNode*, ThinHashSet<cg::OperatorNodeBase*>>&
    reduce_out_var_deps() const {
        return m_reduce_out_var_deps;
    }

    const ThinHashMap<cg::OperatorNodeBase*, cg::OperatorNodeBase*>&
    oprs_depended_by_dimshuffe() const {
        return m_oprs_depended_by_dimshuffle;
    }

private:
    using DepType = cg::OperatorNodeBase::NodeProp::DepType;
    VarNode* const m_output;
    ThinHashSet<cg::OperatorNodeBase*> m_opr_set;
    ThinHashSet<VarNode*> m_graph_input_set;

    //! depedency type for readers of all vars
    ThinHashMap<VarNode*, DepType> m_var_dep_type;

    //! oprs that Reduce and JITExecutor(with Reduce) oprs depend on
    ThinHashMap<VarNode*, ThinHashSet<cg::OperatorNodeBase*>>
            m_reduce_out_var_deps;

    //! oprs that depended by Dimshuffle or JITExecutor(with Dimshuffle)
    //! kw: <opr, the latest dimshuffle or JITExecutors(with Dimshuffle)
    //! that depands on this opr>
    ThinHashMap<cg::OperatorNodeBase*, cg::OperatorNodeBase*>
            m_oprs_depended_by_dimshuffle;

    megdnn::TensorShape m_before_reduce_shape;

    VarNodeArray m_orig_inps, m_placeholders;
    size_t m_input_idx;
    JITFeatureBits m_feature_bits{JITFeatureBits::NONE};

    static PlaceholderArray to_placeholder_opr_arr(const VarNodeArray& vars);
};

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
