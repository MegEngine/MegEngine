/**
 * \file src/gopt/include/megbrain/gopt/framework.h
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
#include "megbrain/gopt/gtrans.h"

namespace mgb {
namespace gopt {
    using cg::OperatorNodeBase;

    class OptState;

    //! forward declaration for structs in inference.h
    struct OptimizeForInferenceOptions;

    /*!
     * \brief represent a computing graph to be optimized by specifying its
     *      endpoints
     *
     * Typical usage is to call auto_replace_outputs() on function entrance and
     * call replace_var() if an output var needs to be replaced.
     */
    class SubGraph {
        friend class OptState;
        struct StableOprHash {
            size_t operator() (OperatorNodeBase *opr) const {
                return opr->id();
            }
        };

        OptState *m_owner_opt_state = nullptr;
        std::unordered_set<OperatorNodeBase*, StableOprHash> m_endpoint_oprs;

        SymbolVarArray m_endpoint_vars;
        ThinHashSet<VarNode*> m_endpoint_vars_set;

        ComputingGraph *m_comp_graph;

        public:
            using Callback = cg::DepOprIter::Callback;
            using ExtraDep = ThinHashMap<OperatorNodeBase*, VarNodeArray>;

            //! rewrite vars in a graph
            class Rewriter;

            SubGraph(const SymbolVarArray &endpoint_vars);

            //! get the associated ComputingGraph
            ComputingGraph* comp_graph() const {
                return m_comp_graph;
            }

            //! iterate in topology order
            void iter(const Callback& cb,
                      std::shared_ptr<ExtraDep> = nullptr) const;

            //! make a Rewriter bound to this graph
            inline Rewriter make_rewriter();

            //! endpoint var nodes
            const SymbolVarArray& endpoint_vars() const {
                return m_endpoint_vars;
            }

            //! whether endpoint contain a given var node
            bool endpoint_contain(VarNode *var) const {
                return m_endpoint_vars_set.count(var);
            }

            OptState* owner_opt_state() const {
                return m_owner_opt_state;
            }

            /*!
             * \brief get map from VarNode to number of oprs that need its value
             *      on device
             *
             * An opr is counted only once, even if a var occupies more than one
             * of its inputs.
             */
            ThinHashMap<VarNode*, size_t> get_var2nr_val_dep_oprs() const;
    };

    class SubGraph::Rewriter {
        SubGraph *m_owner_graph;
        //! var -> (is_auto_replace, new_var)
        ThinHashMap<VarNode*, std::pair<bool, VarNode*>> m_varmap;
        VarNodeArray m_opr_new_inp_cache;

        inline void on_var_replaced(
                VarNode* src, VarNode* dst, const char* msg);

        //! Returns (is_auto_replace, new_var) if var is replaced, otherwise
        //! (true, var).
        std::pair<bool, VarNode*> get_var_internal(VarNode* var);

        public:
            Rewriter(SubGraph *g):
                m_owner_graph{g}
            {}

            /*!
             * \brief must be called on each visited opr, to replace its
             *      output vars if input var has been replaced
             *
             * Note that the caller must take explicit care to process
             * replaced oprs (either by using the returned opr, or by
             * using get_var() appropriately)
             *
             * \return new operator that uses new inputs; it would be
             *      opr if no input is changed
             */
            OperatorNodeBase* auto_replace_outputs(
                    OperatorNodeBase *opr);

            //! get current var: if var has been replaced, return its
            //! new value; otherwise return var itself
            VarNode* get_var(VarNode *var) const {
                auto res = const_cast<Rewriter*>(this)->get_var_internal(var);
                return res.second;
            }

            //! whether a var is replaced by replace_var()
            bool has_manual_replace(VarNode *var) const {
                auto res = const_cast<Rewriter*>(this)->get_var_internal(var);
                return !res.first;
            }

            /*!
             * \brief replace var node *src* by *dst*
             *
             * \param msg see OptState::on_var_replaced
             */
            void replace_var(VarNode *src, VarNode *dst,
                    const char *msg);

            //! apply this rewriter to the owner graph and modify owner
            //! SubGraph inplace
            void apply_inplace() const;
    };
    SubGraph::Rewriter SubGraph::make_rewriter() {
        return {this};
    }

    /*!
     * \brief check whether a var has only one reader opr
     */
    class UniqReaderCheck {
        ThinHashMap<VarNode*, size_t> m_var2nr_val_dep;

        public:
            UniqReaderCheck(const SubGraph &graph);

            bool operator() (VarNode *var) const {
                auto iter = m_var2nr_val_dep.find(var);
                return iter == m_var2nr_val_dep.end() || iter->second <= 1;
            }

            //! update status after Rewriter::auto_replace_outputs
            void update_on_opr_auto_replace(
                    OperatorNodeBase *opr, OperatorNodeBase *repl_opr);
    };

    class GraphOptimizer;

    enum class VarReplaceCheckFlag : uint8_t {
        NOCHECK = 0,
        CHECK_INFER_TYPE = 1 << 0,
        CHECK_DTYPE = 1 << 1,
        CHECK_SHAPE = 1 << 2,
        CHECK_ALL = 255
    };
    MGB_DEF_ENUM_CLASS_BIT_OPR(VarReplaceCheckFlag)

    enum class OprPropertyFlag : uint8_t {
        NONE = 0,
        SOURCE_OPR = 1 << 0,
        PRIORITY = 1 << 1,
        ALL = 255
    };
    MGB_DEF_ENUM_CLASS_BIT_OPR(OprPropertyFlag)

    /*!
     * \brief current optimization state
     *
     * Note that this class listens to opr-inserted event, which must occur in
     * the context in SubGraph::iter(), and properties from currently operator
     * being iterated would be copied to new operator.
     */
    class OptState final: public NonCopyableObj {
        friend class SubGraph;

        VarReplaceCheckFlag m_var_replace_check_flag =
                VarReplaceCheckFlag::CHECK_ALL;

        const GraphOptimizer * const m_owner_optimizer;
        //! map from src to dst var for all current replaces
        ThinHashMap<VarNode*, VarNode*> * const m_var_replace_map;

        SyncEventConnecter::ReceiverHandler m_on_opr_insert_handler;

        OperatorNodeBase *m_cur_iter_src_opr = nullptr;
        int m_cur_iter_opr_priority;
        cg::SeqCompNodeOptimizer::StreamPropType
                m_cur_iter_opr_stream_prop_type;
        OprPropertyFlag m_opr_property_flag;
        cg::SeqCompNodeOptimizer &m_comp_node_opt;

        SubGraph m_graph;
        std::string m_log_msg;
        size_t m_log_nr_item = 0;

        // record oprs inserted into comp_graph, would be reset on on_var_replaced
        ThinHashMap<OperatorNodeBase*, OprPropertyFlag> m_oprs_inserted;

        public:
            OptState(const GraphOptimizer *owner_optimizer,
                    const SubGraph& graph);

            //! graph to be optimized; can be modified by Pass
            SubGraph& graph() {
                return m_graph;
            }

            /*!
             * \brief set whether to check for dtype and shape match when
             *      replacing a var
             *
             * This is set true before applying an optimizer pass, and a pass
             * can set it to false.
             *
             * Currently only used by ExpandVirtualGradPass, because VirtualGrad
             * has shape infer but shape of actual grad may be uninferable.
             */
            OptState& set_var_replace_check_flag(VarReplaceCheckFlag flag) {
                m_var_replace_check_flag = flag;
                return *this;
            }

            /*!
             * \brief called when a var is replaced
             *
             * This method propagates some var properties and records the
             * replace in m_var_replace_map, which can be retrieved via
             * var_replace_map()
             *
             * \param msg_log human-readable diagnostic message to be appended
             *      to replace log; pass nullptr if there is none; Note that log
             *      would be flushed after each pass, so msg needs not contain
             *      the pass name.
             */
            void on_var_replaced(VarNode *src, VarNode *dst,
                    const char *msg_log);

            /*!
             * \brief write current operator replace log by calling mgb_log
             * \param prefix message to be prepended before log entries
             * \return number of log items
             */
            size_t flush_log(const char *prefix);

            /*!
             * \brief call function with a temporary context from given operator's
             * properties.
             * \param opr_property_flag which property should copy to new opr
             */
            void call_with_opr(OperatorNodeBase *opr, thin_function<void(void)> func,
                               OprPropertyFlag opr_property_flag=OprPropertyFlag::ALL);
    };

    class GraphOptimizer;

    //! a single optimization pass to transform a graph
    class Pass {
        GraphOptimizer *m_owner_optimizer = nullptr;

        friend class GraphOptimizer;

        public:

            //! the optimizer that contains this Pass
            GraphOptimizer *owner_optimizer() const {
                return m_owner_optimizer;
            }

            virtual ~Pass() = default;

            //! name of this optimization pass
            virtual const char* name() const = 0;

            /*!
             * \brief apply this pass on a GraphOptimizer
             *
             * Note: \p opt would be prorperly initialized and it can be
             * modified. The subclasses do not need to restore any modified
             * state of \p opt.
             */
            virtual void apply(OptState &opt) const = 0;
    };

    /*!
     * \brief helper used as a base class to implement a Pass that replaces oprs
     *      in a subgraph
     *
     * Only the oprs with a unique output var would be processed, and newly
     * created oprs would be further processed until no opr is replaced (hence
     * the name *recursive*).
     *
     * This is useful in cases like
     * `conv(kx+b,w) => conv(kx,w)+b1 => conv(x, w1)+b1`.
     */
    class RecursiveSubGraphRewriteHelper {
        struct StackFrame {
            VarNode *orig_var;
            OperatorNodeBase *opr;
        };
        OptState &m_opt_state;
        SubGraph::Rewriter m_rewriter;
        std::vector<StackFrame> m_opr_stack;
        std::string m_log_msg;

        void on_opr(OperatorNodeBase *opr);

        protected:
            ~RecursiveSubGraphRewriteHelper() noexcept;

            SubGraph::Rewriter& rewriter() {
                return m_rewriter;
            }

            /*!
             * \brief process the owner opr of given var
             *
             * Please note:
             * 1. It is guaranteed that *out_var* is the only output var of its
             *    owner opr.
             * 2. The implementation should not call rewriter().replace_var()
             *
             * \return transformed var with additional info
             */
            virtual GTransResult process_opr(VarNode *out_var) = 0;

            /*!
             * \brief callback on visiting a new opr
             *
             * This callback is called once for each operator (either in
             * original graph or newly created) in topological order, and it
             * should return whether this opr should be passed to process_opr().
             * An opr would be actually processed only if this callback return
             * true and it has a unique output var.
             *
             * \param opr opr visited
             * \param repl_opr opr returned by Rewriter::auto_replace_outputs
             */
            virtual bool on_new_opr_check_should_process(
                    OperatorNodeBase *opr, OperatorNodeBase *repl_opr) = 0;

            /*!
             * \brief called after replace orig_var with new_var
             */
            virtual void after_replace_var(VarNode *orig_var, VarNode *new_var) = 0;

            //! apply rewrite on current graph
            void apply();

        public:
            RecursiveSubGraphRewriteHelper(OptState &state);
    };

    /*!
     * \brief manage passes and their applying on graphs
     *
     * It is guaranteed that the original graph would not be changed; modified
     * nodes would be copied.
     *
     * When a pass is called, graph_opt_level would be set to 1 with
     * original value recorded, and the pass can modify it without having to
     * restore.
     */
    class GraphOptimizer {
        bool m_enable_check_result = false;
        int m_verbosity = 1;
        std::vector<std::unique_ptr<Pass>> m_passes;

        class VarReplaceMapStorage;

        public:
            ~GraphOptimizer() noexcept;

            //! add an optimization pass
            GraphOptimizer& add_pass(std::unique_ptr<Pass> pass);

            //! add a pass with given type
            template<class Pass, typename ...Params>
            GraphOptimizer& add_pass(Params&& ...params) {
                return add_pass(std::make_unique<Pass>(
                            std::forward<Params>(params)...));
            }

            //! whether to check result by comparing optimized and
            //! non-optimized vars
            GraphOptimizer& enable_check_result(bool flag) {
                m_enable_check_result = flag;
                return *this;
            }

            //! get current verbosity setting
            int verbosity() const {
                return m_verbosity;
            }

            /*!
             * \brief set verbosity
             *
             * 0: no log
             * 1: only summary
             * 2: replacing details
             */
            GraphOptimizer& verbosity(int level) {
                m_verbosity = level;
                return *this;
            }

            /*!
             * \brief add predefined set of passes for given usage
             * \return *this
             */
            GraphOptimizer& add_preset_passes(
                    bool after_grad = false,
                    const OptimizeForInferenceOptions* inference_opt = nullptr,
                    const ComputingGraph::Options* comp_graph_opt = nullptr);

            //! transform given graph into a new optimized graph
            SubGraph apply(const SubGraph &graph) const;

            /*!
             * \brief optimize graph defined by given endpoints and modify them
             *      inplace
             * \return *this
             */
            const GraphOptimizer& apply_inplace(VarNodeArray &vars) const;

            /*!
             * \brief get var replace map associated with a computing graph
             *
             * The map maps from a var to its replaced var during optimization.
             * Note that the map would be cleared when GraphOptimizer is applied
             * on the graph.
             */
            static const ThinHashMap<VarNode*, VarNode*>&
                var_replace_map(ComputingGraph &graph);

            /*!
             * \brief get the final replaced var in
             *      var_replace_map(var->owner_graph()) corresponding to var
             */
            static VarNode* var_replace_lookup(VarNode *var);
    };

    /*!
     * \brief propagate the const property
     *
     * Const property is defined for oprs that only depends on
     * SharedDeviceTensor or ImmutableTensor.
     *
     * Usually you would want to use ConstVarPropogate, and this base class
     * exists to avoid virtual dtor while allowing polymorphism.
     */
    class ConstVarPropogateBase {
        protected:
            ~ConstVarPropogateBase() = default;

            //! memory usage of a var
            static size_t var_mem_size(VarNode *var) {
                return var->dtype().size(var->shape().total_nr_elems());
            }

            //! called after a const but non-source opr is visited
            virtual void on_midconst_opr(
                    OperatorNodeBase *opr, size_t max_src_size) {
                MGB_MARK_USED_VAR(opr);
                MGB_MARK_USED_VAR(max_src_size);
            }

        public:
            explicit ConstVarPropogateBase(ConstVarType const_var_type):
                m_const_var_type{const_var_type}
            {
            }

            //! note that both attrs would be false if opr is impure or it is
            //! not allowed to be replaced
            struct AddOprResult {
                //! whether at least one input is const
                bool has_const_inp;

                //! whether at least one input is const and not source opr
                bool has_midconst_inp;

                //! whether all inputs are const
                bool all_const_inp;
            };

            AddOprResult add_opr(OperatorNodeBase *opr);

            bool is_const(OperatorNodeBase *opr) const {
                return m_oprinfo.at(opr).is_const;
            }
            bool is_const(VarNode *var) const {
                return is_const(var->owner_opr());
            }

            //! whether a var is produced by non-source const opr
            bool is_midconst(OperatorNodeBase *opr) const {
                return is_const(opr) && !is_const_var(m_const_var_type, opr);
            }

            bool is_midconst(VarNode *var) const {
                return is_midconst(var->owner_opr());
            }

        private:
            struct OprInfo {
                bool processed = false, is_const = false;

                //! map max size (bytes) of source oprs that this opr depends on
                size_t max_size = 0;
                AddOprResult result;
            };
            ThinHashMap<OperatorNodeBase*, OprInfo> m_oprinfo;
            ConstVarType m_const_var_type;

    };

    class ConstVarPropogate final: public ConstVarPropogateBase {
        public:
            using ConstVarPropogateBase::ConstVarPropogateBase;
    };

} // namespace gopt
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
