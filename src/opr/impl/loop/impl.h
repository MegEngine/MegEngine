/**
 * \file src/opr/impl/loop/impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/opr/loop.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/internal/mixin_base.h"
#include "megbrain/graph/grad_impl.h"
#include "./output_recorder.h"

#include "megdnn/oprs.h"

#include <list>

namespace mgb {
namespace opr {
namespace intl {

/*!
 * \brief an entry for specifying how to record an output var
 */
class LoopImpl::OutputRecordSpecItem final: public Hashable {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

    static Desc::OutputRecorderBase * const m_dummy_recorder;
    bool m_enabled = true;
    //! var node in subgraph and owner graph
    VarNode *m_var_sub, *m_var_owner = nullptr;
    std::unique_ptr<Desc::OutputRecorderBase> m_recorder;

    bool is_same_st(const Hashable &rhs) const override {
        auto &&robj = static_cast<const OutputRecordSpecItem&>(rhs);
        return m_var_sub == robj.m_var_sub &&
            m_recorder->is_same(*robj.m_recorder);
    }

    public:
        //! any data associated with thie output recorder spec to be used by
        //! its callers (currently only one bit is needed)
        mutable bool user_data = false;

        OutputRecordSpecItem(SymbolVar sub,
                std::unique_ptr<Desc::OutputRecorderBase> recorder):
            m_var_sub(sub.node()), m_recorder(std::move(recorder))
        {}

        size_t hash() const override {
            return hash_pair_combine(
                    std::hash<void*>{}(m_var_sub), m_recorder->hash());
        }

        //! get the output recorder if this output var is enabled, or a dummmy
        //! recorder otherwise
        Desc::OutputRecorderBase* recorder() const {
            return m_enabled ? m_recorder.get() : m_dummy_recorder;
        }

        //! get output mode for the original output recorder
        Desc::OutputMode output_mode() const {
            return m_recorder->output_mode();
        }

        void bind(VarNode *var_owner) {
            mgb_assert(!m_var_owner && var_owner);
            m_var_owner = var_owner;
            m_recorder->bind_var(m_var_sub, var_owner);
        }

        VarNode* var_sub() const {
            return m_var_sub;
        }

        VarNode* var_owner() const {
            return m_var_owner;
        }

        bool enabled() const {
            return m_enabled;
        }

        OutputRecordSpecItem& enable(bool flag) {
            m_enabled = flag;
            return *this;
        }

        //! modify var sub; must be called before bind()
        void var_sub(VarNode *var) {
            mgb_assert(!m_var_owner, "bind() must not be called");
            m_var_sub = var;
        }

}; // OutputRecordSpecItem

/*!
 * \brief copy input from original graph into subgraph
 */
MGB_DEFINE_OPR_CLASS(LoopImpl::InputMaker, cg::SingleCNOperatorNodeBase) // {
    public:
        struct Param {
            bool disable_value_infer;
            bool has_assign;
        };

        InputMaker(DescImplBase *desc, VarNode *orig_var, const Param &param);

        static SymbolVar make(
                DescImplBase *desc, SymbolVar orig_var, const Param &param);

        //! set assignor var
        void set_assignor(VarNode *var) {
            mgb_assert(m_param.has_assign && var && !m_assignor_committed);
            m_assignor_var = var;
        }

        //! setup assignor updator for current assignor; assignor can not be
        //! further changed
        void commit_assignor();

        VarNode* assignor() const {
            mgb_assert(m_assignor_var, "assignment value not set for "
                    "%s (orig: %s)",
                    cname(),
                    cg::dump_var_info({m_orig_var}).c_str());
            return m_assignor_var;
        }

        VarNode *orig_var() const {
            return m_orig_var;
        }

        const Param& param() const {
            return m_param;
        }

        //! clear device memory and reset state
        void on_exec_end() {
            m_first_exec = true;
            m_assignor_value = {};
        }

    private:
        const Param m_param;

        bool m_first_exec = true;
        bool m_assignor_committed = false;

        VarNode *m_orig_var;
        DescImplBase *m_desc;

        VarNode* m_assignor_var = nullptr;
        DeviceTensorND m_assignor_value;

        NodeProp* do_make_node_prop() const override;

        void init_output_comp_node() override {
            comp_node(m_orig_var->comp_node());
        }

        void init_output_static_infer_desc() override;

        void init_output_mem_plan(bool dynamic) override;

        void scn_do_execute() override;

}; // InputMaker

/*!
 * \brief iterate over dep oprs in subgraph of loop
 *
 * It differs from DepOprIter by handling assignor vars of InputMaker
 */
class LoopImpl::SubgraphDepIter: public NonCopyableObj {
    size_t m_input_makers_sorted_size = 0;
    VarNodeArray m_unresolved_assignors;
    std::vector<InputMaker*> m_input_makers;
    cg::OprNodeArray m_oprs;
    cg::DepOprIter m_dep_iter;

    void sort_input_makers();
    void dep_iter_cb(cg::OperatorNodeBase *opr);

    public:

        SubgraphDepIter();
        ~SubgraphDepIter() noexcept;

        //! add a dest var
        void add(VarNode *dest);

        /*!
         * \brief all needed input makers in ascending ID order
         *
         * Note: stable order is important, since loop opr may be copied and
         * copied grad opr relies on input order to determine output order
         */
        auto&& input_makers() {
            if (m_input_makers_sorted_size != m_input_makers.size()) {
                sort_input_makers();
            }
            return m_input_makers;
        }

        //! all oprs, in topological order
        auto&& oprs() const {
            return m_oprs;
        }
}; // SubgraphDepIter

/*!
 * \brief base class for implementing loop desc
 */
class LoopImpl::DescImplBase: public LoopImpl::Desc {
    public:
        // use list to avoid reference being invalidated
        using OutputRecordSpec = std::list<OutputRecordSpecItem>;

        class CounterProvider;

        //! manager for loop condition
        class LoopCondManager final: NonCopyableObj {
            SymbolVar m_var;

            class GetCondOpr;
            GetCondOpr *m_get_cond_opr = nullptr;

            public:
                //! get loop cond var
                SymbolVar var() const {
                    return m_var;
                }

                LoopCondManager& setup(SymbolVar var) {
                    m_var = var;
                    return *this;
                }

                ComputingGraph::OutputSpec::value_type subgraph_outspec_item();

                //! query whether loop should continue
                bool should_loop();
        };

        DescImplBase();

        /* ========= overwrite parent method ========= */

        SymbolVar get_counter_var() override {
            mgb_throw_if(!m_counter_var.node(), GraphError,
                    "could only get counter var "
                    "when there is at least one input");
            return m_counter_var;
        }

        Desc& set_loop_condition(SymbolVar cond) override {
            mgb_throw_if(!check_in_sub_graph(cond),
                    GraphError, "loop condition must be in the sub graph");
            m_loop_cond_manager.setup(cond);
            return *this;
        }

        /* ========= other methods for loop impl ========= */

        //! called in LoopImpl::LoopImpl()
        void set_loop_opr(LoopImpl *opr) {
            mgb_assert(!m_owner_loop_opr);
            m_owner_loop_opr = opr;
        }

        //! graph in which this loop is constructed
        ComputingGraph* owner_graph() const {
            return m_owner_graph;
        }

        //! the graph that corresponds to loop body, managed by this loop opr
        ComputingGraph* sub_graph() const {
            return m_sub_graph.get();
        }

        std::unique_ptr<cg::AsyncExecutable> compile();

        auto&& output_record_spec() const {
            return m_output_record_spec;
        }

        auto&& output_record_spec_no_dedup() const {
            return m_output_record_spec_no_dedup;
        }

        auto&& loop_cond_manager() {
            return m_loop_cond_manager;
        }

        /*!
         * \brief input vars used in current compiled func
         */
        const std::vector<InputMaker*>& cur_func_input() const {
            return m_cur_func_input.val();
        }

        /*!
         * \brief all input vars needed for producing output vars given by
         *      do_add_output()
         *
         * The value is initialized at the first call
         */
        virtual const std::vector<InputMaker*>& all_inputs() = 0;

        cg::static_infer::SubgraphStaticInferHelper&
                sub_graph_static_infer_helper() {
            return *m_sub_graph_static_infer_helper;
        }

        //! reset counter provider to the value before loop starts
        virtual void reset_counter_provider();

        //! update counter provider to next loop value
        virtual void update_counter_provider();

        CounterProvider* counter_provider() const {
            return m_counter_provider;
        }

        //! construct an InputMaker and record it
        SymbolVar do_add_input(SymbolVar inp, const InputMaker::Param &param);

    protected:
        LoopImpl *m_owner_loop_opr = nullptr;

        std::shared_ptr<cg::ComputingGraph> m_sub_graph;

        OutputRecordSpec m_output_record_spec;
        std::vector<OutputRecordSpecItem*> m_output_record_spec_no_dedup;

        bool check_in_owner_graph(SymbolVar var) {
            return m_owner_graph == var.node()->owner_graph();
        }

        bool check_in_sub_graph(SymbolVar var) {
            return m_sub_graph.get() == var.node()->owner_graph();
        }

        size_t do_add_output(
                SymbolVar val,
                std::unique_ptr<OutputRecorderBase> recorder) override;

        /*!
         * \brief subclass can override this function to modify output spec
         *
         * Currently used by grad opr to modify vars for graph optimization
         */
        virtual void on_sub_graph_func_compile(
                ComputingGraph::OutputSpec &out_spec) {
        }

    private:
        struct OutputRecordSpecPtr {
            OutputRecordSpecItem *p;

            bool operator == (const OutputRecordSpecPtr &rhs) const {
                return p->is_same(*rhs.p);
            }

            struct Hash {
                size_t operator() (const OutputRecordSpecPtr &ptr) const {
                    return ptr.p->hash();
                }
            };
        };

        Maybe<std::vector<InputMaker*>> m_cur_func_input;

        cg::ComputingGraph *m_owner_graph = nullptr;
        std::unique_ptr<cg::static_infer::SubgraphStaticInferHelper>
            m_sub_graph_static_infer_helper =
            cg::static_infer::SubgraphStaticInferHelper::make();
        std::unordered_set<OutputRecordSpecPtr, OutputRecordSpecPtr::Hash>
            m_output_record_spec_dedup;

        SymbolVar m_counter_var;
        CounterProvider *m_counter_provider = nullptr;

        LoopCondManager m_loop_cond_manager;

        void on_first_input_added(SymbolVar inp);

}; // DescImplBase

/*!
 * \brief an operator to provider loop counter: updated after each
 *      scn_do_execute
 *
 * Default next_val is 0 and default delta is 1
 */
MGB_DEFINE_OPR_CLASS(LoopImpl::DescImplBase::CounterProvider,
        cg::SingleCNOperatorNodeBase) // {
    HostTensorND m_delta_host, m_next_val_host;
    DeviceTensorND m_delta_dev, m_next_val_dev;

    int m_delta, m_next_val;
    std::unique_ptr<megdnn::AddUpdate> m_add_update;

    void init_output_comp_node() override;
    void init_output_mem_plan(bool dynamic) override;
    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
    NodeProp* do_make_node_prop() const override;

    public:
        CounterProvider(
                ComputingGraph &graph, const OperatorNodeConfig &config);

        static CounterProvider* make(
                ComputingGraph &graph, const OperatorNodeConfig &config);

        //! update next value by adding delta to it
        void update_next_val();

        //! set next valud that this CounterProvider would produce
        void next_val(int v);

        //! set value of delta
        void delta(int v);

        int next_val() {
            return m_next_val;
        }
};

/*!
 * \brief base class for operators that serve as proxy for multiple dependencies
 */
MGB_DEFINE_CLS_WITH_SUPER(MultidepProxyOperatorNodeBase,
        cg::SingleCNOperatorNodeBase) // {
    void init_output_static_infer_desc() override final;

    protected:
        /*!
         * \brief dummy output var is allocated here
         */
        MultidepProxyOperatorNodeBase(const OperatorNodeBaseCtorParam &opr);
};

/*!
 * \brief update a DeviceTensorND by a var, with another dependence var
 *
 * This operator is intended to ensure update is performed after output var has
 * been computed.
 */
MGB_DEFINE_OPR_CLASS(LoopImpl::DepTensorUpdator,
        MultidepProxyOperatorNodeBase) // {
    public:
        //! state for accumulating values into dest tensor
        struct AccumulatorState {
            DeviceTensorND *dest = nullptr;
            bool first_sum = true;
            intl::UniqPtrWithCN<megdnn::Elemwise> adder;

            void reset() {
                first_sum = true;
            }
        };

        DepTensorUpdator(DeviceTensorND *dest,
                const std::shared_ptr<AccumulatorState> &accum_state,
                VarNode *val, VarNode *dep,
                const OperatorNodeConfig &config = {});

        /*!
         * \brief copy value into dest each time this opr is executed
         * \param val valued to be copied
         * \param dep dep var that must have been computed
         */
        static SymbolVar make(DeviceTensorND *dest,
                SymbolVar val, SymbolVar dep);

        /*!
         * \brief accumulate value into dest each time this opr is executed
         * \param val valued to be copied
         * \param dep dep var that must have been computed
         */
        static SymbolVar make(const std::shared_ptr<AccumulatorState> &state,
                SymbolVar val, SymbolVar dep);

        //! copy from this
        cg::OperatorNodeBase* shallow_copy(
                const VarNodeArray &inputs,
                const OperatorNodeConfig &config) const;

    private:
        DeviceTensorND * const m_dest;
        std::shared_ptr<AccumulatorState> const m_accum_state;

        void scn_do_execute() override;

        NodeProp* do_make_node_prop() const override;
};

class LoopImpl::FwdDesc final: public LoopImpl::DescImplBase {
    //! whether an inner inp var declared has_assign has actually been assigned
    ThinHashMap<VarNode*, bool> m_input_assigned;

    //! map from outer var to inner var for add_input without has_assign
    ThinHashMap<VarNode*, VarNode*> m_input_no_assign_dedup;

    //! see output_record_spec_mode_all()
    ThinHashMap<VarNode*, OutputRecordSpecItem*>
        m_output_record_spec_mode_all;

    std::unique_ptr<SubgraphDepIter> m_dep_iter;

    public:

        SymbolVar add_input(SymbolVar inp, bool has_assign) override;

        size_t add_output(SymbolVar val, OutputMode mode) override;

        Desc& assign(SymbolVar dest, SymbolVar val) override;

        VarNode* owner_graph_output_at(size_t idx) const;

        /*!
         * \brief output vars added by user(duplicated ones are replicated)
         */
        SymbolVarArray user_output_vars_including_dup() const;

        /*!
         * \brief map from var in sub graph to corresponding
         *      OutputRecordSpecItem if OutputMode is ALL
         */
        auto&& output_record_spec_mode_all() const {
            return m_output_record_spec_mode_all;
        }

        const std::vector<InputMaker*>& all_inputs() override;

        /*!
         * \brief all oprs in the sub graph needed for producing output vars
         *      given by do_add_output()
         */
        const cg::OprNodeArray& sub_graph_oprs() {
            all_inputs();
            return m_dep_iter->oprs();
        }

        //! called after sub graph has been optimized and endpoints changed
        void on_sub_graph_optimized() {
            m_dep_iter.reset();
        }
};

/*!
 * \brief save all history versions of mutable vars to be used for computing
 *      grad
 *
 * Notes on implementation:
 * 1. When loop executes for one time, all mutable vars would be saved in a
 *    bucket; each bucket has a size of swap_interval, and when it is full, it
 *    would be copied to host
 * 2. Important steps:
 *    2.1. Forward opr call disable() in add_input_layout_constraint()
 *    2.2. Grad opr call init_sub_graph_func() and enable_for_grad() in
 *         add_input_layout_constraint(), so this MutableStateSaver knows what
 *         states are needed
 *    2.3. Forward opr call init_sub_graph_func() in LoopImpl::scn_do_execute(),
 *         which utimately calls update_subgraph_outspec() to add oprs for
 *         saving needed state
 */
class LoopImpl::MutableStateSaver {
    //! recorder for a single var
    class Recorder;

    //! opr to update value of a Recorder
    class ValueUpdator;

    //! opr to update shape of a Recorder
    class ShapeUpdator;

    //! info for a saved var
    struct SavedVarInfo {
        //! var in fwd graph that is saved
        VarNode *var = nullptr;
        bool need_value = false, need_shape = false;
        std::unique_ptr<Recorder> recorder;
        //! updators for the recorder
        SymbolVar value_updator, shape_updator;
    };

    Loop * const m_owner_opr;

    bool m_slowcopy_warn_printed = false;
    bool m_enabled = true;

    //! swap_interval is min(swap_interval_setting, inferred loop time)
    int m_swap_interval_setting = 5;

    //! map from var in forward subgraph to corresponding SavedVarInfo
    ThinHashMap<VarNode*, SavedVarInfo> m_var2info;

    //! all vars that are recorded
    ThinHashSet<VarNode*> m_recorded_vars;

    //! print a warning about copy being slower than loop computation
    void print_slowcopy_warn(const char *msg);

    /*!
     * \brief get the corresponding var in owner graph added by user with
     *      add_output(mode=ALL)
     * \param var var in the fwd sub graph
     * \return nullptr if there is no OutputRecorder(ALL) for that var
     */
    inline VarNode* get_user_recorded_output_all(VarNode *var);

    public:
        MutableStateSaver(Loop *owner_opr);

        ~MutableStateSaver();

        //! set swap interval
        void swap_interval(int v) {
            m_swap_interval_setting = v;
        }

        void add_var_to_record(VarNode *var);

        bool enabled() const {
            return m_enabled;
        }

        void disable();

        //! enable recorders for grad comp seq
        void enable_for_grad(cg::AsyncExecutable *seq);

        //! test whether a var is recorded
        bool is_var_recorded(VarNode *var) const {
            return m_recorded_vars.count(var);
        }

        //! get saved state at current counter value in grad graph
        VarNode* get_state_for_grad(VarNode *fwd_var, DescImplBase *grad_desc);

        //! update subgraph outspec for forward opr
        void update_subgraph_outspec(ComputingGraph::OutputSpec &spec);

        //! callback when forward exec starts
        void on_fwd_begin();

        //! callback when forward exec finishes
        void on_fwd_finish();

        //! callback when grad exec finishes
        void on_grad_finish();

        //! for testing: get map from var to whether it is enabled in recorder
        ThinHashMap<VarNode*, bool> test_get_var_rec_spec();
};

} // intl
} // opr
} // mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

