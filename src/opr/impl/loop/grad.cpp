/**
 * \file src/opr/impl/loop/grad.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./impl.h"
#include "./grad.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/gopt/basic_arith.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/serialization/opr_shallow_copy.h"

#include "megdnn/oprs.h"

using namespace mgb;
using namespace opr;
using namespace intl;

/* ==================== OutputRecorderSumIntoDest ==================== */
namespace {

/*!
 * \brief sum grad values during loop into final grad output
 */
class OutputRecorderSumIntoDest final:
            public LoopImpl::Desc::OutputRecorderBase {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

    /*!
     * if m_sum_last is true, only the last value is summed into dest; note
     * that there might be multiple OutputRecorderSumIntoDest recorders for a
     * single dest var, so we could not simply forward the last value into dest
     */
    const bool m_sum_last;
    bool *m_dest_first_summed;
    bool m_optimize_coalesce_incr_sub = false;
    //! for debug assertions
    int m_dbg_nr_input_replacer_call = 0, m_dbg_nr_val_available_call = 0;
    VarNode *m_dest, *m_grad_incr_subtensor_modifier = nullptr;
    DeviceTensorND m_last_val;
    opr::intl::UniqPtrWithCN<megdnn::Elemwise> m_adder_opr;

    void bind_var(VarNode *, VarNode *var_owner) override {
        mgb_assert(var_owner == m_dest);
    }

    bool has_shape_infer_desc() const override {
        // dest is already allocated by LoopGrad
        return false;
    }

    void on_exec_begin() override {
        *m_dest_first_summed = true;
    }

    void do_sum(const DeviceTensorND &val);

    void on_val_produced(const DeviceTensorND &val) override;

    void on_exec_end() override;

    size_t hash() const override {
        return std::hash<const void*>{}(this);
    }

    bool is_same_st(const Hashable &rhs) const override {
        return this == &rhs;
    }

    SymbolVar get_outgrad_in_iter(
            SymbolVar, SymbolVar ,
            SymbolVar) override {
        mgb_assert(0);
    }

    Loop::Desc::OutputMode output_mode() const override {
        mgb_assert(0);
    }

    virtual std::string name() const override { return "outgradsum"; }

    DeviceTensorND incr_sub_input_replacer(const TensorShape &shape);

    public:
        static bool test_check_optimize_success;

        OutputRecorderSumIntoDest(bool sum_last,
                bool *dest_first_sumed, VarNode *dest):
            m_sum_last(sum_last),
            m_dest_first_summed(dest_first_sumed), m_dest(dest)
        {
        }

        /*!
         * \brief optimize grad computing when possible
         *
         * Internal state may be changed if optimization is successful
         *
         * \return new grad var
         */
        SymbolVar optimize_grad_var(SymbolVar grad);

        //! add extra targets needed for output recording; currently needed by
        //! optimize_grad_var()
        void add_extra_compile_output_spec(ComputingGraph::OutputSpec &spec) {
            if (m_grad_incr_subtensor_modifier) {
                spec.push_back({m_grad_incr_subtensor_modifier, {}});
            }
        }
}; // OutputRecorderSumIntoDest

} // anonymous namespace

bool OutputRecorderSumIntoDest::test_check_optimize_success;
MGB_DYN_TYPE_OBJ_FINAL_IMPL(OutputRecorderSumIntoDest);


void OutputRecorderSumIntoDest::do_sum(const DeviceTensorND &val) {
    auto &&dest = m_dest->dev_tensor();
    mgb_assert(dest.comp_node() == val.comp_node());
    if (*m_dest_first_summed) {
        *m_dest_first_summed = false;
        dest.copy_from_fixlayout(val);
    } else {
        if (!m_adder_opr) {
            m_adder_opr = intl::create_megdnn_opr<megdnn::Elemwise>(
                    dest.comp_node());
            m_adder_opr->param() = {megdnn::Elemwise::Mode::ADD};
        }
        mgb_assert(m_adder_opr.comp_node() == dest.comp_node());
        auto mdn_dest = dest.as_megdnn();
        m_adder_opr->exec({mdn_dest, val.as_megdnn()}, mdn_dest);
    }
}

void OutputRecorderSumIntoDest::on_val_produced(const DeviceTensorND &val) {
    if (m_optimize_coalesce_incr_sub) {
        ++ m_dbg_nr_val_available_call;
        return;
    }
    if (m_sum_last) {
        m_last_val = val;
    } else {
        do_sum(val);
    }
}

void OutputRecorderSumIntoDest::on_exec_end() {
    if (m_optimize_coalesce_incr_sub) {
        mgb_assert(
                m_dbg_nr_input_replacer_call == m_dbg_nr_val_available_call &&
                m_dbg_nr_input_replacer_call);
        m_dbg_nr_input_replacer_call = m_dbg_nr_val_available_call = 0;
        return;
    }
    if (m_sum_last) {
        do_sum(m_last_val);
        m_last_val = {};
    }
}

SymbolVar OutputRecorderSumIntoDest::optimize_grad_var(SymbolVar grad) {
    if (m_sum_last)
        return grad;

    // currently only try to coalesce incr_sub oprs
    auto opr = grad.node()->owner_opr();
    if (!gopt::check_is_incr_subtensor_zero(opr))
        return grad;

    // now we are sure that grad is in the form of incr_sub(0, sub)
    m_optimize_coalesce_incr_sub = true;
    test_check_optimize_success = true;

    {
        using namespace std::placeholders;
        auto replacer = std::bind(
                &OutputRecorderSumIntoDest::incr_sub_input_replacer, this, _1);
        m_grad_incr_subtensor_modifier = gopt::remake_incr_subtensor_zero(
                opr, nullptr, replacer);
    }

    // use a placeholder grad var to ensure sub is computed; result is computed
    // correctly since m_optimize_coalesce_incr_sub has been set
    return opr->input(1);
}

DeviceTensorND OutputRecorderSumIntoDest::incr_sub_input_replacer(
        const TensorShape& shape) {
    ++m_dbg_nr_input_replacer_call;
    auto&& dest = m_dest->dev_tensor();
    if (*m_dest_first_summed) {
        *m_dest_first_summed = false;
        mgb_assert(dest.shape().eq_shape(shape),
                   "output shape changed: %s vs %s",
                   dest.shape().to_string().c_str(), shape.to_string().c_str());
        fill_zero_dev_tensor(dest);
    }
    return dest;
}

/* ==================== AssignorGradOpr ==================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(LoopGrad::AssignorGradOpr);

bool LoopGrad::AssignorGradOpr::should_fwd() const {
    mgb_assert(m_assignee_grads_init);
    return m_assignee_grads_empty && m_assignor_grad;
}

void LoopGrad::AssignorGradOpr::mem_plan_fwd_in2out_readonly() {
    if (should_fwd()) {
        m_rofwd_subspec = SubTensorSpec::make_from_layout(
                m_assignor_grad->layout());
        rofwd_init_mem_plan();
    }
}

void LoopGrad::AssignorGradOpr::mem_plan_fwd_in2out_writable() {
    if (!should_fwd() && m_assignor_grad) {
        mgb_assert(m_assignor_grad == input(0));
        cg::request_fwd_in2out_writable_if_no_mem_ovelap(this, 0, 0);
    }
}

void LoopGrad::AssignorGradOpr::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto &&mgr = owner_graph()->static_infer_manager();
    mgr.register_shape_infer(output(0),
            ShapeInferDesc::make_identity(m_assignor));
}

cg::OperatorNodeBase::NodeProp*
LoopGrad::AssignorGradOpr::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    using DT = NodeProp::DepType;
    if (input().size() == 1) {
        prop->reset_dep_type(input(), {DT::SHAPE});
    } else {
        prop->reset_dep_type(input(), {DT::DEV_VALUE, DT::SHAPE});
    }
    return prop;
}

void LoopGrad::AssignorGradOpr::init_assignee_info(
        const VarNodeArray &assignees, SymbolVar loss) {
    mgb_assert(!m_assignee_grads_init && m_assignee_grads.empty());
    m_assignee_grads.reserve(assignees.size());
    for (auto i: assignees) {
        auto grad = cg::grad(loss, i, false, false).node();
        if (grad) {
            m_assignee_grads.push_back(grad);
        }
    }
    m_assignee_grads_init = true;
    m_assignee_grads_buf_init = true;
    m_assignee_grads_empty = m_assignee_grads.empty();
}

void LoopGrad::AssignorGradOpr::scn_do_execute() {
    if (should_fwd()) {
        rofwd_execute();
        return;
    }
    auto &&prev_gsum = m_state->prev_gsum;
    auto &&dest = output(0)->dev_tensor();
    if (prev_gsum.empty()) {
        // first execution in a loop

        if (m_assignor_grad) {
            auto &&src = m_assignor_grad->dev_tensor();
            if (dest.raw_ptr() != src.raw_ptr()) {
                dest.copy_from_fixlayout(src);
            } else {
                mgb_assert(dest.layout().eq_layout(src.layout()));
            }
        } else {
            fill_zero_dev_tensor(dest);
        }
        return;
    }
    if (m_assignor_grad) {
        auto &&src = m_assignor_grad->dev_tensor();
        opr::Elemwise::perform(opr::Elemwise::Mode::ADD,
                               const_cast<DeviceTensorND&>(dest),
                               {src, prev_gsum}, m_state->accum_state.adder);
    } else {
        dest.copy_from_fixlayout(prev_gsum);
    }
    m_state->accum_state.reset();
}

cg::OperatorNodeBase* LoopGrad::AssignorGradOpr::shallow_copy(
        const VarNodeArray &inputs, const OperatorNodeConfig &config) const {
    mgb_assert(m_assignee_grads_init);

    SymbolVar assignor_grad, assignor;
    if (inputs.size() == 1) {
        assignor = inputs[0];
    } else {
        mgb_assert(inputs.size() == 2);
        assignor_grad = inputs[0];
        assignor = inputs[1];
    }
    auto &&ret = make(
            assignor_grad, assignor, m_state, config).node()->owner_opr()
        ->cast_final_safe<AssignorGradOpr>();
    ret.m_assignee_grads_init = true;
    return &ret;
}

void LoopGrad::AssignorGradOpr::add_extra_compile_output_spec(
        ComputingGraph::OutputSpec &spec) {
    mgb_assert(m_assignee_grads_buf_init);
    auto ovar = output(0);
    for (auto i: m_assignee_grads) {
        auto updator = DepTensorUpdator::make(
                m_state->accum_state_shared(), i, ovar);
        spec.push_back({updator, {}});
    }
}

/* ==================== GradProxy ==================== */
/*!
 * \brief add grads to specific vars
 *
 * This operator is used to create a virtual loss var, so when computing grads
 * of the virtual loss with its input, the grad var would be replaced by given
 * value
 */
MGB_DEFINE_OPR_CLASS(LoopGrad::GradProxy,
        MultidepProxyOperatorNodeBase) // {
    public:
        //! set given grad to wrt
        struct GradInfo {
            VarNode *wrt = nullptr,
                    *grad = nullptr;
        };
        using GradInfoArray = std::vector<GradInfo>;

        GradProxy(ComputingGraph *graph, GradInfoArray &&grad):
            Super({graph, {}, "grad_proxy", {}}),
            m_grad(grad)
        {
            for (auto i: grad)
                add_input({i.wrt});
        }

        /*!
         * \param var the vars whose grads should be overwritten
         * \param grad the vars to provide grad
         * \brief return a placeholder scalar var to get grad w.r.t. inputs
         */
        static SymbolVar make(ComputingGraph *graph, GradInfoArray &&grad) {
            return graph->insert_opr(std::make_unique<GradProxy>(
                        graph, std::move(grad)))->output(0);
        }
    private:
        GradInfoArray m_grad;

        void scn_do_execute() override {
        }

        static VarNode* grad(
                OperatorNodeBase *opr, size_t wrt_idx,
                const VarNodeArray &out_grad) {
            MGB_MARK_USED_VAR(out_grad);
            auto &&info = opr->cast_final_safe<GradProxy>().m_grad.at(wrt_idx);
            return info.grad;
        }

        class _RegGrad {
            static _RegGrad ins;
            public:
                _RegGrad() {
                    cg::register_grad_func(typeinfo(), grad);
                }
        };

}; // GradProxy
MGB_DYN_TYPE_OBJ_FINAL_IMPL(LoopGrad::GradProxy);
LoopGrad::GradProxy::_RegGrad
LoopGrad::GradProxy::_RegGrad::ins;

/* ==================== GraphModifier ==================== */

/*!
 * \brief copy and modify forward graph to be used in backward pass
 */
class LoopGrad::GraphModifier {
    public:
        //! info of an input var
        struct InputInfo {
            //! used for OutputRecorderSumIntoDest::m_dest_first_summed
            mutable bool grad_dest_summed = false;

            //! corresponding vars in grad subgraph; added in add_input()
            VarNodeArray subgraph_var;
        };

        //! info entry for single assignment
        struct AssignmentInfo {
            // rule is assignee := assignor

            VarNode
                //! assignee var in owner graph
                *assignee_owner = nullptr,
                //! assignee var in sub graph (from saved mutable state at
                //! counter)
                *assignee_sub = nullptr,
                //! assignor var in sub graph
                *assignor = nullptr;
        };


        GraphModifier(DescImplBase *grad_desc,
                const ComputingGraph *fwd_graph,
                MutableStateSaver *mutable_state_saver);

        /*!
         * \brief initialize for given output vars
         * \param dest_vars vars in fwd graph needed to give grad
         */
        void init(const VarNodeArray &dest_vars) {
            mgb_assert(m_var_fwd2grad.empty() && !dest_vars.empty());
            SubgraphDepIter iter;
            for (auto i: dest_vars)
                iter.add(i);

            for (auto i: iter.oprs()) {
                process_opr(i);
            }

            for (auto i: iter.input_makers())  {
                process_input_maker(i);
            }

            for (auto &&i: m_var_fwd2grad) {
                mgb_assert(i.second->owner_graph() == m_grad_desc->sub_graph());
            }
        }

        //! get var in grad graph corresponding to given var in fwd graph
        VarNode* map_var(VarNode *fwd_var) const {
            return m_var_fwd2grad.at(fwd_var);
        }

        //! map from assignee var in subgraph to AssigneeInfo
        auto&& assignee2info() const {
            return m_assignee2info;
        }

        //! map from input var in owner graph to corresponding info
        auto&& input_ogvar2info() const {
            return m_input_ogvar2info;
        }

    private:

        MutableStateSaver * const m_mutable_state_saver;
        DescImplBase * const m_grad_desc;

        //! map vars in forward graph to vars in grad graph
        ThinHashMap<VarNode*, VarNode*> m_var_fwd2grad;

        //! used for avoiding mem alloc/free in process_opr()
        VarNodeArray m_new_opr_inputs, m_new_opr_recorded_outputs;

        //! process a single operator in fwd graph
        void process_opr(OperatorNodeBase *opr);

        //! process InputMaker oprs in fwd graph
        void process_input_maker(InputMaker *opr);

        //! see assignee2info()
        ThinHashMap<VarNode*, AssignmentInfo> m_assignee2info;

        //! see input_ogvar2info()
        ThinHashMap<VarNode*, InputInfo> m_input_ogvar2info;

};

LoopGrad::GraphModifier::GraphModifier(
        DescImplBase *grad_desc, const ComputingGraph *fwd_graph,
        MutableStateSaver *mutable_state_saver):
    m_mutable_state_saver{mutable_state_saver},
    m_grad_desc{grad_desc}
{
    auto trans_fwd_var = [this](VarNode *src) -> VarNode* {
        auto iter = m_var_fwd2grad.find(src);
        mgb_throw_if(iter == m_var_fwd2grad.end(),
                GraphError,
                "loop grad: var %s in fwd graph not used by grad opr",
                cg::dump_var_info({src}).c_str());
        return iter->second;
    };
    cg::InterGraphVarTransformer::register_to(
            grad_desc->sub_graph(), fwd_graph, trans_fwd_var);
}

void LoopGrad::GraphModifier::process_input_maker(InputMaker *opr) {
    auto sub_inp = m_var_fwd2grad.at(opr->output(0));
    auto owner_var = opr->orig_var();
    m_input_ogvar2info[owner_var].subgraph_var.push_back(sub_inp);
    if (opr->param().has_assign) {
        auto &&info = m_assignee2info[sub_inp];
        info.assignee_owner = owner_var;
        info.assignee_sub = sub_inp;
        info.assignor = m_var_fwd2grad.at(opr->assignor());
    }
}

/* ==================== GradDesc ==================== */

/*!
 * Grad comes from two sources:
 * 1. output grad in owner graph
 * 2. iterative grad for assignment
 *
 * See init_for_grad() and assign() for more details.
 */
class LoopGrad::GradDesc final: public LoopImpl::DescImplBase {
    struct AssignorInfo {
        VarNodeArray assignees;
        AssignorGradOpr *grad_opr = nullptr;
    };

    std::vector<InputMaker*> m_all_inputs;
    MutableStateSaver * const m_mutable_state_saver;

    GraphModifier m_fwd_graph_modifier;

    //! CounterProvider in fwd graph
    CounterProvider * const m_fwd_counter_provider;

    ThinHashMap<VarNode*, AssignorInfo> m_assignor2info;

    //! newly inserted AssignorGradOpr that have not been initialized; they
    //! would be initialized after grad transformer exits, to avoid recursive
    //! cg::grad call
    std::vector<AssignorGradOpr*> m_uninitialized_assignor_grad_oprs;

    //! cached endpoints so graph optimizer would be called only once
    VarNodeArray m_prev_sub_graph_opt_endpoints_inp,
                 m_prev_sub_graph_opt_endpoints_out;

    SymbolVar m_counter_var_up;
    CounterProvider* m_counter_provider_up = nullptr;

    GradProxy *m_grad_virtual_loss_opr = nullptr;
    SymbolVar m_grad_virtual_loss;

    SymbolVar m_orig_loop_cond_var;

    //! tot counter value for previous forward
    size_t m_counter_tot = 0;

    void init_virtual_loss(
            DescImplBase *fwd_desc, const VarNodeArray &outgrad_owner);

    void init_assignments();

    const std::vector<InputMaker*>& all_inputs() override {
        return m_all_inputs;
    }

    void on_sub_graph_func_compile(
            ComputingGraph::OutputSpec &out_spec) override;

    public:
        GradDesc(Loop *loop,
                MutableStateSaver *mutable_state_saver,
                const VarNodeArray &outgrad_owner);

        SymbolVar add_input(SymbolVar inp) {
            auto ret = do_add_input(inp, {true, false});
            m_all_inputs.push_back(
                    &ret.node()->owner_opr()->cast_final_safe<InputMaker>());
            return ret;
        }

        SymbolVar add_input(SymbolVar inp, bool has_assign) override {
            // used by LoopImpl::MutableStateSaver::get_state_for_grad
            mgb_assert(!has_assign);
            return add_input(inp);
        }

        Desc& assign(SymbolVar, SymbolVar) override {
            mgb_trap();
        }

        Desc& set_loop_condition(SymbolVar) override {
            mgb_trap();
        }

        /*!
         * \brief connect two vars in owner graph, so *owner_dest* would be the
         *      value of grads of *owner_wrt*
         * \param owner_wrt owner var with respect to which to take grad; must
         *      be an input of loop fwd opr
         * \param owner_dest target grad var; an output of loop grad opr
         * \return whether grad is non-zero
         */
        bool bind_grad_var(VarNode *owner_wrt, VarNode *owner_dest);

        void reset_counter_provider() override {
            m_counter_tot = m_fwd_counter_provider->next_val() + 1;
            mgb_assert(m_counter_tot);
            m_counter_provider_up->next_val(0);
            counter_provider()->next_val(m_counter_tot - 1);
        }

        void update_counter_provider() override {
            m_counter_provider_up->update_next_val();
            counter_provider()->update_next_val();
        }

        size_t counter_var_tot() const {
            return m_counter_tot;
        }

        void on_grad_exec_finish() {
            m_mutable_state_saver->on_grad_finish();
            for (auto &&i: m_assignor2info) {
                auto o = i.second.grad_opr;
                if (o)
                    o->on_grad_exec_finish();
            }
        }

        MutableStateSaver* mutable_state_saver() const {
            return m_mutable_state_saver;
        }
};

LoopGrad::GradDesc::GradDesc(
        Loop *loop, MutableStateSaver *mutable_state_saver,
        const VarNodeArray &outgrad_owner):
    m_mutable_state_saver{mutable_state_saver},
    m_fwd_graph_modifier{this, loop->m_desc->sub_graph(), mutable_state_saver},
    m_fwd_counter_provider{loop->m_desc->counter_provider()}
{
    m_counter_provider_up = CounterProvider::make(
            *m_sub_graph,
            OperatorNodeConfig{loop->comp_node()}.name("counter_up"));
    m_counter_var_up = m_counter_provider_up->output(0);

    // add counter var as input to ensure loop executes before grad
    add_input(loop->output_counter_var());
    counter_provider()->delta(-1);

    init_virtual_loss(loop->m_desc.get(), outgrad_owner);
    init_assignments();
    DescImplBase::set_loop_condition(
            m_orig_loop_cond_var = (get_counter_var() > 0));
}

void LoopGrad::GradDesc::init_virtual_loss(
        DescImplBase *fwd_desc, const VarNodeArray &outgrad_owner) {

    GradProxy::GradInfoArray output_grad_info;
    VarNodeArray needed_fwd_outvars;

    // handle user added outputs: forward grads in owner graph to subgraph
    size_t idx = 0;

    // user_data records whether it has been added
    for (auto &&i: fwd_desc->output_record_spec_no_dedup())
        i->user_data = false;
    for (auto &&i: fwd_desc->output_record_spec_no_dedup()) {
        if (!i->user_data) {
            i->user_data = true;
            auto owner_grad = outgrad_owner.at(idx);
            if (owner_grad) {
                auto rec = i->recorder();
                auto all_grad_sub = add_input(owner_grad);
                auto sub_grad = rec->get_outgrad_in_iter(
                        get_counter_var(), m_counter_var_up, all_grad_sub);
                sub_grad.rename(ssprintf("outgrad:%s[%zd]",
                            rec->name().c_str(), idx));
                needed_fwd_outvars.push_back(i->var_sub());
                output_grad_info.push_back({i->var_sub(), sub_grad.node()});
            }
            idx ++;
        }
    }

    mgb_assert(idx == outgrad_owner.size());
    m_fwd_graph_modifier.init(needed_fwd_outvars);
    for (auto &&i: output_grad_info) {
        i.wrt = m_fwd_graph_modifier.map_var(i.wrt);
    }

    m_grad_virtual_loss = GradProxy::make(
            m_sub_graph.get(), std::move(output_grad_info));
    m_grad_virtual_loss_opr =
        &m_grad_virtual_loss.node()->owner_opr()->cast_final_safe<GradProxy>();

}

bool LoopGrad::GradDesc::bind_grad_var(
        VarNode *owner_wrt, VarNode *owner_dest) {
    auto &&input_ogvar2info = m_fwd_graph_modifier.input_ogvar2info();
    auto info_iter = input_ogvar2info.find(owner_wrt);
    if (info_iter == input_ogvar2info.end()) {
        // caused by input vars not needed by grad
        return false;
    }
    auto &&info = m_fwd_graph_modifier.input_ogvar2info().at(owner_wrt);
    auto &&assignee2info = m_fwd_graph_modifier.assignee2info();
    bool nonzero = false;
    for (auto i: info.subgraph_var) {
        auto grad = cg::grad(m_grad_virtual_loss, i, false, false);
        if (!grad.node())
            continue;
        nonzero = true;
        bool sum_last;
        if (!assignee2info.count(i)) {
            // sum all intermediate grads
            mgb_assert(i->owner_opr()->same_type<InputMaker>());
            sum_last = false;
        } else {
            mgb_assert(!i->owner_opr()->same_type<InputMaker>());
            grad.node()->add_flag(VarNode::Flag::NO_MEM_RECLAIM);
            sum_last = true;
        }
        auto grad_sum_recorder = std::make_unique<OutputRecorderSumIntoDest>(
                sum_last, &info.grad_dest_summed, owner_dest);
        grad = grad_sum_recorder->optimize_grad_var(grad);
        do_add_output(grad, std::move(grad_sum_recorder));
        mgb_assert(
                m_output_record_spec_no_dedup.back()->var_sub() == grad.node());
        const_cast<OutputRecordSpecItem&>(
                *m_output_record_spec_no_dedup.back()).bind(owner_dest);
    }
    if (nonzero) {
        auto &&vec = m_uninitialized_assignor_grad_oprs;
        while (!vec.empty()) {
            auto opr = vec.back();
            vec.pop_back();
            opr->init_assignee_info(
                    m_assignor2info.at(opr->assignor()).assignees,
                    m_grad_virtual_loss);
        }
    }
    return nonzero;
}

void LoopGrad::GradDesc::init_assignments() {
    for (auto &&i: m_fwd_graph_modifier.assignee2info()) {
        m_assignor2info[i.second.assignor].assignees.push_back(i.first);
    }

    auto grad_trans = [this](VarNode *target, VarNode *wrt, VarNode *grad) {
        mgb_assert(target == m_grad_virtual_loss.node());
        auto gnew = AssignorGradOpr::make(grad, wrt);
        auto &&d = m_assignor2info.at(wrt);
        mgb_assert(!d.grad_opr);
        d.grad_opr = &gnew.node()->owner_opr()->
            cast_final_safe<AssignorGradOpr>();
        m_uninitialized_assignor_grad_oprs.push_back(d.grad_opr);
        return gnew.node();
    };

    for (auto &&i: m_assignor2info) {
        cg::add_grad_transformer(i.first, grad_trans);

        for (VarNode *j: i.second.assignees) {
            // assignee := assignor, and grads on assignor can be computed if we
            // have grads on assignee; assinee is output var, and assignor is
            // input var
            cg::add_extra_dep_for_grad(i.first, j);
        }
    }
}

void LoopGrad::GradDesc::on_sub_graph_func_compile(
        ComputingGraph::OutputSpec &out_spec) {

    {
        // append extra targets to out_spec
        size_t idx = 0, nr_out_spec = out_spec.size();
        mgb_assert(out_spec[idx ++].first.node() ==
                loop_cond_manager().subgraph_outspec_item().first.node());


        for (auto &&i: output_record_spec()) {
            if (!i.enabled())
                continue;
            auto &&spec = out_spec[idx ++];
            mgb_assert(spec.first.node() == i.var_sub());
            i.recorder()->cast_final_safe<OutputRecorderSumIntoDest>().
                add_extra_compile_output_spec(out_spec);
        }
        mgb_assert(idx == nr_out_spec);

        // add outspec for AssignorGradOpr
        auto cb = [&](OperatorNodeBase *opr) {
            if (opr->same_type<AssignorGradOpr>()) {
                opr->cast_final<AssignorGradOpr>().
                    add_extra_compile_output_spec(out_spec);
            }
        };
        cg::DepOprIter iter{cb};
        for (idx = 0, nr_out_spec = out_spec.size();
                idx < nr_out_spec; ++ idx) {
            iter.add(out_spec[idx].first.node()->owner_opr());
        }
    }
    int opt_level = owner_graph()->options().graph_opt_level;
    if (std::abs(opt_level) < 2)
        return;
    VarNodeArray endpoints;
    endpoints.reserve(out_spec.size());
    endpoints.push_back(m_orig_loop_cond_var.node());
    for (size_t i = 1; i < out_spec.size(); ++ i)
        endpoints.push_back(out_spec[i].first.node());

    if (endpoints == m_prev_sub_graph_opt_endpoints_inp) {
        endpoints = m_prev_sub_graph_opt_endpoints_out;
    } else {
        m_prev_sub_graph_opt_endpoints_inp = endpoints;
        gopt::GraphOptimizer().
            verbosity(0).
            add_preset_passes().
            enable_check_result(opt_level < 0).
            apply_inplace(endpoints);
        m_prev_sub_graph_opt_endpoints_out = endpoints;

        // NO_MEM_RECLAIM flag is required for OutputRecorderSumIntoDest
        for (size_t i = 0; i < endpoints.size(); ++ i) {
            constexpr auto F = VarNode::Flag::NO_MEM_RECLAIM;
            if (m_prev_sub_graph_opt_endpoints_inp[i]->contain_flag(F)) {
                endpoints[i]->add_flag(F);
            }
        }
    }

    auto iter = endpoints.begin();
    out_spec[0] = loop_cond_manager().
        setup(*(iter ++)).
        subgraph_outspec_item();

    for (size_t i = 1; i < out_spec.size(); ++ i) {
        out_spec[i].first = *(iter ++);
    }
    mgb_assert(iter == endpoints.end());
}

/* ==================== GraphModifier (deps on GradDesc) ==================== */
void LoopGrad::GraphModifier::process_opr(OperatorNodeBase *opr) {
    mgb_assert(
            !opr->node_prop().contain(NodeProp::Flag::FORCE_UPDATE_INPUT_VAR),
            "FORCE_UPDATE_INPUT_VAR node in "
            "subgraph of loop currently unsupported: %s{%s}",
            opr->cname(), opr->dyn_typeinfo()->name);

    m_new_opr_recorded_outputs.clear();
    for (auto i: opr->output()) {
        if (m_mutable_state_saver->is_var_recorded(i)) {
            auto out = m_mutable_state_saver->get_state_for_grad(
                    i, m_grad_desc);
            m_var_fwd2grad[i] = out;
            m_new_opr_recorded_outputs.push_back(out);
        }
    }

    bool output_recorded = !m_new_opr_recorded_outputs.empty();
    if (opr->same_type<InputMaker>()) {
        auto &&im = opr->cast_final<InputMaker>();
        mgb_assert(im.output().size() == 1);
        mgb_assert(output_recorded == im.param().has_assign);
        if (!im.param().has_assign) {
            auto new_var = static_cast<GradDesc*>(m_grad_desc)->add_input(
                    im.orig_var());
            m_var_fwd2grad[im.output(0)] = new_var.node();
        }
        // assignments are processed in init_assignments()
        return;
    }

    if (opr->same_type<DescImplBase::CounterProvider>()) {
        mgb_assert(opr->output().size() == 1);
        auto ovar = m_grad_desc->counter_provider();
        m_var_fwd2grad[opr->output(0)] = ovar->output(0);
        return;
    }

    m_new_opr_inputs.clear();
    for (auto i: opr->input()) {
        m_new_opr_inputs.push_back(m_var_fwd2grad.at(i));
    }

    if (output_recorded) {
        // output vars has been replaced by MutableStateSaver, but gradient must
        // be computed by original opr
        cg::add_var_virtual_receiver_reuse_opr_grad(
                m_new_opr_inputs, m_new_opr_recorded_outputs, opr,
                true);
        return;
    }

    auto config = opr->config();
    config.name(opr->name());
    auto new_opr = serialization::copy_opr_shallow(
            *opr, m_new_opr_inputs, opr->config(),
            m_grad_desc->sub_graph());

    for (size_t i = 0; i < opr->output().size(); ++ i)
        m_var_fwd2grad[opr->output(i)] = new_opr->output(i);
}

/* ========= LoopGrad ========= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LoopGrad);

LoopGrad::LoopGrad(Loop *loop_opr, std::unique_ptr<GradDesc> desc,
        const OperatorNodeConfig &config):
    Super({nullptr, config, "loop_grad", {}}, std::move(desc)),
    m_fwd_loop_opr(loop_opr),
    m_grad_result_cache(loop_opr->input().size())
{

    add_input_in_desc();
    for (auto i: loop_opr->input()) {
        add_output(i->name())->
            dtype(i->dtype()).
            add_flag(VarNode::Flag::NO_ALLOC_IF_UNUSED);
    }

    add_equivalence_component<ScalarHash<Loop*>>(loop_opr);

    m_static_loop_time_infer = [this] {
        mgb_assert(m_nr_scn_do_execute_run + 1 ==
                m_fwd_loop_opr->m_nr_scn_do_execute_run);
        return static_cast<GradDesc*>(m_desc.get())->counter_var_tot();
    };
}

LoopGrad* LoopGrad::make(Loop *loop_opr, const VarNodeArray &outgrad,
        const OperatorNodeConfig &config) {
    auto desc = std::make_unique<GradDesc>(
            loop_opr, loop_opr->m_mutable_state_saver.get(), outgrad);
    auto opr = loop_opr->owner_graph()->insert_opr(
            std::make_unique<LoopGrad>(loop_opr, std::move(desc), config));
    auto &&ret = opr->cast_final_safe<LoopGrad>();

    // init ret.m_orig_outgrad_idx_in_input
    ThinHashMap<VarNode*, size_t> var2idx;
    for (size_t i = 1; i < ret.input().size(); ++ i) {
        var2idx[ret.input()[i]] = i;
    }
    ret.m_orig_outgrad_idx_in_input.reserve(outgrad.size());
    for (auto i: outgrad) {
        size_t cur = 0;
        if (i) {
            cur = var2idx.at(i);
        }
        ret.m_orig_outgrad_idx_in_input.push_back(cur);
    }
    return &ret;
}


cg::OperatorNodeBase* LoopGrad::shallow_copy(
        const VarNodeArray &inputs, const OperatorNodeConfig &config) const {
    auto loop = &inputs[0]->owner_opr()->cast_final_safe<Loop>();
    VarNodeArray outgrad;
    outgrad.reserve(m_orig_outgrad_idx_in_input.size());
    for (auto i: m_orig_outgrad_idx_in_input) {
        VarNode *cur = nullptr;
        if (i)
            cur = inputs[i];
        outgrad.push_back(cur);
    }
    auto ret = make(loop, outgrad);
    for (size_t i = 0; i < m_grad_result_cache.size(); ++ i) {
        if (m_grad_result_cache[i].first) {
            ret->get_grad_var(i);
        }
    }
    return ret;
}

cg::OperatorNodeBase::NodeProp *LoopGrad::do_make_node_prop() const {
    // skip LoopImpl::do_make_node_prop because sub_graph_func not ready yet
    auto prop = Super::do_make_node_prop();

    auto &&p0 = m_fwd_loop_opr->node_prop();
    {
        constexpr auto i = NodeProp::Flag::IMPURE_FUNC;
        if (p0.contain(i))
            prop->add_flag(i);
    }

    // add shape deps so shape could be updated for InputMaker shape infer
    auto counter = m_fwd_loop_opr->output_counter_var();
    mgb_assert(input(0) == counter);
    for (size_t i = 1; i < input().size(); ++ i) {
        auto var = input()[i];
        prop->add_dep_type(var, NodeProp::DepType::SHAPE);
        mgb_assert(var != counter);
    }

    const_cast<NodeProp::DepMap&>(prop->dep_map())[counter] =
        NodeProp::DepType::DEV_COMP_ORDER;

    return prop;
}

void LoopGrad::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto &&li = m_fwd_loop_opr->input();
    auto &&mgr = owner_graph()->static_infer_manager();
    for (size_t i = 0; i < li.size(); ++ i) {
        mgr.register_shape_infer(output(i),
                ShapeInferDesc::make_identity(li[i]));
    }
}

VarNode* LoopGrad::get_grad_var(size_t inp_idx) {
    auto &&cache = m_grad_result_cache.at(inp_idx);
    if (!cache.first) {
        if (static_cast<GradDesc*>(m_desc.get())->bind_grad_var(
                    m_fwd_loop_opr->input(inp_idx), output(inp_idx))) {
            cache.second = output(inp_idx);
        } else {
            cache.second = nullptr;
        }
        cache.first = true;
    }
    return cache.second;
}

void LoopGrad::scn_do_execute() {
    mgb_assert(m_nr_scn_do_execute_run + 1 ==
            m_fwd_loop_opr->m_nr_scn_do_execute_run);
    Super::scn_do_execute();
    static_cast<GradDesc*>(m_desc.get())->on_grad_exec_finish();
}

void LoopGrad::add_input_layout_constraint() {
    LoopImpl::add_input_layout_constraint();
    init_sub_graph_func();
    static_cast<GradDesc*>(m_desc.get())->
        mutable_state_saver()->
        enable_for_grad(sub_graph_func());
}

bool& LoopImpl::test_check_grad_output_recorder_sum_optimize_success() {
    return OutputRecorderSumIntoDest::test_check_optimize_success;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
