/**
 * \file src/opr/impl/loop/impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./impl.h"
#include "megbrain/utils/async_worker.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"

#include <atomic>
#include <cmath>

using namespace mgb;
using namespace opr;
using namespace intl;

/*
 * Notes: ComputingGraphImpl::ComputingSequence::execute would not wait for
 * previous exec to finish, so we have to be careful that device status change
 * can be correctly queued (e.g. DepTensorUpdator must copy value to a device
 * buffer, and CounterProvider must use AddUpdate rather than modify a host
 * buffer and copy it to device)
 */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LoopImpl::OutputRecordSpecItem);

/* ============== DescImplBase ============== */

LoopImpl::DescImplBase::DescImplBase():
    m_sub_graph{cg::ComputingGraph::make()}
{
    auto &&opt = m_sub_graph->options();
    opt.log_level = 0;
    opt.async_exec_level = 0;
    opt.allocate_static_mem_after_graph_compile = false;
}

void LoopImpl::DescImplBase::on_first_input_added(SymbolVar inp) {
    m_owner_graph = inp.node()->owner_graph();
    m_sub_graph->set_as_subgraph(*m_owner_graph);
    if (m_owner_graph->options().graph_opt_level < 0) {
        m_sub_graph->options().graph_opt_level = -1;
    } else {
        m_sub_graph->options().graph_opt_level = 1;
    }

    mgb_assert(!m_counter_provider);
    m_counter_provider = CounterProvider::make(
            *m_sub_graph, {inp.node()->comp_node()});
    m_counter_var = m_counter_provider->output(0);
}

SymbolVar LoopImpl::DescImplBase::do_add_input(
        SymbolVar inp, const InputMaker::Param &param) {
    if (!m_owner_graph) {
        on_first_input_added(inp);
    } else {
        mgb_throw_if(!check_in_owner_graph(inp),
                GraphError, "inputs belong to different graphs");
    }
    auto var = LoopImpl::InputMaker::make(this, inp, param);
    auto opr = var.node()->owner_opr();
    mgb_assert(opr->same_type<InputMaker>() || !param.has_assign);
    // opr can be non-InputMaker when immutable static infer
    return var;
}

size_t LoopImpl::DescImplBase::do_add_output(
                SymbolVar val,
                std::unique_ptr<OutputRecorderBase> recorder) {

    mgb_throw_if(!check_in_sub_graph(val),
            GraphError, "output var must be in sub graph");

    OutputRecordSpecItem elem(val, std::move(recorder));
    auto iter = m_output_record_spec_dedup.find({&elem});
    if (iter == m_output_record_spec_dedup.end()) {
        m_output_record_spec.emplace_back(std::move(elem));
        auto siter = m_output_record_spec.end();
        -- siter;
        auto rst = m_output_record_spec_dedup.insert({&*siter});
        mgb_assert(rst.second);
        iter = rst.first;
    }

    auto id = m_output_record_spec_no_dedup.size();
    m_output_record_spec_no_dedup.push_back(iter->p);
    return id;
}

std::unique_ptr<cg::AsyncExecutable> LoopImpl::DescImplBase::compile() {

    // build output spec
    ComputingGraph::OutputSpec out_spec;

    out_spec.push_back(m_loop_cond_manager.subgraph_outspec_item());
    if (auto s = m_owner_loop_opr->m_mutable_state_saver.get()) {
        s->update_subgraph_outspec(out_spec);
    }

    for (auto &&i: m_output_record_spec) {
        if (!i.enabled())
            continue;

        auto cb = [ptr=&i](const DeviceTensorND& dev) {
            ptr->recorder()->on_val_produced(dev);
        };
        out_spec.push_back({i.var_sub(), cb});
    }

    on_sub_graph_func_compile(out_spec);
    auto func = m_sub_graph->compile(out_spec);

    // find used input, and check unique comp node
    if (!m_cur_func_input.valid())
        m_cur_func_input = std::vector<InputMaker*>();
    auto &&inp = m_cur_func_input.val();
    inp.clear();
    CompNode the_comp_node;
    ThinHashSet<OperatorNodeBase*> visited;
    auto cb = [&inp, &the_comp_node, &visited](OperatorNodeBase *opr) {
        visited.insert(opr);
        if (opr->same_type<InputMaker>()) {
            inp.push_back(&opr->cast_final<InputMaker>());
        }
        for (auto i: opr->output()) {
            if (!the_comp_node.valid())
                the_comp_node = i->comp_node();
            else {
                mgb_assert(the_comp_node == i->comp_node(),
                        "different comp nodes encountered in subgraph of loop: "
                        "expect=%s get=%s (from %s)",
                        the_comp_node.to_string().c_str(),
                        i->comp_node().to_string().c_str(),
                        cg::dump_var_info({i}).c_str());
            }
        }
        return true;
    };
    func->iter_opr_seq(cb);
    for (auto &&i: func->get_rt_static_source_deps()) {
        auto opr = i.dest->owner_opr();
        if (visited.insert(opr).second && opr->same_type<InputMaker>()) {
            inp.push_back(&opr->cast_final<InputMaker>());
        }
    }

    return func;
}

void LoopImpl::DescImplBase::reset_counter_provider() {
    m_counter_provider->next_val(0);
}

void LoopImpl::DescImplBase::update_counter_provider() {
    m_counter_provider->update_next_val();
}

/* ============== LoopCondManager ============== */

MGB_DEFINE_OPR_CLASS(LoopImpl::DescImplBase::LoopCondManager::GetCondOpr,
        cg::SingleCNOperatorNodeBase) // {
    bool m_static_infer;
    HostTensorND m_host_val;
    DeviceTensorND m_inferred_val;
    std::unique_ptr<CompNode::Event> m_copy_event;

    void init_output_static_infer_desc() override {
        using namespace cg::static_infer;
        owner_graph()->static_infer_manager().register_shape_infer(
                output(0), ShapeInferDesc::make_const({}));
    }

    NodeProp* do_make_node_prop() const override {
        auto prop = Super::do_make_node_prop();
        if (m_static_infer) {
            prop->reset_dep_type(input(), {NodeProp::DepType::HOST_VALUE});
        }
        prop->add_flag(NodeProp::Flag::DISALLOW_COMP_NODE_OPTIMIZE);
        return prop;
    }

    void scn_do_execute() override {
        if (!m_static_infer) {
            if (!m_copy_event)
                m_copy_event = comp_node().create_event();
            m_host_val.copy_from(input(0)->dev_tensor());
            m_copy_event->record();
        } else {
            m_inferred_val = owner_graph()->static_infer_manager().infer_value(
                    input(0));
        }
    }

    public:

        GetCondOpr(VarNode *inp):
            Super{inp->owner_graph(), {}, "cond", {inp}}
        {
            add_input({inp});
            using VF = VarNode::Flag;
            add_output(None)->
                add_flag(VF::ALLOW_EMPTY_SHAPE).
                add_flag(VF::VOLATILE_CONTENT);
            m_static_infer = cg::is_static_var_value(inp);
        }

        bool should_loop() {
            megdnn::TensorND cond;
            if (m_static_infer) {
                cond = m_inferred_val.as_megdnn();
            } else {
                cond = m_host_val.as_megdnn();
                m_copy_event->host_wait();
            }
            mgb_assert(cond.layout.is_scalar());
            switch (cond.layout.dtype.enumv()) {
                case DTypeEnum::Float32:
                    return std::abs(cond.ptr<float>()[0]) > 1e-5;
#if !MEGDNN_DISABLE_FLOAT16
                case DTypeEnum::Float16:
                    return std::abs(cond.ptr<dt_float16>()[0]) > 1e-5;
                case DTypeEnum::BFloat16:
                    return std::abs(cond.ptr<dt_bfloat16>()[0]) > 1e-5;
#endif

#define cb(_dt) case DTypeTrait<_dt>::enumv: \
                    return cond.ptr<DTypeTrait<_dt>::ctype>()[0] != 0;
                MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
#undef cb
                case DTypeEnum::Byte:
                    break;
                case DTypeEnum::IntB1:
                    break;
                case DTypeEnum::IntB2:
                    break;
                case DTypeEnum::IntB4:
                    break;
                case DTypeEnum::UintB4:
                    break;
                case DTypeEnum::Bool:
                    break;
                case DTypeEnum::Uint16:
                    break;
#define cb(_dt)         \
    case DTypeEnum::_dt: \
        break;
                MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb)
#undef cb
            }
            mgb_throw(GraphError, "unhandled dtype for loop condition: %s",
                    cond.layout.dtype.name());
        }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(
        LoopImpl::DescImplBase::LoopCondManager::GetCondOpr);

bool LoopImpl::DescImplBase::LoopCondManager::should_loop() {
    mgb_assert(m_get_cond_opr);
    return m_get_cond_opr->should_loop();
}

ComputingGraph::OutputSpec::value_type
LoopImpl::DescImplBase::LoopCondManager::subgraph_outspec_item() {
    mgb_assert(m_var.node());
    if (!m_get_cond_opr || m_get_cond_opr->input(0) != m_var.node()) {
        auto ov = m_var.insert_single_output_opr<GetCondOpr>(m_var.node());
        m_get_cond_opr = &ov.node()->owner_opr()->cast_final_safe<GetCondOpr>();
    }
    return {m_get_cond_opr->output(0), {}};
}

/* ============== CounterProvider ============== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LoopImpl::DescImplBase::CounterProvider);

LoopImpl::DescImplBase::CounterProvider::CounterProvider(
        ComputingGraph &graph, const OperatorNodeConfig &config):
    Super(&graph, config, "counter", {})
{
    // disable dedup
    add_equivalence_component<ScalarHash<void*>>(this);

    add_output(None)->dtype(dtype::Int32());
}

LoopImpl::DescImplBase::CounterProvider*
LoopImpl::DescImplBase::CounterProvider::make(
        ComputingGraph &graph, const OperatorNodeConfig &config) {
    auto o = graph.insert_opr(std::make_unique<CounterProvider>(graph, config));
    return &o->cast_final_safe<CounterProvider>();
}

cg::OperatorNodeBase::NodeProp*
LoopImpl::DescImplBase::CounterProvider::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    prop->add_flag(NodeProp::Flag::DISALLOW_COMP_NODE_OPTIMIZE);
    return prop;
}

void LoopImpl::DescImplBase::CounterProvider::init_output_comp_node() {
    mgb_assert(config().has_comp_node_set());
    auto cn = config().get_single_comp_node();
    comp_node(cn);
    m_delta_host.
        comp_node(cn).
        dtype(dtype::Int32()).
        resize({1});
    m_next_val_host.
        comp_node(cn).
        dtype(dtype::Int32()).
        resize({1});

    m_delta_dev.copy_from(m_delta_host);
    m_next_val_dev.copy_from(m_next_val_host);

    delta(1);
    next_val(0);

    m_add_update =
        intl::get_megdnn_handle(cn)->create_operator<megdnn::AddUpdate>();
    m_add_update->param() = {1, 1, 0};
}

void LoopImpl::DescImplBase::CounterProvider::init_output_mem_plan(
        bool dynamic) {
    MGB_MARK_USED_VAR(dynamic);
    output(0)->init_mem_plan(&m_next_val_dev);
}

void LoopImpl::DescImplBase::CounterProvider::scn_do_execute() {
    mgb_assert(output(0)->dev_tensor().raw_ptr() == m_next_val_dev.raw_ptr());
}

void LoopImpl::DescImplBase::CounterProvider::init_output_static_infer_desc() {
    using namespace cg::static_infer;

    auto &&mgr = owner_graph()->static_infer_manager();
    mgr.register_shape_infer(output(0), ShapeInferDesc::make_const({1}));

    auto infer_value = [this](DeviceTensorND &dest, const InpVal &) {
        dest.resize({1}).ptr<int>()[0] = m_next_val;
        return true;
    };
    mgr.register_value_infer(output(0),
            {SourceType::MUTABLE, {}, infer_value});
}

void LoopImpl::DescImplBase::CounterProvider::update_next_val() {
    m_next_val += m_delta;
    m_add_update->exec(m_next_val_dev.as_megdnn(), m_delta_dev.as_megdnn());
}

void LoopImpl::DescImplBase::CounterProvider::next_val(int v) {
    m_next_val = v;
    m_next_val_host.ptr<int>()[0] = v;
    m_next_val_dev.copy_from_fixlayout(m_next_val_host);
}

void LoopImpl::DescImplBase::CounterProvider::delta(int v) {
    m_delta = v;
    m_delta_host.ptr<int>()[0] = v;
    m_delta_dev.copy_from_fixlayout(m_delta_host);
}

/* ========= LoopImpl::InputMaker ========= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LoopImpl::InputMaker);

LoopImpl::InputMaker::InputMaker(
        DescImplBase *desc, VarNode *orig_var, const Param &param):
    Super{desc->sub_graph(), {}, "fwd", {orig_var}},
    m_param{param}, m_orig_var{orig_var}, m_desc{desc}
{
    mgb_assert(!param.has_assign || param.disable_value_infer);

    add_output(None)->dtype(orig_var->dtype());

    // different inputs may be used with different updating rules, and dedup
    // should have been handled by FwdDesc, so disable dedup here
    add_equivalence_component<ScalarHash<void*>>(this);
}

SymbolVar LoopImpl::InputMaker::make(
        DescImplBase *desc, SymbolVar orig_var, const Param &param) {
    return desc->sub_graph()->insert_opr(std::make_unique<InputMaker>(
                desc, orig_var.node(), param))->output(0);
}

cg::OperatorNodeBase::NodeProp*
LoopImpl::InputMaker::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    if (m_param.has_assign) {
        prop->add_flag(NodeProp::Flag::IMPURE_FUNC);
    } else {
        prop->add_flag(NodeProp::Flag::IMPURE_OUTPUT_MEM_PLAN);
    }
    return prop;
}

void LoopImpl::InputMaker::init_output_mem_plan(bool dynamic) {
    if (!m_param.has_assign) {
        auto dv = m_orig_var->dev_tensor();
        if (output(0)->dev_tensor_valid()) {
            auto&& odv = output(0)->dev_tensor();
            if (dv.raw_ptr() == odv.raw_ptr() &&
                dv.layout().eq_layout(odv.layout())) {
                // mem plan already valid, do not re-init
                return;
            }
        }
        output(0)->init_mem_plan(&dv);
    } else {
        mgb_assert(m_assignor_var);
        Super::init_output_mem_plan(dynamic);
    }
}

void LoopImpl::InputMaker::scn_do_execute() {
    if (!m_param.has_assign) {
        mgb_assert(output(0)->dev_tensor().raw_ptr() ==
                m_orig_var->dev_tensor().raw_ptr());
        return;
    }

    if (m_first_exec) {
        m_first_exec = false;
        output(0)->dev_tensor().
            copy_from_fixlayout(m_orig_var->dev_tensor());
    } else {
        output(0)->dev_tensor().
            copy_from_fixlayout(m_assignor_value);
    }
}

void LoopImpl::InputMaker::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto &&mgr = m_desc->sub_graph_static_infer_helper();

    mgr.register_shape_infer_sub(
            output(0), ShapeInferDesc::make_identity(m_orig_var));

    if (!m_param.disable_value_infer) {
        mgr.register_value_infer_sub(
                output(0), ValueInferDesc::make_identity(m_orig_var));
    }
}

void LoopImpl::InputMaker::commit_assignor() {
    mgb_assert(m_assignor_var && !m_assignor_committed);
    owner_graph()->options().extra_vardeps[output(0)].push_back(
            DepTensorUpdator::make(
                &m_assignor_value, m_assignor_var, output(0)).node());
    m_assignor_committed = true;
}

/* ========= LoopImpl::SubgraphDepIter ========= */
LoopImpl::SubgraphDepIter::SubgraphDepIter():
    m_dep_iter{std::bind(&SubgraphDepIter::dep_iter_cb, this,
            std::placeholders::_1)}
{
}

LoopImpl::SubgraphDepIter::~SubgraphDepIter() noexcept = default;

void LoopImpl::SubgraphDepIter::dep_iter_cb(cg::OperatorNodeBase *opr) {
    m_oprs.push_back(opr);
    if (opr->same_type<InputMaker>()) {
        auto &&im = opr->cast_final<InputMaker>();
        m_input_makers.push_back(&im);
        if (im.param().has_assign) {
            m_unresolved_assignors.push_back(im.assignor());
        }
    }
}

void LoopImpl::SubgraphDepIter::add(VarNode *dest) {
    m_dep_iter.add(dest->owner_opr());
    while (!m_unresolved_assignors.empty()) {
        auto var = m_unresolved_assignors.back();
        m_unresolved_assignors.pop_back();
        m_dep_iter.add(var->owner_opr());
    }
}

void LoopImpl::SubgraphDepIter::sort_input_makers() {
    auto cmp = [](InputMaker *a, InputMaker *b) {
        return a->id() < b->id();
    };
    small_sort(m_input_makers.begin(), m_input_makers.end(), cmp);
    m_input_makers_sorted_size = m_input_makers.size();
}

/* ========= LoopImpl::DepTensorUpdator ========= */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(LoopImpl::DepTensorUpdator);

LoopImpl::DepTensorUpdator::DepTensorUpdator(
        DeviceTensorND *dest,
        const std::shared_ptr<AccumulatorState> &accum_state,
        VarNode *val, VarNode *dep, const OperatorNodeConfig &config):
    Super({val->owner_graph(), config, "dev_tensor_update", {dep}}),
    m_dest{accum_state ? accum_state->dest : dest},
    m_accum_state{accum_state}
{
    mgb_assert(!accum_state || (!dest || dest == accum_state->dest));
    mgb_assert(m_dest);
    add_input({val, dep});
    add_equivalence_component<ScalarHash<void*>>(m_dest);
    if (accum_state)
        add_equivalence_component<ScalarHash<void*>>(accum_state.get());
}

SymbolVar LoopImpl::DepTensorUpdator::make(
        DeviceTensorND *dest, SymbolVar val, SymbolVar dep) {
    return val.insert_single_output_opr<DepTensorUpdator>(
            dest, std::shared_ptr<AccumulatorState>(),
            val.node(), dep.node());
}

SymbolVar LoopImpl::DepTensorUpdator::make(
        const std::shared_ptr<AccumulatorState> &state,
        SymbolVar val, SymbolVar dep) {
    return val.insert_single_output_opr<DepTensorUpdator>(
            nullptr, state, val.node(), dep.node());
}

cg::OperatorNodeBase::NodeProp*
LoopImpl::DepTensorUpdator::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    using D = NodeProp::DepType;
    prop->reset_dep_type(input(), {D::DEV_VALUE, D::DEV_COMP_ORDER});
    return prop;
}

void LoopImpl::DepTensorUpdator::scn_do_execute() {
    auto &&src = input(0)->dev_tensor();
    if (m_accum_state && !m_accum_state->first_sum) {
        mgb_assert(m_dest->shape().eq_shape(src.shape()));
        opr::Elemwise::perform(Elemwise::Mode::ADD,
                *m_dest, {*m_dest, src},
                m_accum_state->adder);
    } else {
        m_dest->copy_from(src);
        if (m_accum_state) {
            m_accum_state->first_sum = false;
        }
    }
}

cg::OperatorNodeBase* LoopImpl::DepTensorUpdator::shallow_copy(
        const VarNodeArray &inputs, const OperatorNodeConfig &config) const {
    mgb_assert(inputs.size() == 2);
    return SymbolVar{inputs[0]}.insert_single_output_opr<DepTensorUpdator>(
            m_dest, m_accum_state, inputs[0], inputs[1], config).
        node()->owner_opr();
}

/* ================= LoopImpl ================= */

namespace {
    OperatorNodeBaseCtorParam replace_opr_ctor_owner(
            OperatorNodeBaseCtorParam p, ComputingGraph *g) {
        mgb_assert(!p.owner && g);
        p.owner = g;
        return p;
    }
}

LoopImpl::LoopImpl(
        const OperatorNodeBaseCtorParam &opr_param,
        std::unique_ptr<DescImplBase> desc):
    Super(replace_opr_ctor_owner(opr_param, desc->owner_graph())),
    m_desc(std::move(desc))
{
    m_desc->set_loop_opr(this);
    add_equivalence_component<ScalarHash<void*>>(this);
}

LoopImpl::~LoopImpl() = default;

cg::ComputingGraph* LoopImpl::get_sub_graph() const {
    return m_desc->sub_graph();
}

cg::OperatorNodeBase::NodeProp* LoopImpl::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    using F = NodeProp::Flag;
    prop->add_flag(F::DISALLOW_COMP_NODE_OPTIMIZE);
    prop->add_flag(F::NO_AUTOMATIC_DUP);
    return prop;
}

void LoopImpl::init_sub_graph_func() {
    if (m_sub_graph_func)
        return;

    m_sub_graph_func = m_desc->compile();

    // check used inputs are actually added
    cg::VarNodeSet inputs(input().begin(), input().end());
    for (auto i: m_desc->cur_func_input())
        mgb_assert(inputs.count(i->orig_var()));
}

void LoopImpl::add_input_in_desc() {
    cg::VarNodeSet input_added;
    for (auto i: m_desc->all_inputs()) {
        auto var = i->orig_var();
        if (input_added.insert(var).second)
            add_input({var});
    }
}

void LoopImpl::add_input_layout_constraint() {
    m_sub_graph_func.reset();
    for (auto i: m_desc->all_inputs()) {
        // InputMakers for assigned inputs would copy to contig; for
        // non-assigned inputs, we need to ensure it is contig so forward can
        // always succeed
        if (!i->param().has_assign)
            i->orig_var()->add_layout_constraint_contiguous();
    }
    m_nr_scn_do_execute_run = 0;
    for (auto &&i: m_desc->output_record_spec()) {
        auto used = owner_graph()->var_receiver_in_current_comp_seq(
                i.var_owner()).value_needed();
        const_cast<OutputRecordSpecItem&>(i).enable(used);
    }
}

void LoopImpl::scn_do_execute() {
    init_sub_graph_func();

    for (auto &&i: m_desc->output_record_spec())
        i.recorder()->on_exec_begin();

    if (auto s = m_mutable_state_saver.get())
        s->on_fwd_begin();

    m_desc->reset_counter_provider();
    auto exec_first = true;
    auto exec = [&]() {
        if (exec_first) {
            exec_first = false;
        } else {
            m_desc->update_counter_provider();
        }
        m_sub_graph_func->execute();
    };

    auto &&cond_manager = m_desc->loop_cond_manager();

    if (m_static_loop_time_infer) {
        // use inferred loop time
        auto nr_loop = m_static_loop_time_infer();
        mgb_assert(nr_loop >= 1);

        if (nr_loop > 1) {
            for (size_t i = 0; i < nr_loop - 1; ++ i) {
                exec();
            }
            mgb_assert(cond_manager.should_loop());
        }
        exec();
        mgb_assert(!cond_manager.should_loop());
    } else {
        for (; ; ) {
            exec();
            if (!cond_manager.should_loop())
                break;
        }
    }

    if (auto s = m_mutable_state_saver.get())
        s->on_fwd_finish();

    for (auto &&i: m_desc->output_record_spec())
        i.recorder()->on_exec_end();

    for (auto &&i: m_desc->cur_func_input())
        i->on_exec_end();

    // sub graph device memory is allocated dynamically, so we clean it ASAP
    m_desc->sub_graph()->clear_device_memory();

    ++ m_nr_scn_do_execute_run;
}

ThinHashMap<VarNode*, bool> LoopImpl::test_get_var_rec_spec() {
    mgb_assert(m_mutable_state_saver.get());
    return m_mutable_state_saver->test_get_var_rec_spec();
}

/* ============== MultidepProxyOperatorNodeBase ============== */

MultidepProxyOperatorNodeBase::MultidepProxyOperatorNodeBase(
        const OperatorNodeBaseCtorParam &opr_param):
    Super(opr_param)
{
    add_output(None)->
        dtype(dtype::Float32()).
        add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE).
        add_flag(VarNode::Flag::VOLATILE_CONTENT);
}

void MultidepProxyOperatorNodeBase::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    owner_graph()->static_infer_manager().register_shape_infer(
            output(0), ShapeInferDesc::make_const({}));
}

/* ============== MutableStateSaver::Recorder ============== */

namespace {
    class LoopRecCopyThreadPool final: public UserDataContainer::UserData {
        MGB_TYPEINFO_OBJ_DECL;

        std::atomic_size_t m_nr_start{0};
        FutureThreadPool<void> m_pool;

        public:
            LoopRecCopyThreadPool(CompNode cn):
                m_pool{"looprec:" + cn.to_string()}
            {
            }

            ~LoopRecCopyThreadPool() {
                if (m_nr_start.load()) {
                    m_pool.stop();
                }
            }

            static LoopRecCopyThreadPool& inst(CompNode cn) {
                auto maker = [cn]() {
                    return std::make_shared<LoopRecCopyThreadPool>(cn);
                };
                return CompNodeEnv::from_comp_node(cn).get_user_data<
                    LoopRecCopyThreadPool>(maker);
            }

            void start() {
                if ((++ m_nr_start) == 1) {
                    m_pool.start(1);
                }
            }

            void stop() {
                auto nr = m_nr_start --;
                mgb_assert(nr);
                if (nr == 1) {
                    m_pool.stop();
                }
            }

            template<typename Func, typename ...Args>
            FutureThreadPool<void>::Future launch(Func&& func, Args&&... args) {
                return m_pool.launch(std::forward<Func>(func),
                        std::forward<Args>(args)...);
            }
    };
} // anonymous namespace
MGB_TYPEINFO_OBJ_IMPL(LoopRecCopyThreadPool);

class LoopImpl::MutableStateSaver::Recorder final: public NonCopyableObj {
    /*!
     * A bucket is a swap buffer; we maintain two buckets, one for current
     * device access, the other for doing copy in the background.
     *
     * We need to be very careful about syncrhonization.
     * There are two basic principles:
     * 1. for each copy a => b, we must ensure reading of buffer on b has
     *    finished and value on a is ready before starting copy;
     * 2. before change the state of an event (i.e. calling record()), we must
     *    ensure all of its waitiers (d2d wait, or host_wait) have finished
     */
    struct Bucket {
        struct EventGroup {
            std::unique_ptr<CompNode::Event>
                //! event for copy stream to wait on computing stream
                comp2copy,
                //! sync between host and device, on LOOP_SWAP stream
                hd;
        };

        //! whether currently copy task in LoopRecCopyThreadPool is running
        std::atomic_bool copy_task_running{false};

        DeviceTensorND buf, buf_on_copy_stream;

        //! host buf with pinned allocation as staging area, for DMA
        HostTensorND buf_host;

        //! whether there is a previous D2H copy
        bool ev_d2h_has_prev = false;

        //! use two event groups to solve sync problems (principle 2)
        EventGroup ev_grp[2];
        int ev_grp_cur = 0;

        std::unique_ptr<dt_byte[]> h2d_copy_refhold;

        FutureThreadPool<void>::Future copy_task;
        bool copy_task_need_wait = false;

        //! whether need to call wait_copy() before overwritting host buffer at
        //! next copy_host_to_bucket() call
        bool h2d_wait_copy_in_next_overwrite = false;

        void init(CompNode comp_node, DType dtype, TensorShape shape,
                int shape0) {
            mgb_assert(shape.ndim + 1 < TensorShape::MAX_NDIM,
                    "tensor shape ndim too large");
            ++ shape.ndim;
            for (size_t i = shape.ndim; i; -- i)
                shape[i] = shape[i - 1];
            shape[0] = shape0;

            buf.comp_node(comp_node).dtype(dtype).resize(shape);
            buf_on_copy_stream = buf;
            auto cn_copy = comp_node;
            if (comp_node.contain_flag(CompNode::Flag::HAS_COPY_STREAM)) {
                cn_copy = comp_node.change_stream(CompNode::Stream::LOOP_SWAP);
            }
            buf_on_copy_stream.comp_node(cn_copy);
            mgb_assert(buf_on_copy_stream.raw_ptr() == buf.raw_ptr());

            if (!buf_host.shape().eq_shape(shape)) {
                buf_host = {};
                buf_host.comp_node(cn_copy).dtype(dtype).resize(shape);
            }

            if (!ev_grp[0].comp2copy) {
                for (int i = 0; i < 2; ++ i) {
                    ev_grp[i].comp2copy = comp_node.create_event();
                    ev_grp[i].hd = cn_copy.create_event();
                }
            } else {
                mgb_assert(ev_grp[0].comp2copy->comp_node() == comp_node);
                mgb_assert(ev_grp[0].hd->comp_node() == cn_copy);
            }
        }

        CompNode::Event& ev_comp2copy() {
            return *ev_grp[ev_grp_cur].comp2copy;
        }

        CompNode::Event& ev_hd() {
            return *ev_grp[ev_grp_cur].hd;
        }

        DeviceTensorND buf_sub(size_t idx) {
            auto subs = Slice(idx, idx + 1).apply(buf.layout(), 0);
            subs = SubTensorSpec::make_from_offset_elem(
                    subs.layout().remove_axis(0), subs.offset_elem());
            return buf.sub(subs);
        }

        void wait_copy() {
            if (copy_task_need_wait) {
                copy_task.get();
                copy_task_need_wait = false;
                mgb_assert(!copy_task_running);
            }
        }
    };

    LoopRecCopyThreadPool &m_copy_threadpool;
    MutableStateSaver * const m_owner_saver;
    SavedVarInfo * const m_saved_var_info;
    Bucket m_buckets[2];
    int m_cur_bucket = 0,   //!< circular counter for current bucket used on dev
        m_cur_bucket_used = 0,  //!< number of slots used in current bucket
        m_swap_interval = 0,
        m_elem_unpop = 0;   //! number of not poped elements

    //! number of elements in  m_saved_buckets to be popped
    size_t m_saved_buckets_pop_remain = 0;
    std::vector<std::unique_ptr<dt_byte[]>> m_saved_buckets;

    //! mutex for m_saved_buckets, used between copy_bucket_to_host() and the
    //! async copy task in m_copy_threadpool
    std::mutex m_saved_buckets_mtx;
    //! see on_fwd_finish()
    TensorShape m_var_shape;
    bool m_enabled = false;

    //!< whether current buffer swap is the first one during grad
    bool m_grad_first_swap = true;

    //! pop m_saved_buckets and copy to target bucket
    void copy_host_to_bucket(Bucket &dest) {
        if (dest.h2d_wait_copy_in_next_overwrite) {
            dest.wait_copy();
            dest.h2d_wait_copy_in_next_overwrite = false;
        }
        mgb_assert(!dest.copy_task_need_wait);

        {
            auto p = dest.copy_task_running.exchange(true);
            mgb_assert(!p);
        }
        mgb_assert(m_saved_buckets_pop_remain);
        dest.h2d_copy_refhold = std::move(
                m_saved_buckets[-- m_saved_buckets_pop_remain]);
        mgb_assert(!dest.buf_host.empty() &&
                dest.buf_host.layout().eq_layout(dest.buf.layout()) &&
                dest.buf_host.layout().eq_layout(
                    dest.buf_on_copy_stream.layout()));

        // wait for current comp stream to finish before overwritting device
        // buffer
        //
        // we need two events because otherwise when copy_host_to_bucket() is
        // called twice, the second record() would overwrite the first one, but
        // previous device_wait in do_copy may have not finished yet
        //
        // we don't need more events because ev_hd->host_wait() would make old
        // events usable again
        int ev_grp_idx = (dest.ev_grp_cur ^= 1);
        dest.ev_comp2copy().record();

        auto do_copy = [&dest, ev_grp_idx]() {
            mgb_assert(ev_grp_idx == dest.ev_grp_cur);
            if (dest.ev_d2h_has_prev) {
                // wait for previous copy to finish before overwritting host
                // buffer
                dest.ev_grp[ev_grp_idx^1].hd->host_wait();
            }
            memcpy(dest.buf_host.raw_ptr(), dest.h2d_copy_refhold.get(),
                    dest.buf_host.layout().span().dist_byte());
            dest.h2d_copy_refhold.reset();

            auto &&ev_grp = dest.ev_grp[ev_grp_idx];
            dest.buf_on_copy_stream.comp_node().device_wait_event(
                    *ev_grp.comp2copy);
            dest.buf_on_copy_stream.copy_from_fixlayout(dest.buf_host);

            // full wait chain (assume ev_grp_idx == 0):
            // ev_hd[1]->host_wait() => (due to device wait above)
            // ev_comp2copy[1] on cn_copy => (due to device wait in pop_value)
            // ev_hd[0] on cn =>
            // ev_hd[0]
            //
            // therefore waiters on ev_grp.hd (especially ev_hd[0] on cn) have
            // finished, so we can record it now
            ev_grp.hd->record();
            dest.ev_d2h_has_prev = true;

            auto p = dest.copy_task_running.exchange(false);
            mgb_assert(p);
        };
        dest.copy_task = m_copy_threadpool.launch(do_copy);
        dest.copy_task_need_wait = true;
    }

    //! copy from given bucket to m_saved_buckets
    void copy_bucket_to_host(Bucket &src) {
        mgb_assert(!src.copy_task_need_wait);
        {
            auto p = src.copy_task_running.exchange(true);
            mgb_assert(!p);
        }
        auto size = src.buf_host.layout().span().dist_byte();
        mgb_assert(size);

        size_t save_bucket_pos = m_saved_buckets.size();
        {
            MGB_LOCK_GUARD(m_saved_buckets_mtx);
            m_saved_buckets.emplace_back();
        }

        // perform copy on dedicated thread pool;
        auto do_copy = [&src, size, save_bucket_pos, this]() {
            // DO NOT use make_unique, since it would call constructors for
            // elements, making it very slow
            auto ptr = new dt_byte[size];

            {
                MGB_LOCK_GUARD(m_saved_buckets_mtx);
                m_saved_buckets[save_bucket_pos].reset(ptr);
            }

            src.buf_on_copy_stream.comp_node().device_wait_event(
                    src.ev_comp2copy());
            src.buf_host.copy_from_fixlayout(src.buf_on_copy_stream);
            src.ev_hd().record();
            src.ev_hd().host_wait();
            memcpy(ptr, src.buf_host.raw_ptr(), size);

            auto p = src.copy_task_running.exchange(false);
            mgb_assert(p);
        };
        src.copy_task = m_copy_threadpool.launch(do_copy);
        src.copy_task_need_wait = true;
    }

    //! pop the last saved value for grad computing
    DeviceTensorND pop_value() {
        mgb_assert(m_elem_unpop > 0);
        -- m_cur_bucket_used;
        if (m_cur_bucket_used < 0) {
            if (m_elem_unpop >= m_swap_interval * 2) {
                // speculative copy for values needed in the future
                copy_host_to_bucket(m_buckets[m_cur_bucket]);
            } else {
                mgb_assert(!m_saved_buckets_pop_remain);
            }

            m_cur_bucket_used = m_swap_interval - 1;
            m_cur_bucket ^= 1;

            auto &&bucket = m_buckets[m_cur_bucket];
            if (m_grad_first_swap) {
                // device value is ready at the first swap
                // wait for fwd d2h copy before next overwritting
                bucket.h2d_wait_copy_in_next_overwrite = true;
                m_grad_first_swap = false;
            } else {
                if (bucket.copy_task_running.load()) {
                    m_owner_saver->print_slowcopy_warn(ssprintf(
                                "grad at %d remaining", m_elem_unpop).c_str());
                }
                bucket.wait_copy();
                bucket.buf.comp_node().device_wait_event(bucket.ev_hd());
            }
        }

        -- m_elem_unpop;
        return m_buckets[m_cur_bucket].buf_sub(m_cur_bucket_used);
    }

    public:
        class ReplayOpr;

        Recorder(MutableStateSaver *owner_saver, SavedVarInfo *saved_var_info):
            m_copy_threadpool{
                LoopRecCopyThreadPool::inst(
                        owner_saver->m_owner_opr->input(0)->comp_node())},
            m_owner_saver{owner_saver}, m_saved_var_info{saved_var_info}
        {
            m_copy_threadpool.start();
        }

        ~Recorder() {
            m_copy_threadpool.stop();
        }

        MutableStateSaver * owner_saver() const {
            return m_owner_saver;
        }

        bool enabled() const {
            return m_enabled;
        }

        void enable(bool flag) {
            m_enabled = flag;
        }

        void setup_for_record(int swap_interval) {
            mgb_assert(!m_var_shape.ndim, "on_grad_finish() not called");
            m_swap_interval = swap_interval;
        }

        void on_val_produced(const DeviceTensorND& val) {
            // always record shape, since it may be needed during grad
            on_shape_produced(val.shape());

            if (!m_enabled)
                return;

            mgb_assert(m_swap_interval > 0, "setup_for_record() not called");

            if (m_cur_bucket_used == m_swap_interval) {
                // bucket full, copy to host and swap
                copy_bucket_to_host(m_buckets[m_cur_bucket]);
                m_cur_bucket ^= 1;
                m_cur_bucket_used = 0;
            }

            auto &&bucket = m_buckets[m_cur_bucket];
            auto comp_node = val.comp_node();
            if (!m_cur_bucket_used) {
                if (bucket.copy_task_running.load()) {
                    m_owner_saver->print_slowcopy_warn(ssprintf(
                                "fwd at %d", m_elem_unpop).c_str());
                } else if (bucket.buf.empty()) {
                    bucket.init(comp_node, val.dtype(), m_var_shape,
                            m_swap_interval);
                }
                bucket.wait_copy();
            }
            mgb_assert(bucket.buf.comp_node() == comp_node);
            bucket.buf_sub(m_cur_bucket_used).copy_from_fixlayout(val);
            ++ m_cur_bucket_used;
            if (m_cur_bucket_used == m_swap_interval) {
                // waited in copy_bucket_to_host() at next call
                bucket.ev_comp2copy().record();
            }
            ++ m_elem_unpop;
        }

        void on_shape_produced(const TensorShape &shape) {
            if (!m_var_shape.ndim)
                m_var_shape = shape;
            else
                mgb_assert(m_var_shape.eq_shape(shape));
        }

        //! get recorded value at given grad iter
        SymbolVar get_var_for_replay(SymbolVar counter);

        void on_fwd_finish() {
            if (!m_enabled)
                return;

            // the last saved bucket should be dropped, since its value exists
            // on the other swap buffer on device
            m_saved_buckets_pop_remain =
                std::max<size_t>(m_saved_buckets.size(), 1) - 1;
        }

        void on_grad_finish() {
            mgb_assert(!m_saved_buckets_pop_remain && !m_elem_unpop,
                    "grad opr not executed");
            m_var_shape.ndim = 0;
            if (!m_enabled)
                return;

            m_cur_bucket = m_cur_bucket_used = 0;
            m_grad_first_swap = true;
            for (auto &&i: m_buckets) {
                i.wait_copy();
                i.h2d_wait_copy_in_next_overwrite = false;
                i.ev_d2h_has_prev = false;
                mgb_assert(!i.copy_task_running.load() &&
                        !i.copy_task_need_wait);
                i.buf = {};
                i.buf_on_copy_stream = {};
            }
            m_saved_buckets.clear();
            m_swap_interval = 0;
        }

        SavedVarInfo* saved_var_info() const {
            return m_saved_var_info;
        }
};

MGB_DEFINE_OPR_CLASS(LoopImpl::MutableStateSaver::Recorder::ReplayOpr,
        cg::SingleCNOperatorNodeBase) // {

    int m_prev_idx = -1;
    const void *m_expected_dev_ptr = nullptr;
    Recorder * const m_owner_recorder;

    int get_counter() {
        auto &&mgr = owner_graph()->static_infer_manager();
        auto &&iv = mgr.infer_value(input(0));
        mgb_assert(iv.shape().is_scalar());
        return iv.ptr<int>()[0];
    }

    NodeProp* do_make_node_prop() const override {
        auto prop = Super::do_make_node_prop();
        using DT = NodeProp::DepType;
        prop->reset_dep_type(input(), {DT::HOST_VALUE | DT::HOST_VALUE_DYNOUT});
        return prop;
    }

    void init_output_mem_plan(bool dynamic) override {
        mgb_assert(dynamic);
        m_prev_idx = get_counter();
        auto val = m_owner_recorder->pop_value();
        output(0)->init_mem_plan(&val);
        m_expected_dev_ptr = val.raw_ptr();
        mgb_assert(!output(0)->contain_flag(VarNode::Flag::NO_MEM_RECLAIM));
    }

    void init_output_static_infer_desc() override {
        using namespace cg::static_infer;
        auto infer_shp = [this](TensorShape &dest, const InpVal &) -> bool {
            dest = m_owner_recorder->m_var_shape;
            return dest.ndim;
        };
        owner_graph()->static_infer_manager().register_shape_infer(
                output(0), {SourceType::MUTABLE, {}, infer_shp});
    }

    void scn_do_execute() override {
        mgb_assert(m_prev_idx == get_counter());
        mgb_assert(m_expected_dev_ptr == output(0)->dev_tensor().raw_ptr());
    }

    public:

        ReplayOpr(Recorder *recorder, VarNode *counter):
            Super(counter->owner_graph(),
                    OperatorNodeConfig{ssprintf("replay(%s)",
                            recorder->saved_var_info()->var->cname())}, "",
                    {counter}),
            m_owner_recorder{recorder}
        {
            add_input({counter});
            add_output(None)->dtype(recorder->saved_var_info()->var->dtype());
            add_equivalence_component<ScalarHash<Recorder*>>(recorder);
        }

        Recorder* owner_recorder() const {
            return m_owner_recorder;
        }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(LoopImpl::MutableStateSaver::Recorder::ReplayOpr);

SymbolVar LoopImpl::MutableStateSaver::Recorder::get_var_for_replay(
        SymbolVar counter) {
    return counter.insert_single_output_opr<ReplayOpr>(this, counter.node());
}

/* ============== MutableStateSaver ============== */

MGB_DEFINE_OPR_CLASS(LoopImpl::MutableStateSaver::ValueUpdator,
        MultidepProxyOperatorNodeBase) // {
    Recorder * const m_recorder;

    void scn_do_execute() override {
        m_recorder->on_val_produced(input(0)->dev_tensor());
    }

    bool update_priority() const override {
        node_prop().attribute().priority = std::numeric_limits<int>::min();
        return true;
    }

    public:

        ValueUpdator(VarNode *inp, Recorder *recorder):
            Super({inp->owner_graph(), {}, "record_val", {inp}}),
            m_recorder{recorder}
        {
            add_input({inp});
        }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(LoopImpl::MutableStateSaver::ValueUpdator);

MGB_DEFINE_OPR_CLASS(LoopImpl::MutableStateSaver::ShapeUpdator,
        MultidepProxyOperatorNodeBase) // {
    Recorder * const m_recorder;

    NodeProp* do_make_node_prop() const override {
        auto prop = MultidepProxyOperatorNodeBase::do_make_node_prop();
        prop->reset_dep_type(input(), {NodeProp::DepType::SHAPE});
        return prop;
    }

    void scn_do_execute() override {
        m_recorder->on_shape_produced(input(0)->shape());
    }

    public:

        ShapeUpdator(VarNode *inp, Recorder *recorder):
            Super({inp->owner_graph(), {}, "record_val", {inp}}),
            m_recorder{recorder}
        {
            add_input({inp});
        }
};
MGB_DYN_TYPE_OBJ_FINAL_IMPL(LoopImpl::MutableStateSaver::ShapeUpdator);


LoopImpl::MutableStateSaver::MutableStateSaver(Loop *owner_opr):
    m_owner_opr{owner_opr}
{
}

LoopImpl::MutableStateSaver::~MutableStateSaver() = default;

VarNode*
LoopImpl::MutableStateSaver::get_user_recorded_output_all(VarNode *var) {
    auto &&v2r = static_cast<FwdDesc*>(m_owner_opr->m_desc.get())->
        output_record_spec_mode_all();
    auto iter = v2r.find(var);
    if (iter != v2r.end())
        return iter->second->var_owner();
    return nullptr;
}

void LoopImpl::MutableStateSaver::add_var_to_record(VarNode *var) {
    auto ins = m_recorded_vars.insert(var);
    mgb_assert(ins.second);

    if (get_user_recorded_output_all(var))
        return;

    SavedVarInfo &info = m_var2info[var];
    info.var = var;
    info.recorder.reset(new Recorder{this, &info});
    info.value_updator = SymbolVar{var}.
        insert_single_output_opr<ValueUpdator>(
                var, info.recorder.get());
    info.shape_updator = SymbolVar{var}.
        insert_single_output_opr<ShapeUpdator>(
                var, info.recorder.get());
}

void LoopImpl::MutableStateSaver::disable() {
    m_enabled = false;
    for (auto &&i: m_var2info) {
        i.second.recorder->enable(false);
        i.second.need_shape = false;
        i.second.need_value = false;
    }
}

void LoopImpl::MutableStateSaver::enable_for_grad(cg::AsyncExecutable *seq) {
    mgb_assert(!m_enabled, "multiple loop grads currently not supported");
    m_enabled = true;
    auto cb_val = [this](cg::OperatorNodeBase *opr) {
        if (opr->same_type<Recorder::ReplayOpr>()) {
            auto rec = opr->cast_final<Recorder::ReplayOpr>().owner_recorder();
            rec->saved_var_info()->need_value = true;
            mgb_assert(rec->owner_saver() == this);
            rec->enable(true);
        }
        return true;
    };
    seq->iter_opr_seq(cb_val);

    for (auto &&i: seq->get_rt_static_source_deps()) {
        if (i.dest->owner_opr()->same_type<Recorder::ReplayOpr>()) {
            mgb_assert(i.type == cg::static_infer::DepType::SHAPE);
            auto rec = i.dest->owner_opr()->cast_final<
                Recorder::ReplayOpr>().owner_recorder();
            mgb_assert(rec->owner_saver() == this);
            auto info = rec->saved_var_info();
            if (!info->need_value) {
                info->need_shape = true;
            }
        }
    }
}

VarNode* LoopImpl::MutableStateSaver::get_state_for_grad(
        VarNode *fwd_var, DescImplBase *grad_desc) {
    if (auto rec_all = get_user_recorded_output_all(fwd_var)) {
        // reuse all hist recorded by user
        auto all_hist = grad_desc->add_input(rec_all, false);
        return opr::IndexAt::make(all_hist,
                {{0, grad_desc->get_counter_var()}}).node();
    }
    return m_var2info.at(fwd_var).recorder->get_var_for_replay(
            grad_desc->get_counter_var()).node();
}

void LoopImpl::MutableStateSaver::update_subgraph_outspec(
        ComputingGraph::OutputSpec &spec) {
    for (auto &&i: m_var2info) {
        if (i.second.need_value) {
            spec.push_back({i.second.value_updator, {}});
        } else if (i.second.need_shape) {
            spec.push_back({i.second.shape_updator, {}});
        }
    }
}

void LoopImpl::MutableStateSaver::on_fwd_begin() {
    if (!m_enabled)
        return;

    int swap_interval = m_swap_interval_setting;
    if (m_owner_opr->m_static_loop_time_infer) {
        int infer = m_owner_opr->m_static_loop_time_infer();
        if (swap_interval < 0)
            swap_interval = infer;
        else
            swap_interval = std::min(swap_interval, infer);
    } else {
        if (swap_interval < 0)
            swap_interval = -swap_interval;
    }
    mgb_assert(swap_interval > 0);
    for (auto &&i: m_var2info)
        i.second.recorder->setup_for_record(swap_interval);
}

void LoopImpl::MutableStateSaver::on_fwd_finish() {
    if (!m_enabled)
        return;
    for (auto &&i: m_var2info)
        i.second.recorder->on_fwd_finish();
}

void LoopImpl::MutableStateSaver::on_grad_finish() {
    mgb_assert(m_enabled);
    for (auto &&i: m_var2info)
        i.second.recorder->on_grad_finish();
}

void LoopImpl::MutableStateSaver::print_slowcopy_warn(const char *msg) {
    if (m_slowcopy_warn_printed)
        return;
    mgb_log_warn("Loop %s: %s: copy not finished when new value becomes "
            "available; consider increase swap_interval (cur setting: %d); "
            "this warning would be presented only once",
            m_owner_opr->cname(), msg, m_swap_interval_setting);
    m_slowcopy_warn_printed = true;
}

ThinHashMap<VarNode*, bool>
LoopImpl::MutableStateSaver::test_get_var_rec_spec() {
    ThinHashMap<VarNode*, bool> ret;
    for (auto &&i: m_var2info)
        ret[i.first] = i.second.recorder->enabled();
    return ret;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
