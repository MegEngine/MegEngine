/**
 * \file src/opr/impl/utility.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/graph/grad_impl.h"
#include "megbrain/graph/event.h"
#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/graph/operator_node.h"
#include "megbrain/utils/debug.h"
#include "megbrain/opr/utility.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/comp_node_env.h"

#include <thread>

using namespace mgb;
using namespace opr;

#if !MGB_BUILD_SLIM_SERVING

namespace {
OperatorNodeConfig setup_config_cn(const OperatorNodeConfig& config_,
                                   const CompNode& cn) {
    auto prev_cn = config_.get_single_comp_node();
    mgb_assert(!prev_cn.valid() || cn == prev_cn);
    auto config = config_;
    config.comp_node(cn);
    return config;
}
}  // namespace

/* ===================== Sleep ===================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Sleep);

void Sleep::scn_do_execute() {
#if MGB_HAVE_THREAD
    auto in = input(0), out = output(0);
    if (m_type.device) {
        if (!m_opr || m_opr.comp_node() != comp_node()) {
            m_opr = intl::create_megdnn_opr<megdnn::Sleep>(comp_node());
        }
        m_opr->param().time = m_seconds;
        m_opr->exec();
    }
    if (m_type.host) {
        std::this_thread::sleep_for(std::chrono::microseconds(
                    static_cast<uint64_t>(m_seconds * 1e6)));
    }
    out->dev_tensor().copy_from_fixlayout(in->dev_tensor());
#else
    mgb_throw(MegBrainError, "sleep is unavilable when threading is disabled");
#endif
}

void Sleep::record_execute_deps(ExecDependencyArray& deps) {
    if (m_opr) {
        mixin::MegDNNOprHolder::record_megdnn_opr(std::move(m_opr), deps);
    }
}

void Sleep::sleep(const CompNode &node, double seconds) {
    node.activate();
    auto opr = intl::get_megdnn_handle(node)->create_operator<megdnn::Sleep>();
    opr->param().time = seconds;
    opr->exec();
}

Sleep::Sleep(VarNode *node, double seconds, Type type,
        const OperatorNodeConfig &config):
    Super(node->owner_graph(), config, "sleep", {node}),
    m_seconds{seconds}, m_type{type}
{
    mgb_assert(seconds > 0);
    add_input({node});
    add_output(None);
    add_equivalence_component<PODHash<double>>(&m_seconds);
    add_equivalence_component<PODHash<Type>>(&m_type);
}

SymbolVar Sleep::make(SymbolVar node, double seconds, Type type,
        const OperatorNodeConfig &config) {
    mgb_assert(seconds >= 0);
    if (!seconds)
        return node;
    return node.insert_single_output_opr<Sleep>(node.node(),
            seconds, type, config);
}

MGB_IMPL_OPR_GRAD(Sleep) {
    return out_grad.at(0);
}

/* ===================== Timestamp ===================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(Timestamp);

class Timestamp::GraphStorage final : public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;

    //! whether oprs and event info should be cleared upon next register call
    bool m_should_clear = false;

    SyncEventConnecter::ReceiverHandler m_recv_handler_wait,
            m_recv_handler_compile;
    std::vector<Timestamp*> m_oprs;
    CompNode::UnorderedMap<CompNode::Event*> m_first_event;

public:
    GraphStorage(ComputingGraph* cg) {
        auto on_compile = [this](const cg::event::CompSeqOrderDetermined&) {
            m_should_clear = true;
        };
        auto on_wait = [this](const cg::event::CompSeqExecFinished& event) {
            for (auto i : m_oprs) {
                i->update();
            }
            mgb_assert(event.device_actually_finished,
                       "Timestamp in subgraph is not supported");
        };
        m_recv_handler_compile =
                cg->event()
                        .register_receiver<cg::event::CompSeqOrderDetermined>(
                                on_compile);
        m_recv_handler_wait =
                cg->event().register_receiver<cg::event::CompSeqExecFinished>(
                        on_wait);
    }

    //! return the first event on this comp seq
    CompNode::Event* register_opr(Timestamp* opr) {
        if (m_should_clear) {
            m_oprs.clear();
            m_first_event.clear();
            m_should_clear = true;
        }
        m_oprs.push_back(opr);
        auto ins = m_first_event.insert({opr->comp_node(), opr->m_event.get()});
        return ins.first->second;
    }
};
MGB_TYPEINFO_OBJ_IMPL(Timestamp::GraphStorage);

void Timestamp::add_input_layout_constraint() {
    if (!m_event) {
        m_event = comp_node().create_event(CompNode::Event::Flags::NEED_TIMER);
    }
    auto make = [this]() {
        return std::make_shared<GraphStorage>(owner_graph());
    };
    auto storage =
            owner_graph()
                    ->options()
                    .user_data.get_user_data_or_create<GraphStorage>(make);
    m_first_event = storage->register_opr(this);
    Super::add_input_layout_constraint();
}

void Timestamp::scn_do_execute_finish(const DeviceTensorND&) {
    m_event->record();
}
void Timestamp::on_output_comp_node_stream_changed() {
    m_event.reset();
    Super::on_output_comp_node_stream_changed();
}

void Timestamp::update() {
    mgb_assert(m_dest_off < m_dest->shape(0));
    m_dest->ptr<float>()[m_dest_off] =
            m_first_event->elapsed_time_until(*m_event);
}

Timestamp::Timestamp(VarNode* node, std::shared_ptr<HostTensorND> dest,
                     size_t dest_off, const OperatorNodeConfig& config)
        : Super(node->owner_graph(), config, "timestamp", {node}),
          m_dest{std::move(dest)},
          m_dest_off{dest_off} {
    mgb_assert(m_dest, "empty dest tensor");
    mgb_assert(m_dest->dtype() == dtype::Float32{} &&
                       m_dest->shape().ndim == 1 &&
                       dest_off < m_dest->shape()[0] &&
                       m_dest->layout().stride[0] == 1,
               "dest tensor must be 1-dimensional float32; got %s (%s)",
               m_dest->layout().to_string().c_str(), m_dest->dtype().name());
    add_input({node});
    add_output(None);
    add_equivalence_component<ScalarHash<void*>>(m_dest.get());
    add_equivalence_component<ScalarHash<size_t>>(m_dest_off);
}

SymbolVar Timestamp::make(SymbolVar node, std::shared_ptr<HostTensorND> dest,
                          size_t dest_off, const OperatorNodeConfig& config) {
    return node.insert_single_output_opr<Timestamp>(
            node.node(), std::move(dest), dest_off, config);
}

/* ========================== VirtualDep ============================ */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(VirtualDep);

VirtualDep::VirtualDep(const VarNodeArray& inputs,
                       const OperatorNodeConfig& cfg)
        : Super(inputs[0]->owner_graph(),
                cfg.has_comp_node_set() ? cfg : setup_config_cn(cfg, inputs[0]->comp_node()),
                "virtual_dep", inputs) {
    for (auto inp : inputs) {
        add_input({inp});
    }
    mgb_assert(inputs[0]->dtype().valid());
    add_output(None)
            ->dtype(inputs[0]->dtype())
            .comp_node(config().get_single_comp_node());
}

cg::OperatorNodeBase::NodeProp* VirtualDep::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    if (input().size() > 1) {
        SmallVector<NodeProp::DepType> dep_types{NodeProp::DepType::DEV_VALUE};
        for (size_t i = 1; i < input().size(); ++i) {
            dep_types.push_back(NodeProp::DepType::DEV_COMP_ORDER);
        }
        prop->reset_dep_type(input(), dep_types);
    }
    prop->add_flag(
            cg::OperatorNodeBase::NodeProp::Flag::CROSS_COMP_NODE_MEMORY);
    return prop;
}

SymbolVar VirtualDep::make(const SymbolVarArray& inputs,
                           const OperatorNodeConfig& config) {
    mgb_assert(!inputs.empty());
    auto nodes = to_var_node_array(inputs);
    return inputs[0].insert_single_output_opr<VirtualDep>(nodes, config);
}
MGB_IMPL_OPR_GRAD(VirtualDep) {
    if (wrt_idx == 0) {
      return out_grad.at(0);
    }
    return nullptr;
}
#endif  // MGB_BUILD_SLIM_SERVING

/* ===================== MarkDynamicVar ===================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MarkDynamicVar);

void MarkDynamicVar::scn_do_execute() {
    auto i = input(0), o = output(0);
    o->shape_alloc(i->shape());
    o->dev_tensor().copy_from_fixlayout(i->dev_tensor());
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(MarkDynamicVar) {
    return MarkDynamicVar::make(out_grad.at(0)).node();
}
#endif

MarkDynamicVar::MarkDynamicVar(VarNode *node, const OperatorNodeConfig &config):
    Super{node->owner_graph(), config, "mark_dyn", {node}}
{
    add_input({node});
    add_output(None)
            ->add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC)
            .add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
}

SymbolVar MarkDynamicVar::make(
        SymbolVar node, const OperatorNodeConfig &config) {
    return node.insert_single_output_opr<MarkDynamicVar>(node.node(), config);
}

MarkDynamicVar::NodeProp* MarkDynamicVar::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_dep_type_existing_var(input(0),
                                   NodeProp::DepType::VALUE_ALLOW_EMPTY);
    return ret;
}

/* ===================== CallbackInjector ===================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(CallbackInjector);

CallbackInjector::CallbackInjector(
        VarNode *inp, const Param &param, const OperatorNodeConfig &config):
    Super{inp->owner_graph(), config, "callback", {inp}},
    m_param{param}
{
    add_input({inp});
    add_output(None);

    if (m_param.ignore_side_effect) {
        set_ignore_side_effect();
    }

    // so this opr would not get deduped
    add_equivalence_component<ScalarHash<void*>>(this);
}

CallbackInjector::CallbackInjector(
        VarNodeArray& inps,
        const Param &param,
        const OperatorNodeConfig &config):
        Super{inps[0]->owner_graph(), config, "callback", inps}, m_param{param}
{
    for (auto inp : inps) {
        add_input({inp});
    }
    add_output(None);

    if (m_param.ignore_side_effect) {
        set_ignore_side_effect();
    }

    // so this opr would not get deduped
    add_equivalence_component<ScalarHash<void*>>(this);
}

SymbolVar CallbackInjector::make(mgb::cg::SymbolVarArray inp, const Param &param,
                                 const OperatorNodeConfig &config) {
    auto nodes = to_var_node_array(inp);
    return inp[0].insert_single_output_opr<CallbackInjector>(nodes, param, config);
}


void CallbackInjector::scn_do_execute_finish(const DeviceTensorND &val) {
    SmallVector<DeviceTensorND> input_list = {};
    for(size_t i = 0; i < input().size(); ++i) {
        input_list.push_back(input(i)->dev_tensor());
    }
    m_param.callback(const_cast<SmallVector<DeviceTensorND>&>(input_list));
}

cg::OperatorNodeBase::NodeProp* CallbackInjector::do_make_node_prop() const {
    auto prop = ForwardInputToOutput::do_make_node_prop();
    if (!m_param.allow_auto_dup) {
        prop->add_flag(NodeProp::Flag::NO_AUTOMATIC_DUP);
    }
    return prop;
}

cg::static_infer::ValueInferDesc
CallbackInjector::mixin_get_static_infer_desc(OperatorNodeBase &opr) {
    using namespace cg::static_infer;
    auto infer_val = [this](DeviceTensorND& dst, const InpVal& iv) -> bool {
        dst = iv.val[0].value();
        if (!m_param.invoke_for_static_infer) {
            return true;
        }
        if (m_warn_printed < 10) {
            mgb_log_warn(
                    "[warn %d/10] CallbackInjector %s is called during static "
                    "value inference. The warning can be safely ignored if "
                    "CallbackInjector does nothing other than inspecting the "
                    "tensor value; otherwise it may introduce unexpected "
                    "behavior.",
                    ++m_warn_printed, cname());
        }
        SmallVector<DeviceTensorND> callback_list =  {};
        for (size_t i = 0; i < iv.val.size(); ++i) {
            if (m_append_one_more_shape and i + 1== iv.val.size()) {
                continue;
            }
            callback_list.push_back(iv.val[i].value());
        }
        m_param.callback(callback_list);
        return true;
    };

    DepVal dep_val_list = {};
    for (size_t i = 0; i < input().size(); ++i) {
        dep_val_list.push_back({opr.input(i), DepType::VALUE});
    }
    if (m_param.invoke_for_static_infer) {
        return {SourceType::DEP, {{opr.input(0), DepType::VALUE}}, infer_val};
    } else {
        return {SourceType::DEP, dep_val_list, infer_val};
    }
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(CallbackInjector) {
    MGB_MARK_USED_VAR(wrt_idx);
    return out_grad.at(0);
}
#endif

/* ===================== MarkNoBroadcastElemwise ===================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(MarkNoBroadcastElemwise);

MarkNoBroadcastElemwise::MarkNoBroadcastElemwise(
        VarNode* input, const OperatorNodeConfig &config):
    Super(input->owner_graph(), config, "no_brdcst", {input})
{
    add_input({input});
    add_output(None);
    set_ignore_side_effect();
}

SymbolVar MarkNoBroadcastElemwise::make(
        SymbolVar input, const OperatorNodeConfig &config) {
    return input.insert_single_output_opr<MarkNoBroadcastElemwise>(
            input.node(), config);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(MarkNoBroadcastElemwise) {
    return out_grad.at(0);
}
#endif

/* ===================== Identity ===================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(Identity);

Identity::Identity(VarNode* input, const OperatorNodeConfig &config):
    Super(input->owner_graph(), config, "identity", {input})
{
    add_input({input});
    add_output(None);
    set_ignore_side_effect();
}

SymbolVar Identity::make(
        SymbolVar input, const OperatorNodeConfig &config) {
    if (input.node()->owner_opr()->same_type<Identity>()) {
        // collapse consecutive Identity oprs
        // this is also necessary for megskull GradWrt in loop to work
        return input;
    }
    return input.insert_single_output_opr<Identity>(input.node(), config);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Identity) {
    return out_grad.at(0);
}
#endif

/* ===================== AssertEqual ===================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(AssertEqual);

AssertEqual::AssertEqual(
        VarNode *expect, VarNode *get, VarNode *err,
        const Param &param, const OperatorNodeConfig &config):
    Super(err->owner_graph(), config, "assert_eq", {expect, get}),
    m_param{param}
{
    add_input({expect, get, err});
    add_output(None);
    add_equivalence_component<PODHash<Param>>(&m_param);
}

SymbolVar AssertEqual::make(SymbolVar expect, SymbolVar get,
        const Param &param, const OperatorNodeConfig &config) {
    auto err = opr::reduce_max(
            opr::abs(expect - get) /
            opr::max(
                opr::min(opr::abs(expect), opr::abs(get)),
                expect.make_scalar_dt(1)),
            expect.make_scalar(1));
    return make(expect, get, err, param, config);
}

SymbolVar AssertEqual::make(
        SymbolVar expect, SymbolVar get, SymbolVar err,
        const Param &param, const OperatorNodeConfig &config) {
    return expect.insert_single_output_opr<AssertEqual>(
            expect.node(), get.node(), err.node(), param, config);
}

void AssertEqual::scn_do_execute_finish(const DeviceTensorND &) {
    if (owner_graph()->options().comp_node_seq_record_level >= 2) {
        mgb_log_error("AssertEqual %s disabled due to seq rec", cname());
        return;
    }
    m_hv.copy_from(input(2)->dev_tensor()).sync();
    mgb_assert(m_hv.shape().is_scalar());
    auto err = DTypeScalar::make_from_raw(
            m_hv.dtype(), m_hv.raw_ptr()).get_cast<float>();
    if (m_param.verbose) {
        //! FIXME: stderr will be slow when build windows with VS clang-cl (test in VM),
        //! but I can`t find the root case. fix it when you figure out
        fprintf(stdout,
                "AssertEqual: err=%g (name=%s id=%zu)\n", err, cname(), id());
    }
    if (!(err >= 0 && err <= m_param.maxerr)) {
        HostTensorND expect, get;
        expect.copy_from(input(0)->dev_tensor());
        get.copy_from(input(1)->dev_tensor()).sync();
        auto msg = debug::compare_tensor_value(
                expect, cg::dump_var_info({input(0)}).c_str(),
                get, cg::dump_var_info({input(1)}).c_str(),
                m_param.maxerr);
        mgb_assert(msg.valid());
        if (m_throw_on_error) {
            owner_graph()->record_async_error(
                cg::OperatorNodeExcExtraInfo::ExcMaker{
                    input(1)->owner_opr()}.make_unique<UnequalError>(msg.val()));
        } else {
            mgb_log_error("%s", msg->c_str());
        }
    }
}

#if MGB_ENABLE_GRAD
/* ===================== SetGrad ===================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(SetGrad);

SetGrad::SetGrad(
        VarNode* input, const GradGetter& grad_getter,
        const OperatorNodeConfig &config):
    Super(input->owner_graph(), config, "set_grad", {input}),
    m_grad_getter{grad_getter}
{
    add_input({input});
    add_output(None);
    set_ignore_side_effect();

    if (grad_getter) {
        // dedup not allowed
        add_equivalence_component<ScalarHash<void*>>(this);
    } else {
        // force to be zero_grad if no callback, and we can safely enable dedup
        m_grad_getter = zero_grad;
    }
}

SymbolVar SetGrad::make(SymbolVar input, const GradGetter& grad_getter,
        const OperatorNodeConfig &config) {
    return input.insert_single_output_opr<SetGrad>(
            input.node(), grad_getter, config);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(SetGrad) {
    MGB_MARK_USED_VAR(wrt_idx);
    MGB_MARK_USED_VAR(out_grad);
    auto grad = opr.grad_getter()(opr);
    mgb_assert(!grad.node() || grad.node()->owner_graph() == opr.owner_graph(),
            "var returned by grad_getter belongs to a different comp graph");
    return grad.node();
}
#endif

/* ===================== InvalidGrad ===================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(InvalidGrad);

void InvalidGrad::scn_do_execute() {
    mgb_assert(0);
}

InvalidGrad::InvalidGrad(VarNode* vinp, const OperatorNodeBase* grad_opr,
                         size_t inp_idx)
        : Super{vinp->owner_graph(), {}, "invalid_grad", {vinp}},
          m_grad_opr(grad_opr),
          m_inp_idx(inp_idx) {
    add_input({vinp});
    add_output(None);
}

void InvalidGrad::add_input_layout_constraint() {
    MGB_MARK_USED_VAR(m_grad_opr);
    mgb_throw(GraphError,
              "invalid grad: can not take grad with respect to the %zu'th "
              "input var of operator {id:%zu, name:%s, type:%s}; "
              "(w.r.t. var: %s)",
              m_inp_idx, m_grad_opr->id(), m_grad_opr->cname(),
              m_grad_opr->dyn_typeinfo()->name,
              cg::dump_var_info(input()).c_str());
}

VarNode* InvalidGrad::make(const OperatorNodeBase& grad_opr, size_t inp_idx) {
    return SymbolVar(grad_opr.input(inp_idx))
            .insert_single_output_opr<InvalidGrad>(grad_opr.input(inp_idx),
                                                   &grad_opr, inp_idx)
            .node();
}

/* ===================== VirtualGrad ===================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(VirtualGrad);

VirtualGrad::VirtualGrad(VarNode *target, VarNode *wrt,
        const OperatorNodeConfig &config):
    Super(target->owner_graph(), config, "grad", {target, wrt})
{
    add_input({target, wrt});
    add_output(None)->dtype(wrt->dtype());
}

SymbolVar VirtualGrad::make(SymbolVar target, SymbolVar wrt,
        Param, const OperatorNodeConfig &config) {
    return target.insert_single_output_opr<VirtualGrad>(
            target.node(), wrt.node(), config);
}

void VirtualGrad::do_execute(ExecEnv &) {
    mgb_throw(MegBrainError, "VirtualGrad opr must be removed by "
            "gopt::ExpandVirtualGradPass");
}

void VirtualGrad::init_output_comp_node() {
    output(0)->comp_node(input(1)->comp_node());
}

void VirtualGrad::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto &&mgr = owner_graph()->static_infer_manager();
    auto ovar = output(0), ivar = input(1);
    mgr.register_shape_infer(ovar, ShapeInferDesc::make_identity(ivar));
}

void VirtualGrad::on_output_comp_node_stream_changed() {
}

VirtualGrad::NodeProp* VirtualGrad::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_flag(NodeProp::Flag::CROSS_COMP_NODE_MEMORY);
    return ret;
}

/* ===================== VirtualLoss ===================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(VirtualLoss);

VirtualLoss::VirtualLoss(const VarNodeArray& inputs,
                         const OperatorNodeConfig& config)
        : Super(inputs.at(0)->owner_graph(), config, "internal_grad",
                {inputs.at(0)}) {
    mgb_assert(inputs.size() % 2 == 0);
    for (size_t i = 0, it = inputs.size() / 2; i < it; ++i) {
        auto yi = inputs[i], gradi = inputs[i + it];
        mgb_assert(yi && gradi);
        auto&& shp0 = yi->shape();
        auto&& shp1 = gradi->shape();
        mgb_assert((!shp0.ndim && !shp1.ndim) || shp0.eq_shape(shp1),
                   "grad shape mismatch: %s vs %s", shp0.to_string().c_str(),
                   shp1.to_string().c_str());
        mgb_assert(yi->comp_node() == gradi->comp_node());
        add_input({yi});
    }
    for (size_t i = inputs.size() / 2; i < inputs.size(); ++i) {
        add_input({inputs[i]});
    }
    add_output(None)->dtype(dtype::Float32{});
}

SymbolVar VirtualLoss::make(const SymbolVarArray& ys,
                            const SymbolVarArray& y_grads, Param,
                            const OperatorNodeConfig& config) {
    mgb_assert(ys.size() == y_grads.size() && !ys.empty());
    VarNodeArray inputs = to_var_node_array(ys);
    // sort for better dedup
    auto cmp = [](VarNode* a, VarNode* b) { return a->id() < b->id(); };
    std::sort(inputs.begin(), inputs.end(), cmp);
    ThinHashMap<VarNode*, VarNode*> var2grad;
    for (size_t i = 0; i < inputs.size(); ++i) {
        var2grad[ys[i].node()] = y_grads[i].node();
    }
    inputs.resize(inputs.size() * 2);
    for (size_t i = 0, it = inputs.size() / 2; i < it; ++i) {
        inputs[i + it] = var2grad.at(inputs[i]);
    }
    return ys[0].insert_single_output_opr<VirtualLoss>(inputs, config);
}

void VirtualLoss::do_execute(ExecEnv&) {
    mgb_throw_if(
#if MGB_BUILD_SLIM_SERVING
            true,
#else
            !owner_graph()->options().eager_evaluation,
#endif
            MegBrainError, "InternalGradLoss should never be executed");
}

void VirtualLoss::init_output_comp_node() {
    output(0)->comp_node(input(0)->comp_node());
}

void VirtualLoss::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    mgr.register_shape_infer(output(0), ShapeInferDesc::make_const({1}));
}

void VirtualLoss::on_output_comp_node_stream_changed() {}

VirtualLoss::NodeProp* VirtualLoss::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_flag(NodeProp::Flag::CROSS_COMP_NODE_MEMORY);
    return ret;
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(VirtualLoss) {
    mgb_assert(out_grad.size() == 1);
    auto mid = opr.input().size() / 2;
    if (wrt_idx < mid) {
        return opr.input(wrt_idx + mid);
    }
    return nullptr;
}
#endif

#else
VarNode* InvalidGrad::make(const OperatorNodeBase&, size_t) {
    mgb_throw(MegBrainError, "grad disabled at compile time");
}
#endif  // MGB_ENABLE_GRAD

/* ================== PersistentOutputStorage =================== */

class PersistentOutputStorage::StorageHolder final
        : public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;
    using Key = std::pair<CompNode, int>;
    struct KeyHash {
        size_t operator()(const Key& key) const {
            return hash_pair_combine(HashTrait<CompNode>::eval(key.first),
                                     key.second);
        }
    };
    std::mutex m_mtx;
    std::unordered_map<Key, DeviceTensorStorage, KeyHash> m_storage;

public:
    void set_tensor(DeviceTensorND& dst, int key, CompNode comp_node,
                    const TensorLayout& layout) {
        MGB_LOCK_GUARD(m_mtx);
        DeviceTensorStorage* storage;
        Maybe<DeviceTensorStorage> local_storage;
        if (key == -1) {
            storage = &local_storage.emplace(dst.storage());
        } else {
            storage = &m_storage[{comp_node, key}];
        }
        if (!storage->comp_node_valid()) {
            storage->comp_node(comp_node);
        }
        auto s = layout.span().dist_byte();
        if (s > storage->size()) {
            if (storage->size()) {
                // exponential growth if size gets increased
                s = s * 3 / 2;
            }
            storage->ensure_size(s);
        }
        dst.reset(*storage, layout);
    }
};

MGB_DYN_TYPE_OBJ_FINAL_IMPL(PersistentOutputStorage);
MGB_TYPEINFO_OBJ_IMPL(PersistentOutputStorage::StorageHolder);

class PersistentOutputStorage::DevValueExecDep final : public ExecDependency {
    DeviceTensorStorage m_val;

public:
    explicit DevValueExecDep(DeviceTensorStorage val) : m_val{std::move(val)} {}
};

PersistentOutputStorage::PersistentOutputStorage(
        VarNode* inp, const Param& param, const OperatorNodeConfig& config)
        : Super{inp->owner_graph(), config, "persist", {}}, m_param{param} {
    add_input({inp});
    add_output(None)
            ->add_flag(VarNode::Flag::NO_MEM_RECLAIM)
            .add_flag(VarNode::Flag::DISALLOW_RT_FORCE_DYNAMIC_MEM_ALLOC);
}

SymbolVar PersistentOutputStorage::make(SymbolVar inp, const Param& param,
                                        const OperatorNodeConfig& config) {
    return inp.insert_single_output_opr<PersistentOutputStorage>(inp.node(),
                                                                 param, config);
}

void PersistentOutputStorage::record_execute_deps(ExecDependencyArray& deps) {
    mgb_assert(!m_dev_tensor.empty());
    deps.emplace_back(
            std::make_unique<DevValueExecDep>(m_dev_tensor.storage()));
}

void PersistentOutputStorage::scn_do_execute() {
    auto &&od = output(0)->dev_tensor(), &&id = input(0)->dev_tensor();
    mgb_assert(od.raw_ptr() == m_dev_tensor.raw_ptr());
    od.copy_from_fixlayout(id);
}

void PersistentOutputStorage::init_output_mem_plan(bool dynamic) {
    mgb_throw_if(
            dynamic, GraphError,
            "PersistentOutputStorage can not be used in dynamic storage case");
    auto cn = comp_node();
    auto ovar = output(0);
    mgb_assert(cg::is_static_var_storage(ovar));
    // note that this method is called after static shape infer, so it is safe
    // to access var shapes here
    auto&& shape = ovar->shape();
    if (!m_dev_tensor.shape().eq_shape(shape) ||
        m_dev_tensor.comp_node() != cn) {
        TensorLayout layout{shape, ovar->dtype(), ovar->format()};
        auto holder =
                owner_graph()
                        ->options()
                        .user_data.get_user_data_or_create<StorageHolder>();
        holder->set_tensor(m_dev_tensor, m_param.share_key, cn, layout);
    }
    ovar->init_mem_plan(&m_dev_tensor);
}

/* ================ RequireInputDynamicStorage ================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(RequireInputDynamicStorage);

RequireInputDynamicStorage::RequireInputDynamicStorage(
        VarNode* input, const OperatorNodeConfig& config)
        : Super{input->owner_graph(),
                config,
                "require_input_dynamic_storage",
                {input}} {
    input->add_flag(VarNode::Flag::NO_SYS_STATIC_MEM_ALLOC);
    add_input({input});
    add_output(None);
}

SymbolVar RequireInputDynamicStorage::make(const SymbolVar input,
                                           const OperatorNodeConfig& config) {
    return input.insert_single_output_opr<RequireInputDynamicStorage>(
            input.node(), config);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
