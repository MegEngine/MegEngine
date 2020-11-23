/**
 * \file src/opr/impl/loop/forward.cpp
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

#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/gopt/framework.h"

#include "megdnn/oprs.h"

#include <cmath>

using namespace mgb;
using namespace opr;
using namespace intl;

/* ========= FwdDesc ========= */

SymbolVar LoopImpl::FwdDesc::add_input(SymbolVar inp, bool has_assign) {
    if (!has_assign) {
        auto &&var = m_input_no_assign_dedup[inp.node()];
        if (var)
            return var;
        var = do_add_input(inp, {false, false}).node();
        return var;
    }
    auto var = do_add_input(inp, {true, true});
    m_input_assigned[var.node()] = false;
    return var;
}

size_t LoopImpl::FwdDesc::add_output(SymbolVar val, OutputMode mode) {
    auto ret = DescImplBase::add_output(val, mode);
    if (mode == OutputMode::ALL) {
        auto &&d = m_output_record_spec_mode_all[val.node()];
        auto s = m_output_record_spec_no_dedup.at(ret);
        mgb_assert(!d || d == s);
        d = s;
    }
    return ret;
}

Loop::Desc& LoopImpl::FwdDesc::assign(SymbolVar dest, SymbolVar val) {
    mgb_throw_if(!check_in_sub_graph(dest) || !check_in_sub_graph(val),
            GraphError, "assign dest and val must be in sub graph");

    auto iter = m_input_assigned.find(dest.node());
    mgb_throw_if(iter == m_input_assigned.end(), GraphError,
            "assign dest must be InputMaker declared as has_assign; "
            "got %s", cg::dump_var_info({dest.node()}).c_str());
    mgb_throw_if(iter->second, GraphError,
            "a single var should only be assigned once; found multiple assigns"
            " to %s", cg::dump_var_info({dest.node()}).c_str());
    mgb_throw_if(!dest.node()->owner_opr()->same_type<InputMaker>(), GraphError,
            "assignment dest must be input var");

    auto opr = &dest.node()->owner_opr()->cast_final<InputMaker>();

    mgb_throw_if(dest.dtype() != val.dtype(), GraphError,
            "assignment dtype mismatch: dest is %s, value is %s",
            dest.dtype().name(), val.dtype().name());
    mgb_throw_if(dest.shape().ndim && val.shape().ndim &&
            !dest.shape().eq_shape(val.shape()), GraphError,
            "assignment shape mismatch: %s",
            cg::dump_var_info({dest.node(), val.node()}).c_str());

    opr->set_assignor(val.node());
    iter->second = true;
    return *this;
}

SymbolVarArray
LoopImpl::FwdDesc::user_output_vars_including_dup() const {
    for (auto &&i: m_input_assigned) {
        mgb_throw_if(!i.second, GraphError,
                "%s is declared to have assign, "
                "but has not actually been assigned",
                cg::dump_var_info({i.first}).c_str());
    }

    mgb_throw_if(m_output_record_spec_no_dedup.empty(), GraphError,
            "add_output not called on loop desc");
    SymbolVarArray ret;
    ret.reserve(m_output_record_spec_no_dedup.size());
    for (auto i: m_output_record_spec_no_dedup) {
        ret.push_back(i->var_owner());
    }
    return ret;
}

VarNode* LoopImpl::FwdDesc::owner_graph_output_at(size_t idx) const {
    auto rst = m_output_record_spec_no_dedup.at(idx)->var_owner();
    mgb_assert(rst);
    return rst;
}

const std::vector<LoopImpl::InputMaker*>& LoopImpl::FwdDesc::all_inputs() {
    if (!m_dep_iter) {
        m_dep_iter.reset(new SubgraphDepIter);
        auto &&iter = *m_dep_iter;
        for (auto &&i: m_output_record_spec)
            iter.add(i.var_sub());

        auto cond = loop_cond_manager().var().node();
        mgb_throw_if(!cond, GraphError,
                "loop condition not set");
        iter.add(cond);
    }
    return m_dep_iter->input_makers();
}

/* ========= Loop ========= */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Loop);

Loop::Loop(std::unique_ptr<FwdDesc> desc, DescMaker desc_maker,
        const Param &param, const OperatorNodeConfig &config):
    Super({nullptr, config, "loop", {}}, std::move(desc)),
    m_param{param}, m_desc_maker{desc_maker}
{
    add_input_in_desc();
    mgb_assert(!input().empty(), "Loop must have some input vars");

    // add and bind output vars
    for (auto &&i: m_desc->output_record_spec()) {
        auto oname = i.recorder()->name() + ":" + i.var_sub()->name();
        auto out = add_output(oname);
        const_cast<OutputRecordSpecItem&>(i).bind(out);
        using F = VarNode::Flag;
        out->add_flag(F::ALLOW_EMPTY_SHAPE).
            dtype(i.var_sub()->dtype());
        if (!i.recorder()->has_shape_infer_desc())
            out->add_flag(F::NO_SYS_MEM_ALLOC);
    }

    init_mutable_state_saver();

    m_output_counter_var = add_output("virtual_counter");
    m_output_counter_var->dtype(dtype::Int32());
}

SymbolVarArray Loop::make(
        DescMaker desc_maker,
        const Param &param, const OperatorNodeConfig &config) {
    auto desc = std::make_unique<FwdDesc>();
    desc_maker(*desc);

    auto graph = desc->owner_graph();
#if !MGB_BUILD_SLIM_SERVING
    if (std::abs(graph->options().graph_opt_level) >= 2) {
        optimize_fwd_graph(graph->options().graph_opt_level, *desc);
    }
#endif
    for (auto i: desc->all_inputs()) {
        if (i->param().has_assign)
            i->commit_assignor();
    }
    auto opr = graph->insert_opr(std::make_unique<Loop>(
                std::move(desc), desc_maker, param, config));
    return static_cast<FwdDesc*>(
            opr->cast_final_safe<Loop>().m_desc.get())->
        user_output_vars_including_dup();
}

void Loop::add_input_layout_constraint() {
    LoopImpl::add_input_layout_constraint();
    // disable all state saver, and the needed would be enabled in
    // LoopGrad::add_input_layout_constraint
    m_mutable_state_saver->disable();
}

void Loop::init_output_static_infer_desc() {
    using namespace cg::static_infer;

    node_prop(); // initialize m_static_final_counter_value_infer

    auto output_cnt = output_counter_var();

    // register counter shape infer
    owner_graph()->static_infer_manager().register_shape_infer(
            output_cnt, {SourceType::CONSTANT, {},
            [](TensorShape &dest, const InpVal &){
                dest = {1};
                return true;
            }
    });

    // register counter value infer
    if (m_static_final_counter_value_infer.first) {
        auto &&cnt_infer_trait = m_static_final_counter_value_infer;
        m_static_loop_time_infer = [&cnt_infer_trait]() -> size_t {
            auto &&iv = cnt_infer_trait.first->
                owner_graph()->static_infer_manager().infer_value(
                        cnt_infer_trait.first);
            return cnt_infer_trait.second(iv) + 1;
        };

        auto infer_val = [&cnt_infer_trait](
                DeviceTensorND &dest, const InpVal &val) {
            auto &&iv = val.val.at(0).value();
            dest.resize({1}).ptr<int>()[0] = cnt_infer_trait.second(iv);
            return true;
        };

        m_desc->sub_graph_static_infer_helper().register_value_infer_par(
                output_cnt,
                {SourceType::DEP,
                {{cnt_infer_trait.first, DepType::VALUE}}, infer_val});
    };

    // register shape infer for add_output vars
    for (auto &&i: m_desc->output_record_spec()) {
        auto rec = i.recorder();
        if (rec->has_shape_infer_desc()) {
            rec->register_infer_desc(
                    m_desc->sub_graph_static_infer_helper());
            auto out = i.var_owner();
            using F = VarNode::Flag;
            if (!out->contain_flag(F::NO_SYS_MEM_ALLOC))
                out->add_flag(F::NO_ALLOC_IF_UNUSED);
        }
    }
}

void Loop::init_mutable_state_saver() {
    auto desc = static_cast<FwdDesc*>(m_desc.get());
    auto saver = std::make_unique<MutableStateSaver>(this);

    for (auto i: desc->sub_graph_oprs()) {
        if (i->node_prop().contain(NodeProp::Flag::IMPURE_FUNC)) {
            mgb_assert(!i->same_type<Loop>(), "nested loop with impure nodes "
                    "currently not supported");
            for (auto j: i->output()) {
                if (!j->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
                    saver->add_var_to_record(j);
                }
            }
        }
    };

    saver->swap_interval(m_param.swap_interval);
    m_mutable_state_saver = std::move(saver);
}

VarNode* Loop::grad(Loop &opr, size_t wrt_idx, const VarNodeArray &out_grad) {
    LoopGrad* &gopr =
        opr.m_loss2grad_opr[cg::current_grad_target(*opr.owner_graph()).node()];
    if (!gopr) {
        // extra output is counter var
        mgb_assert(out_grad.size() ==
                opr.m_desc->output_record_spec().size() + 1 &&
                !out_grad.back());
        VarNodeArray out_grad_used(out_grad);
        out_grad_used.pop_back();
        gopr = LoopGrad::make(&opr, out_grad_used);
    }
    return gopr->get_grad_var(wrt_idx);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Loop) {
    return Loop::grad(const_cast<Loop&>(opr), wrt_idx, out_grad);
}
#endif

cg::OperatorNodeBase::NodeProp* Loop::do_make_node_prop() const {
    auto prop = LoopImpl::do_make_node_prop();

    // check whether sub graph is impure
    for (auto i: static_cast<FwdDesc*>(m_desc.get())->sub_graph_oprs()) {
        constexpr auto IMPURE = NodeProp::Flag::IMPURE_FUNC;
        if (i->node_prop().contain(IMPURE) && !i->same_type<InputMaker>()) {
            prop->add_flag(IMPURE);
            break;
        }
    }

    auto cond_opr = m_desc->loop_cond_manager().var().node()->owner_opr();

    // add static infer deps to opr deps
    auto extend = [&](const cg::static_infer::DepVal &deps) {
        using namespace cg::static_infer;
        using NDT = NodeProp::DepType;

        for (auto &&i: deps) {

            if (i.dest == m_desc->get_counter_var().node())
                continue;

            auto dt = i.type == DepType::SHAPE ?
                NDT::SHAPE : NDT::HOST_VALUE;
            auto opr = i.dest->owner_opr();
            if (opr->same_type<InputMaker>()) {
                prop->add_dep_type(
                        opr->cast_final<InputMaker>().orig_var(), dt);
            }
        }
    };

    extend(m_desc->compile()->get_rt_static_source_deps());

    auto setup_static_infer = [&]() {
        if (cond_opr->dyn_typeinfo() != opr::Elemwise::typeinfo() ||
                cond_opr->input().size() != 2)
            return;

        auto mode = cond_opr->cast_final<opr::Elemwise>().param().mode;
        using Mode = opr::Elemwise::Mode;
        if (mode != Mode::LT && mode != Mode::LEQ)
            return;
        {
            // check whether is the form counter < X
            auto inp0 = cond_opr->input(0),
                 cnt = m_desc->get_counter_var().node();
            if (inp0 != cnt) {
                auto inp0_opr = inp0->owner_opr();
                if (inp0_opr->dyn_typeinfo() != opr::TypeCvt::typeinfo())
                    return;
                inp0 = inp0_opr->input(0);
                if (inp0 != cnt)
                    return;
            }
        }
        auto cnt_end = cond_opr->input(1);
        if (!cg::is_static_var_value(cnt_end))
            return;

        // infer counter value at loop exit
        auto infer_counter_val = [cnt_end, contain_eq=mode==Mode::LEQ](
                const DeviceTensorND &val) -> size_t {
            MGB_MARK_USED_VAR(cnt_end);
            mgb_assert(val.comp_node() == CompNode::default_cpu());
            mgb_assert(val.shape().is_scalar(),
                    "loop condition is counter < t, "
                    "but t is not scalar: %s",
                    cg::dump_var_info({cnt_end}).c_str());
            switch(val.dtype().enumv()) {
                case DTypeEnum::Uint8:
                    {
                        auto iv = val.ptr<dt_uint8>()[0];
                        iv += contain_eq;
                        return std::max<int>(iv, 0);
                    }
                case DTypeEnum::Int8:
                    {
                        auto iv = val.ptr<dt_int8>()[0];
                        iv += contain_eq;
                        return std::max<int>(iv, 0);
                    }
                case DTypeEnum::Int16:
                    {
                        auto iv = val.ptr<dt_int16>()[0];
                        iv += contain_eq;
                        return std::max<int>(iv, 0);
                    }
                case DTypeEnum::Int32:
                    {
                        auto iv = val.ptr<int>()[0];
                        iv += contain_eq;
                        return std::max(iv, 0);
                    }
                case DTypeEnum::Uint16:
                    {
                        auto iv = val.ptr<dt_uint16>()[0];
                        iv += contain_eq;
                        return std::max<int>(iv, 0);
                    }
                case DTypeEnum::Float32:
#if !MEGDNN_DISABLE_FLOAT16
                case DTypeEnum::Float16:
                    {
                        float iv;
                        if (val.dtype().enumv() == DTypeEnum::Float16)
                            iv = val.ptr<dt_float16>()[0];
                        else
                            iv = val.ptr<float>()[0];
                        auto inext = std::ceil(iv);
                        if (iv == inext && contain_eq)
                            ++ inext;
                        return std::max<int>(inext, 0);
                    }
                case DTypeEnum::BFloat16:
                    {
                        float iv;
                        if (val.dtype().enumv() == DTypeEnum::BFloat16)
                            iv = val.ptr<dt_bfloat16>()[0];
                        else
                            iv = val.ptr<float>()[0];
                        auto inext = std::ceil(iv);
                        if (iv == inext && contain_eq)
                            ++ inext;
                        return std::max<int>(inext, 0);
                    }
#endif
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

                #define cb(x) case DTypeEnum::x: break;
                MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb)
                #undef cb
            }
            mgb_throw(MegBrainError, "unhandled dtype: %s", val.dtype().name());
        };

        m_static_final_counter_value_infer = {cnt_end, infer_counter_val};

        auto &&mgr = cnt_end->owner_graph()->static_infer_manager();
        extend(mgr.get_rt_static_source_deps(
                    {cnt_end, cg::static_infer::DepType::VALUE}));
    };
    setup_static_infer();

    // add shape deps so shape could be updated for InputMaker shape infer
    for (auto i: input())
        prop->add_dep_type(i, NodeProp::DepType::SHAPE);

    return prop;
}

void Loop::optimize_fwd_graph(int level, FwdDesc &desc) {
    // setup endpoints
    VarNodeArray endpoints;
    endpoints.reserve(desc.all_inputs().size() +
            desc.output_record_spec().size());
    for (auto i: desc.all_inputs()) {
        if (i->param().has_assign) {
            endpoints.push_back(i->assignor());
        }
    }
    for (auto &&i: desc.output_record_spec()) {
        endpoints.push_back(i.var_sub());
    }
    auto cond = desc.loop_cond_manager().var();
    mgb_throw_if(!cond.node(), GraphError, "loop condition not set");
    endpoints.push_back(cond.node());

    // optimize also extra_vardeps
    size_t nr_extra_deps = 0;
    auto &&extra_deps = desc.sub_graph()->options().extra_vardeps;
    {
        auto on_opr = [&](OperatorNodeBase *opr) {
            for (auto i: opr->output()) {
                auto &&iter = extra_deps.find(i);
                if (iter != extra_deps.end()) {
                    nr_extra_deps += iter->second.size();
                    endpoints.insert(endpoints.end(),
                            iter->second.begin(),
                            iter->second.end());
                    extra_deps.erase(iter);
                }
            }
        };
        cg::DepOprIter opr_iter{on_opr};
        for (size_t i = 0; i < endpoints.size(); ++ i) {
            opr_iter.add(endpoints[i]->owner_opr());
        }
    }

    // apply opt and reset vars
    gopt::GraphOptimizer().
        add_preset_passes().
        verbosity(0).
        enable_check_result(level < 0).
        apply_inplace(endpoints);

    auto ep_iter = endpoints.begin();
    for (auto i: desc.all_inputs()) {
        if (i->param().has_assign) {
            i->set_assignor(*(ep_iter ++));
        }
    }
    for (auto &&i: desc.output_record_spec()) {
        const_cast<OutputRecordSpecItem&>(i).var_sub(*(ep_iter ++));
    }
    desc.loop_cond_manager().setup(*(ep_iter ++));

    auto &&cond_deps = extra_deps[desc.loop_cond_manager().var().node()];
    for (size_t i = 0; i < nr_extra_deps; ++ i) {
        cond_deps.push_back(*(ep_iter ++));
    }

    mgb_assert(ep_iter == endpoints.end());

    desc.on_sub_graph_optimized();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

