/**
 * \file src/core/impl/graph/grad_manager.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./cg_impl.h"
#include "./grad_manager.h"

#include "megbrain/graph/helper.h"
#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/utility.h"

using namespace mgb;
using namespace cg;

#if MGB_ENABLE_GRAD

namespace {

/*!
 * \brief check that grad shape is always same as original var shape
 * \param opr fwd opr, only used in error message and can be nullptr
 */
class GradShapeChecker {
    AsyncExecutable *m_cur_comp_seq = nullptr;
    size_t m_cur_run_id = 0;

    int m_nr_wrt = 0, m_nr_grad = 0;
    OperatorNodeBase * const m_opr;
    VarNode * const m_wrt, * const m_grad;
    TensorShape m_prev_wrt_shape, m_prev_grad_shape;

    void do_on_var_shape(VarNode *var) {
        MGB_MARK_USED_VAR(m_opr);
        auto graph = ComputingGraphImpl::downcast(var->owner_graph());

        auto seq = graph->current_comp_seq();
        if (seq) {
            auto run_id = seq->get_run_id();
            if (seq != m_cur_comp_seq || run_id != m_cur_run_id) {
                m_cur_comp_seq = seq;
                m_cur_run_id = run_id;
                m_nr_wrt = m_nr_grad = 0;
            }
        }
        if (var == m_wrt) {
            ++ m_nr_wrt;
            m_prev_wrt_shape = var->shape();
        } else {
            mgb_assert(var == m_grad);
            ++ m_nr_grad;
            m_prev_grad_shape = var->shape();
        }

        if (m_nr_wrt == m_nr_grad) {
            mgb_throw_if(!m_prev_wrt_shape.eq_shape(m_prev_grad_shape),
                    GraphError,
                    "grad should have same shape as original var "
                    "(opr: %s{%s}, wrt_var: %s, grad_var: %s): "
                    "wrt_shape=%s grad_shape=%s",
                    m_opr ? m_opr->cname() : "unknown",
                    m_opr ? m_opr->dyn_typeinfo()->name : "virtual_recv",
                    cg::dump_var_info({m_wrt}).c_str(),
                    cg::dump_var_info({m_grad}).c_str(),
                    m_wrt->shape().to_string().c_str(),
                    m_grad->shape().to_string().c_str());
        }
    }


    static void on_var_shape(
            const std::shared_ptr<GradShapeChecker> &checker,
            VarNode *var) {
        checker->do_on_var_shape(var);
    }

    public:

        GradShapeChecker(OperatorNodeBase *opr, VarNode *wrt, VarNode *grad):
            m_opr(opr), m_wrt(wrt), m_grad(grad)
        {
        }

        static void make(OperatorNodeBase *opr, VarNode *wrt, VarNode *grad) {
            if (ComputingGraphImpl::downcast(wrt->owner_graph())
                    ->eager_eval_manager().enabled())
                return;
            using namespace std::placeholders;
            auto checker = std::make_shared<GradShapeChecker>(opr, wrt, grad);
            auto func = std::bind(&on_var_shape, checker, _1);
            wrt->add_shape_update_callback(grad, func);
            grad->add_shape_update_callback(wrt, func);

            if (wrt->shape().ndim && grad->shape().ndim) {
                // eager check if shape available
                checker->do_on_var_shape(wrt);
                checker->do_on_var_shape(grad);
            }
        }
}; // GradShapeChecker

struct StaticData {
    ThinHashMap<Typeinfo*, OprGradFunc> grad_func_registry;
};

StaticData& static_data() {
    static StaticData sd;
    return sd;
}


} // anonymous namespace

VarNodeArray& OprGradResult::all(OperatorNodeBase* opr) {
    mgb_assert(m_all.size() == opr->input().size(),
               "input grad size mismatch: opr=%s{%s} inputs=%zu grads=%zu",
               opr->cname(), opr->dyn_typeinfo()->name, opr->input().size(),
               m_all.size());
    return m_all;
}

void cg::register_grad_func(Typeinfo *opr_type, OprGradFunc grad) {
    auto ins = static_data().grad_func_registry.emplace(opr_type, grad);
    mgb_assert(ins.second, "duplicated grad registry for type %s",
            opr_type->name);
}

OprGradFunc* cg::lookup_grad_func(Typeinfo *opr_type) {
    auto giter = static_data().grad_func_registry.find(opr_type);
    if (giter != static_data().grad_func_registry.end()) {
        return &giter->second;
    } else {
        return nullptr;
    }
}

class GradManager::StreamStrongPropInfer {
    DepOprIter m_opr_iter;
    ThinHashSet<OperatorNodeBase*> m_strong_oprs;

    void on_opr(OperatorNodeBase *opr) {
        // grads for vars that only depend on SharedDeviceTensor should be moved
        // to copy stream
        if (!opr->same_type<opr::SharedDeviceTensor>()) {
            auto &&dep_map = opr->node_prop().dep_map();
            for (auto i: opr->input()) {
                if (need_device_computing_on_var(i, dep_map.at(i)) &&
                        !m_strong_oprs.count(i->owner_opr())) {
                    return;
                }
            }
        }

        m_strong_oprs.insert(opr);
    }

    public:

        StreamStrongPropInfer():
            m_opr_iter([this](OperatorNodeBase *o){on_opr(o);})
        {
        }

        bool need_strong_stream(VarNode *var) {
            auto opr = var->owner_opr();
            m_opr_iter.add(opr);
            return m_strong_oprs.count(opr);
        }
};

GradManager::GradManager(ComputingGraphImpl *graph):
    m_owner_graph(graph),
    m_stream_strong_prop_infer{new StreamStrongPropInfer}
{
}

GradManager::~GradManager() noexcept = default;

VarNode* GradManager::grad(VarNode *target, VarNode *wrt) {
    mgb_assert(target->owner_graph() == wrt->owner_graph());

    auto check_target_shape = [](VarNode *var) {
        if (!var->shape().is_scalar()) {
            mgb_throw(OperatorNodeExcExtraInfo::ExcMaker{var->owner_opr()}.
                    make<GraphError>,
                    "grad target var must be scalar; got shape %s",
                    var->shape().to_string().c_str());
        }
    };
    if (target->shape().ndim) {
        check_target_shape(target);
    }
    target->add_shape_update_callback(this, check_target_shape);

    m_target_stack.push_back(target);
    MGB_TRY {
        auto ret = do_grad_with_cache(target, wrt);
        m_target_stack.pop_back();
        return ret;
    } MGB_CATCH(..., {
        m_target_stack.pop_back();
        throw;
    })
}

VarNode* GradManager::do_grad_with_cache(VarNode* target, VarNode *wrt) {
    mgb_assert(target->owner_graph() == m_owner_graph);
    auto tgt_wrt_pair = std::make_pair(target, wrt);

    if (m_in_stack.count(tgt_wrt_pair)) {
        mgb_throw(OperatorNodeExcExtraInfo::ExcMaker{wrt->owner_opr()}.
                make<GraphError>,
                "infinite recursion detected while computing grad: "
                "target=%s wrt=%s", cg::dump_var_info({target}).c_str(),
                cg::dump_var_info({wrt}).c_str());
    }

    auto &&tgt_context = m_target_context[target];
    tgt_context.init(this, target);
    auto &&cache = tgt_context.cache;
    {
        auto iter = cache.find(wrt);
        if (iter != cache.end())
            return iter->second;
    }

    auto deps = get_dep_seq(wrt, tgt_context);

    m_in_stack.insert(tgt_wrt_pair);
    VarNodeArray tmp_var_arrs[2];
    MGB_TRY {
        VarNode *ret = nullptr;
        for (auto &&dep: deps) {
            auto ins = cache.emplace(dep.first, nullptr);
            if (ins.second) {
                auto rst = compute_grad_of_single_var(target, dep.first,
                                                      tgt_context, dep.second,
                                                      tmp_var_arrs);
                auto trans_iter = m_grad_transformers.find(dep.first);
                if (trans_iter != m_grad_transformers.end()) {
                    for (auto &&i: trans_iter->second) {
                        rst = i(target, dep.first, rst);
                    }
                }
                ins.first->second = rst;
                ret = rst;
            } else {
                // cache may already exists due to SetGrad recursiive calls
                ret = ins.first->second;
            }
        }

        m_in_stack.erase(tgt_wrt_pair);
        return ret;
    } MGB_CATCH (..., {
        m_in_stack.erase(tgt_wrt_pair);
        throw;
    })
}

void GradManager::ContextForTargetVar::init(
        GradManager *manager, VarNode *target) {
    if (m_virtual_receiver_version == manager->m_virtual_receiver_version &&
            !m_dep_oprs.empty())
        return;

    m_dep_oprs.clear();
    VarNodeArray stack;
    VarNodeSet visited;
    auto add_var = [&](VarNode *var) {
        if (visited.insert(var).second) {
            stack.push_back(var);
            m_dep_oprs.insert(var->owner_opr());
        }
    };

    add_var(target);
    while (!stack.empty()) {
        auto var = stack.back();
        stack.pop_back();

        // add input vars
        for (auto i: var->owner_opr()->input()) {
            add_var(i);
        }

        // find virtual receivers
        {
            auto iter = manager->m_var2virtual_receiver_inv.find(var);
            if (iter != manager->m_var2virtual_receiver_inv.end()) {
                for (VarVirtualReceiverDesc* desc: iter->second) {
                    for (auto i: desc->inputs) {
                        add_var(i);
                    }
                }
            }
        }

        // add extra deps
        {
            auto iter = manager->m_extra_deps_inv_lookup.find(var);
            if (iter != manager->m_extra_deps_inv_lookup.end()) {
                for (auto i: iter->second) {
                    add_var(i);
                }
            }
        }
    }

    m_virtual_receiver_version = manager->m_virtual_receiver_version;
}

struct GradManager::GetDepSeqStackFrame {
    VarNode * const var;
    size_t cur_output_idx = 0;

    OprNodeArray::const_iterator opr_recv_iter;
    const OprNodeArray::const_iterator opr_recv_begin;
    const OprNodeArray::const_iterator opr_recv_end;

    VarVirtualReceiverArray::const_iterator vrt_recv_iter;
    const VarVirtualReceiverArray::const_iterator vrt_recv_begin;
    const VarVirtualReceiverArray::const_iterator vrt_recv_end;

    size_t const tot_nr_recv;

    GetDepSeqStackFrame(
            VarNode *var, const OprNodeArray &opr_recv,
            const VarVirtualReceiverArray &vrt_recv):
        var{var},
        opr_recv_iter{opr_recv.begin()}, opr_recv_begin{opr_recv_iter},
        opr_recv_end{opr_recv.end()},
        vrt_recv_iter{vrt_recv.begin()}, vrt_recv_begin{vrt_recv_iter},
        vrt_recv_end{vrt_recv.end()},
        tot_nr_recv{opr_recv.size() + vrt_recv.size()}
    {
    }

    bool opr_recv_done() const {
        return opr_recv_iter == opr_recv_end;
    }

    bool vrt_recv_done() const {
        return vrt_recv_iter == vrt_recv_end;
    }
};

GradManager::DepSeq GradManager::get_dep_seq(
        VarNode *start_var, const ContextForTargetVar &tgt_context) {
    DepSeq seq;
    VarNodeSet visited;
    std::vector<GetDepSeqStackFrame> stack;

    auto push_stack = [&](VarNode *var)  {
        if (!tgt_context.cache.count(var) && visited.insert(var).second) {
            VarVirtualReceiverArray *vptr;
            auto viter = m_var2virtual_receiver.find(var);
            if (viter != m_var2virtual_receiver.end()) {
                vptr = &viter->second;
            } else {
                static VarVirtualReceiverArray e;
                vptr = &e;
            }
            mgb_assert(var->owner_graph() == m_owner_graph);
            stack.emplace_back(
                    var, m_owner_graph->var_receiver(var), *vptr);
        }
    };

    push_stack(start_var);
    while (!stack.empty()) {
        auto &&frame = stack.back();

        if (frame.opr_recv_done() && frame.vrt_recv_done()) {
            // pop stack if all receivers have been processed
            seq.emplace_back(frame.var, VarReceiverArray{});
            auto &&arr = seq.back().second;
            arr.reserve(frame.tot_nr_recv);
            for (auto i = frame.opr_recv_begin; i != frame.opr_recv_end; ++ i) {
                // for oprs that tgt does not depend on, we do not need to
                // consider it as a receiver
                if (tgt_context.has_dep_opr(*i)) {
                    arr.push_back(*i);
                }
            }
            for (auto i = frame.vrt_recv_begin; i != frame.vrt_recv_end; ++ i) {
                arr.push_back(i->get());
            }
            stack.pop_back();
            continue;
        }

        // process opr receiver
        if (!frame.opr_recv_done()) {
            auto opr = *frame.opr_recv_iter;
            if (!frame.cur_output_idx && (opr->same_type<opr::SetGrad>() ||
                        !tgt_context.has_dep_opr(opr))) {
                // For SetGrad: we do not need to compute its output gradients
                // eagerly, since its callback would call cg::grad if it needs
                // output grad.
                // For oprs that tgt does not depend on, no need to compute its
                // output grad.
                //
                // In these two cases we just ignore the receiver opr.
                ++ frame.opr_recv_iter;
                continue;
            }
            auto &&output = opr->output();
            if (frame.cur_output_idx == output.size()) {
                ++ frame.opr_recv_iter;
                frame.cur_output_idx = 0;
            } else {
                push_stack(output[frame.cur_output_idx ++]);
            }

            continue;
        }

        // process virtual receiver
        auto &&output = frame.vrt_recv_iter->get()->outputs;
        if (frame.cur_output_idx == output.size()) {
            ++ frame.vrt_recv_iter;
            frame.cur_output_idx = 0;
        } else {
            push_stack(output[frame.cur_output_idx ++]);
        }
    }

    return seq;
}

VarNode* GradManager::compute_grad_of_single_var(
        VarNode *target, VarNode *wrt, ContextForTargetVar& context,
        const  VarReceiverArray &wrt_recv, VarNodeArray *tmp_var_arrs) {
    if (target == wrt)
        return SymbolVar{wrt}.make_scalar_dt(1.f).node();

    // grads of the receivers that should be summed to get final grad
    auto &&recv_grads_to_sum = tmp_var_arrs[0];
    recv_grads_to_sum.clear();

    // current outgrad when append_opr_grad() is called
    auto &&outgrad = tmp_var_arrs[1];

    auto &&grad_func_registry = static_data().grad_func_registry;

    auto add_to_recv_grads_to_sum = [&](OperatorNodeBase *opr, VarNode *grad) {
        mgb_assert(grad->comp_node() == wrt->comp_node(),
                "grad comp node must be the same of original var");
        mgb_assert(
                grad->dtype() == wrt->dtype(),
                "grad dtype must be the same of original dtype, or opr is "
                "Reduce; got "
                "opr=%s grad=%s wrt=%s\ndetails: %s",
                opr->cname(), grad->dtype().name(), wrt->dtype().name(),
                dump_var_info({grad, wrt}).c_str());
        GradShapeChecker::make(opr, wrt, grad);
        recv_grads_to_sum.push_back(grad);
    };

    // append grad of a single operator to recv_grads_to_sum
    auto append_opr_grad = [&](OperatorNodeBase* opr) {
        if (!opr->same_type<opr::SetGrad>()) {
            // if opr is SetGrad, call its grad() unconditionally; otherwise
            // check whether target does not depend on any output
            bool all_null = true;
            for (VarNode* i : outgrad)
                if (i) {
                    all_null = false;
                    break;
                }
            if (all_null)
                return;
        }

        const VarNodeArray* inp_grad_result = nullptr;
        {
            auto iter = context.holistic_input_grads.find(opr);
            if (iter != context.holistic_input_grads.end()) {
                inp_grad_result = &iter->second;
            }
        }

        bool found = false;
        for (size_t i = 0; i < opr->input().size(); ++i) {
            if (opr->input()[i] == wrt) {
                found = true;
                VarNode* cur;
                if (inp_grad_result) {
                    cur = inp_grad_result->at(i);
                } else {
                    auto gfunc_iter =
                            grad_func_registry.find(opr->dyn_typeinfo());
                    mgb_assert(gfunc_iter != grad_func_registry.end(),
                               "grad for type %s not implemented",
                               opr->dyn_typeinfo()->name);
                    auto res = gfunc_iter->second(opr, i, outgrad);
                    if (res.from_single()) {
                        cur = res.single();
                    } else {

                        auto ins = context.holistic_input_grads.emplace(
                                opr, std::move(res.all(opr)));
                        mgb_assert(ins.second);
                        inp_grad_result = &ins.first->second;
                        cur = inp_grad_result->at(i);
                    }
                }
                if (cur) {
                    add_to_recv_grads_to_sum(opr, cur);
                    cur->name(ssprintf("grad[var%zu,opr%zu]:%zu", wrt->id(),
                                       opr->id(), i));
                }
            }
        }
        mgb_assert(found);
    };

    auto &&opr_list = m_owner_graph->m_opr_refkeeper;

    auto setup_outgrad = [&](const VarNodeArray &output) {
        for (auto i: output)
            outgrad.push_back(context.cache.at(i));
    };

    for (auto &&recv_item: wrt_recv) {
        outgrad.clear();

        if (recv_item.vrt) {
            auto vrecv = recv_item.vrt;
            setup_outgrad(vrecv->outputs);
            bool found = false;
            for (size_t i = 0; i < vrecv->inputs.size(); ++ i) {
                auto inp = vrecv->inputs[i];
                if (inp == wrt) {
                    found = true;
                    auto cur = vrecv->grad(
                            vrecv->inputs, vrecv->outputs, i, outgrad);
                    if (cur) {
                        add_to_recv_grads_to_sum(nullptr, cur);
                        cur->name(ssprintf("grad[var%zu,virtual_recv]:%zu",
                                    wrt->id(), i));
                    }
                }
            }
            mgb_assert(found);
            continue;
        }

        auto recv_opr = recv_item.opr;
        if (!recv_opr->same_type<opr::SetGrad>()) {
            setup_outgrad(recv_opr->output());
        }

        // add grad source tracker to opr attr
        auto add_grad_src = [&, orig_size=opr_list.size()]() {
            for (size_t i = orig_size; i < opr_list.size(); ++ i) {
                auto &&attr = opr_list[i]->node_prop().attribute();
                auto &&tk = attr.grad_tracker;
                if (!tk.valid()) {
                    tk.emplace(recv_opr, target, wrt);
                }
                // do not mark as copied from other
                attr.src_opr = nullptr;
                // set reverse priority
                attr.priority = -recv_opr->node_prop().attribute().priority;
            }
        };

        // take grad and add extra info on error
        MGB_TRY {
            append_opr_grad(recv_opr);
        } MGB_CATCH(MegBrainError &exc, {
            if (!exc.extra_info()) {
                mgb_log_warn("error while taking grad to %s{%s}; "
                        "but exc extra info has not been set; "
                        "use original operator",
                        recv_opr->cname(), recv_opr->dyn_typeinfo()->name);
                OperatorNodeExcExtraInfo::record(recv_opr, exc);
            }
            add_grad_src();
            throw;
        })

        add_grad_src();

    }
    if (recv_grads_to_sum.empty())
        return nullptr;

    // prompt copy stream vars to strong
    auto &&comp_node_opt = m_owner_graph->components().seq_comp_node_opt;
    for (auto i: recv_grads_to_sum) {
        using T = SeqCompNodeOptimizer::StreamPropType;
        auto stream_prop_type = comp_node_opt.stream_prop_type(i);
        if (stream_prop_type.prop_type != T::NONE) {
            if (m_stream_strong_prop_infer->need_strong_stream(i)) {
                comp_node_opt.register_stream_var(
                        i, {stream_prop_type.stream, T::STRONG});
            }
        }
    }

    VarNode *result = opr::Elemwise::sum_grad_list(wrt, recv_grads_to_sum);

    result->name(ssprintf("grad[var%zu:%s]", wrt->id(), wrt->cname()));

    if (m_owner_graph->options().enable_grad_var_static_reshape &&
            is_static_var_shape(wrt) && !is_static_var_shape(result)) {
        // use static shape to facilitate memory allocation
        result = SymbolVar{result}.reshape(SymbolVar{wrt}.symshape()).node();
    }
    return result;
}

void GradManager::add_var_virtual_receiver(
        const std::shared_ptr<VarVirtualReceiverDesc> &desc) {
    ++ m_virtual_receiver_version;
    mgb_assert(!desc->inputs.empty() && !desc->outputs.empty());
    for (auto i: desc->inputs) {
        mgb_assert(i->owner_graph() == m_owner_graph);
    }
    for (auto i: desc->outputs) {
        mgb_assert(i->owner_graph() == m_owner_graph);
    }

    VarNodeSet vars_dedup;
    for (size_t i = 0; i < desc->inputs.size(); ++ i) {
        auto inp = desc->inputs[i];
        if (vars_dedup.insert(inp).second) {
            m_var2virtual_receiver[inp].push_back(desc);
        }
    }

    vars_dedup.clear();
    for (size_t i = 0; i < desc->outputs.size(); ++ i) {
        auto out = desc->outputs[i];
        if (vars_dedup.insert(out).second) {
            m_var2virtual_receiver_inv[out].push_back(desc.get());
        }
    }
}

void cg::add_grad_transformer(VarNode *var, const GradTransformer &cb) {
    ComputingGraphImpl::downcast(var->owner_graph())->
        grad_manager().
        add_grad_transformer(var, cb);
}

void cg::add_extra_dep_for_grad(VarNode *inp, VarNode *out) {
    ComputingGraphImpl::downcast(inp->owner_graph())->grad_manager().
        add_extra_dep_for_grad(inp, out);
}

void cg::add_var_virtual_receiver(
        const VarNodeArray &inputs, const VarNodeArray &outputs,
        const VarVirtualReceiverGrad &grad) {
    auto desc = std::make_shared<GradManager::VarVirtualReceiverDesc>();
    desc->inputs = inputs;
    desc->outputs = outputs;
    desc->grad = grad;
    ComputingGraphImpl::downcast(inputs.at(0)->owner_graph())->
        grad_manager().
        add_var_virtual_receiver(desc);
}

VarNode* cg::call_opr_grad_on_given_io(
        OperatorNodeBase *opr,
        const VarNodeArray &inputs, const VarNodeArray &outputs,
        size_t idx, const VarNodeArray &out_grad,
        bool add_volatile_out) {

    VarNodeArray *cur_out = const_cast<VarNodeArray*>(&outputs),
                 outputs_with_volatile;
    if (add_volatile_out) {
        if (outputs.size() != opr->output().size()) {
            size_t used = 0;
            for (auto i: opr->output()) {
                if (i->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
                    outputs_with_volatile.push_back(nullptr);
                } else {
                    outputs_with_volatile.push_back(outputs.at(used ++));
                }
            }
            mgb_assert(used == outputs.size());
            cur_out = &outputs_with_volatile;
        } else {
            for (auto i: opr->output())
                mgb_assert(!i->contain_flag(VarNode::Flag::VOLATILE_CONTENT));
        }
    }

    mgb_assert(inputs.size() == opr->input().size() &&
            cur_out->size() == opr->output().size());
    auto giter = static_data().grad_func_registry.find(opr->dyn_typeinfo());
    mgb_assert(giter != static_data().grad_func_registry.end(),
            "grad for type %s not implemented",
            opr->dyn_typeinfo()->name);

    auto &&opr_inp = const_cast<VarNodeArray&>(opr->input()),
         &&opr_out = const_cast<VarNodeArray&>(opr->output()),
         &&cur_inp = const_cast<VarNodeArray&>(inputs);
    opr_inp.swap(cur_inp);
    opr_out.swap(*cur_out);
    OprGradResult res;
    MGB_TRY {
        res = giter->second(opr, idx, out_grad);
    } MGB_FINALLY({
        opr_inp.swap(cur_inp);
        opr_out.swap(*cur_out);
    });
    if (res.from_single())
        return res.single();
    return res.all(opr).at(idx);
}

void cg::add_var_virtual_receiver_reuse_opr_grad(
        const VarNodeArray &inputs, const VarNodeArray &outputs,
        OperatorNodeBase *opr, bool add_volatile_out) {
    using namespace std::placeholders;
    auto grad = std::bind(call_opr_grad_on_given_io, opr,
            _1, _2, _3, _4, add_volatile_out);
    add_var_virtual_receiver(inputs, outputs, grad);
}

#else

void cg::register_grad_func(Typeinfo*, OprGradFunc) {}

void cg::add_grad_transformer(VarNode*, const GradTransformer&) {}

void cg::add_extra_dep_for_grad(VarNode*, VarNode*) {}

void cg::add_var_virtual_receiver(const VarNodeArray&, const VarNodeArray&,
                                  const VarVirtualReceiverGrad&) {}

VarNode* cg::call_opr_grad_on_given_io(OperatorNodeBase*, const VarNodeArray&,
                                       const VarNodeArray&, size_t,
                                       const VarNodeArray&, bool) {
    mgb_throw(MegBrainError, "grad disabled at compile time");
}

void cg::add_var_virtual_receiver_reuse_opr_grad(const VarNodeArray&,
                                                 const VarNodeArray&,
                                                 OperatorNodeBase*, bool) {}

#endif  // MGB_ENABLE_GRAD

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
