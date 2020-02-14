/**
 * \file src/gopt/impl/misc.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/gopt/misc.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/cond.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/serialization/opr_shallow_copy.h"

using namespace mgb;
using namespace gopt;

/* ================ RemoveNonComputingOprPass ================ */

const char* RemoveNonComputingOprPass::name() const {
    return "remove_non_computing_opr";
}

void RemoveNonComputingOprPass::apply(OptState& opt) const {
    auto rewriter = opt.graph().make_rewriter();
    auto on_opr = [&](OperatorNodeBase* opr) {
        auto type = opr->dyn_typeinfo();
        if (type == opr::MarkNoBroadcastElemwise::typeinfo() ||
#if MGB_ENABLE_GRAD
            type == opr::SetGrad::typeinfo() ||
#endif
            type == opr::Identity::typeinfo()) {
            // remove marker oprs
            auto src = opr->output(0);
            auto dst = rewriter.get_var(opr->input(0));
            rewriter.replace_var(src, dst, mgb_cstr_log(type->name));
            return;
        }

        if (type == opr::Split::typeinfo()) {
            // check split on const scalar: useful for grad wrt Concat
            auto iv = SymbolVar{opr->input(0)}.as_immutable_scalar();
            if (iv.valid()) {
                bool shape_known = true;
                for (auto i : opr->output()) {
                    if (!cg::is_static_var_shape(i)) {
                        shape_known = false;
                        break;
                    }
                }
                if (shape_known) {
                    for (auto i : opr->output()) {
                        auto iv_src = opr::ImmutableTensor::make(
                                *i->owner_graph(), iv.val(), i->comp_node());
                        auto vnew = opr::Broadcast::make(
                                            iv_src, opr::GetVarShape::make(i))
                                            .node();
                        rewriter.replace_var(
                                i, vnew, mgb_cstr_log("const split output"));
                    }
                    return;
                }
            }
        }

        rewriter.auto_replace_outputs(opr);
    };

    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
}

/* ================ ExpandVirtualGradPass ================ */

const char* ExpandVirtualGradPass::name() const {
    return "expand_virtual_grad";
}

void ExpandVirtualGradPass::apply(OptState& opt) const {
#if MGB_ENABLE_GRAD
    opt.set_var_replace_check_flag(VarReplaceCheckFlag::NOCHECK);
    auto rewriter = opt.graph().make_rewriter();
    auto on_opr = [&](OperatorNodeBase* opr) {
        if (!opr->same_type<opr::VirtualGrad>()) {
            rewriter.auto_replace_outputs(opr);
            return;
        }
        // Create opr and replace var but no need to copy old opr_properties
        // to new oprs because grad_manager would handle it.
        opt.call_with_opr(opr, [&]{
            auto target = opr->input(0), wrt = opr->input(1),
                 grad = cg::grad(target, wrt).node();
            auto src = opr->output(0);
            grad = GraphOptimizer::var_replace_lookup(grad);
            rewriter.replace_var(
                    src, grad,
                    mgb_ssprintf_log("grad(%s, %s)", target->cname(), wrt->cname())
                            .c_str());
        }, OprPropertyFlag::NONE);
    };

    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
#else
    MGB_MARK_USED_VAR(opt);
#endif
}

/* ================= DelayBroadcastPass ================ */

bool DelayBroadcastPass::allowed_opr(OperatorNodeBase* opr) {
    static const ThinHashSet<Typeinfo*> allowed_opr_type{
            opr::Broadcast::typeinfo(),

            // should include all oprs below that doesn't explictly change the
            // input's shape.
            opr::TypeCvt::typeinfo(),
            opr::Elemwise::typeinfo(),
    };
    return allowed_opr_type.count(opr->dyn_typeinfo());
};

const char* DelayBroadcastPass::name() const {
    return "delay_broadcast";
}

void DelayBroadcastPass::apply(OptState& opt) const {
    // Extract a chain, make sure the oprs on the chain are
    // read by only one operator.
    // The endpoint of the chain meet one of the following three conditions:
    //      1. more than one opr depend on it.
    //      2. only one opr depends on it, but can not be on the chain.
    //      3. endponit of the graph.
    // When processing the chain from endpoint,
    // find the opr that is not the Broadcast, set it to the new endpoint
    // Find all broadcasts from the chain from new endpoint,
    // remove them from the chain, and add them back right after the endpoint.

    // TypeCvt's order may change, so disable the check.
    opt.set_var_replace_check_flag(VarReplaceCheckFlag::NOCHECK);

    auto unique_reader_chk = UniqReaderCheck{opt.graph()};
    auto rewriter = opt.graph().make_rewriter();
    ThinHashSet<OperatorNodeBase*> visited;
    ThinHashMap<OperatorNodeBase*, bool> dep_on_bcast;

    // map from opr to the input in unary-bcast chain
    // value is (valid, input idx)
    ThinHashMap<OperatorNodeBase*, std::pair<bool, uint32_t>> opr2chain_inp_idx;

    auto is_opr_endpoint = [&](OperatorNodeBase* opr) -> bool {
        if (!unique_reader_chk(opr->output(0)))
            return true;
        if (opt.graph().endpoint_contain(opr->output(0)))
            return true;
        return false;
    };

    auto opr_in_chain = [&](OperatorNodeBase* opr, VarNode** chain_input,
                            bool could_be_endpoint) {
        if (!allowed_opr(opr))
            return false;
        auto chain_input_ins = opr2chain_inp_idx.insert({opr, {}});
        auto&& chain_input_pair = chain_input_ins.first->second;
        if (chain_input_ins.second) {
            if (opr->same_type<opr::Broadcast>()) {
                mgb_assert(opr->input().size() == 2);
                chain_input_pair = {true, 0};
            } else {
                int idx = -1;
                chain_input_pair = {false,
                                    std::numeric_limits<uint32_t>::max()};
                for (size_t i = 0; i < opr->input().size(); ++i) {
                    auto var = opr->input()[i];
                    if (!(cg::is_const_var_shape(var) &&
                          var->shape().is_scalar())) {
                        if (idx < 0) {
                            idx = i;
                        } else {
                            return false;
                        }
                    }
                }
                if (idx != -1) {
                    chain_input_pair = {true, static_cast<uint32_t>(idx)};
                }
            }
        }
        if (!chain_input_pair.first) {
            return false;
        }
        *chain_input = opr->input()[chain_input_pair.second];
        if (!could_be_endpoint)
            return unique_reader_chk(opr->output(0));
        return true;
    };

    auto build_chain =
            [&](const std::vector<cg::OperatorNodeBase*>& oprs) -> VarNode* {
        VarNode* prev = nullptr;
        // note that reversed opr seq is the correct topo order
        for (auto opr : reverse_adaptor(oprs)) {
            auto inp_idx = opr2chain_inp_idx.at(opr).second;
            if (!prev)
                prev = rewriter.get_var(opr->input(inp_idx));
            if (!opr->same_type<opr::Broadcast>()) {
                VarNodeArray new_inp = opr->input();
                new_inp.at(inp_idx) = prev;
                opt.call_with_opr(opr, [&] {
                    // create new opr with the original opr's properties
                    auto new_opr = serialization::copy_opr_shallow(
                        *opr, new_inp, opr->config());
                    prev = new_opr->output(0);
                });
            }
        }
        return prev;
    };

    auto process_chain_from_endpoint = [&](OperatorNodeBase* opr) {

        auto auto_replace_with_context = [&](OperatorNodeBase* opr) {
            opt.call_with_opr(opr, [&]{
                rewriter.auto_replace_outputs(opr);
            });
        };

        if (!dep_on_bcast[opr]) {
            auto_replace_with_context(opr);
            return;
        }
        SmallVector<OperatorNodeBase*> trailing_bcasts;

        auto replace_trailing_bcasts = [&]() {
            for (auto opr : reverse_adaptor(trailing_bcasts)) {
                auto_replace_with_context(opr);
            }
        };

        // Find the latest opr that is not the Broadcast.
        VarNode* chain_input;
        for (; opr_in_chain(opr, &chain_input, true) && !visited.count(opr);
             opr = chain_input->owner_opr()) {
            if (!opr->same_type<opr::Broadcast>()) {
                break;
            }
            visited.insert(opr);
            trailing_bcasts.push_back(opr);
        }

        std::vector<cg::OperatorNodeBase*> all_oprs, broadcasts;
        // Get the varnode array and find all broadcasts.
        for (OperatorNodeBase* iter = opr;
             opr_in_chain(iter, &chain_input, iter == opr);
             iter = chain_input->owner_opr()) {
            if (visited.count(iter))
                break;
            if (iter->same_type<opr::Broadcast>()) {
                broadcasts.push_back(iter);
            }
            visited.insert(iter);
            all_oprs.push_back(iter);
        }
        if (broadcasts.empty()) {
            auto_replace_with_context(opr);
            replace_trailing_bcasts();
            return;
        }

        // we only need to process the chain from first broadcast
        while (all_oprs.back() != broadcasts.back()) {
            all_oprs.pop_back();
        }

        auto prev = build_chain(all_oprs);
        for (auto broadcast : reverse_adaptor(broadcasts)) {
            // add it back to operator.
            opt.call_with_opr(broadcast, [&]{
                // create new opr with the original opr's properties
                auto new_broadcast =
                    opr::Broadcast::make(
                        prev, rewriter.get_var(broadcast->input(1)), {})
                        .node();
                prev = new_broadcast;
            });
        }
        // Following line would not trigger opr properties check.
        // The new oprs created before are all constructed in a temporary
        // context, so no opr insertion registered in current context.
        // We have reordered the oprs on the chain, so check the last
        // opr on the chain is meaningless since sometimes prev->owner_opr()
        // is a broadcast but \p opr not.
        rewriter.replace_var(opr->output(0), prev,
                             mgb_cstr_log("insert broadcast %s"));
        replace_trailing_bcasts();
    };

    auto on_opr = [&](OperatorNodeBase* opr) {
        VarNode* chain_input;
        dep_on_bcast[opr] = opr->same_type<opr::Broadcast>() ||
                            (opr_in_chain(opr, &chain_input, true) &&
                             dep_on_bcast[chain_input->owner_opr()]);
        if (opr_in_chain(opr, &chain_input, true)) {
            if (is_opr_endpoint(opr))
                process_chain_from_endpoint(opr);
            else
                rewriter.auto_replace_outputs(opr);
        } else {
            for (auto inp : opr->input()) {
                if (opr_in_chain(inp->owner_opr(), &chain_input, true) &&
                    !visited.count(inp->owner_opr())) {
                    process_chain_from_endpoint(inp->owner_opr());
                }
            }
            rewriter.auto_replace_outputs(opr);
        }
    };

    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
}

/* ======================= RecompTypeCvtPass ====================== */

const char* RecompTypeCvtPass::name() const {
    return "recomp_typecvt_pass";
}

void RecompTypeCvtPass::apply(OptState& opt) const {
    auto rewriter = opt.graph().make_rewriter();

    auto allowed_typecvt = [](OperatorNodeBase* opr) -> OperatorNodeBase* {
        if (!opr->same_type<opr::TypeCvt>())
            return nullptr;
        if (opr->input().size() != 1 || opr->output().size() != 1)
            return nullptr;
        if (opr->input(0)->dtype().size() < opr->output(0)->dtype().size()) {
            return opr;
        }
        return nullptr;
    };

    size_t step = 0;
    auto opr2step = ThinHashMap<OperatorNodeBase*, size_t>();
    auto on_opr = [&](OperatorNodeBase* opr) {
        VarNodeArray rewritten_inputs;
        step++;
        bool any_inp_changed = false;
        for (auto inp : opr->input()) {
            bool inp_changed = false;
            if (auto typecvt = allowed_typecvt(inp->owner_opr())) {
                auto iter = opr2step.find(typecvt);
                if (iter != opr2step.end()) {
                    size_t prev_step = iter->second;
                    if (step - prev_step > m_threshold) {
                        OperatorNodeConfig config = opr->config();
                        config.instance_id(opr);
                        opt.call_with_opr(typecvt, [&]{
                            auto new_typecvt =
                                    opr::TypeCvt::make(
                                            rewriter.get_var(typecvt->input(0)),
                                            typecvt->output(0)->dtype(), config)
                                            .node();
                            new_typecvt->owner_opr()
                                    ->node_prop()
                                    .attribute()
                                    .priority = std::numeric_limits<int>::max();
                            rewritten_inputs.push_back(new_typecvt);
                        }, OprPropertyFlag::ALL ^ OprPropertyFlag::PRIORITY);
                        inp_changed = true;
                    }
                } else {
                    opr2step[typecvt] = step;
                }
            }
            if (!inp_changed)
                rewritten_inputs.push_back(rewriter.get_var(inp));
            if (inp_changed || inp != rewriter.get_var(inp))
                any_inp_changed = true;
        }
        if (any_inp_changed) {
            auto new_opr = serialization::copy_opr_shallow(
                    *opr, rewritten_inputs, opr->config());
            if (new_opr != opr) {
                for (size_t i = 0; i < opr->output().size(); ++i)
                    if (!opr->output(i)->contain_flag(
                                VarNode::Flag::VOLATILE_CONTENT))
                        rewriter.replace_var(opr->output(i), new_opr->output(i),
                                             mgb_cstr_log(""));
            }
        }
    };
    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
}

/* ======================= CombineAstypeAndReducePass ====================== */

const char* CombineAstypeAndReducePass::name() const {
    return "combine_astype_and_reduce";
}

void CombineAstypeAndReducePass::apply(OptState& opt) const {
    auto rewriter = opt.graph().make_rewriter();

    using DataType = opr::Reduce::Param::DataType;

    auto get_data_type = [](DType before, DType after) {
#if !MEGDNN_DISABLE_FLOAT16
        if (before == dtype::Float16() && after == dtype::Float32())
            return DataType::FLOAT_O32xC32;
#endif
        return DataType::DEFAULT;
    };

    auto on_opr = [&](OperatorNodeBase* opr) {
        if (auto reduce = try_cast_as_op<opr::Reduce>(opr)) {
            auto inp = rewriter.get_var(reduce->input(0));
            if (inp->owner_opr()->same_type<opr::TypeCvt>()) {
                auto data_type = get_data_type(
                        inp->owner_opr()->input(0)->dtype(), inp->dtype());

                if (data_type != DataType::DEFAULT) {
                    opr::Reduce::Param param = reduce->param();
                    param.data_type = data_type;
                    VarNode* target_shape = nullptr;
                    if (param.axis < -MEGDNN_MAX_NDIM ||
                        param.axis >= MEGDNN_MAX_NDIM) {
                        mgb_assert(reduce->input().size() > 1);
                        target_shape = reduce->input(1);
                    } else {
                        mgb_assert(reduce->input().size() == 1);
                    }
                    auto new_var =
                            opr::Reduce::make(inp->owner_opr()->input(0), param,
                                              target_shape, opr->config())
                                    .node();
                    rewriter.replace_var(opr->output(0), new_var,
                                         mgb_cstr_log("replace reduce"));
                    return;
                }
            }
        }
        rewriter.auto_replace_outputs(opr);
    };

    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
}

/* ================ CondExecConstPredicateFolding ================ */
const char* CondExecConstPredicateFolding::name() const {
    return "cond_exec_const_predicate_folding";
}

void CondExecConstPredicateFolding::apply(OptState& opt) const {
#if MGB_ENABLE_COND_EXEC
    if (!cg::ExecutionMask::have_alive_instance()) {
        return;
    }

    // replace var with unmasked version for active branches, and mark inactive
    // branches in const_mask

    ConstVarPropogate const_prop{ConstVarType::IMMUTABLE};

    auto&& mgr = opt.graph().comp_graph()->static_infer_manager();
    // value of PPV
    auto get_ppvv = [&](VarNode* var) -> const int* {
        const_prop.add_opr(var->owner_opr());
        if (const_prop.is_const(var)) {
            return mgr.infer_value(var).ptr<int>();
        }
        return nullptr;
    };
    // mask to ppvv value
    ThinHashMap<cg::ExecutionMask*, int> const_mask;

    auto rewriter = opt.graph().make_rewriter();

    auto handle_merge = [&](opr::CondExecMerge& opr) -> bool {
        SmallVector<size_t> active_br;
        size_t nr_out = opr.param().nr_output,
               nr_branch = opr.branch_masks().size();
        for (size_t i = 0; i < nr_branch; ++i) {
            auto iter = const_mask.find(opr.branch_masks()[i]);
            if (iter == const_mask.end()) {
                return false;
            }
            if (iter->second) {
                active_br.push_back(i);
            }
        }

        using Mode = opr::CondExecMerge::Param::Mode;
        auto mode = opr.param().mode;

        if (mode == Mode::EXACT_ONE || mode == Mode::EXACT_ONE_SAME_SHAPE) {
            mgb_assert(active_br.size() == 1,
                       "%zu branches are active for EXACT_ONE CondExecMark %s",
                       active_br.size(), opr.cname());
        }

        SymbolVarArray ovars(nr_out);
        if (active_br.empty()) {
            if (mode == Mode::SUM) {
                auto shp_inp = opr.input().data() + nr_out * nr_branch;
                for (size_t i = 0; i < nr_out; ++i) {
                    auto shp = rewriter.get_var(shp_inp[i]);
                    if (cg::ExecutionMask::get_from_opr(shp->owner_opr())) {
                        // output should have no mask
                        return false;
                    }
                    ovars[i] = SymbolVar{opr.output(i)}
                                       .make_scalar_dt(0)
                                       .broadcast(shp);
                }
            } else {
                mgb_assert(mode == Mode::SUM_COND_OUT);
                auto mask = cg::ExecutionMask::get_from_opr(&opr);
                mgb_assert(mask && mask->owner() == opr.input().back());
                auto ppvv = get_ppvv(mask->owner());
                mgb_assert(ppvv && !ppvv[0]);
                const_mask[mask] = 0;
                // mark as false and do nothing more
                return false;
            }
        } else {
            auto inp = [&](size_t br, size_t oidx) {
                return rewriter.get_var(opr.input(br * nr_out + oidx));
            };
            for (auto br_idx : active_br) {
                for (size_t i = 0; i < nr_out; ++i) {
                    auto sum = ovars[i];
                    if (!sum.node()) {
                        sum = inp(br_idx, i);
                    } else {
                        sum = sum + inp(br_idx, i);
                    }
                    ovars[i] = sum;
                }
            }
        }

        for (size_t i = 0; i < nr_out; ++i) {
            rewriter.replace_var(opr.output(i), ovars[i].node(),
                                 mgb_cstr_log("const merge"));
        }

        return true;
    };

    auto on_opr = [&](OperatorNodeBase* opr) {
        auto opr_type = opr->dyn_typeinfo();
        if (opr_type->is<opr::CondExecMark>()) {
            if (auto ppvv = get_ppvv(opr->input().back())) {
                auto mask = cg::ExecutionMask::get_from_opr(opr);
                mgb_assert(mask && mask->owner() == opr->input().back());
                if (ppvv[0]) {
                    for (size_t i = 0; i < opr->output().size(); ++i) {
                        rewriter.replace_var(opr->output(i),
                                             rewriter.get_var(opr->input(i)),
                                             mgb_cstr_log("const true mark"));
                    }
                    const_mask[mask] = 1;
                } else {
                    const_mask[mask] = 0;
                }
            }
            return;
        }
        if (opr_type->is<opr::CondExecMerge>()) {
            if (!handle_merge(opr->cast_final<opr::CondExecMerge>())) {
                for (auto i : opr->output()) {
                    rewriter.replace_var(
                            i, i,
                            mgb_cstr_log("keep when not all inputs have const "
                                         "mask"));
                }
            }
            return;
        }
        rewriter.auto_replace_outputs(opr);
    };

    opt.graph().iter(on_opr);
    for (auto i : opt.graph().endpoint_vars()) {
        auto mask = cg::ExecutionMask::get_from_opr(i.node()->owner_opr());
        if (mask) {
            auto iter = const_mask.find(mask);
            if (iter != const_mask.end()) {
                mgb_throw_if(!iter->second, GraphError,
                             "endpoint is not reachable due to conditional "
                             "execution: %s",
                             cg::dump_var_info({i}).c_str());
            }
        }
    }

    rewriter.apply_inplace();

#endif  // MGB_ENABLE_COND_EXEC
}

/* ======================= RemoveRedundantTypeCvtPass ====================== */

const char* RemoveRedundantTypeCvtPass::name() const {
    return "remove_redundant_typecvt";
}

bool RemoveRedundantTypeCvtPass::should_remove(DType A, DType B) {
    if (A.category() == B.category() &&
        (B.category() == DTypeCategory::INT ||
         B.category() == DTypeCategory::FLOAT) &&
        B.size() >= A.size()) {
        return true;
    }
    if (B.enumv() == DTypeEnum::Float32 &&
        (A.category() == DTypeCategory::QUANTIZED ||
         // Integers with <= 24 bits can be expressed precisely in Float32.
         (A.category() == DTypeCategory::INT && A.size() * 8 <= 24))) {
        return true;
    }
    return false;
}

void RemoveRedundantTypeCvtPass::apply(OptState& opt) const {
    auto rewriter = opt.graph().make_rewriter();

    auto on_opr = [&](OperatorNodeBase* opr) {
        if (auto tc0 = try_cast_as_op<opr::TypeCvt>(opr)) {
            if (auto tc1 = try_cast_as_op<opr::TypeCvt>(tc0->input(0))) {
                if (should_remove(tc0->param(), tc1->param())) {
                    // TypeCvt returns the input var if its dtype is already
                    // dest_type
                    auto fold = opr::TypeCvt::make(tc1->input(0), tc0->param());
                    rewriter.replace_var(
                            tc0->output(0), fold.node(),
                            mgb_cstr_log("cvt_b(cvt_a(x)) -> cvt_b(x)"));
                }
            }
        }
        rewriter.auto_replace_outputs(opr);
    };

    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
