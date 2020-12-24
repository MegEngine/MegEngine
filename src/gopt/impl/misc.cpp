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
#include "../../core/impl/graph/cg_impl.h"

#include "megbrain/utils/hash_ct.h"
#include "midout.h"

MIDOUT_DECL(megbrain_misc)
#define MIDOUT_B(tag) \
    MIDOUT_BEGIN(megbrain_misc, midout_iv(MGB_HASH_STR(tag))) {
#define MIDOUT_E \
    }            \
    MIDOUT_END();

using namespace mgb;
using namespace gopt;

/* ================ RemoveNonComputingOprPass ================ */

const char* RemoveNonComputingOprPass::name() const {
    return "remove_non_computing_opr";
}

void RemoveNonComputingOprPass::apply(OptState& opt) const {
    MIDOUT_B("RemoveNonComputingOprPass::apply")
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
    MIDOUT_E
}

/* ================ ExpandVirtualGradPass ================ */

const char* ExpandVirtualGradPass::name() const {
    return "expand_virtual_grad";
}

void ExpandVirtualGradPass::apply(OptState& opt) const {
    MIDOUT_B("ExpandVirtualGradPass::apply")
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
    MIDOUT_E
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
    MIDOUT_B("DelayBroadcastPass::apply")
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
    MIDOUT_E
}

/* ======================= RecompTypeCvtPass ====================== */

const char* RecompTypeCvtPass::name() const {
    return "recomp_typecvt_pass";
}

void RecompTypeCvtPass::apply(OptState& opt) const {
    MIDOUT_B("RecompTypeCvtPass::apply")
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
                        config.update_instance_id(opr);
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
    MIDOUT_E
}

/* ======================= CombineAstypeAndReducePass ====================== */

const char* CombineAstypeAndReducePass::name() const {
    return "combine_astype_and_reduce";
}

void CombineAstypeAndReducePass::apply(OptState& opt) const {
    MIDOUT_B("CombineAstypeAndReducePass::apply")
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
    MIDOUT_E
}

/* ================ CondExecConstPredicateFolding ================ */
const char* CondExecConstPredicateFolding::name() const {
    return "cond_exec_const_predicate_folding";
}

void CondExecConstPredicateFolding::apply(OptState& opt) const {
#if MGB_ENABLE_COND_EXEC
    MIDOUT_B("CondExecConstPredicateFolding::apply")
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
    MIDOUT_E

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
    MIDOUT_B("RemoveRedundantTypeCvtPass::apply")
    auto rewriter = opt.graph().make_rewriter();

    auto on_opr = [&](OperatorNodeBase* opr) {
        if (auto tc0 = try_cast_as_op<opr::TypeCvt>(opr)) {
            auto inp0 = rewriter.get_var(tc0->input(0));
            if (auto tc1 = try_cast_as_op<opr::TypeCvt>(inp0)) {
                if (should_remove(tc0->param(), tc1->param())) {
                    auto inp1 = tc1->input(0);
                    mgb_assert(!rewriter.has_manual_replace(inp1));
                    // TypeCvt returns the input var if its dtype is already
                    // dest_type
                    auto fold = opr::TypeCvt::make(inp1, tc0->param());
                    rewriter.replace_var(
                            tc0->output(0), fold.node(),
                            mgb_cstr_log("cvt_b(cvt_a(x)) -> cvt_b(x)"));
                }
                return;
            }
        }
        rewriter.auto_replace_outputs(opr);
    };

    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
    MIDOUT_E
}

/* ======================= RemoveRedundantCopyPass ====================== */

const char* RemoveRedundantCopyPass::name() const {
    return "remove_redundant_copy";
}

bool RemoveRedundantCopyPass::should_remove(const CompNode& A,
                                            const CompNode& B) {
    //! if A and B has the same memnode and cpu <-> atlas/cpu <-> cuda, as only
    //! these two compnode support crosscncopy
    if (A.mem_node() == B.mem_node() ||
        ((A.device_type() == CompNode::DeviceType::CPU ||
          A.device_type() == CompNode::DeviceType::MULTITHREAD) &&
         (B.device_type() == CompNode::DeviceType::ATLAS ||
          B.device_type() == CompNode::DeviceType::CUDA)) ||
        ((B.device_type() == CompNode::DeviceType::CPU ||
          B.device_type() == CompNode::DeviceType::MULTITHREAD) &&
         (A.device_type() == CompNode::DeviceType::ATLAS ||
          A.device_type() == CompNode::DeviceType::CUDA))) {
        return true;
    } else {
        return false;
    }
}

void RemoveRedundantCopyPass::apply(OptState& opt) const {
    MIDOUT_B("RemoveRedundantCopyPass::apply")
    auto rewriter = opt.graph().make_rewriter();

    auto on_opr = [&](OperatorNodeBase* opr) {
        if (auto copy0 = try_cast_as_op<opr::Copy>(opr)) {
            auto inp0 = rewriter.get_var(copy0->input(0));
            if (auto copy1= try_cast_as_op<opr::Copy>(inp0)) {
                auto inp1 = copy1->input(0);
                if (should_remove(inp1->comp_node(),
                                  copy0->output(0)->comp_node())) {
                    mgb_assert(!rewriter.has_manual_replace(inp1));
                    if (inp1->comp_node() == copy0->output(0)->comp_node()) {
                        rewriter.replace_var(
                                copy0->output(0), inp1,
                                mgb_cstr_log("copy(copy(a0, a1), a0) -> "
                                             "a0"));
                        return;
                    } else {
                        auto fold = opr::Copy::make(
                                inp1, copy0->output(0)->comp_node());
                        rewriter.replace_var(
                                copy0->output(0), fold.node(),
                                mgb_cstr_log("copy(copy(a0, a1), a2) -> "
                                             "copy(a0, a2)"));
                        return;
                    }
                }
            }
        }
        rewriter.auto_replace_outputs(opr);
    };

    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
    MIDOUT_E
}

#if MGB_ENABLE_OPR_MM
#include "megbrain/opr/collective_comm.h"

/* ======================= PackAllReduceScanPass ====================== */

const char* PackAllReduceScanPass::name() const {
    return "pack_allreduce_scan";
}

void PackAllReduceScanPass::apply(OptState& opt) const {
    MIDOUT_B("PackAllReduceScanPass::apply")
    auto comp_graph = opt.graph().comp_graph();
    if (comp_graph->options().allreduce_pack_max_size == 0) return;
    auto cb_scan = [this] (OperatorNodeBase* opr) {
        if (check_pattern(opr)) {
            auto& comm = opr->cast_final_safe<opr::CollectiveComm>();
            VarNode* target = comm.input(0)->owner_opr()->input(0);
            // only pack allreduces of grads of the same target
            // in case two allreduces depend on each other
            size_t id = target->id();
            uint64_t hash = XXHash().update(&id, sizeof(size_t)).digest();
            comm.set_pack_hash(hash);
        }
    };
    opt.graph().iter(cb_scan);
    MIDOUT_E
}

bool PackAllReduceScanPass::check_pattern(OperatorNodeBase* opr) {
    if (!opr->same_type<opr::CollectiveComm>()) return false;
    auto& comm = opr->cast_final_safe<opr::CollectiveComm>();
    if (comm.param().mode != opr::CollectiveComm::Param::Mode::ALL_REDUCE_SUM) return false;
    if (comm.local_grad()) return false;
    if (comm.input().size() != 1) return false;

    auto grad = comm.input(0)->owner_opr();
    if (!grad->same_type<opr::VirtualGrad>()) return false;
    if (grad->input().size() != 2 or grad->output().size() != 1) return false;

    auto param = grad->input(1)->owner_opr();
    if (!param->same_type<opr::SharedDeviceTensor>() and
        !param->same_type<opr::VolatileSharedDeviceTensor>()) return false;
    if (param->input().size() != 0) return false;

    return true;
}

/* ======================= PackAllReduceReplacePass ====================== */

const char* PackAllReduceReplacePass::name() const {
    return "pack_allreduce_replace";
}

class PackAllReduceReplacePass::GroupInfo {
public:
    GroupInfo(int _device, DType _dtype,
            size_t _nr_devices, bool _is_root, int _rank,
            std::shared_ptr<opr::GroupClient> _group_client,
            const std::string& _backend);

    uint64_t hash(uint64_t extra) const;

    int device;
    DType dtype;
    size_t nr_devices;
    bool is_root;
    int rank;
    std::shared_ptr<opr::GroupClient> group_client;
    std::string backend;
};

PackAllReduceReplacePass::GroupInfo::GroupInfo(
        int _device, DType _dtype,
        size_t _nr_devices, bool _is_root, int _rank,
        std::shared_ptr<opr::GroupClient> _group_client,
        const std::string& _backend) :
    device(_device), dtype(_dtype),
    nr_devices(_nr_devices), is_root(_is_root), rank(_rank),
    group_client(_group_client), backend(_backend) {
}

uint64_t PackAllReduceReplacePass::GroupInfo::hash(uint64_t extra) const {
    DTypeEnum ev = dtype.enumv();
    const std::string& server_addr = group_client->get_addr();
    return XXHash()
        .update(&extra, sizeof(uint64_t))
        .update(&device, sizeof(int))
        .update(&ev, sizeof(DTypeEnum))
        .update(&nr_devices, sizeof(size_t))
        .update(&is_root, sizeof(bool))
        .update(&rank, sizeof(int))
        .update(server_addr.c_str(), server_addr.size())
        .update(backend.c_str(), backend.size())
        .digest();
}

uint64_t PackAllReduceReplacePass::collect_groups(OperatorNodeBase* opr,
        ThinHashMap<uint64_t, std::shared_ptr<GroupInfo>>& group_info,
        ThinHashMap<uint64_t, cg::OprNodeArray>& groups) {
    // check CollectiveComm oprs that have been marked in PackAllReduceScanPass
    if (!opr->same_type<opr::CollectiveComm>()) return 0;
    opr::CollectiveComm& comm = opr->cast_final_safe<opr::CollectiveComm>();
    if (comm.pack_hash() == 0) return 0;  // pack_hash not set

    VarNode* var = comm.input(0);
    auto info = std::make_shared<GroupInfo>(
        var->comp_node().locator().device,
        var->dtype(),
        comm.nr_devices(),
        comm.is_root(),
        comm.rank(),
        comm.group_client(),
        comm.backend()
    );
    uint64_t hash = info->hash(comm.pack_hash());
    if (group_info.find(hash) == group_info.end()) {
        group_info.emplace(hash, info);
    }
    groups[hash].push_back(opr);
    return hash;
}

void PackAllReduceReplacePass::divide_packs(
        const ThinHashMap<uint64_t, cg::OprNodeArray>& groups,
        ThinHashMap<uint64_t, std::vector<cg::OprNodeArray>>& packs,
        size_t max_size) {
    cg::OprNodeArray pack;
    size_t sum = 0;
    for (auto it : groups) {
        uint64_t hash = it.first;
        const cg::OprNodeArray& group = it.second;
        for (size_t i = 0; i < group.size(); i++) {
            OperatorNodeBase* opr = group[i];
            VarNode* var = opr->input(0);
            const TensorShape* shape = var->owner_graph()
                    ->static_infer_manager().infer_shape_fallible(var);
            if (shape == nullptr) continue;
            pack.push_back(opr);
            sum += var->dtype().size(shape->total_nr_elems());
            if (sum >= max_size) {
                if (pack.size() > 1) packs[hash].push_back(pack);
                pack.clear();
                sum = 0;
            }
        }
        if (pack.size() > 1) packs[hash].push_back(pack);
        pack.clear();
        sum = 0;
    }
}

void PackAllReduceReplacePass::insert_packed_oprs(
        size_t pack_id,
        const cg::OprNodeArray& pack,
        std::shared_ptr<GroupInfo> info,
        ThinHashMap<VarNode*, VarNode*>& replace_map, int priority) {
    // set priority
    mgb_assert(pack.size() > 0);
    auto graph = pack[0]->owner_graph();
    auto on_opr_inserted = [priority] (const cg::event::OprInserted& event) {
        event.opr->node_prop().attribute().priority = priority;
    };
    auto handler = graph->event().register_receiver<cg::event::OprInserted>(on_opr_inserted);

    // flatten inputs and record shapes and partition
    std::vector<SymbolVar> shapes;
    SymbolVarArray flattens;
    SymbolVarArray partition;
    for (size_t i = 0; i < pack.size(); i++) {
        VarNode* var = pack[i]->input(0);
        auto shape = opr::GetVarShape::make(SymbolVar(var));
        shapes.push_back(shape);
        SymbolVar flatten = SymbolVar(var).flatten();
        flattens.push_back(flatten);
        partition.push_back(opr::Reduce::make(shape, {opr::Reduce::Mode::PRODUCT, 0}));
    }

    // concat
    SymbolVar concat = opr::Concat::make(flattens, 0);

    // allreduce
    std::string key = ssprintf("grad_pack_%zu", pack_id);
    auto param = opr::CollectiveComm::Param::Mode::ALL_REDUCE_SUM;
    SymbolVar allreduce = opr::CollectiveComm::make({concat}, graph,
        key, info->nr_devices, info->is_root, info->rank, false,
        info->group_client, param, info->dtype, info->backend)[0];

    // split according to recorded partition
    SymbolVarArray splits = opr::Split::make(allreduce,
        opr::Split::Options::make_partition(0, partition));

    // reshape and insert results into replace_map
    mgb_assert(pack.size() == splits.size());
    for (size_t i = 0; i < pack.size(); i++) {
        VarNode* reshape = splits[i].reshape(shapes[i]).node();
        replace_map[pack[i]->output(0)] = reshape;
    }
}

void PackAllReduceReplacePass::apply(OptState& opt) const {
    MIDOUT_B("PackAllReduceReplacePass::apply")
    // get graph options
    auto comp_graph = opt.graph().comp_graph();
    size_t max_size = comp_graph->options().allreduce_pack_max_size * 1024 * 1024;
    size_t ignore_first = comp_graph->options().allreduce_pack_ignore_first;
    if (max_size == 0) return;

    // get topo order
    auto& topo_sorter = static_cast<cg::ComputingGraphImpl*>(comp_graph)->topo_sorter();
    cg::CompSeqExtraInfo extra_info;
    VarNodeArray endpoints = to_var_node_array(opt.graph().endpoint_vars());
    const cg::OprNodeArray* seq = topo_sorter.get_comp_seq(extra_info, endpoints);
    topo_sorter.restore_opr_prop();

    // collect allreduce groups from topo sequence
    ThinHashMap<uint64_t, std::shared_ptr<GroupInfo>> group_info;
    ThinHashMap<uint64_t, cg::OprNodeArray> groups;
    for (size_t i = 0; i < seq->size(); i++) {
        if (seq->at(i)->same_type<opr::CollectiveComm>()) {
            // ignore the first several allreduces
            if (ignore_first > 0) {
                --ignore_first;
            } else {
                collect_groups(seq->at(i), group_info, groups);
            }
        }
    }

    // divide groups into packs
    ThinHashMap<uint64_t, std::vector<cg::OprNodeArray>> packs;
    divide_packs(groups, packs, max_size);

    // make sure that oprs inserted in this pass (reshape, concat, allreduce,
    // split, reshape) have higher priority than existing operators
    int priority = -seq->size() - 100;

    // insert packed operators and generate replace_map
    ThinHashMap<VarNode*, VarNode*> replace_map;
    size_t pack_id = 0;
    for (auto it : packs) {
        uint64_t hash = it.first;
        for (auto pack : it.second) {
            opt.call_with_opr(pack[0], [&]() {
                insert_packed_oprs(pack_id, pack, group_info[hash], replace_map, priority);
            }, OprPropertyFlag::NONE);
            pack_id += 1;
        }
    }

    // replace vars
    auto rewriter = opt.graph().make_rewriter();
    auto cb_replace = [&](OperatorNodeBase* opr) {
        for (auto i : opr->input()) {
            auto iter = replace_map.find(i);
            if (iter != replace_map.end()) {
                rewriter.replace_var(i, iter->second, nullptr);
            }
        }
        rewriter.auto_replace_outputs(opr);
    };
    opt.graph().iter(cb_replace);
    rewriter.apply_inplace();
    MIDOUT_E
}

#else

/* ======================= PackAllReduceScanPass ====================== */

const char* PackAllReduceScanPass::name() const {
    return "pack_allreduce_scan";
}

void PackAllReduceScanPass::apply(OptState& opt) const {
}

bool PackAllReduceScanPass::check_pattern(OperatorNodeBase* opr) {
    return true;
}

/* ======================= PackAllReduceReplacePass ====================== */

const char* PackAllReduceReplacePass::name() const {
    return "pack_allreduce_replace";
}

void PackAllReduceReplacePass::apply(OptState& opt) const {}

uint64_t PackAllReduceReplacePass::collect_groups(
        OperatorNodeBase* opr,
        ThinHashMap<uint64_t, std::shared_ptr<GroupInfo>>& group_info,
        ThinHashMap<uint64_t, cg::OprNodeArray>& groups) {
    return 0;
}

void PackAllReduceReplacePass::divide_packs(
        const ThinHashMap<uint64_t, cg::OprNodeArray>& groups,
        ThinHashMap<uint64_t, std::vector<cg::OprNodeArray>>& packs,
        size_t max_size) {
}

void PackAllReduceReplacePass::insert_packed_oprs(
        size_t pack_id,
        const cg::OprNodeArray& pack,
        std::shared_ptr<GroupInfo> info,
        ThinHashMap<VarNode*, VarNode*>& replace_map, int priority) {
}

#endif  // MGB_ENABLE_OPR_MM

/* ======================= RemoveShapeHintPass ====================== */

const char* RemoveShapeHintPass::name() const {
    return "remove_shape_hint";
}

void RemoveShapeHintPass::apply(OptState& opt) const {
    MIDOUT_B("RemoveShapeHintPass::apply")
    opt.set_var_replace_check_flag(VarReplaceCheckFlag::CHECK_DTYPE);
    auto rewriter = opt.graph().make_rewriter();

    auto on_opr = [&](OperatorNodeBase* opr) {
        if (auto sh = try_cast_as_op<opr::ShapeHint>(opr)) {
            auto inp = rewriter.get_var(sh->input(0));
            rewriter.replace_var(sh->output(0), inp,
                mgb_cstr_log("remove shape hint"));
            return;
        }
        rewriter.auto_replace_outputs(opr);
    };

    opt.graph().iter(on_opr);
    rewriter.apply_inplace();
    MIDOUT_E
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
