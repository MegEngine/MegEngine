/**
 * \file src/opr/impl/cond.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/cond.h"
#include "megbrain/graph/event.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/utility.h"

using namespace mgb;
using namespace opr;

#if MGB_ENABLE_COND_EXEC

namespace {

//! return whether ``lhs -> rhs`` can be proved
bool can_prove_imply(cg::ExecutionMask* lhs, cg::ExecutionMask* rhs) {
    // this function is neither sound nor complete (and it can never be due
    // to the NP-completeness of SAT); here we only handle the most common
    // cases

    if (rhs == lhs->parent()) {
        // nested cond exec oprs
        return true;
    }

    using Mode = CondExecPredLogical::Mode;
    auto is_pred_logical = [](cg::OperatorNodeBase* opr, Mode mode) {
        auto as_p = opr->try_cast_final<CondExecPredLogical>();
        return as_p && as_p->param().mode == mode;
    };

    auto opr = rhs->owner()->owner_opr();

    if (is_pred_logical(opr, Mode::AND) && opr->input().size() == 1) {
        // cross-cn copy of predicate
        opr = opr->input(0)->owner_opr();
    }

    if (is_pred_logical(opr, Mode::OR)) {
        // in the grad of SUM_COND_OUT CondExecMerge
        auto lvar = lhs->owner();
        for (auto i : opr->input()) {
            if (lvar == i) {
                return true;
            }
        }
        return false;
    }
    return false;
}

VarNode* proxy_var_from_mask(cg::ExecutionMask* mask) {
    auto var = mask->owner();
    mgb_assert(var);
    auto opr = var->owner_opr();
    auto type = opr->dyn_typeinfo();
    mgb_assert(type->is<CondExecPred>() || type->is<CondExecPredLogical>(),
               "mask not from CondExec opr: %s",
               cg::dump_var_info({var}).c_str());
    return var;
}

#if MGB_ENABLE_LOGGING
std::string mask2str(cg::ExecutionMask* mask) {
    if (!mask) {
        return "null";
    }
    auto var = mask->owner();
    mgb_assert(var);
    if (var->owner_opr()->same_type<CondExecPred>()) {
        return ssprintf("CondExecPred(%s)", var->cname());
    }
    mgb_assert(var->owner_opr()->same_type<CondExecPredLogical>());
    return ssprintf("CondExecPredLogical(%s)", var->cname());
}
#else

std::string mask2str(cg::ExecutionMask*) {
    return "";
}
#endif

}  // anonymous namespace

/* ============================= CondExecPred ============================= */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(CondExecPred);

class CondExecPred::PredEvaluator {
public:
    enum Result { LT, EQ, GT };
    PredEvaluator(const CondExecPred& opr, const DeviceTensorND& pred);

    Result operator()(const DeviceTensorND& key) {
        pre_check(key);
        return m_compare(key);
    }

private:
    CompNode default_cpu = CompNode::default_cpu();
    thin_function<Result(const DeviceTensorND&)> m_compare;

    void pre_check(const DeviceTensorND& val) {
        mgb_assert(val.comp_node() == default_cpu);
        mgb_throw_if(!val.shape().is_scalar(), GraphError,
                     "CondExec predicate or branch key is not scalar: %s",
                     val.shape().to_string().c_str());
    }
};

CondExecPred::PredEvaluator::PredEvaluator(const CondExecPred& opr,
                                           const DeviceTensorND& pred) {
    pre_check(pred);
    switch (pred.dtype().enumv()) {
#define cbf(dt)                                                            \
    case DTypeTrait<dt>::enumv: {                                          \
        using ct = DTypeTrait<dt>::ctype;                                  \
        m_compare = [ eps = opr.m_param.eps,                               \
                      p = pred.ptr<ct>()[0] ](const DeviceTensorND& key) { \
            ct k = key.ptr<ct>()[0];                                       \
            return std::abs(p - k) < eps ? EQ : (p < k ? LT : GT);         \
        };                                                                 \
        break;                                                             \
    }
#define cbi(dt)                                                          \
    case DTypeTrait<dt>::enumv: {                                        \
        using ct = DTypeTrait<dt>::ctype;                                \
        m_compare = [p = pred.ptr<ct>()[0]](const DeviceTensorND& key) { \
            ct k = key.ptr<ct>()[0];                                     \
            return p == k ? EQ : (p < k ? LT : GT);                      \
        };                                                               \
        break;                                                           \
    }

        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cbf);
        MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cbi)
#undef cbf
#undef cbi

        default:
            mgb_throw(GraphError, "unsupported pred dtype: %s",
                      pred.dtype().name());
    }
}

class CondExecPred::GlobalRegistry final : public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;
    SyncEventConnecter::ReceiverHandler m_opr_insert_handler;
    ThinHashMap<VarNode*, ExecutionMask*> m_var2mask;

    void on_new_opr(OperatorNodeBase* opr);

public:
    static GlobalRegistry* get(ComputingGraph& graph) {
        using namespace cg::event;
        auto ptr = graph.options()
                           .user_data.get_user_data_or_create<GlobalRegistry>();
        if (!ptr->m_opr_insert_handler) {
            ptr->m_opr_insert_handler =
                    graph.event().register_receiver<OprInserted>(
                            [ptr](const OprInserted& ev) {
                                if (!ev.is_dedup && !ev.exc) {
                                    ptr->on_new_opr(ev.opr);
                                }
                            });
        }
        return ptr;
    }

    //! get mask if var is conditional, or nullptr otherwise
    ExecutionMask* get_mask_from_var(VarNode* var) const {
        auto iter = m_var2mask.find(var);
        return iter == m_var2mask.end() ? nullptr : iter->second;
    }

    //! throw error if var is not controlled by ExecutionMask
    ExecutionMask* require_mask_from_var(VarNode* var) const {
        auto mask = get_mask_from_var(var);
        mgb_throw_if(!mask, GraphError,
                     "var is not controlled by ExecutionMask: %s",
                     cg::dump_var_info({var}).c_str());
        return mask;
    }

    //! assert that a var is a PPV
    ExecutionMask* check_ppv(VarNode* var) const {
        auto mask = require_mask_from_var(var);
        mgb_throw_if(mask->owner() != var, GraphError,
                     "a conditional var is not PPV: mask=%s var=%s",
                     mask2str(mask).c_str(), cg::dump_var_info({var}).c_str());
        return mask;
    }
};
MGB_TYPEINFO_OBJ_IMPL(CondExecPred::GlobalRegistry);

void CondExecPred::GlobalRegistry::on_new_opr(OperatorNodeBase* const opr) {
    // mask that controls execution of this opr
    ExecutionMask* mask = nullptr;

    auto opr_type = opr->dyn_typeinfo();
    bool opr_is_mark = opr_type->is<CondExecMark>(),
         opr_is_merge = opr_type->is<CondExecMerge>(),
         opr_is_pred_logical = opr_type->is<CondExecPredLogical>();

    using MergeMode = CondExecMerge::Mode;
    MergeMode merge_mode =
            opr_is_merge ? opr->cast_final<CondExecMerge>().param().mode
                         : static_cast<MergeMode>(-1);
    bool opr_follow_pred =
            opr_is_mark ||
            (opr_is_merge && merge_mode == MergeMode::SUM_COND_OUT);

    // find mask from inputs
    auto&& inputs = opr->input();
    for (size_t idx = 0; idx < inputs.size(); ++idx) {
        auto i_var = inputs[idx];
        ExecutionMask* i_mask = nullptr;
        auto i_owner = i_var->owner_opr();

        bool i_is_pred = false;
        if (i_owner->same_type<CondExecPred>() ||
            i_owner->same_type<CondExecPredLogical>()) {
            i_is_pred = true;
            mgb_throw_if(!((opr_follow_pred && i_var == opr->input().back()) ||
                           opr_is_pred_logical),
                         GraphError,
                         "predicate proxy var not received by CondExec "
                         "mark/merge opr: var=%s recv_opr=%s{%s}",
                         cg::dump_var_info({i_var}).c_str(), opr->cname(),
                         opr->dyn_typeinfo()->name);
        }

        if (opr_follow_pred && i_var == opr->input().back()) {
            // CondExecMerge(with SUM_COND_OUT) and CondExecMark are controlled
            // by given pred
            mgb_assert(i_is_pred);
            i_mask = m_var2mask.at(i_var);
            if (mask) {
                // here we handle the nested case; note that pred is the last
                // input, so other inputs have been processed and mask is
                // derived from other inputs
                mgb_throw_if(!can_prove_imply(i_mask, mask), GraphError,
                             "can not prove opr mask implies inputs mask: "
                             "opr=%s{%s}: opr_mask=%s "
                             "inputs_mask=%s",
                             opr->cname(), opr->dyn_typeinfo()->name,
                             mask2str(i_mask).c_str(), mask2str(mask).c_str());
            }
            mask = i_mask;
            break;
        }

        if (!i_mask) {
            auto iter = m_var2mask.find(i_var);
            i_mask = iter == m_var2mask.end() ? nullptr : iter->second;
        }

        if (opr_is_pred_logical && i_mask) {
            // CondExecPredLogical should only combine preds from the
            // higher-level same mask
            i_mask = i_mask->parent();
        }

        if (opr_is_merge) {
            if (merge_mode == MergeMode::SUM &&
                idx >= inputs.size() - opr->output().size()) {
                // the remaining inputs are output shapes; if they can not be
                // statically inferred, their execution mask must be on the same
                // level of this CondExecMerge, so we do not modify i_mask
                if (cg::is_static_var_value(i_var)) {
                    // no need to add execution mask for statically inferrable
                    // values
                    i_mask = nullptr;
                }
            } else if (i_mask) {
                // execution of merge opr is controlled by mask at a higher
                // level
                i_mask = i_mask->parent();
            }
        }

        if (i_mask) {
            auto lower = ExecutionMask::find_direct_lowest(mask, i_mask);
            mgb_throw_if(!lower, GraphError,
                         "different ExecutionMask trees on inputs of a single "
                         "opr: opr=%s{%s} mask0=%s mask1=%s",
                         opr->cname(), opr->dyn_typeinfo()->name,
                         mask2str(mask).c_str(), mask2str(i_mask).c_str());
            mask = lower;
        }
    }

    if (mask) {
        mask->register_to_opr(opr);
        for (auto i : opr->output()) {
            m_var2mask[i] = mask;
        }
    }

    // register nested masks and record var2mask map
    if (opr_type->is<CondExecPred>()) {
        size_t idx = 0;
        for (auto&& i : opr->cast_final<CondExecPred>().masks()) {
            if (mask) {
                mask->add_nested(i.get());
            }
            m_var2mask[opr->output(idx++)] = i.get();
        }
    } else if (opr_is_pred_logical) {
        auto m = opr->cast_final<CondExecPredLogical>().mask();
        if (mask) {
            mask->add_nested(m);
        }
        m_var2mask[opr->output(0)] = m;
    }
}

CondExecPred::CondExecPred(VarNode* pred, const VarNodeArrayView& keys,
                           const Param& param, const OperatorNodeConfig& config)
        : Super(pred->owner_graph(), config, "cond_pred", {pred}),
          m_param{param} {
    m_masks.reserve(keys.size() + 1);
    auto add_out = [this](const std::string& name) {
        auto var = add_output(name);
        var->add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC).dtype(dtype::Int32{});
        m_masks.emplace_back(std::make_shared<ExecutionMask>(var));
    };
    for (size_t i = 0; i < keys.size(); ++i) {
        mgb_throw_if(keys[i]->dtype() != pred->dtype(), GraphError,
                     "dtype mismatch: pred=%s input[%zu]=%s",
                     pred->dtype().name(), i, keys[i]->dtype().name());
        add_input({keys[i]});
        if (param.mode == Param::Mode::PIECEWISE) {
            if (!i) {
                add_out("[-inf,k0]");
            }
            if (i != keys.size() - 1) {
                add_out(ssprintf("[k%zu,k%zu]", i, i + 1));
            } else {
                add_out(ssprintf("[k%zu,inf]", i));
            }
        } else {
            add_out(ssprintf("branch%zu", i));
        }
    }
    if (param.mode == Param::Mode::CASE_FALLBACK) {
        add_out("fallback");
    }
    add_input({pred});
    add_equivalence_component<PODHash<Param>>(&m_param);

    // ensure listener is registered
    GlobalRegistry::get(*owner_graph());
}

cg::OperatorNodeBase* CondExecPred::make_opr(SymbolVar pred,
                                             const VarNodeArrayView& keys,
                                             const Param& param,
                                             const OperatorNodeConfig& config) {
    return pred.node()->owner_graph()->insert_opr(
            std::make_unique<CondExecPred>(pred.node(), keys, param, config));
}

void CondExecPred::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();

    for (auto i : output()) {
        mgr.register_shape_infer(i, ShapeInferDesc::make_const({1}));
    }

    auto reg_value_infer_no_const = [&mgr](VarNode* var, ValueInferDesc& desc) {
        auto orig_size = desc.deps.size();
        mixin::ForwardInputToOutput::ensure_not_replaced_by_const_folding(desc);
        mgr.register_value_infer(var, desc);
        if (desc.deps.size() != orig_size) {
            // remove newly added dep
            mgb_assert(desc.deps.size() == orig_size + 1);
            desc.deps.pop_back();
        }
    };

    size_t nr_key = input().size() - 1;

    auto mode = m_param.mode;
    if (mode == Mode::CASE || mode == Mode::CASE_FALLBACK) {
        auto infer_val_eq = [this](DeviceTensorND& dest, const InpVal& inp) {
            auto&& pv = inp.val[0].value();
            auto&& key = inp.val[1].value();
            dest.resize({1}).ptr<int>()[0] =
                    (PredEvaluator{*this, pv}(key) == PredEvaluator::EQ);
            return true;
        };
        ValueInferDesc desc{
                SourceType::DEP,
                {{input().back(), DepType::VALUE}, {nullptr, DepType::VALUE}},
                infer_val_eq};
        for (size_t i = 0; i < nr_key; ++i) {
            desc.deps[1].dest = input(i);
            reg_value_infer_no_const(output(i), desc);
        }

        if (mode == Mode::CASE_FALLBACK) {
            desc.deps.clear();
            for (size_t i = 0; i < nr_key; ++i) {
                desc.deps.push_back({output(i), DepType::VALUE});
            }
            desc.infer_func = [](DeviceTensorND& dest, const InpVal& inp) {
                int r = 1;
                for (auto&& i : inp.val) {
                    if (i.value().ptr<int>()[0]) {
                        r = 0;
                        break;
                    }
                }
                dest.resize({1}).ptr<int>()[0] = r;
                return true;
            };
            reg_value_infer_no_const(output().back(), desc);
        }
    } else {
        mgb_assert(mode == Mode::PIECEWISE);
        auto infer_first = [this](DeviceTensorND& dest, const InpVal& inp) {
            auto&& pv = inp.val[0].value();
            auto&& key = inp.val[1].value();
            dest.resize({1}).ptr<int>()[0] =
                    (PredEvaluator{*this, pv}(key) == PredEvaluator::LT);
            return true;
        };
        auto infer_mid = [this](DeviceTensorND& dest, const InpVal& inp) {
            auto&& pv = inp.val[0].value();
            auto&& left = inp.val[1].value();
            auto&& right = inp.val[2].value();
            PredEvaluator eval{*this, pv};
            auto el = eval(left), er = eval(right);
            dest.resize({1}).ptr<int>()[0] =
                    (el != PredEvaluator::LT && er == PredEvaluator::LT);
            return true;
        };
        auto infer_last = [this](DeviceTensorND& dest, const InpVal& inp) {
            auto&& pv = inp.val[0].value();
            auto&& key = inp.val[1].value();
            dest.resize({1}).ptr<int>()[0] =
                    (PredEvaluator{*this, pv}(key) != PredEvaluator::LT);
            return true;
        };

        // (-inf, key[0])
        ValueInferDesc desc{
                SourceType::DEP,
                {{input().back(), DepType::VALUE}, {input(0), DepType::VALUE}},
                infer_first};
        reg_value_infer_no_const(output(0), desc);

        // [key[i-1], key[i])
        desc.deps.push_back({nullptr, DepType::VALUE});
        desc.infer_func = infer_mid;
        for (size_t i = 1; i < nr_key; ++i) {
            desc.deps[1].dest = input(i - 1);
            desc.deps[2].dest = input(i);
            reg_value_infer_no_const(output(i), desc);
        }

        // [key[n-1], inf)
        desc.deps.resize(2);
        desc.deps[1].dest = input(nr_key - 1);
        desc.infer_func = infer_last;
        reg_value_infer_no_const(output(nr_key), desc);
    }
}

CondExecPred::NodeProp* CondExecPred::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    for (auto&& i : ret->dep_map()) {
        i.second = NodeProp::DepType::HOST_VALUE;
    }
    return ret;
}

void CondExecPred::scn_do_execute() {
    auto&& mgr = owner_graph()->static_infer_manager();
    PredEvaluator eval{*this, mgr.infer_value(input().back())};
    auto mode = m_param.mode;
    if (mode == Mode::CASE || mode == Mode::CASE_FALLBACK) {
        bool enabled = false;
        for (size_t i = 0; i < input().size() - 1; ++i) {
            auto cur = eval(mgr.infer_value(input(i))) == PredEvaluator::EQ;
            m_masks[i]->enable(cur);
            enabled |= cur;
        }
        if (mode == Mode::CASE_FALLBACK) {
            m_masks.back()->enable(!enabled);
        }
    } else {
        mgb_assert(mode == Mode::PIECEWISE);
        const DeviceTensorND *val_prev = nullptr, *val_cur = nullptr;
        for (size_t i = 0; i < input().size(); ++i) {
            val_prev = val_cur;
            if (i == input().size() - 1) {
                val_cur = nullptr;
            } else {
                val_cur = &mgr.infer_value(input(i));
            }

            PredEvaluator::Result el, er;
            if (!val_prev) {
                el = PredEvaluator::GT;
            } else {
                el = eval(*val_prev);
            }
            if (!val_cur) {
                er = PredEvaluator::LT;
            } else {
                er = eval(*val_cur);
            }
            m_masks[i]->enable(el != PredEvaluator::LT &&
                               er == PredEvaluator::LT);
        }
    }
}

VarNode* CondExecPred::out_var_from_mask(ExecutionMask* mask) const {
    for (size_t i = 0; i < output().size(); ++i) {
        if (mask == m_masks[i].get()) {
            return output(i);
        }
    }
    mgb_throw(AssertionError, "bad mask");
}

/* ========================== CondExecPredLogical ========================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(CondExecPredLogical);

class CondExecPredLogical::PredEvaluator {
    //! return false to early stop
    bool (*m_updater)(int*, int);
    int m_cur_val, m_negate = 0;

public:
    explicit PredEvaluator(Mode mode, int init) : m_cur_val{init} {
        auto fn_or = [](int* dst, int v) -> bool {
            *dst |= v;
            return !*dst;
        };
        auto fn_and = [](int* dst, int v) -> bool {
            *dst &= v;
            return *dst;
        };
        auto fn_xor = [](int* dst, int v) -> bool {
            *dst ^= v;
            return true;
        };
        switch (mode) {
            case Mode::NOR:
                m_negate = 1;
                // falls through
            case Mode::OR:
                m_updater = fn_or;
                break;
            case Mode::NAND:
                m_negate = 1;
                // falls through
            case Mode::AND:
                m_updater = fn_and;
                break;
            case Mode::XNOR:
                m_negate = 1;
                // falls through
            case Mode::XOR:
                m_updater = fn_xor;
                break;
            default:
                mgb_throw(MegBrainError, "invalid CondExecPredLogical mode");
        }
    }

    //! return false to early stop
    bool update(int val) { return m_updater(&m_cur_val, val); }

    bool get() const { return m_cur_val ^ m_negate; }
};

CondExecPredLogical::CondExecPredLogical(const VarNodeArrayView& preds,
                                         const Param& param,
                                         const OperatorNodeConfig& config)
        : Super(preds.at(0)->owner_graph(), config,
                mgb_cstr_log(mode2str(param.mode)), preds),
          m_param{param} {
    m_input_masks.resize(preds.size());
    auto gr = CondExecPred::GlobalRegistry::get(*owner_graph());
    for (size_t i = 0; i < preds.size(); ++i) {
        m_input_masks[i] = gr->require_mask_from_var(preds[i]);
        add_input({preds[i]}, i == preds.size() - 1 ? AddInputSortType::ALL
                                                    : AddInputSortType::NONE);
    }
    add_output(None)
            ->dtype(dtype::Int32{})
            .add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC);
    m_mask = std::make_shared<ExecutionMask>(output(0));
    add_equivalence_component<PODHash<Param>>(&m_param);
}

SymbolVar CondExecPredLogical::make(const VarNodeArrayView& preds,
                                    const Param& param,
                                    const OperatorNodeConfig& config) {
    mgb_assert(!preds.empty());
    if (preds.size() == 1) {
        if (!config.has_comp_node_set() ||
            config.get_single_comp_node() == preds[0]->comp_node()) {
            auto m = param.mode;
            if (m == Mode::OR || m == Mode::XOR || m == Mode::AND) {
                return preds[0];
            }
        }
    }
    return SymbolVar{preds[0]}.insert_single_output_opr<CondExecPredLogical>(
            preds, param, config);
}

void CondExecPredLogical::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    mgr.register_shape_infer(output(0), ShapeInferDesc::make_const({1}));

    auto infer_val = [mode = m_param.mode](DeviceTensorND & dst,
                                           const InpVal& inp) {
        PredEvaluator eval{mode, inp.val[0].value().ptr<int>()[0]};
        for (size_t i = 1; i < inp.val.size(); ++i) {
            if (!eval.update(inp.val[i].value().ptr<int>()[0])) {
                break;
            }
        }
        dst.resize({1}).ptr<int>()[0] = eval.get();
        return true;
    };
    ValueInferDesc desc;
    desc.src_type = SourceType::DEP;
    desc.deps.reserve(input().size());
    for (auto i : input()) {
        desc.deps.push_back({i, DepType::VALUE});
    }
    desc.infer_func = infer_val;
    mgr.register_value_infer(output(0), desc);
}

void CondExecPredLogical::scn_do_execute() {
    PredEvaluator eval{m_param.mode, m_input_masks[0]->enabled()};
    for (size_t i = 1; i < m_input_masks.size(); ++i) {
        if (!eval.update(m_input_masks[i]->enabled())) {
            break;
        }
    }
    m_mask->enable(eval.get());
}

CondExecPredLogical::NodeProp* CondExecPredLogical::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    for (auto&& i : ret->dep_map()) {
        i.second = NodeProp::DepType::DEV_COMP_ORDER;
    }
    ret->add_flag(NodeProp::Flag::CROSS_COMP_NODE_MEMORY);
    return ret;
}

const char* CondExecPredLogical::mode2str(Mode mode) {
    switch (mode) {
#define CASE(n)   \
    case Mode::n: \
        return #n
        CASE(OR);
        CASE(AND);
        CASE(XOR);
        CASE(NOR);
        CASE(NAND);
        CASE(XNOR);
        default:
            mgb_throw(MegBrainError, "bad CondExecPredLogical mode: %d",
                      static_cast<int>(mode));
    }
}

/* ============================= CondExecMark ============================= */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(CondExecMark);

CondExecMark::CondExecMark(VarNode* ppv, const VarNodeArrayView& inputs,
                           const Param& param, const OperatorNodeConfig& config)
        : Super(ppv->owner_graph(), config, "cond_mark", {ppv}),
          m_param{param} {
    CondExecPred::GlobalRegistry::get(*owner_graph())->check_ppv(ppv);

    for (size_t i = 0; i < inputs.size(); ++i) {
        add_input({inputs[i]});
        add_output(ssprintf("fwd%zu", i))
                ->dtype(inputs[i]->dtype())
                .add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    }
    add_input({ppv});
    add_equivalence_component<PODHash<Param>>(&m_param);
    if (has_no_shape_infer()) {
        for (auto i : input()) {
            // force dynamic allocation of input so storage can be forwarded
            i->add_flag(VarNode::Flag::NO_SYS_STATIC_MEM_ALLOC);
        }
        for (auto i : output()) {
            i->add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC);
        }
    } else {
        m_mem_fwd_success.resize(inputs.size(), false);
    }
}

void CondExecMark::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    using InferMode = Param::StaticInfer;
    auto infer_mode = param().static_infer;
    if (infer_mode == InferMode::NONE) {
        return;
    }
    for (size_t i = 0; i < output().size(); ++i) {
        auto s = input(i), t = output(i);
        mgr.register_shape_infer(t, ShapeInferDesc::make_identity(s));
        if (infer_mode != InferMode::SHAPE_ONLY) {
            auto desc = ValueInferDesc::make_identity(s);
            mixin::ForwardInputToOutput::ensure_not_replaced_by_const_folding(
                    desc);
            mgr.register_value_infer(t, desc);
        }
    }
}

void CondExecMark::scn_do_execute() {
    bool no_sys_alloc = has_no_shape_infer();
    for (size_t i = 0; i < output().size(); ++i) {
        if (no_sys_alloc) {
            bool succ = output(i)->reset_dev_tensor_from_other_var(input(i));
            MGB_MARK_USED_VAR(succ);
        } else {
            auto &&out = output(i)->dev_tensor(),
                 &&inp = input(i)->dev_tensor();
            if (m_mem_fwd_success[i]) {
                mgb_assert(inp.raw_ptr() == out.raw_ptr() &&
                           out.layout().eq_layout(inp.layout()));
            } else {
                out.copy_from_fixlayout(inp);
            }
        }
    }
}

void CondExecMark::init_rt_force_dynamic_mem_alloc_imply_chain() {
    if (has_no_shape_infer()) {
        return;
    }
    for (size_t i = 0; i < output().size(); ++i) {
        auto s = input(i), t = output(i);
        s->add_rt_force_dynamic_mem_alloc_imply_chain(t);
        t->add_rt_force_dynamic_mem_alloc_imply_chain(s);
    }
}

void CondExecMark::mem_plan_fwd_in2out_readonly() {
    if (has_no_shape_infer()) {
        return;
    }
    for (size_t i = 0; i < output().size(); ++i) {
        auto s = input(i), t = output(i);
        m_mem_fwd_success[i] = t->set_fwd_in2out_readonly(
                s, SubTensorSpec::make_from_layout(s->layout()));
    }
}

void CondExecMark::add_input_layout_constraint() {
    if (has_no_shape_infer()) {
        for (auto i : input()) {
            // reset_dev_tensor_from_other_var already has such requirement
            i->add_layout_constraint_contiguous();
        }
    }
}

CondExecMark::NodeProp* CondExecMark::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->dep_map().at(input().back()) = NodeProp::DepType::DEV_COMP_ORDER;
    for (size_t i = 0; i < input().size() - 1; ++ i) {
        ret->add_dep_type_existing_var(input(i),
                NodeProp::DepType::VALUE_ALLOW_EMPTY);
    }
    return ret;
}

cg::OperatorNodeBase* CondExecMark::make_opr(SymbolVar ppv,
                                             const VarNodeArrayView& inputs,
                                             const Param& param,
                                             const OperatorNodeConfig& config) {
    return ppv.node()->owner_graph()->insert_opr(
            std::make_unique<CondExecMark>(ppv.node(), inputs, param, config));
}

SymbolVar CondExecMark::mark_if_need(SymbolVar maybe_ppv, SymbolVar input,
                                     const Param& param,
                                     const OperatorNodeConfig& config) {
    auto mask =
            CondExecPred::GlobalRegistry::get(*maybe_ppv.node()->owner_graph())
                    ->get_mask_from_var(maybe_ppv.node());
    if (mask) {
        return make_opr(mask->owner(), {input}, param, config)->output(0);
    }
    return input;
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(CondExecMark) {
    if (wrt_idx == opr.input().size() - 1 || !out_grad.at(wrt_idx)) {
        return nullptr;
    }
    using GradMode = CondExecMark::Param::GradMode;
    using MergeMode = CondExecMerge::Param::Mode;
    MergeMode grad_mode;
    SymbolVarArray grad_shapes;
    switch (opr.param().grad_mode) {
        case GradMode::SUM:
            grad_mode = MergeMode::SUM;
            grad_shapes.emplace_back(SymbolVar{opr.input(wrt_idx)}.symshape());
            break;
        case GradMode::SUM_COND_OUT:
            grad_mode = MergeMode::SUM_COND_OUT;
            break;
        default:
            mgb_throw(MegBrainError, "invalid grad_mode");
    }
    return CondExecMerge::make_opr({out_grad[wrt_idx]}, grad_shapes,
                                   {1, grad_mode}, OperatorNodeConfig{})
            ->output(0);
}
#endif

/* ============================= CondExecMerge ============================= */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(CondExecMerge);

CondExecMerge::CondExecMerge(const VarNodeArrayView& inputs,
                             const VarNodeArrayView& out_shapes,
                             const Param& param,
                             const OperatorNodeConfig& config)
        : Super(inputs[0]->owner_graph(), config, "cond_merge", {}),
          m_param{param} {
    mgb_throw_if(inputs.size() % param.nr_output, GraphError,
                 "input size can not divide nr_output: %zu %u", inputs.size(),
                 param.nr_output);
    auto global_registry = CondExecPred::GlobalRegistry::get(*owner_graph());
    auto nr_branch = inputs.size() / param.nr_output;
    mgb_assert(param.nr_output);
    for (size_t i = 0; i < param.nr_output; ++i) {
        auto ovar = add_output(ssprintf("out%zu", i));
        ovar->dtype(inputs[i]->dtype());
        // disable system memory allocation because:
        //  1. we can directly forward input storage to output
        //  2. dynamic allocator would wait for all inputs to become ready (see
        //     VarNodeMemManager::DynamicAllocOprInfo::host_wait_input_ready),
        //     which would cause infinite waiting for unselected inputs.
        ovar->add_flag(VarNode::Flag::NO_SYS_MEM_ALLOC)
            .add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    }

    MGB_MARK_USED_VAR(mask2str);
    m_branch_masks.resize(nr_branch, nullptr);
    for (size_t i = 0; i < nr_branch; ++i) {
        ExecutionMask* br_mask = nullptr;
        for (size_t j = 0; j < param.nr_output; ++j) {
            auto ivar = inputs[i * param.nr_output + j];
            auto mask = global_registry->require_mask_from_var(ivar);
            mgb_throw_if(
                    output(j)->dtype() != ivar->dtype(), GraphError,
                    "CondExecMerge input dtypes mismatch: branch=%zu %s vs %s",
                    i, output(j)->dtype().name(), ivar->dtype().name());
            if (!j) {
                br_mask = mask;
            } else {
                mgb_throw_if(br_mask != mask, GraphError,
                             "CondExecMerge branch %zu have different masks: "
                             "%s vs %s",
                             i, mask2str(br_mask).c_str(),
                             mask2str(mask).c_str());
            }
            // this flag is added by ExecutionMask; we require flag because
            // output var might forward input var storage
            mgb_assert(
                    ivar->contain_flag(VarNode::Flag::NO_SYS_STATIC_MEM_ALLOC));
            add_input({ivar});
        }
        m_branch_masks[i] = br_mask;
    }
    add_equivalence_component<PODHash<Param>>(&m_param);

    // handle extra inputs for special modes
    if (param.mode == Mode::SUM) {
        mgb_assert(out_shapes.size() == param.nr_output);
        for (auto i : out_shapes) {
            add_input({i});
        }
    } else {
        mgb_assert(out_shapes.empty(),
                   "out_shapes should not be given if mode is not SUM");
    }
    if (param.mode == Mode::SUM_COND_OUT) {
        VarNodeArray preds;
        preds.reserve(nr_branch);
        for (auto i : m_branch_masks) {
            preds.emplace_back(proxy_var_from_mask(i));
        }
        auto cn = mixin_infer_output_comp_node(*this, true);
        auto preds_or = CondExecPredLogical::make(
                preds, CondExecPredLogical::Mode::OR, cn);
        add_input({preds_or.node()});
    }
}

cg::OperatorNodeBase* CondExecMerge::make_opr(
        const VarNodeArrayView& inputs, const VarNodeArrayView& out_shapes,
        const Param& param, const OperatorNodeConfig& config) {
    mgb_assert(!inputs.empty());
    const VarNodeArrayView* out_shapes_ptr = &out_shapes;
    Maybe<VarNodeArrayView> out_shapes_from_inp;
    VarNodeArray out_shapes_from_inp_storage;
    if (out_shapes.empty() && param.mode == Mode::SUM) {
        // find out_shapes from inputs
        mgb_assert(inputs.size() % param.nr_output == 0);
        size_t nr_branch = inputs.size() / param.nr_output;
        auto inp = [&](size_t br, size_t oidx) {
            return inputs[br * param.nr_output + oidx];
        };
        for (size_t oidx = 0; oidx < param.nr_output; ++oidx) {
            bool found = false;
            for (size_t br = 0; br < nr_branch; ++br) {
                auto ivar = inp(br, oidx);
                if (cg::is_static_var_shape(ivar)) {
                    found = true;
                    out_shapes_from_inp_storage.push_back(
                            SymbolVar{ivar}.symshape().node());
                    break;
                }
            }
            mgb_throw_if(!found, GraphError,
                         "out_shapes is omitted but no input shape is "
                         "inferrable for output %zu",
                         oidx);
        }

        out_shapes_ptr =
                &out_shapes_from_inp.emplace(out_shapes_from_inp_storage);
    }
    return inputs[0]->owner_graph()->insert_opr(std::make_unique<CondExecMerge>(
            inputs, *out_shapes_ptr, param, config));
}

void CondExecMerge::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();

    auto nr_out = m_param.nr_output;
    auto inp = [this, nr_out](size_t branch, size_t oidx) {
        return input(branch * nr_out + oidx);
    };

    static auto select_one_branch = [](size_t nr_branch,
                                       const InpVal& bval) -> size_t {
        bool found = false;
        size_t ret;
        for (size_t i = 0; i < nr_branch; ++i) {
            if (bval.val[i].value().ptr<int>()[0]) {
                if (!found) {
                    found = true;
                    ret = i;
                } else {
                    mgb_throw(GraphError,
                              "multiple branches are active in EXACT_ONE mode: "
                              "%zu and %zu",
                              ret, i);
                }
            }
        }
        mgb_throw_if(!found, GraphError,
                     "no branch is active in EXACT_ONE mode");
        return ret;
    };

    DepVal branch_deps;
    auto nr_branch = m_branch_masks.size();
    branch_deps.reserve(nr_branch);
    for (size_t i = 0; i < nr_branch; ++i) {
        branch_deps.push_back(
                {proxy_var_from_mask(m_branch_masks[i]), DepType::VALUE});
    }

    // register shape and value infers for each output
    for (size_t oidx = 0; oidx < nr_out; oidx++) {
        if (m_param.mode == Mode::EXACT_ONE_SAME_SHAPE ||
            m_param.mode == Mode::SUM_COND_OUT) {
            // all branches should have the same shape
            bool found = false;
            // find any inferrable input var
            for (size_t i = 0; i < nr_branch; ++i) {
                if (cg::is_static_var_shape(inp(i, oidx))) {
                    mgr.register_shape_infer(
                            output(oidx),
                            ShapeInferDesc::make_identity(inp(i, oidx)));
                    found = true;
                    break;
                }
            }
            if (!found) {
                mgr.register_shape_infer(
                        output(oidx),
                        ShapeInferDesc::make_identity(inp(0, oidx)));
            }
        } else if (m_param.mode == Mode::SUM) {
            auto infer_fn = [](TensorShape& dst, const InpVal& inp) {
                cg::copy_tensor_value_to_shape(dst, inp.val[0].value());
                return true;
            };
            mgr.register_shape_infer(output(oidx),
                                     {SourceType::DEP,
                                      {{inp(nr_branch, oidx), DepType::VALUE}},
                                      infer_fn});
        } else {
            // general shape inference for EXACT_ONE mode
            auto infer_fn = [this](TensorShape& dest, const InpVal& inp) {
                auto nr_branch = m_branch_masks.size();
                size_t branch = select_one_branch(nr_branch, inp);
                dest = inp.val.at(nr_branch + branch).shape();
                return true;
            };
            ShapeInferDesc desc{SourceType::DEP, branch_deps, infer_fn};
            for (size_t i = 0; i < nr_branch; ++i) {
                desc.deps.push_back({inp(i, oidx), DepType::SHAPE});
            }
            mgr.register_shape_infer(output(oidx), desc);
        }

        // general value inference
        ValueInferDesc desc{SourceType::DEP, branch_deps, {}};
        for (size_t i = 0; i < nr_branch; ++i) {
            desc.deps.push_back({inp(i, oidx), DepType::VALUE});
        }

        if (is_exact_one()) {
            desc.infer_func = [this](DeviceTensorND& dest, const InpVal& inp) {
                auto nr_branch = m_branch_masks.size();
                size_t branch = select_one_branch(nr_branch, inp);
                dest = inp.val.at(nr_branch + branch).value();
                return true;
            };
        } else {
            mgb_assert(m_param.mode == Mode::SUM ||
                       m_param.mode == Mode::SUM_COND_OUT);
            desc.infer_func = [this](DeviceTensorND& dest, const InpVal& inp) {
                auto nr_branch = m_branch_masks.size();
                bool found = false, first = true;
                auto&& shape = inp.val.at(nr_branch).shape();

                for (size_t i = 0; i < nr_branch && !shape.is_empty(); ++i) {
                    if (!inp.val[i].value().ptr<int>()[0])
                        continue;
                    auto&& cur = inp.val.at(nr_branch + i).value();

                    // add cur value to dest
                    if (!found) {
                        found = true;
                        dest = cur;
                    } else {
                        if (first) {
                            first = false;
                            DeviceTensorND tmp;
                            tmp.copy_from(dest);
                            dest = std::move(tmp);
                        }
                        // comp node is cpu default, so it is safe to use a
                        // temporary megdnn opr here
                        auto dnn_opr =
                                intl::create_megdnn_opr<megdnn::Elemwise>(
                                        dest.comp_node());
                        dnn_opr->param().mode = Elemwise::Mode::ADD;
                        dnn_opr->exec({dest.as_megdnn(), cur.as_megdnn()},
                                      dest.as_megdnn());
                    }
                }
                if (!found) {
                    if (dest.storage().raw_storage().use_count() > 1) {
                        // likely to be assigned from some input in previous
                        // runs; we create a new tensor to avoid modifying input
                        // value
                        DeviceTensorND tmp{dest.comp_node(), shape,
                                           dest.dtype()};
                        dest = std::move(tmp);
                    } else {
                        dest.resize(shape);
                    }
                    fill_zero_dev_tensor(dest);
                }
                return true;
            };
        }

        mgr.register_value_infer(output(oidx), desc);
    }
}

void CondExecMerge::scn_do_execute() {
    auto nr_out = m_param.nr_output;
    auto inp = [this, nr_out](size_t branch, size_t oidx) {
        return input(branch * nr_out + oidx);
    };

    auto cn = this->comp_node();
    mgb_assert(cn == output(0)->comp_node());

    bool first = true;
    auto&& forwarded = m_mem_forwarded;
    std::vector<bool> is_shape_empty(nr_out, false);
    for (size_t br = 0; br < m_branch_masks.size(); ++br) {
        if (!m_branch_masks[br]->enabled()) {
            continue;
        }

        if (first) {
            first = false;
            for (size_t oidx = 0; oidx < nr_out; ++oidx) {
                bool succ = output(oidx)->reset_dev_tensor_from_other_var(
                        inp(br, oidx));
                if (inp(br, oidx)->shape().is_empty()) {
                    is_shape_empty[oidx] = true;
                    continue;
                }
                if (!is_exact_one()) {
                    if (forwarded.empty()) {
                        forwarded.resize(nr_out);
                    }
                    forwarded[oidx] = succ;
                }
            }
        } else {
            mgb_throw_if(is_exact_one(), GraphError,
                         "multiple branches are active in EXACT_ONE mode");
            auto&& dnn_opr = m_exec_dnn_opr;
            if (!dnn_opr || dnn_opr.comp_node() != cn) {
                dnn_opr = intl::create_megdnn_opr<megdnn::Elemwise>(cn);
                dnn_opr->param().mode = Elemwise::Mode::ADD;
            }
            for (size_t oidx = 0; oidx < nr_out; ++oidx) {
                auto ovar = output(oidx);
                auto&& src = inp(br, oidx)->dev_tensor().as_megdnn();
                auto&& dest = ovar->dev_tensor().as_megdnn();
                mgb_assert(src.layout.eq_shape(dest.layout),
                        "shape mismatch: %s vs %s in CondExecMerge",
                        src.layout.to_string().c_str(),
                        dest.layout.to_string().c_str());
                if (is_shape_empty[oidx]) continue;
                if (forwarded[oidx]) {
                    ovar->shape_alloc(ovar->shape());
                    auto&& own_dest = ovar->dev_tensor().as_megdnn();
                    mgb_assert(own_dest.raw_ptr != dest.raw_ptr);
                    dnn_opr->exec({dest, src}, own_dest);
                    forwarded[oidx] = false;
                } else {
                    dnn_opr->exec({dest, src}, dest);
                }
            }
        }
    }

    if (first) {
        mgb_throw_if(is_exact_one(), GraphError,
                     "no branch is selected in EXACT_ONE mode");
        mgb_assert(m_param.mode == Param::Mode::SUM);
        auto&& mgr = owner_graph()->static_infer_manager();
        for (auto var : output()) {
            auto&& dv = var->shape_alloc(mgr.infer_shape(var)).dev_tensor();
            fill_zero_dev_tensor(dv);
        }
    } else if (m_param.mode == Param::Mode::SUM) {
        auto&& mgr = owner_graph()->static_infer_manager();
        for (auto var : output()) {
            auto&& shp_infer = mgr.infer_shape(var);
            auto&& shp_got = var->shape();
            mgb_throw_if(!shp_infer.eq_shape(shp_got), GraphError,
                         "inferred shape is %s, actual shape is %s",
                         shp_infer.to_string().c_str(),
                         shp_got.to_string().c_str());
        }
    }
}

void CondExecMerge::add_input_layout_constraint() {
    for (auto i : input()) {
        // reset_dev_tensor_from_other_var already has such requirement
        i->add_layout_constraint_contiguous();
    }
}

CondExecMerge::NodeProp* CondExecMerge::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    using DT = NodeProp::DepType;
    if (m_param.mode == Mode::SUM) {
        SmallVector<DT> inp_dt(input().size(), DT::DEV_VALUE);
        for (size_t i = 0; i < m_param.nr_output; ++i) {
            inp_dt[inp_dt.size() - i - 1] = DT::HOST_VALUE;
        }
        ret->reset_dep_type(input(), inp_dt);
    } else if (m_param.mode == Mode::SUM_COND_OUT) {
        // PPV can not be used as a usual input, so we can modify dep_map
        // directly
        ret->dep_map().at(input().back()) = NodeProp::DepType::DEV_COMP_ORDER;
    }
    for (size_t i = 0; i < m_param.nr_output * m_branch_masks.size(); ++ i) {
        ret->add_dep_type_existing_var(input(i),
                NodeProp::DepType::VALUE_ALLOW_EMPTY);
    }
    return ret;
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(CondExecMerge) {
    using Mode = CondExecMerge::Param::Mode;
    if (opr.param().mode == Mode::SUM_COND_OUT &&
        wrt_idx == opr.input().size() - 1) {
        return nullptr;
    }
    if (opr.param().mode == Mode::SUM &&
        wrt_idx >= opr.input().size() - opr.output().size()) {
        return InvalidGrad::make(opr, wrt_idx);
    }
    size_t wrt_branch = wrt_idx / opr.param().nr_output,
           wrt_oidx = wrt_idx % opr.param().nr_output;
    auto og = out_grad.at(wrt_oidx);
    if (!og) {
        return nullptr;
    }
    auto ppv = proxy_var_from_mask(opr.branch_mask(wrt_branch));
    if (ppv->comp_node().mem_node() != og->comp_node().mem_node()) {
        ppv = CondExecPredLogical::make({ppv}, CondExecPredLogical::Mode::AND,
                                        og->comp_node())
                      .node();
    }
    CondExecMark::Param gparam;
    if (opr.param().mode == Mode::EXACT_ONE) {
        // only in this mode different branches may have different shapes, so to
        // avoid shape inference failure we simply skip shape inference here;
        // see TestCondExec.MultiShape
        // TODO: remove this if static infer considers execution mask
        gparam.static_infer = CondExecMark::Param::StaticInfer::NONE;
    }
    return CondExecMark::make_opr(ppv, {og}, gparam,
                                  OperatorNodeConfig{og->comp_node()})
            ->output(0);
}
#endif

void CondExecMerge::modify_grad_sum_list(VarNode* wrt, VarNodeArray& grads) {
    if (!ExecutionMask::have_alive_instance()) {
        return;
    }

    auto global_registry_vec =
            grads.at(0)
                    ->owner_graph()
                    ->options()
                    .user_data.get_user_data<CondExecPred::GlobalRegistry>();
    if (!global_registry_vec.second) {
        // no cond exec related oprs
        return;
    }
    auto global_registry = global_registry_vec.first[0];

    size_t nr_var_remove = 0, nr_merge_opr = 0;
    VarNodeArray merged_branches;
    static constexpr Param::Mode BAD_MODE = static_cast<Param::Mode>(-1);
    Param::Mode merged_mode = BAD_MODE;
    ExecutionMask* part_exec_mask = nullptr;
    bool have_multiple_exec_mask = false;

    auto check_multiple_mask = [&part_exec_mask,
                                &have_multiple_exec_mask](ExecutionMask* mask) {
        if (!part_exec_mask) {
            part_exec_mask = mask;
        } else if (part_exec_mask != mask) {
            have_multiple_exec_mask = true;
        }
    };

    // loop in reverse order, and put vars to be merged at end
    for (size_t i = grads.size(); i;) {
        --i;
        auto opr = grads[i]->owner_opr();
        if (opr->same_type<CondExecMerge>()) {
            // merge sum of CondExecMerge by expanding their inputs
            mgb_assert(opr->output().size() == 1,
                       "CondExecMerge in grad list has multiple outputs: "
                       "name=%s out=%zu",
                       opr->cname(), opr->output().size());
            auto cur_mode = opr->cast_final<CondExecMerge>().param().mode;
            mgb_assert(cur_mode == Param::Mode::SUM ||
                       cur_mode == Param::Mode::SUM_COND_OUT);
            if (merged_mode != Param::Mode::SUM_COND_OUT) {
                // only allow promoting merge mode to be cond out (if any of the
                // components are conditional)
                merged_mode = cur_mode;
            }
            merged_branches.insert(merged_branches.end(), opr->input().begin(),
                                   opr->input().end());

            if (cur_mode == Param::Mode::SUM_COND_OUT) {
                // remove the predicate input
                mgb_assert(opr->input().size() == opr->output().size() + 1);
                merged_branches.pop_back();

                check_multiple_mask(
                        global_registry->require_mask_from_var(opr->output(0)));
            } else if (cur_mode == Param::Mode::SUM) {
                // remove shape input
                mgb_assert(opr->input().size() >= opr->output().size() * 2);
                merged_branches.resize(merged_branches.size() -
                                       opr->output().size());
            }
            ++nr_merge_opr;
            ++nr_var_remove;
            std::swap(grads[grads.size() - nr_var_remove], grads[i]);
        } else if (auto mask = global_registry->get_mask_from_var(grads[i])) {
            check_multiple_mask(mask);
            merged_branches.push_back(grads[i]);

            ++nr_var_remove;
            std::swap(grads[grads.size() - nr_var_remove], grads[i]);
            merged_mode = Param::Mode::SUM_COND_OUT;
        }
    }

    if (have_multiple_exec_mask || nr_merge_opr > 1) {
        mgb_assert(merged_mode != BAD_MODE);
        grads.resize(grads.size() - nr_var_remove);
        SymbolVarArray grad_shapes;
        if (merged_mode == Param::Mode::SUM) {
            grad_shapes.emplace_back(SymbolVar{wrt}.symshape());
        }
        grads.push_back(CondExecMerge::make_opr(merged_branches, grad_shapes,
                                                {1, merged_mode},
                                                OperatorNodeConfig{})
                                ->output(0));
    }
}

#endif  // MGB_ENABLE_COND_EXEC

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
