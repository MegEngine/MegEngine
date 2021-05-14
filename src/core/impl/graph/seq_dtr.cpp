/**
 * \file src/core/impl/graph/seq_dtr.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./seq_dtr.h"

#if MGB_ENABLE_DTR

using namespace mgb;
using namespace cg;

namespace {

bool is_bad_opr(OperatorNodeBase* opr) {
    using F = OperatorNodeBase::NodeProp::Flag;
    return opr->node_prop().contain(
        F::IMPURE_FUNC | F::NO_AUTOMATIC_DUP | F::FORCE_UPDATE_INPUT_VAR);
}
    
} // namespace

class SeqModifierForDTR::ModifyActionPlanner : public ModifyActionPlannerBase {
public:
    ModifyActionPlanner(SeqModifierBase* par) : ModifyActionPlannerBase{par} {}

    void prepare(const OprNodeArray& opr_seq);

    SeqModifyAction perform_dtr(CompNode comp_node, const OprNodeArray& seq, Config* config);
};


SeqModifierForDTR::SeqModifierForDTR(ComputingGraphImpl* owner, Config* config_g)
    : SeqModifierBase(owner), m_config(config_g) {}

void SeqModifierForDTR::modify_endpoint_vars(VarNodeArray& endpoints) {
    var_map().clear();
    auto comp_seq = MemoryOptimizerHelper::CompSeq(owner_graph(), endpoints);
    auto config =
        MemoryOptimizerHelper::SubGraphConfig()
                /*.add_bad_opr_flag(
                        OperatorNodeBase::NodeProp::Flag::IMPURE_FUNC)
                .add_bad_opr_flag(
                        OperatorNodeBase::NodeProp::Flag::NO_AUTOMATIC_DUP)
                .add_bad_opr_flag(OperatorNodeBase::NodeProp::Flag::
                                            FORCE_UPDATE_INPUT_VAR)*/
                // NOTE: it should not actually involve any opr with the above
                // flags, but for better results, some ops(e.g. CudnnBatchNorm)
                // should be involved and they are guaranteed to NEVER recompute.
                .add_bad_var_flag(VarNode::Flag::VOLATILE_CONTENT)
                .add_bad_var_flag(VarNode::Flag::NO_SYS_STATIC_MEM_ALLOC)
                .add_bad_var_flag(VarNode::Flag::NO_SYS_MEM_ALLOC)
                .add_bad_var_flag(VarNode::Flag::PERSISTENT_DEVICE_VALUE);
    auto cn2oprseq = mem_opt().split_into_cn2oprseq(*comp_seq.m_seq, config);

    if (cn2oprseq->empty()) {
        return;
    }
    SeqModifyAction action;
    ModifyActionPlanner* planner = new ModifyActionPlanner(this);
    for (auto && i : *cn2oprseq) {
        auto&& cur = planner->perform_dtr(i.first, i.second, m_config);
        action.insert(cur.begin(), cur.end());
    }
    apply_action(action, *comp_seq.m_seq);
    for (auto&& i : endpoints) {
        auto iter = var_map().find(i);
        if (iter != var_map().end()) {
            i = iter->second;
        }
    }
}

void SeqModifierForDTR::ModifyActionPlanner::prepare(const OprNodeArray& opr_seq) {
    init_seq(opr_seq, false);

    for (size_t i = 0; i < seq().size(); ++i) {
        auto opr = seq()[i].get();
        size_t est = 0;
        for (auto i : opr->input) {
            est += i->size;
        }
        for (auto i : opr->output) {
            est += i->size;
        }
        opr->estimate_compute_time = static_cast<double>(est) / 1e8;
    }
}

SeqModifierForDTR::SeqModifyAction SeqModifierForDTR::ModifyActionPlanner::perform_dtr(
        CompNode comp_node, const OprNodeArray& opr_seq, Config* config) {
    prepare(opr_seq);
    SeqModifyAction action;

    if (comp_node.locator().stream < 0) {
        // do not modify system stream oprs
        return action;
    }

    ThinHashSet<Var*> alive_vars;
    size_t cur_usage = 0;

    //! map from original var to latest var
    ThinHashMap<VarNode*, Var*> latest_var;
    ThinHashMap<VarNode*, size_t> pin;

    auto need_regen = [&](Var* var) {
        return alive_vars.find(var) == alive_vars.end();
    };

    auto add_alive = [&](Var* var) {
        auto&& ins = alive_vars.insert(var);
        mgb_assert(ins.second);
        cur_usage += var->size;
    };

    auto remove_alive = [&](Var* var) {
        if (alive_vars.erase(var)) {
            auto size = var->size;
            mgb_assert(size <= cur_usage);
            cur_usage -= size;
        }
    };

    auto get_latest = [&](Var* var) {
        auto iter = latest_var.find(var->orig_var);
        if (iter == latest_var.end()) {
            return var;
        } else {
            return iter->second;
        }
    };

    double est_time = 0;

    ThinHashMap<Var*, double> dfs_back;
    ThinHashMap<Var*, double> dfs_front;

    auto regen_time = [&](Var* var) {
        thin_function<double(Var*)> dfs_b;
        thin_function<double(Var*)> dfs_f;
        dfs_b = [&](Var* var) {
            if (dfs_back.find(var) != dfs_back.end()) {
                return dfs_back[var];
            }
            auto opr = var->owner_opr();
            double sum_time = opr->estimate_compute_time;
            for (auto i : opr->input) {
                auto ivar = get_latest(i);
                if (need_regen(ivar)) {
                    sum_time += dfs_b(ivar);
                }
            }
            dfs_back[var] = sum_time;
            return sum_time;
        };
        dfs_f = [&](Var* var) {
            if (dfs_front.find(var) != dfs_front.end()) {
                return dfs_front[var];
            }
            double sum_time = 1;
            for (size_t j = 1; j < var->access_rec.size();j ++) {
                auto dep_opr = var->access_rec[j].opr;
                for (auto o : dep_opr->output) {
                    o = get_latest(o);
                    if (need_regen(o)) {
                        sum_time += dfs_f(o);
                    }
                }
            }
            dfs_front[var] = sum_time;
            return sum_time;
        };
        return dfs_f(var) * dfs_b(var);
    };

    static constexpr double MAX_EVAL_VALUE = std::numeric_limits<double>::max();
    auto find_best = [&]() {
        Var* best = nullptr;
        double min_eval_value = MAX_EVAL_VALUE;
        dfs_back.clear();
        dfs_front.clear();
        for (auto var : alive_vars) {
            if (var->size < config->evictee_minimum_size 
                    || pin[var->orig_var] > 0
                    || is_bad_opr(var->owner_opr()->orig_opr)) {
                continue;
            }
            double regen = regen_time(var);
            double eval_value = regen / static_cast<double>(var->size)
                                / (est_time - var->last_access_time + 1e-8);
            if (eval_value < min_eval_value) {
                min_eval_value = eval_value;
                best = var;
            }
        }
        return best;
    };

    auto do_evict = [&](Var* var) {
        remove_alive(var);
    };

    auto auto_evict = [&](size_t needed) {
        while (cur_usage + needed >= config->eviction_threshold) {
            Var* v = find_best();
            if (!v) {
                break;
            }
            do_evict(v);
        }
    };

    thin_function<Var*(Opr*, Var*)> regenerate;
    regenerate = [&](Opr* reader, Var* var) {
        auto opr = var->owner_opr();
        // FIXME: if var can not be recomputed, the previous eviction may fail
        if (is_bad_opr(opr->orig_opr)) {
            return var;
        }

        auto new_opr_storage = opr_mempool().alloc_unique(opr->orig_opr, static_cast<size_t>(DUPOPR_TIME));
        auto new_opr = new_opr_storage.get();

        new_opr->input.reserve(opr->input.size());
        new_opr->output.reserve(opr->output.size());

        for (auto i : opr->input) {
            i->last_access_time = est_time;
            pin[i->orig_var] ++;
        }
        for (auto o : opr->output) {
            auto lo = get_latest(o);
            if (!need_regen(lo)) {
                remove_alive(lo);
            }
        }
        for (auto i : opr->input) {
            auto ivar = get_latest(i);
            if (need_regen(ivar)) {
                ivar = regenerate(reader, ivar);
            }
            new_opr->input.push_back(ivar);
            ivar->access_rec.emplace_back(new_opr);
        }

        reader->oprs_insert_before.emplace_back(std::move(new_opr_storage));

        size_t needed = 0;
        for (auto o : opr->output) {
            needed += o->size;
        }
        auto_evict(needed);
        Var* new_var = nullptr;
        for (auto o : opr->output) {
            auto lo = get_latest(o);
            auto&& ovar = var_mempool().alloc_unique(lo->orig_var, lo->size,
                                                     new_opr);
            ovar->recomp_id = lo->recomp_id + 1;
            new_opr->output.push_back(ovar.get());
            if (o == var) {
                new_var = ovar.get();
            }
            add_alive(ovar.get());
            ovar->last_access_time = est_time;
            latest_var[o->orig_var] = ovar.get();
            var_storage().emplace_back(std::move(ovar));
        }
        est_time += opr->estimate_compute_time;
        for (auto i : opr->input) {
            pin[i->orig_var] --;
        }
        return new_var;
    };

    for (size_t j = 0; j < seq().size(); ++j) {
        auto opr = seq()[j].get();
        for (auto i : opr->input) {
            pin[i->orig_var] ++;
        }
        for (auto i : opr->input) {
            i = get_latest(i);
            if (need_regen(i)) {
                i = regenerate(opr, i);
            }
            i->last_access_time = est_time;
        }
        size_t needed = 0;
        for (auto o : opr->output) {
            needed += o->size;
        }
        auto_evict(needed);
        est_time += opr->estimate_compute_time;
        for (auto o : opr->output) {
            add_alive(o);
            o->last_access_time = est_time;
        }
        for (auto i : opr->input) {
            pin[i->orig_var] --;
        }
        for (auto i : opr->input) {
            i = get_latest(i);
            if (opr == i->last_access_opr())
                remove_alive(i);
        }
    }
    for (size_t j = 0; j < seq().size(); ++j) {
        auto opr = seq()[j].get();
        auto&& arr = opr->oprs_insert_before;
        if (arr.empty()) {
            continue;
        }
        auto&& dest = action[opr->orig_opr];
        dest.reserve(arr.size());
        for (auto&& i : arr) {
            dest.push_back(i->orig_opr);
        }
    }
    return action;
}

void SeqModifierForDTR::apply_action(SeqModifyAction& action,
                                     const OprNodeArray& oprseq) {
    auto cur_priority = std::numeric_limits<decltype(
            OperatorNodeBase::NodeProp::Attribute::priority)>::min();

    ThinHashSet<OperatorNodeBase*> modified_opr;
    ThinHashMap<OperatorNodeBase*, size_t> recomp_id;
    auto set_priority = [&](OperatorNodeBase* opr) {
        mgb_assert(modified_opr.insert(opr).second);
        mem_opt().set_priority(opr, cur_priority++);
    };

    auto on_opr_visited = [&](OperatorNodeBase* opr) {
        if (replace_vars(opr->input())) {
            recomp_id[opr] ++;
            opr = copy_opr_from_new_inputs(opr, true, recomp_id[opr] - 1);
        }
        set_priority(opr);
    };

    DepOprIter dep_iter{on_opr_visited};
    
    for (auto opr : oprseq) {
        auto iter = action.find(opr);
        if (iter != action.end()) {
            for (auto i : iter->second) {
                replace_vars(i->input());
                recomp_id[i] ++;
                auto opr_new = copy_opr_from_new_inputs(i, false, recomp_id[i] - 1);
                set_priority(opr_new);
            }
            action.erase(iter);
        }
        dep_iter.add(opr);
    }
    mgb_assert(action.empty());
}

#endif  // !MGB_ENABLE_DTR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
