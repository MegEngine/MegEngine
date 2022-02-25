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

}  // namespace

class SeqModifierForDTR::ModifyActionPlanner : public ModifyActionPlannerBase {
public:
    ModifyActionPlanner(SeqModifierBase* par) : ModifyActionPlannerBase{par} {}

    void prepare(const OprNodeArray& opr_seq);

    SeqModifyAction perform_dtr(
            CompNode comp_node, const OprNodeArray& seq, Config* config);
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
    for (auto&& i : *cn2oprseq) {
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
    size_t cur_op_cnt = 0;

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
            return true;
        }
        return false;
    };

    auto get_latest = [&](Var* var) {
        auto iter = latest_var.find(var->orig_var);
        if (iter == latest_var.end()) {
            return var;
        } else {
            return iter->second;
        }
    };

    ThinHashMap<Var*, double> dfs_back;
    ThinHashMap<Var*, double> dfs_ops;
    ThinHashMap<Var*, double> dfs_front;
    ThinHashMap<Var*, double> dfs_mem;
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
            for (size_t j = 1; j < var->access_rec.size(); j++) {
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

    auto regen_mem = [&](Var* var) {
        thin_function<double(Var*)> dfs_b;
        dfs_b = [&](Var* var) {
            if (dfs_mem.find(var) != dfs_mem.end()) {
                return dfs_mem[var];
            }
            auto opr = var->owner_opr();
            double mem_sum = var->size;
            for (auto i : opr->input) {
                auto ivar = get_latest(i);
                if (need_regen(ivar)) {
                    mem_sum += dfs_b(ivar);
                }
            }
            dfs_mem[var] = mem_sum;
            return mem_sum;
        };
        return dfs_b(var);
    };

    auto next_used = [&](Var* var) {
        var = get_latest(var);
        size_t t = DUPOPR_TIME;
        for (auto rec : var->access_rec) {
            if (rec.time > cur_op_cnt - 1 && rec.time < t)
                t = rec.time;
        }
        if (t < DUPOPR_TIME) {
            return t + 1 - cur_op_cnt;
        } else {
            return t;
        }
    };

    double tim_factor = 1;
    double mem_factor = 1;
    if (config->recomp_memory_factor >= 0) {
        mem_factor = config->recomp_memory_factor;
    }
    if (config->recomp_time_factor >= 0) {
        tim_factor = config->recomp_time_factor;
    }

    static constexpr double MAX_EVAL_VALUE = std::numeric_limits<double>::max();
    auto find_best = [&]() {
        Var* best = nullptr;
        double min_eval_value = MAX_EVAL_VALUE;
        dfs_back.clear();
        dfs_front.clear();
        dfs_mem.clear();
        for (auto var : alive_vars) {
            if (var->size < config->evictee_minimum_size || pin[var->orig_var] > 0 ||
                is_bad_opr(var->owner_opr()->orig_opr)) {
                continue;
            }
            double regen_t = regen_time(var);
            double regen_m = regen_mem(var);
            double eval_value = pow(regen_t, tim_factor) * pow(regen_m, mem_factor) /
                                static_cast<double>(var->size) / next_used(var);
            if (eval_value < min_eval_value) {
                min_eval_value = eval_value;
                best = var;
            }
        }
        return best;
    };

    auto do_evict = [&](Var* var) { remove_alive(var); };

    thin_function<void(Var*)> recursive_free;
    auto auto_evict = [&](size_t needed) {
        // proactively remove end-of-life vars
        std::vector<Var*> to_free(0);
        for (auto i : alive_vars) {
            if (next_used(get_latest(i)) == DUPOPR_TIME && pin[i->orig_var] == 0) {
                to_free.push_back(get_latest(i));
            }
        }
        for (auto i : to_free) {
            recursive_free(get_latest(i));
        }
        while (cur_usage + needed >= config->eviction_threshold) {
            Var* v = find_best();
            if (!v) {
                break;
            }
            do_evict(v);
        }
    };

    recursive_free = [&](Var* var) {
        if (pin[var->orig_var] > 0)
            return;
        auto opr = var->owner_opr();
        bool need = false;
        for (auto i : var->access_rec) {
            if (i.time >= cur_op_cnt) {
                need = true;
                break;
            }
        }
        if (!need) {
            if (remove_alive(var)) {
                for (auto i : opr->input) {
                    recursive_free(get_latest(i));
                }
            }
        }
    };

    thin_function<Var*(Opr*, Var*)> regenerate;
    regenerate = [&](Opr* reader, Var* var) {
        auto opr = var->owner_opr();
        // FIXME: if var can not be recomputed, the previous eviction may fail
        if (is_bad_opr(opr->orig_opr)) {
            return var;
        }

        auto new_opr_storage = opr_mempool().alloc_unique(
                opr->orig_opr, static_cast<size_t>(DUPOPR_TIME));
        auto new_opr = new_opr_storage.get();

        new_opr->input.reserve(opr->input.size());
        new_opr->output.reserve(opr->output.size());
        new_opr->estimate_compute_time = opr->estimate_compute_time;

        for (auto i : opr->input) {
            pin[i->orig_var]++;
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
            auto&& ovar = var_mempool().alloc_unique(lo->orig_var, lo->size, new_opr);
            ovar->recomp_id = lo->recomp_id + 1;
            new_opr->output.push_back(ovar.get());
            if (need_regen(lo)) {  // latest output is not in memory
                if (o == var) {
                    new_var = ovar.get();
                    for (size_t i = 1; i < lo->access_rec.size(); i++) {
                        new_var->access_rec.push_back(lo->access_rec[i]);
                    }
                    add_alive(new_var);
                    latest_var[o->orig_var] = new_var;
                }
            }
            var_storage().emplace_back(std::move(ovar));
        }
        for (auto i : opr->input) {
            pin[i->orig_var]--;
        }
        return new_var;
    };

    for (size_t j = 0; j < seq().size(); ++j) {
        ++cur_op_cnt;
        auto opr = seq()[j].get();
        for (auto i : opr->input) {
            pin[i->orig_var]++;
        }
        for (auto i : opr->inputs_size) {
            if (i > 0)
                cur_usage += i;
        }
        for (auto i : opr->input) {
            i = get_latest(i);
            if (need_regen(i)) {
                i = regenerate(opr, i);
            }
        }
        size_t needed = 0;
        for (auto o : opr->output) {
            needed += o->size;
        }
        auto_evict(needed);
        for (auto o : opr->output) {
            o = get_latest(o);
            add_alive(o);
        }
        for (auto i : opr->input) {
            pin[i->orig_var]--;
        }
        for (auto i : opr->input) {
            if (opr == i->last_access_opr()) {
                recursive_free(get_latest(i));
            }
        }
        for (auto o : opr->output) {
            if (opr == o->last_access_opr()) {
                recursive_free(get_latest(o));
            }
        }
        for (auto i : opr->inputs_size) {
            if (i < 0)
                cur_usage += i;
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

void SeqModifierForDTR::apply_action(
        SeqModifyAction& action, const OprNodeArray& oprseq) {
    auto cur_priority = std::numeric_limits<
            decltype(OperatorNodeBase::NodeProp::Attribute::priority)>::min();

    ThinHashSet<OperatorNodeBase*> modified_opr;
    ThinHashMap<OperatorNodeBase*, size_t> recomp_id;
    auto set_priority = [&](OperatorNodeBase* opr) {
        mgb_assert(modified_opr.insert(opr).second);
        mem_opt().set_priority(opr, cur_priority++);
    };

    auto on_opr_visited = [&](OperatorNodeBase* opr) {
        if (replace_vars(opr->input())) {
            recomp_id[opr]++;
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
                recomp_id[i]++;
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
