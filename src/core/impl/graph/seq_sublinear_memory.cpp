/**
 * \file src/core/impl/graph/seq_sublinear_memory.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./seq_sublinear_memory.h"

#if MGB_ENABLE_SUBLINEAR

using namespace mgb;
using namespace cg;

#include "megbrain/comp_node_env.h"
#include "megbrain/plugin/opr_footprint.h"
#include "megbrain/serialization/opr_shallow_copy.h"
#include "megbrain/system.h"
#include "megbrain/utils/arith_helper.h"
#include "megbrain/utils/mempool.h"
#include "megbrain/utils/timer.h"

#include <cmath>
#include <random>

namespace {

class RNGxorshf {
    uint64_t s[2];

public:
#if __cplusplus >= 201703L
    typedef uint64_t result_type;
    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return UINT64_MAX; }
#endif
    RNGxorshf(uint64_t seed) {
        std::mt19937_64 gen(seed);
        s[0] = gen();
        s[1] = gen();
    }

    uint64_t operator()() {
        uint64_t x = s[0];
        uint64_t const y = s[1];
        s[0] = y;
        x ^= x << 23;                          // a
        s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);  // b, c
        return s[1] + y;
    }
};

bool is_bad_opr(OperatorNodeBase* opr) {
    using F = OperatorNodeBase::NodeProp::Flag;
    return opr->node_prop().contain(
        F::IMPURE_FUNC | F::NO_AUTOMATIC_DUP | F::FORCE_UPDATE_INPUT_VAR);
}

}  // namespace
/* ======================  Abstract Opr & Var ======================  */
struct SeqModifierForSublinearMemory::Opr {
    OperatorNodeBase* const orig_opr;
    std::vector<Var*> input, output;
    const size_t time;  //!< index in opr sequence
    const bool is_endpoint;

    //! input vars that have been discarded and need to be recomputed before
    //! this opr; for internal use by apply_discard_plan()
    std::vector<Var*> inputs_to_recompute;

    //! new oprs to be inserted before this opr; setup by apply_discard_plan()
    std::vector<MemPool<Opr>::UniquePtr> oprs_insert_before;

    //! [begin, end) interval of *time* for oprs belonging to this block; setup
    //! by make_discard_plan()
    size_t block_begin_time = 0, block_end_time = 0;

    Opr(OperatorNodeBase* opr, size_t t)
            : orig_opr{opr},
              time{t},
              is_endpoint{opr->owner_graph()
                                  ->options()
                                  .opr_attribute.get_sublinear_memory_endpoint(
                                          opr)} {}
};

struct SeqModifierForSublinearMemory::Var {
    //! write or read access of a var
    struct AccessRecord {
        Opr* const opr;
        const size_t time;
        size_t stride;  //!< time distance until next read; 0 for last access

        explicit AccessRecord(Opr* o = nullptr)
                : opr{o}, time{o->time}, stride{0} {}
    };

    VarNode* const orig_var;
    const size_t size;  //!< memory usage in bytes of this var

    //! access_rec[0] is the creation opr, and others are reader oprs
    std::vector<AccessRecord> access_rec;

    /*!
     * An index in access_rec
     *
     * if valid, then the var should be discarded after
     * discard_tailing_access->opr finishes
     *
     * setup by make_discard_plan
     */
    Maybe<size_t> discard_tailing_access;

    /*!
     * An index in access_rec
     * maintained during make_discard_plan(), for the next access relative to
     * current operator
     */
    Maybe<size_t> next_access;

    AccessRecord* visit_discard_tailing_access() {
        return discard_tailing_access.valid()
                       ? &access_rec.at(discard_tailing_access.val())
                       : nullptr;
    }

    AccessRecord* visit_next_access() {
        return next_access.valid() ? &access_rec.at(next_access.val())
                                   : nullptr;
    }

    auto owner_opr() const { return access_rec[0].opr; }

    auto last_access_opr() const { return access_rec.back().opr; }

    Var(VarNode* var, size_t s, Opr* opr) : orig_var{var}, size{s} {
        access_rec.emplace_back(opr);
    }
};
/* ======================  ModifyActionPlanner ======================  */
class SeqModifierForSublinearMemory::ModifyActionPlanner {
    //! special creation time used for oprs duplicated from others
    static constexpr size_t DUPOPR_TIME =
            std::numeric_limits<size_t>::max() - 1;

    using VarArray = std::vector<Var*>;
    using VarSet = ThinHashSet<Var*>;
    using OprArray = std::vector<Opr*>;

    const SeqModifierForSublinearMemory* const m_par_modifier;
    const OprNodeArray* m_orig_opr_seq;

    MemPool<Var> m_var_mempool;
    MemPool<Opr> m_opr_mempool;
    std::vector<MemPool<Var>::UniquePtr> m_var_storage;
    std::vector<MemPool<Opr>::UniquePtr> m_seq;

    size_t m_nr_endpoint_oprs = 0;

    VarSet m_prev_block_discard_vars;
    std::vector<OprArray> m_blocks;

    //! split_point_set to block
    void split_into_blocks(const SplitPointSet& split_point_set);

    //! setup Var::discard_tailing_access
    void make_discard_plan();

    //! modify oprs and vars according to Var::discard_tailing_access
    void apply_discard_plan();

    /*!
     * \brief cleanup request for discarding vars that are immediately
     *      accessed in the next block
     * \param all_inputs all oprs in this block
     * \param discard_vars vars discarded after this block; this sequence
     *      may be modified inplace, but the resulting value has no
     *      specific meaning for the caller (i.e. as temporary var)
     */
    void refine_block_discard_rec(const OprArray& all_oprs, size_t block_num,
                                  VarSet& discard_vars);

    size_t calc_bottleneck_from_discard_plan();

public:
    ModifyActionPlanner(SeqModifierForSublinearMemory* par)
            : m_par_modifier{par} {}

    ~ModifyActionPlanner() noexcept {
        m_opr_mempool.disable_freelist();
        m_var_mempool.disable_freelist();
    }
    //! init m_orig_opr_seq from opr_seq, should be called first.
    void init_seq(const OprNodeArray& opr_seq);

    //! generate split point set from thresh
    SplitPointSet get_split_point_set(size_t block_size_thresh);
    /*!
     * \brief get memory bottleneck after imposing a block size threshold
     *
     * The result can be retrieved by get_prev_action()
     */
    size_t get_memory_bottleneck(const SplitPointSet& split_point_set);

    //! get action for previous get_memory_bottleneck() call
    void get_prev_action(SeqModifyAction& action);
};

void SeqModifierForSublinearMemory::ModifyActionPlanner::get_prev_action(
        SeqModifyAction& action) {
    action.clear();
    for (auto&& opr : m_seq) {
        auto&& arr = opr->oprs_insert_before;
        if (arr.empty())
            continue;
        auto&& dest = action[opr->orig_opr];
        dest.reserve(arr.size());
        for (auto&& i : opr->oprs_insert_before)
            dest.push_back(i->orig_opr);
    }
}

size_t
SeqModifierForSublinearMemory::ModifyActionPlanner::get_memory_bottleneck(
        const SplitPointSet& split_point_set) {
    split_into_blocks(split_point_set);
    make_discard_plan();
    apply_discard_plan();
    return calc_bottleneck_from_discard_plan();
}

SeqModifierForSublinearMemory::SplitPointSet
SeqModifierForSublinearMemory::ModifyActionPlanner::get_split_point_set(
        size_t block_size_thresh) {
    auto split_point_set = make_split_point_set();
    size_t cur_block_usage = 0;

    ThinHashSet<Var*> cur_block_alive_vars;

    auto add_alive = [&](Var* var) {
        auto&& ins = cur_block_alive_vars.insert(var);
        mgb_assert(ins.second);
        cur_block_usage += var->size;
    };

    auto remove_alive = [&](Var* var) {
        if (cur_block_alive_vars.erase(var)) {
            auto size = var->size;
            mgb_assert(size <= cur_block_usage);
            cur_block_usage -= size;
        }
    };

    auto flush_block_member = [&](size_t p) {
        split_point_set->push_back(p);
        cur_block_usage = 0;
        cur_block_alive_vars.clear();
    };

    for (size_t i = 0; i < m_seq.size(); ++i) {
        auto opr = m_seq[i].get();

        for (auto i : opr->output)
            add_alive(i);

        for (auto i : opr->input) {
            if (opr == i->last_access_opr())
                remove_alive(i);
        }

        if (i + 1 < m_seq.size() && (cur_block_usage < block_size_thresh ||
                                     (m_nr_endpoint_oprs && !opr->is_endpoint)))
            continue;

        flush_block_member(i);
    }
    return split_point_set;
}

void SeqModifierForSublinearMemory::ModifyActionPlanner::init_seq(
        const OprNodeArray& opr_seq) {
    m_orig_opr_seq = &opr_seq;

    m_var_storage.clear();
    m_seq.clear();
    m_var_mempool.reorder_free();
    m_opr_mempool.reorder_free();
    m_nr_endpoint_oprs = 0;

    ThinHashMap<VarNode*, Var*> varmap;
    for (auto orig_opr : *m_orig_opr_seq) {
        auto time = m_seq.size();
        m_seq.emplace_back(m_opr_mempool.alloc_unique(orig_opr, time));
        auto opr = m_seq.back().get();
        m_nr_endpoint_oprs += opr->is_endpoint;

        for (auto&& dep : orig_opr->node_prop().dep_map()) {
            if (!OperatorNodeBase::NodeProp::is_device_value_dep(dep.second))
                continue;

            auto iter = varmap.find(dep.first);
            if (iter == varmap.end()) {
                // input var needs not to be considered
                continue;
            }

            auto ivar = iter->second;
            bool exist = false;
            for (auto i : opr->input) {
                if (i == ivar) {
                    exist = true;
                    break;
                }
            }
            if (exist) {
                // same var for different inputs
                continue;
            }

            opr->input.push_back(ivar);
            auto&& prev_rec = ivar->access_rec.back();
            prev_rec.stride = time - prev_rec.opr->time;
            ivar->access_rec.emplace_back(opr);
        }

        for (auto i : orig_opr->output()) {
            auto var2memsize = m_par_modifier->m_mem_opt.var2memsize();
            auto iter = var2memsize->find(i);
            if (iter == var2memsize->end()) {
                // some vars are ignored; see split_into_cn2oprseq()
                continue;
            }
            m_var_storage.emplace_back(
                    m_var_mempool.alloc_unique(i, iter->second, opr));
            auto ovar = m_var_storage.back().get();
            varmap[i] = ovar;
            opr->output.push_back(ovar);
        }
        mgb_assert(!opr->output.empty());
    }

    // remove unused output
    for (auto&& i : m_seq) {
        auto&& oarr = i->output;
        for (size_t j = 0; j < oarr.size();) {
            if (oarr[j]->access_rec.size() == 1) {
                std::swap(oarr[j], oarr.back());
                oarr.pop_back();
            } else
                ++j;
        }
    }
}

size_t SeqModifierForSublinearMemory::ModifyActionPlanner::
        calc_bottleneck_from_discard_plan() {
    size_t cur_usage = 0, max_usage = 0;

    size_t time = 0;

    // map from var to insert time
    // use unordered_map<> in dbg because ThinHashMap does not support copy
    ThinHashMap<Var*, size_t> alive_vars;

    auto remove_alive = [&](Opr* opr, const std::vector<Var*>& vars) {
        for (auto i : vars) {
            if (opr == i->last_access_opr()) {
                cur_usage -= i->size;
                auto nr = alive_vars.erase(i);
                mgb_assert(nr == 1);
            }
        }
    };

    auto process_opr = [&](Opr* opr) {
        for (auto i : opr->output) {
            cur_usage += i->size;
            auto&& ins = alive_vars.insert({i, time});
            mgb_assert(ins.second);
        }

        update_max(max_usage, cur_usage);

        if (opr->output.size() > 1) {
            // a single output may be unused if this opr has multiple outputs
            // and some of them are discarded
            remove_alive(opr, opr->output);
        }
        remove_alive(opr, opr->input);
        ++time;
    };

    for (auto&& opr : m_seq) {
        for (auto&& i : opr->oprs_insert_before)
            process_opr(i.get());
        process_opr(opr.get());
    }
    mgb_assert(alive_vars.empty());

    return max_usage;
}

void SeqModifierForSublinearMemory::ModifyActionPlanner::apply_discard_plan() {
    ThinHashSet<Var*> alive_vars;

    // map from original var to duplicated var
    ThinHashMap<Var*, Var*> var_map;

    auto add_alive = [&](Var* var) {
        auto&& ins = alive_vars.insert(var);
        mgb_assert(ins.second);
    };

    auto remove_alive = [&](Var* var) {
        auto nr = alive_vars.erase(var);
        mgb_assert(nr);
    };

    auto check_and_remove = [&](size_t timestamp, Var* var) {
        auto acc = var->visit_discard_tailing_access();
        if (!acc || (acc && acc->opr->time >= timestamp)) {
            mgb_assert(var->owner_opr()->output.size() > 1);
            for (size_t i = 0; i < var->access_rec.size(); ++ i) {
                if (var->access_rec[i].time >= timestamp) {
                    mgb_assert(i > 0);
                    auto acc_rec_begin = var->access_rec.data();
                    var->access_rec.resize(i);
                    var->discard_tailing_access = i - 1;
                    mgb_assert(var->access_rec.data() == acc_rec_begin);
                    break;
                }
            }
        }
    };

    auto try_discard = [&](Opr* opr, Var* var) {
        auto acc = var->visit_discard_tailing_access();
        if (acc && acc->opr == opr) {
            remove_alive(var);
            acc[1].opr->inputs_to_recompute.push_back(var);
            auto acc_rec_begin = var->access_rec.data();

            // make this opr as the last reader for original var
            var->access_rec.resize(acc - acc_rec_begin + 1);
            mgb_assert(var->access_rec.data() == acc_rec_begin);
        }
    };

    // recompute a var by inserting new oprs
    auto recompute = [&](Opr* reader, Var* var) {
        mgb_assert(!alive_vars.count(var));

        auto block_begin = var->owner_opr()->block_begin_time,
             block_end = var->owner_opr()->block_end_time;

        thin_function<Var*(Var*)> add_dep;
        add_dep = [&](Var* var) {
            if (alive_vars.count(var))
                return var;
            {
                auto iter = var_map.find(var);
                if (iter != var_map.end())
                    return iter->second;
            }

            auto opr = var->owner_opr();

            if (opr->time < block_begin) {
                // do not recompute vars outside this block
                return var;
            }

            if (is_bad_opr(opr->orig_opr)) {
                return var;
            }

            mgb_assert(opr->time < block_end);

            auto new_opr_storage = m_opr_mempool.alloc_unique(
                    opr->orig_opr, static_cast<size_t>(DUPOPR_TIME));
            auto new_opr = new_opr_storage.get();

            new_opr->input.reserve(opr->input.size());
            new_opr->output.reserve(opr->output.size());

            for (auto i : opr->input) {
                auto ivar = add_dep(i);
                new_opr->input.push_back(ivar);
                ivar->access_rec.emplace_back(new_opr);
            }

            reader->oprs_insert_before.emplace_back(std::move(new_opr_storage));

            Var* new_var = nullptr;
            for (auto i : opr->output) {
                auto&& ovar = m_var_mempool.alloc_unique(i->orig_var, i->size,
                                                         new_opr);
                new_opr->output.push_back(ovar.get());
                if (i == var)
                    new_var = ovar.get();

                add_alive(ovar.get());
                auto ins = var_map.insert({i, ovar.get()});
                mgb_assert(ins.second);

                m_var_storage.emplace_back(std::move(ovar));
            }
            mgb_assert(new_var);
            return new_var;
        };
        add_dep(var);
    };

    for (auto&& _raw_opr : m_seq) {
        auto opr = _raw_opr.get();

        for (auto i : opr->inputs_to_recompute)
            recompute(opr, i);

        for (auto&& i : opr->input) {
            // find in recomputed vars and record access
            auto iter = var_map.find(i);
            if (iter != var_map.end()) {

                // handle the vars which haven't been discard after recomputing
                // try to remove access records which redirect to dup-opr
                check_and_remove(opr->time, i);

                i = iter->second;
                i->access_rec.emplace_back(opr);
                mgb_assert(alive_vars.count(i));
                continue;
            }

            if (opr == i->last_access_opr()) {
                remove_alive(i);
            } else {
                try_discard(opr, i);
            }
        }
        for (auto i : opr->output) {
            add_alive(i);
            try_discard(opr, i);
        }
    }
}

void SeqModifierForSublinearMemory::ModifyActionPlanner::make_discard_plan() {
    ThinHashSet<Var*> cur_block_alive_vars;
    std::vector<Opr*> cur_block_member;
    VarSet cur_block_discard_vars;

    size_t nr_blocks = 0;

    auto flush_block_member = [&]() {
        nr_blocks++;
        auto begin = cur_block_member.front()->time,
             end = cur_block_member.back()->time + 1;
        for (auto i : cur_block_member) {
            i->block_begin_time = begin;
            i->block_end_time = end;
        }
        cur_block_member.clear();
        cur_block_alive_vars.clear();
        cur_block_discard_vars.clear();
    };

    for (auto&& block : m_blocks) {
        for (auto&& opr : block) {
            cur_block_member.push_back(opr);

            for (auto i : opr->output) {
                cur_block_alive_vars.insert(i);
                i->next_access = 1;
            }

            for (auto i : opr->input) {
                if (opr == i->last_access_opr()) {
                    cur_block_alive_vars.erase(i);
                    i->next_access = None;
                } else if (opr == i->visit_next_access()->opr) {
                    ++i->next_access.val();
                }
            }
        }

        // TODO: should rewrite for multi-outputs opr
        // This loop only make sense for single-output oprs. Since all oprs
        // only recompute once, it should serach best recomputing-time in opr-level
        // rather than find best discarding-time in var-level for multi-outputs opr.
        for (auto var : cur_block_alive_vars) {
            if (is_bad_opr(var->owner_opr()->orig_opr))
                continue;

            Var::AccessRecord* best = nullptr;
            auto&& rec = var->access_rec;
            mgb_assert(var->next_access.val() >= 1);

            // find best future time to discard
            for (size_t i = var->next_access.val() - 1; i < rec.size() - 1;
                 ++i) {
                if (!i && var->owner_opr()->output.size() == 1) {
                    // never discard output var directly
                    continue;
                }

                auto cur = &rec[i], next = &rec[i + 1];
                if (cur->stride > next->opr->input.size()) {
                    if (!best || cur->stride > best->stride)
                        best = cur;
                } else {
                    // if cur stride too small, it would be immediately used by
                    // next and should not be discarded
                }
            }

            if (best) {
                var->discard_tailing_access = best - rec.data();
                cur_block_discard_vars.insert(var);
            } else {
                var->discard_tailing_access = None;
            }
        }
        // the endpoint vars of the block shouldn't be duplicated
        for (auto&& i : block.back()->output) {
            i->discard_tailing_access = None;
        }
        refine_block_discard_rec(cur_block_member, nr_blocks,
                                 cur_block_discard_vars);
        flush_block_member();
    }
}

void SeqModifierForSublinearMemory::ModifyActionPlanner::split_into_blocks(
        const SplitPointSet& split_point_set) {
    m_blocks.clear();
    std::vector<Opr*> cur_block_member;
    size_t i, j;
    for (i = j = 0; i < m_seq.size() && j < split_point_set->size(); ++i) {
        auto opr = m_seq[i].get();
        cur_block_member.push_back(opr);
        if (i != split_point_set->at(j))
            continue;
        m_blocks.push_back(cur_block_member);
        cur_block_member.clear();
        j++;
    }
    mgb_assert(i >= m_seq.size());
    mgb_assert(j >= split_point_set->size());
}

void SeqModifierForSublinearMemory::ModifyActionPlanner::
        refine_block_discard_rec(const OprArray& all_oprs, size_t block_num,
                                 VarSet& discard_vars) {
    if (block_num) {
        for (auto&& opr : all_oprs) {
            for (auto i : opr->input) {
                auto discard = i->visit_discard_tailing_access();
                if (discard && discard[1].opr == opr &&
                    m_prev_block_discard_vars.count(i)) {
                    // i is discarded after previous block, but used in this
                    // block, so do not discard it
                    i->discard_tailing_access = None;
                }
            }
        }
    }
    m_prev_block_discard_vars.swap(discard_vars);
}

/* ====================  ActionSearcherSingleCN ====================  */
class SeqModifierForSublinearMemory::ActionSearcherSingleCN {
    SeqModifierForSublinearMemory* const m_par_modifier;
    const OprNodeArray* m_cur_opr_seq;

    std::vector<std::pair<size_t, size_t>> m_history;
    size_t m_min_bottleneck, m_best_thresh;
    using Record = std::pair<SplitPointSet, size_t>;
    SplitPointSet m_best_sps;
    std::vector<Record> m_cur_records;
    SeqModifyAction m_action;
    std::vector<std::future<void>> m_futures;
    std::mutex m_mtx;

    /*!
     * \brief check given thresh, and update states
     * \return bottleneck value for given thresh
     */
    void do_search_update_thresh(size_t thresh);
    void do_search_update_split_point_set(SplitPointSet& split_point_set);

    //! invoke search asynchronously in m_planner_thread_pool
    void invoke_search(size_t thresh);
    void invoke_search(SplitPointSet&& split_point_set);

    //! wait for all unfinished asynchronous invoke_search() calls
    void wait_all();

    //! search for initial solutions
    void search_preset();
    //! genetic algorithm
    void search_genetic();
    void search_refine();

    static inline bool cmp_sps(const SplitPointSet &a, const SplitPointSet &b) {
        if (a->size() != b->size()) {
            return a->size() < b->size();
        } else {
            size_t length = a->size();
            for (size_t i = 0; i < length; ++i) {
                if (a->at(i) != b->at(i))
                    return a->at(i) < b->at(i);
            }
            return false;
        }
    }

public:
    ActionSearcherSingleCN(SeqModifierForSublinearMemory* par)
            : m_par_modifier{par} {
        auto & m_config = m_par_modifier->m_config;
        //! allow environmental variable to overwrite the setting
        if (auto env = MGB_GETENV("MGB_SUBLINEAR_MEMORY_THRESH_NR_TRY")) {
            m_config->thresh_nr_try = std::stoi(env);
        }
        if (auto env = MGB_GETENV("MGB_SUBLINEAR_MEMORY_GENETIC_NR_ITER")) {
            m_config->genetic_nr_iter = std::stoi(env);
        }
        if (auto env = MGB_GETENV("MGB_SUBLINEAR_MEMORY_GENETIC_POOL_SIZE")) {
            auto psize = static_cast<size_t>(std::stoi(env));
            mgb_assert(psize > 0 || m_config->genetic_nr_iter == 0,
                       "invalid pool size %zu in genetic algorithm,", psize);
            m_config->genetic_pool_size = psize;
        }
        if (auto env = MGB_GETENV("MGB_SUBLINEAR_MEMORY_LOWER_BOUND_MB")) {
            m_config->lb_memory = std::stoi(env) * 1024 * 1024;
        }
    }

    const SeqModifyAction& search(CompNode comp_node, const OprNodeArray* seq);
};

void SeqModifierForSublinearMemory::ActionSearcherSingleCN::
        do_search_update_thresh(size_t thresh) {
    ModifyActionPlanner* planner =
            m_par_modifier->m_thread2planner.at(std::this_thread::get_id())
                    .get();

    planner->init_seq(*m_cur_opr_seq);
    SplitPointSet split_point_set = planner->get_split_point_set(thresh);
    auto cur = planner->get_memory_bottleneck(split_point_set);

    MGB_LOCK_GUARD(m_mtx);
    if (cur < m_min_bottleneck || (cur == m_min_bottleneck && m_best_thresh < thresh)) {
        m_best_thresh = thresh;
        m_min_bottleneck = cur;
        m_best_sps = split_point_set;
        planner->get_prev_action(m_action);
    }
    m_history.emplace_back(thresh, cur);
    m_cur_records.emplace_back(std::move(split_point_set), cur);
}

void SeqModifierForSublinearMemory::ActionSearcherSingleCN::
        do_search_update_split_point_set(SplitPointSet& split_point_set) {
    ModifyActionPlanner* planner =
            m_par_modifier->m_thread2planner.at(std::this_thread::get_id())
                    .get();

    planner->init_seq(*m_cur_opr_seq);
    auto cur = planner->get_memory_bottleneck(split_point_set);

    MGB_LOCK_GUARD(m_mtx);
    if (cur < m_min_bottleneck || (cur == m_min_bottleneck &&
                cmp_sps(split_point_set, m_best_sps))) {
        m_min_bottleneck = cur;
        m_best_sps = split_point_set;
        planner->get_prev_action(m_action);
    }
    m_cur_records.emplace_back(std::move(split_point_set), cur);
}

void SeqModifierForSublinearMemory::ActionSearcherSingleCN::invoke_search(
        size_t thresh) {
    m_futures.emplace_back(m_par_modifier->m_planner_thread_pool.launch(
            &ActionSearcherSingleCN::do_search_update_thresh, this, thresh));
}

void SeqModifierForSublinearMemory::ActionSearcherSingleCN::invoke_search(
        SplitPointSet&& split_point_set) {
    m_futures.emplace_back(m_par_modifier->m_planner_thread_pool.launch(
            &ActionSearcherSingleCN::do_search_update_split_point_set, this,
            split_point_set));
}

void SeqModifierForSublinearMemory::ActionSearcherSingleCN::wait_all() {
    for (auto&& i : m_futures)
        i.get();
    m_futures.clear();
}

void SeqModifierForSublinearMemory::ActionSearcherSingleCN::search_preset() {
    auto init_thresh = m_min_bottleneck;

    // search in log space
    for (size_t thresh = init_thresh >> 1; thresh >= 1024; thresh >>= 1) {
        invoke_search(thresh);
    }

    size_t NR_TRY = m_par_modifier->m_config->thresh_nr_try;

    // search in linear space
    auto step = init_thresh / (NR_TRY + 1);
    for (size_t i = 1; i <= NR_TRY; ++i) {
        invoke_search(step * i);
    }

    wait_all();

    // search around current best thresh
    auto start = m_best_thresh / 2;
    step = (m_best_thresh * 2 - start) / (NR_TRY - 1);
    for (size_t i = 0; i < NR_TRY; ++i) {
        invoke_search(start + step * i);
    }
    wait_all();
}

void SeqModifierForSublinearMemory::ActionSearcherSingleCN::search_genetic() {
    RNGxorshf rng(2333);
    size_t POOL_SIZE = m_par_modifier->m_config->genetic_pool_size;
    size_t NR_ITER = m_par_modifier->m_config->genetic_nr_iter;
    auto mutation = [&](const SplitPointSet& sps) {
        auto s = *sps;
        size_t length = s.size();
        mgb_assert(length > 0);
        size_t ri = rng() & 3;
        auto ret = make_split_point_set();
        thin_function<void(size_t)> on_split_point;
        if (ri < 1) {
            // insert a split point randomly
            on_split_point = [&](size_t id) {
                size_t st = id > 0 ? s[id - 1] + 1 : 0;
                if (s[id] - st + 1 > 1)
                    ret->push_back(st + rng() % (s[id] - st));
                ret->push_back(s[id]);
            };
        } else if (ri < 2) {
            // remove a split point randomly
            on_split_point = [&](size_t id) {
                if (id == length - 1) {
                    ret->push_back(s[id]);
                } else {
                    /* do nothing */
                }
            };
        } else if (ri < 3) {
            // move a split point randomly
            on_split_point = [&](size_t id) {
                if (id == length - 1) {
                    ret->push_back(s[id]);
                } else {
                    size_t st = id > 0 ? s[id - 1] + 1 : 0;
                    size_t ed = s[id + 1];
                    mgb_assert(ed - st + 1 > 1);
                    ret->push_back(st + rng() % (ed - st));
                }
            };
        } else {
            // no action
            on_split_point = [&](size_t id) { ret->push_back(s[id]); };
        }
        size_t p = rng() % length;
        for (size_t i = 0; i < length; ++i) {
            if (i == p) {
                on_split_point(i);
            } else {
                ret->push_back(s[i]);
            }
        }
        return ret;
    };
    auto crossover = [&](const SplitPointSet& s1, const SplitPointSet& s2) {
        auto ret = make_split_point_set();
        size_t p = rng() % (m_cur_opr_seq->size() / 2);
        for (auto&& x : *s1) {
            if (x < p)
                ret->push_back(x);
        }
        for (auto&& x : *s2) {
            if (x >= p)
                ret->push_back(x);
        }
        return ret;
    };
    for (size_t time = 0; time < NR_ITER; time++) {
        auto cmp = [&](const Record& a, const Record& b) {
            if (a.second != b.second)
                return a.second < b.second;
            return cmp_sps(a.first, b.first);
        };
        std::sort(m_cur_records.begin(), m_cur_records.end(), cmp);

#if MGB_ENABLE_LOGGING
#define LOG_STEP 10
        if (time % LOG_STEP == 0) {
            constexpr double SIZE2MB = 1.0 / 1024 / 1024;
            std::string msg{ssprintf(
                    "Searching in sublinear memory, genetic algorithm:\n"
                    "     Iter: %zu"
                    " cur_min_bottleneck: %.2lf"
                    " his_min_bottleneck: %.2lf\n",
                    time, m_cur_records[0].second * SIZE2MB,
                    m_min_bottleneck * SIZE2MB)};
            mgb_log_debug("%s", msg.c_str());
        }
#endif

        size_t length = std::min(POOL_SIZE, m_cur_records.size());
        std::vector<size_t> perm;
        std::vector<Record> records;
        auto it = m_cur_records.begin();
        // random selection
        for (size_t i = 0; i < length; ++i) {
            perm.push_back(i);
            while (true) {
                if (it == m_cur_records.end())
                    it = m_cur_records.begin();
                if (8 * (rng() % std::max((size_t)1, m_cur_records.begin()->second)) <
                    7 * std::max((size_t)1, it->second)) {
                    records.push_back(*it);
                    it = m_cur_records.erase(it);
                    break;
                } else {
                    it++;
                }
            }
        }
        m_cur_records = records;
#if __cplusplus >= 201703L
        std::shuffle(perm.begin(), perm.end(), rng);
#else
        std::random_shuffle(perm.begin(), perm.end(),
                            [&](size_t x) { return rng() % x; });
#endif
        for (size_t i = 0; i < length; ++i) {
            invoke_search(mutation(mutation(records[i].first)));
            invoke_search(crossover(records[i].first, records[perm[i]].first));
        }
        wait_all();
    }
}

void SeqModifierForSublinearMemory::ActionSearcherSingleCN::search_refine() {
    size_t lower_bound = m_par_modifier->m_config->lb_memory;
    if (m_min_bottleneck >= lower_bound)
        return;
    OprFootprint footprint;
    ThinHashSet<OperatorNodeBase*> dup_oprs_set;
    auto get_computation = [&](OperatorNodeBase* opr) {
        return footprint.get_computation(opr);
    };
    auto cmp = [&](size_t idx_a, size_t idx_b) {
        auto a = m_cur_opr_seq->at(idx_a);
        auto b = m_cur_opr_seq->at(idx_b);
        return get_computation(a) > get_computation(b);
    };
    for (auto&& i : m_action) {
        for (auto&& opr : i.second) {
            dup_oprs_set.insert(opr);
        }
    }
    std::vector<size_t> opr_idx;
    for (size_t idx = 0; idx < m_cur_opr_seq->size(); ++idx)
        if (dup_oprs_set.count(m_cur_opr_seq->at(idx)))
            opr_idx.push_back(idx);
    std::sort(opr_idx.begin(), opr_idx.end(), cmp);

    auto split_point_set = make_split_point_set(*m_best_sps);
    for (size_t i = 0; i < opr_idx.size(); ++i) {
        bool flag = true;
        split_point_set->push_back(opr_idx[i]);
        sort(split_point_set->begin(), split_point_set->end());
        auto f = [&] {
            ModifyActionPlanner* planner =
                    m_par_modifier->m_thread2planner
                            .at(std::this_thread::get_id())
                            .get();
            planner->init_seq(*m_cur_opr_seq);
            auto cur = planner->get_memory_bottleneck(split_point_set);
            if (cur >= lower_bound) {
                planner->get_prev_action(m_action);
                flag = false;
            }
        };
        m_par_modifier->m_planner_thread_pool.launch(f).get();
        if (!flag)
            break;
    }
}

const SeqModifierForSublinearMemory::SeqModifyAction&
SeqModifierForSublinearMemory::ActionSearcherSingleCN::search(
        CompNode comp_node, const OprNodeArray* seq) {
    m_action.clear();

    if (comp_node.locator().stream < 0) {
        // do not modify system stream oprs
        return m_action;
    }

    m_cur_opr_seq = seq;
    m_futures.clear();
    m_history.clear();
    m_cur_records.clear();

    RealTimer timer;
    m_best_thresh = m_min_bottleneck = std::numeric_limits<size_t>::max();

    //! init search
    invoke_search(m_best_thresh);
    wait_all();

    search_preset();
    auto t0 = timer.get_msecs_reset();
    search_genetic();
    auto t1 = timer.get_msecs_reset();
    search_refine();
    auto t2 = timer.get_msecs_reset();

    std::sort(m_history.begin(), m_history.end());
    m_par_modifier->m_prev_min_bottleneck.at(comp_node) = m_min_bottleneck;

#if MGB_ENABLE_LOGGING
    constexpr double SIZE2MB = 1.0 / 1024 / 1024;
    std::string msg{
            ssprintf("finished searching for sublinear memory: "
                     "comp_node=%s seq_len=%zu nr_search=%zu "
                     "time=%.1fms(init%.2f genetic%.2f refine%.2f)\n"
                     "thresh     bottleneck",
                     comp_node.to_string().c_str(), seq->size(),
                     m_history.size(), t0 + t1 + t2, t0, t1, t2)};
    for (auto&& i : m_history) {
        msg.push_back('\n');
        msg.append(ssprintf("%-10.2f %-10.2f", i.first * SIZE2MB,
                            i.second * SIZE2MB));
        if (i.second == m_min_bottleneck) {
            msg.append(" // best; ");
        }
    }
    msg.push_back('\n');
    msg.append(ssprintf("m_min_bottleneck: %-10.2f\n",
                        m_min_bottleneck * SIZE2MB));
    if(!m_par_modifier->m_config->genetic_nr_iter) {
        msg.append(ssprintf(
            "\nGenetic algorithm is currently DISABLED, "
            "set MGB_SUBLINEAR_MEMORY_GENETIC_NR_ITER [default = 0]"
            " to a positive integer to set the number of iterations"
            " in genetic algorithm.\n"));
    }
    mgb_log_debug("%s", msg.c_str());
#else
    MGB_MARK_USED_VAR(t0 + t1 + t2);
#endif
    return m_action;
}

/* ====================  SeqModifierForSublinearMemory ====================  */
void SeqModifierForSublinearMemory::InternalDeleter::operator()(
        ActionSearcherSingleCN* p) const {
    delete p;
}

void SeqModifierForSublinearMemory::InternalDeleter::operator()(
        ModifyActionPlanner* p) const {
    delete p;
}

void SeqModifierForSublinearMemory::reset_opr_seq(const OprNodeArray& oprseq) {
    m_var_map.clear();
    m_opr2replace_info.clear();
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

    auto cn2oprseq = m_mem_opt.split_into_cn2oprseq(oprseq, config);

    if (cn2oprseq->empty()) {
        // empty graph
        return;
    }

    SeqModifyAction action;

    MGB_TRY { action = search_action(cn2oprseq); }
    MGB_FINALLY(m_planner_thread_pool.stop(););
    mgb_log_debug("apply sublinear memory action: %zu opr groups to be inserted",
            action.size());
    apply_action(action, oprseq);
}

SeqModifierForSublinearMemory::SeqModifyAction
SeqModifierForSublinearMemory::search_action(
        const CompNode::UnorderedMap<OprNodeArray>* cn2oprseq) {
    m_thread2planner.clear();

    size_t planner_concur;
    if (auto env = MGB_GETENV("MGB_SUBLINEAR_MEMORY_WORKERS")) {
        auto set = static_cast<size_t>(std::stoi(env));
        mgb_assert(set && set <= static_cast<size_t>(sys::get_cpu_count()) * 4,
                   "invalid planner concurrency: %zu", set);
        planner_concur = set;
    } else {
        planner_concur = m_config->num_worker;
    }

    mgb_log_debug("use %zu threads to search for sublinear memory plan; "
            "this can be changed via MGB_SUBLINEAR_MEMORY_WORKERS env var",
            planner_concur);
    for (auto&& i : m_planner_thread_pool.start(planner_concur))
        m_thread2planner[i].reset(new ModifyActionPlanner{this});

    std::vector<std::unique_ptr<ActionSearcherSingleCN>> searchers;
    searchers.reserve(cn2oprseq->size());

    using WorkerPool = FutureThreadPool<const SeqModifyAction&>;
    WorkerPool workers;
    workers.start(cn2oprseq->size());

    m_prev_min_bottleneck.clear();
    for (auto&& i : *cn2oprseq) {
        m_prev_min_bottleneck[i.first] = 0;
    }

    std::vector<WorkerPool::Future> futures;
    for (auto&& i : *cn2oprseq) {
        searchers.emplace_back(std::make_unique<ActionSearcherSingleCN>(this));
        futures.emplace_back(workers.launch(&ActionSearcherSingleCN::search,
                                            searchers.back().get(), i.first,
                                            &i.second));
    }

    SeqModifyAction action;
    for (auto&& i : futures) {
        auto&& cur = i.get();
        action.insert(cur.begin(), cur.end());
    }
    m_thread2planner.clear();
    return action;
}

void SeqModifierForSublinearMemory::apply_action(SeqModifyAction& action,
                                                 const OprNodeArray& oprseq) {
    auto cur_priority = std::numeric_limits<decltype(
            OperatorNodeBase::NodeProp::Attribute::priority)>::min();

    ThinHashSet<OperatorNodeBase*> modified_opr;

    // each operator should be set no more than once
    auto set_priority = [&](OperatorNodeBase* opr) {
        mgb_assert(modified_opr.insert(opr).second);
        m_mem_opt.set_priority(opr, cur_priority++);
    };

    auto on_opr_visited = [&](OperatorNodeBase* opr) {
        if (replace_vars(opr->input())) {
            auto&& repl_info = m_opr2replace_info[opr];
            mgb_assert(!repl_info.recomp,
                       "input of operator %s{%s} already replaced",
                       opr->cname(), opr->dyn_typeinfo()->name);
            opr = copy_opr_from_new_inputs(opr, true);
            repl_info.recomp = opr;
        }
        set_priority(opr);
    };

    // use a DepOprIter rather than directly iterate on oprseq because shape-dep
    // oprs would be omitted in the opr_seq generated by topo sorter; but they
    // should be replaced too
    DepOprIter dep_iter{on_opr_visited};

    // setup m_var_map and priority
    for (auto opr : oprseq) {
        auto iter = action.find(opr);

        if (iter != action.end()) {
            // insert duplicated oprs
            for (auto i : iter->second) {
                replace_vars(i->input());
                auto&& repl_info = m_opr2replace_info[i];
                mgb_assert(!repl_info.dup, "operator %s{%s} already duplicated",
                           i->cname(), i->dyn_typeinfo()->name);
                auto opr_new = copy_opr_from_new_inputs(i, false);
                repl_info.dup = opr_new;
                set_priority(opr_new);
            }
            action.erase(iter);
        }

        dep_iter.add(opr);
    }
    mgb_assert(action.empty());
}

bool SeqModifierForSublinearMemory::replace_vars(const VarNodeArray& inputs) {
    m_new_inputs.assign(inputs.begin(), inputs.end());
    bool changed = false;
    for (auto&& i : m_new_inputs) {
        auto iter = m_var_map.find(i);
        if (iter != m_var_map.end()) {
            i = iter->second;
            changed = true;
        }
    }
    return changed;
}

OperatorNodeBase* SeqModifierForSublinearMemory::copy_opr_from_new_inputs(
        OperatorNodeBase* opr, bool recomp) {
    auto config = opr->config();
    // update operator instance id to bybass the shallow copy's cache if
    // it's a dup-opr-copying due to discarding.
    // Don't update instance id by `this` pointer if it's a recomp-opr-copying
    // because:
    // 0) recomp-opr would be copied iff its input vars is changed
    // 1) some pair of recomp-opr and dup-opr have the same inputs, params
    //    and config, we use instance id to differentiate them.
    config.name(opr->name() + (recomp ? ":recomp" : ":dup"));
    if (!recomp) {
       config.update_instance_id(this);
    }

    // Note: if all outputs of op were placed on the same comp_node, since its
    // stream maybe changed during seq_comp_node_opt, output's comp_node has
    // higher priority than opr->config()
    auto out_cn = opr->output(0)->comp_node();
    for (auto i : opr->output()) {
        auto cn = i->comp_node();
        if (out_cn != cn) {
            out_cn = {};
            break;
        }
    }
    if (out_cn.valid())
        config.comp_node(out_cn);

    auto opr_new = serialization::copy_opr_shallow(*opr, m_new_inputs, config);
    mgb_assert(opr_new != opr);

    auto&& out0 = opr->output();
    auto&& out1 = opr_new->output();
    mgb_assert(out0.size() == out1.size());
    bool stream_changed = false;
    for (size_t i = 0; i < out0.size(); ++i) {
        auto &&cn0 = out0[i]->comp_node(),
             &&cn1 = out1[i]->comp_node();
        if (cn0 != cn1) {
            mgb_assert(recomp);
            mgb_assert(cn0.locator().type == cn1.locator().type &&
                       cn0.locator().device == cn1.locator().device);
            out1[i]->comp_node(cn0);
            stream_changed = true;
        }
        m_var_map[out0[i]] = out1[i];
    }
    if (stream_changed) {
        opr_new->on_output_comp_node_stream_changed();
    }
    return opr_new;
}

void SeqModifierForSublinearMemory::modify_endpoint_vars(
        VarNodeArray& endpoints) {
    auto comp_seq = MemoryOptimizerHelper::CompSeq(m_owner_graph, endpoints);
    reset_opr_seq(*comp_seq.m_seq);
    for (auto&& i : endpoints) {
        auto iter = m_var_map.find(i);
        if (iter != m_var_map.end()) {
            i = iter->second;
        }
    }
}

void SeqModifierForSublinearMemory::sanity_check(const OprNodeArray& opr_seq) {
    OperatorNodeBase* first_bad_opr = nullptr;
    for (auto i : opr_seq) {
        auto iter = m_opr2replace_info.find(i);
        if (iter != m_opr2replace_info.end() && iter->second.recomp &&
            !first_bad_opr) {
            first_bad_opr = i;
            break;
        }
    }
    if (first_bad_opr) {
        VarNodeSet bad_vars[2];
        std::string err_msg;
        size_t nr_bad_opr = 0;
        auto add_bad_opr = [&](int type, OperatorNodeBase* opr) {
            err_msg += ssprintf(" %d#%zu: %s{%s} id=%zu\n", type, nr_bad_opr++,
                                opr->cname(), opr->dyn_typeinfo()->name,
                                opr->id());
            for (auto i : opr->input()) {
                err_msg += ssprintf("    inp var%zu %s\n", i->id(), i->cname());
            }
            for (auto i : opr->output()) {
                bad_vars[type].insert(i);
                err_msg += ssprintf("    out var%zu %s\n", i->id(), i->cname());
            }
        };
        OperatorNodeBase* bad_opr[] = {
                first_bad_opr, m_opr2replace_info.at(first_bad_opr).recomp};

        for (auto i : opr_seq) {
            bool bad[2] = {i == bad_opr[0], i == bad_opr[1]};
            for (auto j : i->input()) {
                if (bad_vars[0].count(j)) {
                    bad[0] = true;
                }
                if (bad_vars[1].count(j)) {
                    bad[1] = true;
                }
            }
            if (bad[0]) {
                add_bad_opr(0, i);
            }
            if (bad[1]) {
                add_bad_opr(1, i);
            }
        }
        mgb_throw(InternalError,
                  "sublinear memory: opreator input already replaced, but the "
                  "orignal operator is still used. operator chain: {\n%s}",
                  err_msg.c_str());
    }
}

const CompNode::UnorderedMap<size_t>&
SeqModifierForSublinearMemory::prev_min_bottleneck() {
    return m_prev_min_bottleneck;
}

SeqModifierForSublinearMemory::SeqModifierForSublinearMemory(
        ComputingGraphImpl* owner, Config* config_p)
    : m_config(config_p), m_mem_opt(owner), m_owner_graph(owner) {}

#endif  // !MGB_ENABLE_SUBLINEAR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
