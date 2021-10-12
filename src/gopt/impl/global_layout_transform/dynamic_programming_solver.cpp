/**
 * \file src/gopt/impl/dynamic_programming_solver.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include <queue>
#include "./utils.h"
#include "megbrain/gopt/layout_transform_context.h"
#include "megbrain/gopt/profiler.h"
#include "megbrain/gopt/solver.h"

using namespace mgb;
using namespace gopt;
using namespace cg;

/* ================= DynamicProgrammingSolver::Impl ==================*/
class DynamicProgrammingSolver::Impl {
public:
    Impl(size_t max_states) : m_max_states{max_states} {}
    ~Impl() = default;
    Solution solve(const ProfilerBase* profiler, const Problem& problem);

private:
    using TensorFormatsBitSet = uint32_t;
    using State = SmallVector<TensorFormatsBitSet>;
    /// 1bit represents one kind of tensor formats
    static constexpr uint32_t BITS_PER_BYTE = 8;
    static constexpr uint32_t MAX_TENSOR_FORMATS =
            sizeof(TensorFormatsBitSet) * BITS_PER_BYTE;
    TensorFormatsBitSet add(TensorFormatsBitSet& set, TensorFormats fmt) {
        mgb_assert(static_cast<uint32_t>(fmt) < MAX_TENSOR_FORMATS);
        set |= (1 << static_cast<uint32_t>(fmt));
        return set;
    }
    bool valid(const TensorFormatsBitSet& set, TensorFormats fmt) {
        mgb_assert(static_cast<uint32_t>(fmt) < MAX_TENSOR_FORMATS);
        bool val = set & (1 << static_cast<uint32_t>(fmt));
        return val;
    }
    struct Value {
        OperatorNodeBase* opr;
        const State* prev;
        OprFormat opr_fmt;
        float time;
        ///! index in the topo order of the correspoding operator
        size_t opr_idx;
    };

    struct StateHash {
        size_t operator()(const State& key) const {
            size_t h = 0;
            for (auto&& v : key) {
                h = mgb::hash_pair_combine(h, std::hash<TensorFormatsBitSet>{}(v));
            }
            return h;
        }
    };
    struct StateEqual {
        size_t operator()(const State& lhs, const State& rhs) const {
            if (lhs.size() != rhs.size())
                return false;
            for (size_t i = 0; i < lhs.size(); ++i) {
                if (lhs[i] != rhs[i])
                    return false;
            }
            return true;
        }
    };
    using StateTable = std::unordered_map<State, Value, StateHash, StateEqual>;
    struct Cut {
        StateTable states;
    };
    using ProfilingResult = ProfilerBase::ProfilingResult;
    using OprConfigTrait = LayoutTransformContext::OprConfigTrait;
    struct Context {
        const std::vector<OperatorNodeBase*>& topo;
        const ProfilingResult& rst;
        const OprConfigTrait& opr_configs;
        const SmallVector<TensorFormats>& available_tensor_formats;
    };
    /*!
     * \brief get the tensor formats configuration for the operator with
     * particular op format \param[out] var2fmts hashmap that maps varnode to
     * actual tensor formats of the op format configuration \param[in] opr given
     * operator \param[in] opr_fmt given op format, an enum type argument which
     * indicates the op format configuration. \param[in] ctx context
     */
    TensorFormats get_io_formats(
            ThinHashMap<VarNode*, TensorFormats>& var2fmts, const OperatorNodeBase* opr,
            OprFormat opr_fmt, const Context& ctx);
    /*!
     * \brief compute the distace of two states of the given varnode
     * \param[in] from the source state
     * \param[in] to the target state
     * \param[in] var given varnode
     * \param[in] ctx context
     */
    float distance(
            const TensorFormatsBitSet& from, const TensorFormatsBitSet& to,
            VarNode* var, const Context& ctx);
    /*!
     * \brief compute the distace of two states of the given cut edges
     * \param[in] from the source state
     * \param[in] to the target state
     * \param[in] edge a VarNodeArry, the given cut edges
     * \param[in] ctx context
     */
    float state_distance(
            const State& from, const State& to, const VarNodeArray& edge,
            const Context& ctx);
    /*!
     * \brief analyze the edges of each cut
     * \param[out] edges the return edges of the cuts
     * \param[out] edge2idx hashmaps, that maps edge(varnode) to its index
     * \param[in] ctx contex
     */
    void analyze_edges(
            SmallVector<VarNodeArray>& edges,
            SmallVector<std::unordered_map<VarNode*, int>>& edge2idx,
            const Context& ctx);
    /*!
     * \brief prune states using the distance of states
     */
    void prune(StateTable& states, const VarNodeArray& edge, const Context& ctx);
    /*!
     * \brief force prune states, reserve the smallest MAX_STATES states
     */
    void force_prune(StateTable& states);

private:
    size_t m_max_states;
};

TensorFormats DynamicProgrammingSolver::Impl::get_io_formats(
        ThinHashMap<VarNode*, TensorFormats>& var2fmts, const OperatorNodeBase* opr,
        OprFormat opr_fmt, const Context& ctx) {
    auto&& rst = ctx.rst;
    auto&& opr_configs = ctx.opr_configs;

    auto iter = opr_configs.find(opr->dyn_typeinfo());
    Maybe<OprTensorFormatsConfiguration> fmtcfg = None;
    if (iter != opr_configs.end()) {
        fmtcfg = (*iter->second.at(opr_fmt))(opr);
    }
    TensorFormats out_fmt;
    if (fmtcfg.valid())
        out_fmt = fmtcfg.val().output_tensor_formats[0];
    else
        out_fmt = opr_format_to_tensor_formats(opr_fmt);
    for (size_t i = 0; i < opr->input().size(); ++i) {
        auto&& var = opr->input(i);
        auto iter = rst.var_record.find(var);
        if (iter != rst.var_record.end()) {
            if (fmtcfg.valid())
                var2fmts[var] = fmtcfg.val().input_tensor_formats[i];
            else
                var2fmts[var] = opr_format_to_tensor_formats(opr_fmt);
        }
    }
    return out_fmt;
}

float DynamicProgrammingSolver::Impl::distance(
        const TensorFormatsBitSet& from, const TensorFormatsBitSet& to, VarNode* var,
        const Context& ctx) {
    auto&& costs = ctx.rst.var_record.at(var).costs;
    auto&& available_tensor_formats = ctx.available_tensor_formats;

    float dist = 0.f;
    if ((from & to) == to)
        return dist;
    auto to_set = ((from | to) ^ from);
    for (auto o : available_tensor_formats) {
        if (valid(to_set, o)) {
            float o_cost = std::numeric_limits<float>::max();
            for (auto i : available_tensor_formats) {
                if (valid(from, i)) {
                    float cost = costs.at({i, o});
                    o_cost = std::min(o_cost, cost);
                }
            }
            dist += o_cost;
        }
    }
    return dist;
}

float DynamicProgrammingSolver::Impl::state_distance(
        const State& from, const State& to, const VarNodeArray& edge,
        const Context& ctx) {
    float dist = 0.f;
    mgb_assert(from.size() == to.size() && from.size() == edge.size());
    for (size_t i = 0; i < edge.size(); ++i) {
        dist += distance(from[i], to[i], edge[i], ctx);
    }
    return dist;
}

void DynamicProgrammingSolver::Impl::analyze_edges(
        SmallVector<VarNodeArray>& edges,
        SmallVector<std::unordered_map<VarNode*, int>>& edge2idx, const Context& ctx) {
    auto&& topo = ctx.topo;
    auto&& rst = ctx.rst;

    size_t nr_oprs = topo.size();

    edges.resize(nr_oprs);
    edge2idx.resize(nr_oprs);

    ThinHashSet<VarNode*> cur_edge;
    size_t cur = nr_oprs - 1;
    int idx = 0;
    for (auto&& ov : topo[cur]->usable_output()) {
        edges[cur].push_back(ov);
        edge2idx[cur].emplace(ov, idx++);
    }
    cur--;
    for (const auto& opr : reverse_adaptor(topo)) {
        for (const auto& i : opr->input()) {
            if (rst.var_record.count(i) > 0) {
                cur_edge.insert(i);
            }
        }
        for (auto&& ov : opr->usable_output()) {
            cur_edge.erase(ov);
        }
        edges[cur].insert(edges[cur].begin(), cur_edge.begin(), cur_edge.end());
        int i = 0;
        for (auto&& e : edges[cur]) {
            edge2idx[cur][e] = i++;
        }
        if (cur == 0)
            break;
        cur--;
    }
}

void DynamicProgrammingSolver::Impl::prune(
        StateTable& states, const VarNodeArray& edge, const Context& ctx) {
    struct Item {
        decltype(states.begin()) iter;
    };
    std::list<Item> list;
    for (auto it = states.begin(); it != states.end(); ++it) {
        list.emplace_back(Item{it});
    }
    SmallVector<State> removed_states;
    for (auto i = list.begin(); i != list.end();) {
        bool advanced_i = false;
        for (auto j = std::next(i, 1); j != list.end();) {
            if (i->iter->second.time > j->iter->second.time &&
                state_distance(j->iter->first, i->iter->first, edge, ctx) <
                        i->iter->second.time - j->iter->second.time) {
                removed_states.push_back(i->iter->first);
                i = list.erase(i);
                advanced_i = true;
                break;
            } else if (
                    i->iter->second.time < j->iter->second.time &&
                    state_distance(i->iter->first, j->iter->first, edge, ctx) <
                            j->iter->second.time - i->iter->second.time) {
                removed_states.push_back(j->iter->first);
                j = list.erase(j);
            } else {
                j = std::next(j, 1);
            }
        }
        if (!advanced_i)
            i = std::next(i, 1);
    }
    for (auto&& state : removed_states)
        states.erase(state);
}

void DynamicProgrammingSolver::Impl::force_prune(StateTable& states) {
    if (states.size() < m_max_states)
        return;
    struct Item {
        decltype(states.begin()) iter;
    };
    auto cmp = [](Item lhs, Item rhs) {
        return lhs.iter->second.time < rhs.iter->second.time;
    };
    std::priority_queue<Item, std::vector<Item>, decltype(cmp)> pq(cmp);
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (pq.size() < m_max_states)
            pq.push(Item{it});
        else {
            auto i = pq.top();
            if (it->second.time < i.iter->second.time) {
                pq.pop();
                pq.push(Item{it});
            }
        }
    }
    StateTable active_state;
    while (!pq.empty()) {
        auto i = pq.top();
        active_state.insert(*i.iter);
        pq.pop();
    }
    states.swap(active_state);
}

DynamicProgrammingSolver::Solution DynamicProgrammingSolver::Impl::solve(
        const ProfilerBase* profiler, const Problem& problem) {
    const auto rst = profiler->profile(problem);
    const auto& partition = problem.graph_partition();
    const auto& opr_configs = problem.opr_configs();
    const auto& base_fmt = problem.base_format();
    const auto& available_tensor_formats = problem.available_tensor_formats();
    const auto& topo = partition.all_oprs();
    Context ctx{topo, rst, opr_configs, available_tensor_formats};

    SmallVector<VarNodeArray> edges;
    SmallVector<std::unordered_map<VarNode*, int>> edge2idx;
    /// analyze edges of each cuts
    analyze_edges(edges, edge2idx, ctx);

    SmallVector<Cut> cuts;
    size_t cur = 0;

    /// initialize states
    auto init = [&, this](OperatorNodeBase* opr) {
        auto it = rst.opr_record.find(opr);
        if (it == rst.opr_record.end())
            return;
        ThinHashSet<VarNode*> ovar_set;
        for (auto&& ov : opr->usable_output()) {
            ovar_set.insert(ov);
        }
        const auto& records = it->second.costs;
        cuts.emplace_back(Cut{});
        auto& states = cuts.back().states;
        for (const auto& record : records) {
            auto opr_fmt = record.first;
            float opr_time = record.second;
            ThinHashMap<VarNode*, TensorFormats> ivar2fmts;
            auto out_fmt = get_io_formats(ivar2fmts, opr, opr_fmt, ctx);
            const auto& edge = edges[cur];
            State state(edge.size(), 0);
            Value value{opr, nullptr, opr_fmt, 0.f, cur};
            float ovar_time = 0.f;
            for (size_t i = 0; i < edge.size(); ++i) {
                auto&& var = edge[i];
                auto&& costs = rst.var_record.at(var).costs;
                if (ovar_set.count(var) > 0) {
                    add(state[i], out_fmt);
                    if (partition.output().count(var) > 0 && out_fmt != base_fmt) {
                        ovar_time += costs.at({out_fmt, base_fmt});
                        add(state[i], base_fmt);
                    }
                } else {
                    add(state[i], base_fmt);
                }
            }
            float ivar_time = 0.f;
            for (const auto& kv : ivar2fmts) {
                auto&& v = kv.first;
                auto&& costs = rst.var_record.at(v).costs;
                auto to = kv.second;
                float min_time = std::numeric_limits<float>::max();
                if (base_fmt == to) {
                    min_time = 0.f;
                } else {
                    min_time = costs.at({base_fmt, to});
                    if (edge2idx[cur].count(v) > 0) {
                        add(state[edge2idx[cur][v]], to);
                    }
                }
                ivar_time += min_time;
            }
            value.time = opr_time + ivar_time + ovar_time;
            states[state] = value;
        }
    };

    /// update the states
    auto body = [&, this](OperatorNodeBase* opr) {
        auto it = rst.opr_record.find(opr);
        if (it == rst.opr_record.end())
            return;
        ThinHashSet<VarNode*> ovar_set;
        for (auto&& ov : opr->usable_output()) {
            ovar_set.insert(ov);
        }
        const auto& records = it->second.costs;
        StateTable states;
        for (const auto& record : records) {
            auto opr_fmt = record.first;
            float opr_time = record.second;
            ThinHashMap<VarNode*, TensorFormats> ivar2fmts;
            auto out_fmt = get_io_formats(ivar2fmts, opr, opr_fmt, ctx);
            for (const auto& kv : cuts.back().states) {
                auto&& prev_state = kv.first;
                float prev_time = kv.second.time;
                const auto& edge = edges[cur];
                State state(edge.size(), 0);
                Value value{opr, &prev_state, opr_fmt, 0.f, cur};
                float ovar_time = 0.f;
                for (size_t i = 0; i < edge.size(); ++i) {
                    auto&& var = edge[i];
                    auto&& costs = rst.var_record.at(var).costs;
                    auto iter = edge2idx[cur - 1].find(var);
                    if (iter != edge2idx[cur - 1].end()) {
                        state[i] = prev_state[iter->second];
                    } else {
                        mgb_assert(ovar_set.count(var) > 0);
                        add(state[i], out_fmt);
                        if (partition.output().count(var) > 0 && out_fmt != base_fmt) {
                            ovar_time += costs.at({out_fmt, base_fmt});

                            add(state[i], base_fmt);
                        }
                    }
                }
                float ivar_time = 0.f;
                for (const auto& kv : ivar2fmts) {
                    auto&& v = kv.first;
                    auto&& costs = rst.var_record.at(v).costs;
                    auto to = kv.second;
                    auto it1 = edge2idx[cur - 1].find(v);
                    float min_time = std::numeric_limits<float>::max();
                    if (valid(prev_state[it1->second], to)) {
                        min_time = 0.f;
                    } else {
                        for (auto&& from : available_tensor_formats) {
                            if (valid(prev_state[it1->second], from)) {
                                float cost = costs.at({from, to});
                                min_time = std::min(min_time, cost);
                            }
                        }
                    }
                    auto it2 = edge2idx[cur].find(v);
                    if (it2 != edge2idx[cur].end()) {
                        add(state[it2->second], to);
                    }
                    ivar_time += min_time;
                }
                value.time = prev_time + opr_time + ivar_time + ovar_time;
                auto iter = states.find(state);
                if (iter == states.end()) {
                    states[state] = value;
                } else {
                    float time = iter->second.time;
                    if (value.time < time) {
                        iter->second = value;
                    }
                }
            }
        }
        cuts.emplace_back(Cut{});
        cuts.back().states.swap(states);
    };

    /// forward pass to generate all states
    for (auto&& opr : topo) {
        if (cuts.empty()) {
            init(opr);
        } else {
            body(opr);
        }
        if (!cuts.empty()) {
            auto& states = cuts.back().states;
            prune(states, edges[cur], ctx);
            force_prune(states);
        }
        cur++;
    }

    Solution solution;

    /// backward pass to generate the solution
    float min_time = std::numeric_limits<float>::max();
    OperatorNodeBase* cur_opr = nullptr;
    OprFormat min_fmt = OprFormat::NCHW;
    const State* pstate = nullptr;
    for (auto&& kv : cuts.back().states) {
        auto&& v = kv.second;
        if (v.time < min_time) {
            cur_opr = v.opr;
            pstate = v.prev;
            min_time = v.time;
            min_fmt = v.opr_fmt;
            ///! just to check the tensor formats of the output varnode
            auto&& k = kv.first;
            size_t opr_idx = v.opr_idx;
            for (size_t i = 0; i < k.size(); ++i) {
                auto&& fmt_set = k[i];
                auto&& var = edges[opr_idx][i];
                if (partition.output().count(var)) {
                    mgb_assert(valid(fmt_set, base_fmt));
                }
            }
        }
    }
    mgb_assert(cur_opr != nullptr);
    mgb_log_debug(
            "opr:%s;format:%s;time:%f", cur_opr->cname(), opr_format_to_string(min_fmt),
            min_time);

    solution.insert({cur_opr, min_fmt});
    cur = cuts.size() - 2;
    while (pstate) {
        auto val = cuts[cur].states[*pstate];
        ///! just to check the tensor formats of the output varnode
        size_t opr_idx = val.opr_idx;
        for (size_t i = 0; i < pstate->size(); ++i) {
            auto&& fmt_set = pstate->operator[](i);
            auto&& var = edges[opr_idx][i];
            if (partition.output().count(var)) {
                mgb_assert(valid(fmt_set, base_fmt));
            }
        }
        mgb_log_debug(
                "opr:%s;format:%s;time:%f", val.opr->cname(),
                opr_format_to_string(val.opr_fmt), val.time);
        solution.insert({val.opr, val.opr_fmt});
        pstate = val.prev;
        cur--;
    }
    return solution;
}

/* =================== DynamicProgrammingSolver ======================*/
DynamicProgrammingSolver::Solution DynamicProgrammingSolver::do_solve(
        const Problem& problem) const {
    constexpr size_t MAX_STATES = 1024;
    Impl impl(MAX_STATES);
    return impl.solve(m_profiler.get(), problem);
}

bool DynamicProgrammingSolver::can_solve(const Problem& problem) const {
    auto&& available_tensor_formats = problem.available_tensor_formats();
    for (auto&& tensor_format : available_tensor_formats) {
        if (static_cast<uint32_t>(tensor_format) >= 32)
            return false;
    }
    return true;
}

// vim: syntax=cpp.doxygen
