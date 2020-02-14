/**
 * \file src/core/impl/graph/swap/memory_swap.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./memory_swap.h"
#include "./swap_opr.h"

#include "../cg_impl.h"

#include "megbrain/gopt/framework.h"
#include "megbrain/opr/io.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/serialization/opr_shallow_copy.h"

#include <queue>

#if MGB_ENABLE_MEMORY_SWAP
using namespace mgb;
using namespace swap;
using namespace swap::opr;

MGB_TYPEINFO_OBJ_IMPL(SwapCopyThreadPool);

using SharedDeviceTensor = mgb::opr::SharedDeviceTensor;

/* ================ SegmentTree ================ */
class SegmentTree {
    size_t m_len = 0;

public:
    std::vector<NodeInfo> m_tree;
    std::vector<long long> m_lzt;
    ThinHashMap<size_t, ThinHashSet<int>> m_seg_info;
    PLS query_max() { return PLS(m_tree[1].max, m_tree[1].idx); }

    SegmentTree(size_t len) {
        auto segT_size = len * 8 + 1;
        m_len = len;
        m_tree = std::vector<NodeInfo>(segT_size);
        m_lzt = std::vector<long long>(segT_size);
        init(1, 1, len);
    }

    void init(int k, int l, int r) {
        m_lzt[k] = 0;
        if (l == r) {
            m_tree[k].max = 0;
            m_tree[k].idx = l;
            return;
        }
        int mid = (l + r) >> 1;
        init(k << 1, l, mid);
        init(k << 1 | 1, mid + 1, r);
        m_tree[k] = m_tree[k << 1];
    }

    void insert(int k, int l, int r, int ll, int rr, PLI info) {
        if (l == ll && rr == r) {
            m_tree[k].max += info.first;
            m_lzt[k] += info.first;
            if (info.second > -1) {
                if (m_seg_info.find(k) == m_seg_info.end()) {
                    ThinHashSet<int> tmp;
                    m_seg_info[k] = tmp;
                }
                m_seg_info[k].insert(info.second);
            }
            return;
        }
        int mid = (l + r) >> 1;
        if (rr <= mid)
            insert(k << 1, l, mid, ll, rr, info);
        else if (ll > mid)
            insert(k << 1 | 1, mid + 1, r, ll, rr, info);
        else {
            insert(k << 1, l, mid, ll, mid, info);
            insert(k << 1 | 1, mid + 1, r, mid + 1, rr, info);
        }
        if (m_tree[k << 1] <= m_tree[k << 1 | 1]) {
            m_tree[k].max = m_tree[k << 1 | 1].max + m_lzt[k];
            m_tree[k].idx = m_tree[k << 1 | 1].idx;
        } else {
            m_tree[k].max = m_tree[k << 1].max + m_lzt[k];
            m_tree[k].idx = m_tree[k << 1].idx;
        }
    }

    void remove(int k, int l, int r, int ll, int rr, PLI info) {
        if (l == ll && rr == r) {
            m_tree[k].max -= info.first;
            m_lzt[k] -= info.first;
            if (m_seg_info.find(k) != m_seg_info.end() && info.second > -1) {
                mgb_assert(m_seg_info[k].find(info.second) !=
                           m_seg_info[k].end());
                m_seg_info[k].erase(info.second);
            }
            return;
        }
        int mid = (l + r) >> 1;
        if (rr <= mid)
            remove(k << 1, l, mid, ll, rr, info);
        else if (ll > mid)
            remove(k << 1 | 1, mid + 1, r, ll, rr, info);
        else {
            remove(k << 1, l, mid, ll, mid, info);
            remove(k << 1 | 1, mid + 1, r, mid + 1, rr, info);
        }
        if (m_tree[k << 1] <= m_tree[k << 1 | 1]) {
            m_tree[k].max = m_tree[k << 1 | 1].max + m_lzt[k];
            m_tree[k].idx = m_tree[k << 1 | 1].idx;
        } else {
            m_tree[k].max = m_tree[k << 1].max + m_lzt[k];
            m_tree[k].idx = m_tree[k << 1].idx;
        }
    }

    void query(int k, int l, int r, int x, ThinHashSet<int>& cover_seg) {
        if (m_seg_info.find(k) != m_seg_info.end()) {
            for (auto x : m_seg_info[k]) {
                cover_seg.insert(x);
            }
        }
        if (l == r) {
            mgb_assert(x == l,
                       "bug occurs in memory_swap's Segment Tree in line %s:%d "
                       "%s\n",
                       __FILE__, __LINE__, __FUNCTION__);
            return;
        }
        int mid = (l + r) >> 1;
        if (x <= mid)
            query(k << 1, l, mid, x, cover_seg);
        else
            query(k << 1 | 1, mid + 1, r, x, cover_seg);
    }
};

/* ================ MemorySwap ================ */

MemorySwap::MemorySwap(ComputingGraph* graph) : m_owner_graph(graph){};

MemorySwap::~MemorySwap() noexcept = default;

void MemorySwap::determine_swap_edge(PIPSet& heap, size_t loss_idx,
                                     const cg::OprNodeArray& opr_seq,
                                     std::vector<std::vector<size_t>>& g,
                                     std::vector<std::vector<size_t>>& tg) {
    auto&& infer_mgr = m_owner_graph->static_infer_manager();
    static_cast<void>(infer_mgr);

    size_t fin = opr_seq.size() + 10;
    auto segT = new SegmentTree(fin);

    for (auto x : m_var_map) {
        auto v = x.second;
        mgb_assert(tg[v->id()].size() == 1);
        std::vector<size_t> a;
        std::vector<int> m_consume_opr_set;
        std::vector<PPI> segment_set;
        auto sz = m_var_map[x.first]->dtype().size(
                infer_mgr.infer_shape(m_var_map[x.first]).total_nr_elems());
        ++m_segment_race_id;
        size_t last = x.first;
        std::vector<size_t> b;
        for (auto y : g[x.first]) {
            if (m_opr_seq_dist.find(y) == m_opr_seq_dist.end())
                continue;
            auto s = m_opr_map[y]->node_prop().dep_map().find(
                    m_var_map[x.first]);
            if (s == m_opr_map[y]->node_prop().dep_map().end())
                continue;
            if (s->second != cg::OperatorNodeBase::NodeProp::DepType::DEV_VALUE)
                continue;
            b.push_back(y);
        }
        sort(b.begin(), b.end(), [&](const int& lhs, const int& rhs) {
            return m_opr_seq_dist[lhs] < m_opr_seq_dist[rhs];
        });
        for (auto y : b) {
            if (m_opr_seq_dist.find(y) == m_opr_seq_dist.end())
                continue;
            bool flag = true;
            a.push_back(m_opr_seq_dist[y]);
            m_consume_opr_set.push_back(y);
            if (m_opr_seq_dist[last] + 1 <= m_opr_seq_dist[y] - 1) {
                if (flag) {
                    m_all_valid_segments.push_back(
                            PPI(PII(m_opr_seq_dist[last] + 1,
                                    m_opr_seq_dist[y] - 1),
                                m_segment_race_id));
                }
                int seg_id = flag ? (int)m_all_valid_segments.size() - 1 : -1;
                segment_set.push_back(PPI(
                        PII(m_opr_seq_dist[last] + 1, m_opr_seq_dist[y] - 1),
                        seg_id));
                segT->insert(1, 1, fin, m_opr_seq_dist[last] + 1,
                             m_opr_seq_dist[y] - 1, PII(sz, seg_id));
            }
            last = y;
        }
        if (a.empty())
            continue;
        sort(a.begin(), a.end());
        auto s = new SegmentRace(sz, m_segment_race_id, x.first, segment_set,
                                 m_consume_opr_set);
        m_segmentRaceList.push_back(s);

        for (auto id : m_consume_opr_set) {
            segT->insert(1, 1, fin, m_opr_seq_dist[id], m_opr_seq_dist[id],
                         PII(sz, -1));
        }
        for (auto x : segment_set)
            if (x.second > -1)
                m_segmentToRace[x.second] = s;
    }
    int trash_counter = 0;
    long long involved = 0;
    long long dec_tot = 0;
    ThinHashMap<int, bool> race_has_been_swapped;
    long long i = 0;

    while ((i++ < m_n_tensors) || ((m_n_tensors < 0))) {
        PLS s = segT->query_max();
        int place = s.second;
        ThinHashSet<int> covering_idx;
        segT->query(1, 1, fin, place, covering_idx);
        std::vector<int> tmp_vec;
        std::vector<int> tmp_vec_weak;
        if (covering_idx.empty())
            break;
        for (auto x : covering_idx) {
            auto u = m_segmentToRace[x]->m_st;
            auto v = opr_seq[m_all_valid_segments[x].first.second + 1]->id();
            if (m_var_map.find(u) == m_var_map.end())
                continue;
            if (m_opr_map.find(v) == m_opr_map.end())
                continue;
            if (m_opr_seq_dist.find(u) == m_opr_seq_dist.end() ||
                m_opr_seq_dist.find(v) == m_opr_seq_dist.end())
                continue;
            if (m_var_map[u]->owner_opr()->same_type<SharedDeviceTensor>())
                continue;
            auto t = m_opr_map[v]->node_prop().dep_map().find(m_var_map[u]);
            if (t == m_opr_map[v]->node_prop().dep_map().end())
                continue;
            if (t->second != cg::OperatorNodeBase::NodeProp::DepType::DEV_VALUE)
                continue;
            if (m_opr_seq_dist[v] - m_opr_seq_dist[u] <= m_swap_in_prev)
                continue;

            tmp_vec_weak.push_back(x);
            /*!
             * if we cannot find segments that staisfys all conditions,
             * those who does meet all exact the conditions below
             * will be considered
             */
            if (m_opr_seq_dist[v] - m_opr_seq_dist[u] < m_lb_for_distance)
                continue;
            if (m_segmentToRace[x]->m_mem <= m_swap_out_var_size_lb)
                continue;
            /*
            // coarse method to filter edge from forward phase to
            // backward phase
            if (m_topo_layer[loss_idx]) {
                if (m_topo_layer[u] > m_topo_layer[loss_idx])
                    continue;
                if (m_topo_layer[v] < m_topo_layer[loss_idx])
                    continue;
            }
            */
            tmp_vec.push_back(x);
            tmp_vec_weak.pop_back();
        }
        if (tmp_vec.empty()) {
            if (tmp_vec_weak.empty()) {
                break;
            }
            for (auto s : tmp_vec_weak)
                tmp_vec.push_back(s);
        }

        ThinHashMap<int, std::pair<std::pair<double, double>, long long>>
                peak_decrease_res;

        /*!
         * Compute the memory usage reduction
         * after the target segment is removed
         */
        auto peak_decrease = [&](const int& lhs) {
            auto origin_peak = segT->query_max().first;
            segT->remove(1, 1, fin, m_all_valid_segments[lhs].first.first,
                         m_all_valid_segments[lhs].first.second,
                         PII(m_segmentToRace[lhs]->m_mem, lhs));
            if (m_swap_in_prev > 1) {
                segT->insert(1, 1, fin,
                             m_all_valid_segments[lhs].first.second -
                                     m_swap_in_prev + 1,
                             m_all_valid_segments[lhs].first.second - 1,
                             PII(m_segmentToRace[lhs]->m_mem, -1));
            }
            auto ret = origin_peak - segT->query_max().first;
            segT->insert(1, 1, fin, m_all_valid_segments[lhs].first.first,
                         m_all_valid_segments[lhs].first.second,
                         PII(m_segmentToRace[lhs]->m_mem, lhs));
            if (m_swap_in_prev > 1) {
                segT->remove(1, 1, fin,
                             m_all_valid_segments[lhs].first.second -
                                     m_swap_in_prev + 1,
                             m_all_valid_segments[lhs].first.second - 1,
                             PII(m_segmentToRace[lhs]->m_mem, -1));
            }
            return ret;
        };

        for (auto s : tmp_vec) {
            auto fst = peak_decrease(s);
            peak_decrease_res[s] = std::make_pair(
                    std::make_pair(
                            1.0 * fst *
                                    (m_all_valid_segments[s].first.second -
                                     m_all_valid_segments[s].first.first + 1) /
                                    m_segmentToRace[s]->m_mem,
                            1.0 * fst / m_segmentToRace[s]->m_mem),
                    -m_segmentToRace[s]->m_mem);
            if (peak_decrease_res[s].first.second < 0.5)
                peak_decrease_res[s].first.first = 0;
        }
        sort(tmp_vec.begin(), tmp_vec.end(),
             [&](const int& lhs, const int& rhs) {
                 return peak_decrease_res[lhs] > peak_decrease_res[rhs];
             });

        auto pkd = peak_decrease(tmp_vec[0]);
        segT->remove(1, 1, fin, m_all_valid_segments[tmp_vec[0]].first.first,
                     m_all_valid_segments[tmp_vec[0]].first.second,
                     PII(m_segmentToRace[tmp_vec[0]]->m_mem, tmp_vec[0]));
        if (m_swap_in_prev > 1) {
            segT->insert(1, 1, fin,
                         m_all_valid_segments[tmp_vec[0]].first.second -
                                 m_swap_in_prev + 1,
                         m_all_valid_segments[tmp_vec[0]].first.second - 1,
                         PII(m_segmentToRace[tmp_vec[0]]->m_mem, -1));
        }
        auto u = m_segmentToRace[tmp_vec[0]]->m_st;
        auto v = opr_seq[m_all_valid_segments[tmp_vec[0]].first.second + 1]
                         ->id();
        if (m_var_map.find(u) == m_var_map.end())
            continue;
        if (m_opr_map.find(v) == m_opr_map.end())
            continue;
        if (m_opr_seq_dist.find(u) == m_opr_seq_dist.end() ||
            m_opr_seq_dist.find(v) == m_opr_seq_dist.end())
            continue;
        if (m_var_map[u]->owner_opr()->same_type<SharedDeviceTensor>())
            continue;
        auto t = m_opr_map[v]->node_prop().dep_map().find(m_var_map[u]);
        if (t == m_opr_map[v]->node_prop().dep_map().end())
            continue;
        if (t->second != cg::OperatorNodeBase::NodeProp::DepType::DEV_VALUE)
            continue;
        /*!
         * Here is the 'true' lower bound for swap_out_var size, the class
         * member var is a weak one
         */
        if (m_segmentToRace[tmp_vec[0]]->m_mem <= 8)
            continue;
        /*
        if (m_topo_layer[loss_idx]) {
            if (m_topo_layer[u] > m_topo_layer[loss_idx])
                continue;
            if (m_topo_layer[v] < m_topo_layer[loss_idx])
                continue;
        }
        */

        heap.insert(PIP(++trash_counter, PSS(u, v)));

        int ratio = 1;
        if (race_has_been_swapped.find(m_segmentToRace[tmp_vec[0]]->m_id) ==
            race_has_been_swapped.end()) {
            ratio = 2;
            race_has_been_swapped[m_segmentToRace[tmp_vec[0]]->m_id] = true;
        }

        dec_tot += pkd;
        involved += m_segmentToRace[tmp_vec[0]]->m_mem * ratio;
        m_max_swap_out_var_size = std::max(m_max_swap_out_var_size,
                                           m_segmentToRace[tmp_vec[0]]->m_mem);
        m_swapped_pair.insert(PSS(u, v));

        if (involved / m_cpu_gpu_bandwidth > m_swap_time_limit)
            break;
    }

    if (involved > 0) {
        mgb_log_debug("Total Swap in/out computation size : %lld byte(s), static "
                "memory "
                "allocation reduction : %lld byte(s), ratio : %.4f\n",
                involved, dec_tot, 1.0 * dec_tot / involved);
    }
    /*!
     * Sum of swap in/out tensor size : involved
     * Reduction of the static memory usage : dec_tot
     * but this approximation may slightly be disturbed by the phase below
     */

    for (auto t : m_segmentRaceList) {
        bool flag = 0;
        for (size_t i = 0; i < t->m_consume_opr.size(); ++i) {
            bool now = m_swapped_pair.count(PSS(t->m_st, t->m_consume_opr[i]));
            if ((!now) && flag) {
                mgb_assert(m_max_swap_out_var_size >= t->m_mem);
                heap.insert(PIP(++trash_counter,
                                PSS(t->m_st, t->m_consume_opr[i])));
            }
            flag |= now;
        }
    }
}

void MemorySwap::modify_dest_var_inplace(VarNodeArray& vars) {
    const cg::OprNodeArray* opr_seqs = nullptr;
    auto tmp = (static_cast<cg::ComputingGraphImpl*>(m_owner_graph));
    cg::CompSeqExtraInfo extra_info;
    opr_seqs = tmp->topo_sorter().get_comp_seq(extra_info, vars);
    tmp->topo_sorter().restore_opr_prop();

    auto opr_seq = *opr_seqs;
    auto nr_gpu = CompNode::get_device_count(CompNode::DeviceType::CUDA);
    if (!nr_gpu) {
        mgb_log_debug("No device exists, stop memory swap phase");
        return;
    }

    /*
     * change params through env-vars
     */
    auto env_bucket_implement =
            MGB_GETENV("MGB_MEMORY_SWAP_PARAM_BUCKET_IMPLEMENT");
    if (env_bucket_implement) {
        int tmp;
        sscanf(env_bucket_implement, "%d", &tmp);
        mgb_assert(tmp == 0 || tmp == 1);
        m_bucket_implement = tmp & 1;
    }

    auto env_fuse_swap_in_bound =
            MGB_GETENV("MGB_MEMORY_SWAP_PARAM_FUSE_SWAP_IN_BOUND");
    if (env_fuse_swap_in_bound) {
        sscanf(env_fuse_swap_in_bound, "%zu", &m_fuse_swap_in_bound);
    } else {
        m_fuse_swap_in_bound =
                std::max(m_fuse_swap_in_bound, opr_seq.size() / 60);
    }

    auto env_n_tensors = MGB_GETENV("MGB_MEMORY_SWAP_PARAM_N_TENSORS");
    if (env_n_tensors) {
        sscanf(env_n_tensors, "%lld", &m_n_tensors);
    }

    auto env_swap_in_prev = MGB_GETENV("MGB_MEMORY_SWAP_PARAM_SWAP_IN_PREV");
    if (env_swap_in_prev) {
        sscanf(env_swap_in_prev, "%d", &m_swap_in_prev);
        mgb_assert(m_swap_in_prev > 0);
    }

    auto env_swap_time_limit =
            MGB_GETENV("MGB_MEMORY_SWAP_PARAM_SWAP_TIME_LIMIT");
    if (env_swap_time_limit) {
        sscanf(env_swap_time_limit, "%lf", &m_swap_time_limit);
        mgb_assert(m_swap_time_limit + 1e-12 > 0);
    }

    auto env_swap_out_var_size_lb =
            MGB_GETENV("MGB_MEMORY_SWAP_PARAM_SWAP_OUT_VAR_SIZE_LB");
    if (env_swap_out_var_size_lb) {
        sscanf(env_swap_out_var_size_lb, "%zu", &m_swap_out_var_size_lb);
    }

    auto env_lb_for_distance =
            MGB_GETENV("MGB_MEMORY_SWAP_PARAM_LB_FOR_DISTANCE");
    if (env_lb_for_distance) {
        sscanf(env_lb_for_distance, "%lld", &m_lb_for_distance);
        mgb_assert(m_lb_for_distance > 0);
    } else {
        m_lb_for_distance =
                std::min(m_lb_for_distance, (long long)opr_seq.size() / 20);
    }
    if (!m_bucket_implement)
        m_swap_in_prev = 1;

    std::queue<OperatorNodeBase*> rst;
    std::queue<VarNode*> lst;
    SymbolVarArray sva;
    size_t max_idx = 0;
    size_t loss_idx = 0;
    for (size_t i = 0; i < vars.size(); ++i) {
        if (std::string(vars[i]->name()).compare(0, 4, "loss") == 0) {
            loss_idx = vars[i]->id();
        }
        lst.push(vars[i]);
        sva.push_back(vars[i]);
        m_color[vars[i]->id()] = 2;
    }
    for (size_t i = 0; i < opr_seq.size(); ++i) {
        m_opr_seq_dist[opr_seq[i]->id()] = i;
        // reserve numeric_limits<>::min() for swap oprs
        if (opr_seq[i]->node_prop().attribute().priority <
            std::numeric_limits<int>::max())
            opr_seq[i]->node_prop().attribute().priority++;
    }

    while (!lst.empty() || !rst.empty()) {
        while (!lst.empty()) {
            auto x = lst.front();
            m_var_map[x->id()] = x;
            max_idx = std::max(max_idx, x->id());
            if (m_opr_seq_dist.find(x->owner_opr()->id()) !=
                m_opr_seq_dist.end())
                m_opr_seq_dist[x->id()] = m_opr_seq_dist[x->owner_opr()->id()];
            lst.pop();
            size_t owner_id = x->owner_opr()->id();
            m_edges.push_back(std::make_pair(owner_id, x->id()));
            if (m_color.find(owner_id) != m_color.end())
                continue;
            m_color[owner_id] = 1;
            rst.push(x->owner_opr());
        }
        while (!rst.empty()) {
            auto u = rst.front();
            m_opr_map[u->id()] = u;
            max_idx = std::max(max_idx, u->id());
            rst.pop();
            for (auto&& v : u->input()) {
                size_t idx = v->id();
                m_edges.push_back(std::make_pair(v->id(), u->id()));
                if (m_color.find(idx) != m_color.end())
                    continue;
                m_color[idx] = 2;
                lst.push(v);
            }
        }
    }

    if (m_opr_seq_dist.find(loss_idx) == m_opr_seq_dist.end()) {
        mgb_log_debug("Computation of Loss is not found in opr seq\n");
    }

    gopt::SubGraph subgraph(sva);
    max_idx += 1;
    std::vector<int> deg(max_idx);

    std::vector<std::vector<size_t>> g, tg;
    g = std::vector<std::vector<size_t>>(max_idx),
    tg = std::vector<std::vector<size_t>>(max_idx);
    m_topo_layer = std::vector<int>(max_idx);

    ThinHashMap<size_t, std::vector<size_t>> topo_map;
    std::queue<size_t> q[2];

    for (auto& x : m_edges) {
        g[x.first].push_back(x.second);
        tg[x.second].push_back(x.first);
        deg[x.second]++;
    }

    for (size_t i = 0; i < max_idx; ++i) {
        if (!deg[i] && !g[i].empty()) {
            q[0].push(i);
        }
    }

    int f = 1;
    int cnt = 1;
    while (!q[0].empty() || !q[1].empty()) {
        f ^= 1;
        while (!q[f].empty()) {
            auto x = q[f].front();
            m_topo_layer[x] = cnt;
            if (topo_map.find(cnt) != topo_map.end()) {
                std::vector<size_t> s;
                s.push_back(x);
                topo_map[cnt] = s;
            } else
                topo_map[cnt].push_back(x);

            q[f].pop();
            for (auto v : g[x])
                if (--deg[v] == 0)
                    q[f ^ 1].push(v);
        }
        ++cnt;
    }

    PIPSet heap;
    determine_swap_edge(heap, loss_idx, opr_seq, g, tg);

#if 0
    /*!
     * Swap the split_points found in Sublinear phase
     * Need to merge zxr/sublinear to enable the sentence below
     */
    auto split_point_set =
        m_owner_graph->options().opr_attribute.swap_inout_endpoint;
    for (auto u : split_point_set) {
        for (auto v : g[u->id()]) {

            auto x = std::make_pair(u->id(), v);
            if (m_opr_seq_dist.find(x.first) == m_opr_seq_dist.end() ||
                m_opr_seq_dist.find(x.second) == m_opr_seq_dist.end())
                continue;
            if (-m_opr_seq_dist[x.second] + m_opr_seq_dist[x.first] >= -20)
                continue;
            if (m_var_map[x.first]->owner_opr()->same_type<SharedDeviceTensor>())
                continue;
            auto s = m_opr_map[x.second]->node_prop().dep_map().find(
                    m_var_map[x.first]);
            if (s == m_opr_map[x.second]->node_prop().dep_map().end())
                continue;
            if (s->second != cg::OperatorNodeBase::NodeProp::DepType::DEV_VALUE)
                continue;
            if (m_var_map[x.first]->owner_opr()->same_type<opr::Subtensor>())
                continue;
            auto sz = m_var_map[x.first]->dtype().size(
                    infer_mgr.infer_shape(m_var_map[x.first]).total_nr_elems());
            static_cast<void>(sz);
            //heap.insert(PIP(-m_opr_seq_dist[x.second] + m_opr_seq_dist[x.first],
            //                PSS(x.first, x.second)));
            heap.insert(PIP(0, PSS(x.first, x.second)));

        }
    }
#elif 0
    for (auto& x : m_edges)
        if ((m_color[x.first] == 2) && (m_color[x.second] == 1)) {
            if (m_opr_seq_dist.find(x.first) == m_opr_seq_dist.end() ||
                m_opr_seq_dist.find(x.second) == m_opr_seq_dist.end())
                continue;

            if (m_opr_seq_dist[x.second] - m_opr_seq_dist[x.first] <=
                m_lb_for_distance)
                continue;
            if (m_var_map[x.first]
                        ->owner_opr()
                        ->same_type<SharedDeviceTensor>())
                continue;
            auto s = m_opr_map[x.second]->node_prop().dep_map().find(
                    m_var_map[x.first]);
            if (s == m_opr_map[x.second]->node_prop().dep_map().end())
                continue;
            if (s->second != cg::OperatorNodeBase::NodeProp::DepType::DEV_VALUE)
                continue;
            if (m_topo_layer[x.first] > m_topo_layer[loss_idx])
                continue;
            if (m_topo_layer[x.second] < m_topo_layer[loss_idx])
                continue;
            if (m_var_map[x.first]->owner_opr()->same_type<opr::Subtensor>())
                continue;
            auto sz = m_var_map[x.first]->dtype().size(
                    infer_mgr.infer_shape(m_var_map[x.first]).total_nr_elems());
            static_cast<void>(sz);
            if (sz <= m_swap_out_var_size_lb)
                continue;
            heap.insert(PIP(-m_opr_seq_dist[x.second] + m_opr_seq_dist[x.first],
                            PSS(x.first, x.second)));
        }
#endif
    auto rewriter = subgraph.make_rewriter();
    ThinHashMap<size_t, ThinHashSet<size_t>> burden;
    std::vector<PSS> arr;

    // size_t limit = m_n_tensors;
    // if (limit < 0)
    //     limit = heap.size();
    // for (size_t i = 0; i < limit && !heap.empty(); ++i) {
    while (!heap.empty()) {
        auto ret = heap.begin();
        arr.push_back(ret->second);
        heap.erase(ret);
    }
    sort(arr.begin(), arr.end(), [](const PSS& lhs, const PSS& rhs) {
        return lhs.second < rhs.second;
    });
    std::vector<VarNode*> cur;
    ThinHashMap<size_t, std::vector<size_t>> fuse_swap;
    ThinHashMap<size_t, ThinHashMap<size_t, size_t>> fuse_dep_node;

    for (size_t i = 0; i < arr.size(); ++i) {
        if (fuse_swap.find(arr[i].first) == fuse_swap.end()) {
            std::vector<size_t> tmp;
            tmp.push_back(arr[i].second);
            fuse_swap[arr[i].first] = tmp;
        } else {
            fuse_swap[arr[i].first].push_back(arr[i].second);
        }
        if (!cur.empty()) {
            if (arr[i].second != arr[i - 1].second) {
                ThinHashSet<size_t> tmp;
                tmp.insert(cur[0]->id());
                burden[arr[i - 1].second] = tmp;
                for (size_t j = 1; j < cur.size(); ++j)
                    burden[arr[i - 1].second].insert(cur[j]->id());
                cur.clear();
            }
        }
        cur.push_back(m_var_map[arr[i].first]);
    }

    int fail_counter = 0;
    for (auto x : fuse_swap) {
        sort((x.second).begin(), (x.second).end(),
             [&](const size_t& lhs, const size_t& rhs) {
                 return m_opr_seq_dist[lhs] < m_opr_seq_dist[rhs];
             });
        for (size_t i = 0; i < x.second.size(); ++i) {
            int dep_idx = 0;
            if (m_opr_seq_dist[x.second[i]] >= m_swap_in_prev)
                dep_idx = opr_seq[m_opr_seq_dist[x.second[i]] - m_swap_in_prev]
                                  ->output(0)
                                  ->id() +
                          1;
            if (dep_idx > 0) {
                size_t j = i;
                for (; j < x.second.size(); ++j) {
                    if (m_opr_seq_dist[x.second[i]] +
                                (long long)m_fuse_swap_in_bound >
                        m_opr_seq_dist[x.second[j]]) {
                        fuse_dep_node[x.first][x.second[j]] = dep_idx;
                    } else
                        break;
                }
                i = j - 1;
            } else {
                fuse_dep_node[x.first][x.second[i]] = 0;
            }
        }
        for (auto& y : x.second) {
            if (fuse_dep_node[x.first][y] == 0)
                fail_counter++;
        }
    }

    if (!cur.empty()) {
        ThinHashSet<size_t> tmp;
        tmp.insert(cur[0]->id());
        burden[arr.back().second] = tmp;
        for (size_t j = 1; j < cur.size(); ++j)
            burden[arr.back().second].insert(cur[j]->id());
    }
    auto gao = [&](OperatorNodeBase* opr) {
        if (burden.find(opr->id()) == burden.end()) {
            rewriter.auto_replace_outputs(opr);
        } else {
            VarNodeArray swapped_input;
            bool flag = false;
            for (auto& x : opr->input()) {
                auto y = rewriter.get_var(x);
                while (y != rewriter.get_var(y))
                    y = rewriter.get_var(y);
                swapped_input.push_back(y);
            }
            for (size_t i = 0; i < opr->input().size(); ++i) {
                if (burden[opr->id()].find(opr->input()[i]->id()) !=
                    burden[opr->id()].end()) {
                    auto dep_idx =
                            fuse_dep_node[opr->input()[i]->id()][opr->id()];
                    if (!dep_idx)
                        continue;
                    else
                        dep_idx--;
                    flag = 1;
                    auto dep_node = rewriter.get_var(m_var_map[dep_idx]);
                    VarNode* swap_res_var;
                    if (!m_bucket_implement) {
                        auto vd1_idx =
                                opr_seq[m_opr_seq_dist[opr->input()[i]
                                                               ->owner_opr()
                                                               ->id()] +
                                        4]
                                        ->output(0)
                                        ->id();
                        auto cpi_idx = opr_seq[m_opr_seq_dist[opr->id()] -
                                               m_swap_in_prev / 2]
                                               ->output(0)
                                               ->id();
                        auto vd1_dep = rewriter.get_var(m_var_map[vd1_idx]);
                        auto cpi_dep = rewriter.get_var(m_var_map[cpi_idx]);

                        swap_res_var = apply(rewriter.get_var(opr->input()[i]),
                                             vd1_dep, cpi_dep, dep_node);
                    } else {
                        auto wait_dep_idx =
                                opr_seq[m_opr_seq_dist[opr->id()] - 1]
                                        ->output(0)
                                        ->id();
                        auto wait_dep =
                                rewriter.get_var(m_var_map[wait_dep_idx]);
                        swap_res_var =
                                apply_bucket(rewriter.get_var(opr->input()[i]),
                                             dep_node, wait_dep);
                    }
                    swapped_input[i] = swap_res_var;
                } else if (rewriter.get_var(opr->input()[i]) !=
                           opr->input()[i]) {
                    auto x = rewriter.get_var(opr->input()[i]);
                    while (x != rewriter.get_var(x))
                        x = rewriter.get_var(x);
                    swapped_input[i] = x;
                }
            }
            if (flag) {
                auto neo_opr = mgb::serialization::copy_opr_shallow(
                        *opr, swapped_input, opr->config());
                for (size_t i = 0; i < opr->output().size(); ++i) {
                    rewriter.replace_var(opr->output()[i], neo_opr->output()[i],
                                         nullptr);
                }
            } else
                rewriter.auto_replace_outputs(opr);
        }
    };

    // As the selection of swap edges are based on the opr seq,
    // using subgraph.iter(gao) may cause bug
    for (size_t i = 0; i < opr_seq.size(); ++i)
        gao(opr_seq[i]);

    for (size_t i = 0; i < vars.size(); ++i) {
        vars[i] = rewriter.get_var(vars[i]);
    }

    rewriter.apply_inplace();
}

VarNode* MemorySwap::apply_bucket(VarNode* lhs, VarNode* dep_node,
                                  VarNode* wait_dep) {
    if (m_swap_map.find(lhs) != m_swap_map.end()) {
        if (m_swap_map[lhs].find(dep_node) != m_swap_map[lhs].end()) {
            return m_swap_map[lhs][dep_node];
        }
    }
    auto graph = this->m_owner_graph;
    if (m_swap_out_map.find(lhs) == m_swap_out_map.end()) {
        auto internal = SwapOutMS::make(*graph, lhs, {}, {}).node();
        mgb_assert(internal->owner_opr()->same_type<opr::SwapOutMS>(),
                   "fail to cast OperatorNodeBase to SwapOutMS");
        auto soo = static_cast<opr::SwapOutMS*>(internal->owner_opr());
        SwapVarInfo svi;
        svi.var = lhs;
        std::shared_ptr<SwapVarRecorder> swapVarRecorder;
        if (!m_firstSwapVarRecorderOwner) {
            swapVarRecorder = std::make_shared<SwapVarRecorder>(
                    &svi, m_max_swap_out_var_size);
            swapVarRecorder->enable(true);
            m_firstSwapVarRecorderOwner = soo;
        } else {
            mgb_assert(m_firstSwapVarRecorderOwner->same_type<opr::SwapOutMS>(),
                       "fail to cast OperatorNodeBase to SwapOutMS");
            swapVarRecorder =
                    (static_cast<opr::SwapOutMS*>(m_firstSwapVarRecorderOwner))
                            ->recorder();
        }
        mgb_assert(swapVarRecorder);
        soo->set_recorder(swapVarRecorder);

        auto mid = opr::SwapInMS::make(*graph, internal, dep_node,
                                       {swapVarRecorder}, {})
                           .node();

        /*!
         * after enabling value_infer, sometimes the varnode above may
         * may be replaced other var, its owner is not SwapInMS;
         */
        if (!(mid->owner_opr()->same_type<opr::SwapInMS>())) {
            return lhs;
        }

        auto ret = opr::WaitSwapInMS::make(*graph, {mid, wait_dep}, {}).node();
        mgb_assert(soo->recorder() ==
                   static_cast<opr::SwapInMS*>(mid->owner_opr())->recorder());

        internal->owner_opr()->node_prop().attribute().priority =
                mid->owner_opr()->node_prop().attribute().priority =
                        std::numeric_limits<int>::min();
        ret->owner_opr()->node_prop().attribute().priority =
                std::numeric_limits<int>::min();
        m_swap_out_map[lhs] = internal;
        m_swap_map[lhs][dep_node] = ret;
        return ret;
    } else {
        auto internal = m_swap_out_map[lhs];
        mgb_assert(internal->owner_opr()->same_type<opr::SwapOutMS>(),
                   "fail to cast OperatorNodeBase to SwapOutMS");
        auto mid = opr::SwapInMS::make(
                           *graph, internal, dep_node,
                           {(static_cast<SwapOutMS*>(internal->owner_opr()))
                                    ->recorder()},
                           {})
                           .node();
        auto ret = opr::WaitSwapInMS::make(*graph, {mid, wait_dep}, {}).node();
        mid->owner_opr()->node_prop().attribute().priority =
                std::numeric_limits<int>::min();
        ret->owner_opr()->node_prop().attribute().priority =
                std::numeric_limits<int>::min();
        m_swap_map[lhs][dep_node] = ret;
        return ret;
    }
    return nullptr;
}

VarNode* MemorySwap::apply(VarNode* lhs, VarNode* vd1_dep, VarNode* cpi_dep,
                           VarNode* dep_node) {
    if (m_swap_map.find(lhs) != m_swap_map.end()) {
        if (m_swap_map[lhs].find(dep_node) != m_swap_map[lhs].end()) {
            return m_swap_map[lhs][dep_node];
        }
    }
    auto graph = lhs->owner_opr()->owner_graph();
    if (m_swap_out_map.find(lhs) == m_swap_out_map.end()) {
        HostTensorND tms(lhs->comp_node(), lhs->dtype());
        std::shared_ptr<HostTensorND> tmp;
        tmp = std::make_shared<HostTensorND>(tms);
        auto internal = opr::SwapOut::make(*graph, lhs, {tmp}).node();
        auto ret =
                opr::SwapIn::make(*graph, {internal, dep_node}, tmp, {}).node();
        internal->owner_opr()->node_prop().attribute().priority =
                std::numeric_limits<int>::min();
        ret->owner_opr()->node_prop().attribute().priority =
                std::numeric_limits<int>::min();
        m_swap_out_map[lhs] = internal;
        m_swap_map[lhs][dep_node] = ret;
        return ret;
    } else {
        auto internal = m_swap_out_map[lhs];  // in fact do not need this..
        mgb_assert(internal->owner_opr()->same_type<SwapOut>());
        auto ret =
                opr::SwapIn::make(*graph, {internal, dep_node},
                                  (static_cast<SwapOut*>(internal->owner_opr()))
                                          ->host_data(),
                                  {})
                        .node();
        ret->owner_opr()->node_prop().attribute().priority =
                std::numeric_limits<int>::min();
        m_swap_map[lhs][dep_node] = ret;
        return ret;
    }
}
#endif  // MGB_ENABLE_MEMORY_SWAP

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
