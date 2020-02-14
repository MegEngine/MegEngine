/**
 * \file src/core/impl/graph/swap/memory_swap.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megbrain/graph.h"

#include <set>

#if MGB_ENABLE_MEMORY_SWAP
namespace mgb {
namespace swap {

using PSS = std::pair<size_t, size_t>;
using PIP = std::pair<int, PSS>;
using PIS = std::pair<int, size_t>;
using PII = std::pair<int, int>;
using PPI = std::pair<PII, int>;
using PLS = std::pair<long long, size_t>;
using PLI = std::pair<long long, int>;
using PSSSet = std::unordered_set<PSS, pairhash>;
using PIPSet = std::set<PIP>;

struct NodeInfo {
public:
    long long max;
    long long idx;
    bool operator<=(const NodeInfo& rhs) const { return this->max <= rhs.max; }
};

class SegmentRace {
public:
    size_t m_mem;
    size_t m_id;
    int m_st;
    std::vector<PPI> m_segments;
    std::vector<int> m_consume_opr;
    SegmentRace(const size_t mem, const size_t id, const int st,
                const std::vector<PPI>& segs,
                const std::vector<int>& _consume_opr)
            : m_mem(mem),
              m_id(id),
              m_st(st),
              m_segments(segs),
              m_consume_opr(_consume_opr) {}
};

/*!
 * Support large models by swapping some of the vars from GPU to CPU
 * Ideas are mainly copied from :
 * https://github.com/tensorflow/tensorflow/pull/19845 and
 * https://arxiv.org/abs/1807.02037
 */
class MemorySwap {
    using OperatorNodeBase = cg::OperatorNodeBase;

    /*!
     * whether use buckets to swap tensors between h and d
     * if (max_swap_out_var_size / static_memory_allocation) is large, this
     * method is not recommended because of the extra memory usage
     */
    bool m_bucket_implement = false;

    /*!
     * fuse the 'close' swap-in operations, according to the position
     * of their consuming oprs in opr seq
     *
     * be disabled when 0
     *
     * modifiying it automatically corresponding to the length of opr_seq
     * may make the module more flexible
     */
    size_t m_fuse_swap_in_bound = 100;

    /*!
     * the maximum number of tensors to be swapped
     * < 0 : swap all tensors that meet the conditions
     */
    long long m_n_tensors = -1;

    /*!
     * the swap-in opr may start several oprs before its consuming opr,
     * this param controls it; increaseing this param may improve parallelism
     * but increase memory usage
     *
     * in serial mode, this will be modified to 1
     *
     * TODO :: in tensorflow, there is a method named
     * EstimateEarliestExecutionTimes, which may behave better in finding
     * tigger for swap
     */
    int m_swap_in_prev = 5;

    /*!
     * roughly limit the time increase of each iter
     * after this module is enabled hard to control in parallel mode
     */
    double m_swap_time_limit = 0.4;

    /*!
     * minimum size of the VarNode to be swapped
     */
    size_t m_swap_out_var_size_lb = 1024 * 1024;

    /*!
     * lower bound of dist between the producing opr and consuming opr in opr
     * seq
     */
    long long m_lb_for_distance = 500;

    /*!
     * all the params above are preset for ResNet50 in model zoo, while the
     * opr_seq's length is about 7400; for other cases, the params may need to
     * be modified through environment variable.
     */

    /*!
     * maximum size of the swapped out var, this determines the size
     * of the bucket
     */
    size_t m_max_swap_out_var_size = 0;

    const double m_cpu_gpu_bandwidth = 10000000000.0;

    ComputingGraph* m_owner_graph;
    /*!
     * Find the opr/var by their id
     */
    ThinHashMap<size_t, OperatorNodeBase*> m_opr_map;
    ThinHashMap<size_t, VarNode*> m_var_map;
    ThinHashMap<int, SegmentRace*> m_segmentToRace;
    std::vector<SegmentRace*> m_segmentRaceList;

    /*!
     * Record the first SwapOutMS, it owns the shared_ptr to SwapVarRecorder
     */
    OperatorNodeBase* m_firstSwapVarRecorderOwner = nullptr;
    std::vector<int> m_topo_layer;
    std::vector<std::pair<size_t, size_t>> m_edges;

    /*!
     * Ensure each var shoule be swapped out at most once
     */
    ThinHashMap<VarNode*, VarNode*> m_swap_out_map;
    ThinHashMap<VarNode*, ThinHashMap<VarNode*, VarNode*>> m_swap_map;

    VarNode* m_last_vd3 = nullptr;

    size_t m_segment_race_id = 0;
    std::vector<PPI> m_all_valid_segments;

    ThinHashMap<size_t, long long> m_opr_seq_dist;
    ThinHashMap<size_t, int> m_color;
    PSSSet m_swapped_pair;

    void determine_swap_edge(PIPSet& edges, size_t loss_idx,
                             const cg::OprNodeArray& opr_seq,
                             std::vector<std::vector<size_t>>&,
                             std::vector<std::vector<size_t>>&);

    /*!
     * serial mode :
     *   swap-out        swap-in
     *      * ------------- *
     *     /               / \
     *    /               / 1 \
     *   * ------------- * --- *
     *  lhs          dep_node  rhs
     *
     *  currently vd1_dep and cpi_dep are not in use
     */
    VarNode* apply(VarNode* lhs, VarNode* vd1_dep, VarNode* cpi_dep,
                   VarNode* dep_node);

    /*!
     * use buckets to swap vars in another stream, refer to the implementations
     * of opr::Loop
     *     swap-out  swap-in      wait-swap-in
     *        * ------- * -------------- *
     *       /         /                / \
     *      /         / swap_in_prev   / 1 \
     *     * ------- * -------------- * --- *
     *     lhs    dep_node        wait_node rhs
     * the swap-in opr trigger copy_host_to_bucket(), start copying, and the
     * following wait-swap-in opr ensure its completion before being consumed
     */
    VarNode* apply_bucket(VarNode* lhs, VarNode* dep_node, VarNode* wait_node);

public:
    MemorySwap(ComputingGraph* graph);
    ~MemorySwap() noexcept;
    //! Entrance
    void modify_dest_var_inplace(VarNodeArray& vars);
};
}  // namespace swap
}  // namespace mgb
#endif  // MGB_ENABLE_MEMORY_SWAP

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
