/**
 * \file src/core/impl/graph/var_node_mem_mgr/seq_mem_opt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./seq_mem_opt.h"
#include "./static_mem_alloc.h"
#include "../cg_impl.h"

#include "megbrain/graph/event.h"
#include "megbrain/graph/helper.h"
#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/utils/metahelper.h"
#include "megbrain/utils/arith_helper.h"

using namespace mgb;
using namespace cg;

constexpr double BYTE2MB = 1.0 / 1024.0 / 1024;

class SeqMemOptimizer::StaticMemAllocLogger {
    public:
        virtual ~StaticMemAllocLogger() = default;
        virtual void flush() = 0;
        virtual void push(const CompNode &comp_node, size_t size, size_t size_lb,
                size_t size_ub) = 0;

        class LogImpl;
        class FakeImpl;
};

class SeqMemOptimizer::StaticMemAllocLogger::FakeImpl final:
            public StaticMemAllocLogger {
    public:

        void flush() override {}
        void push(const CompNode &, size_t, size_t, size_t) override {}

};

class SeqMemOptimizer::StaticMemAllocLogger::LogImpl final:
            public StaticMemAllocLogger {

    std::vector<std::pair<std::string, std::string>> m_logs;
    public:
        void flush() {
            std::sort(m_logs.begin(), m_logs.end());

            std::string log = "static memory allocation:\n";
            log += " comp_node           alloc                    "
                "  lower_bound         upper_bound\n";
            for (auto const &i: m_logs) {
                log += i.second;
            }
            log.pop_back(); // remove trailing '\n'
            mgb_log_debug("%s", log.c_str());
        }

        void push(const CompNode &comp_node, size_t size, size_t size_lb,
                size_t size_ub) {
            auto msg = ssprintf(
                    "%9s%10.2fMiB(%10zubytes)%10.2fMiB(%6.2f%%)"
                    "%10.2fMiB(%6.2f%%)\n",
                    comp_node.to_string().c_str(), size * BYTE2MB, size,
                    size_lb * BYTE2MB, size_lb * 100.0 / size,
                    size_ub * BYTE2MB, size_ub * 100.0 / size);
            m_logs.push_back(std::make_pair(comp_node.to_string(), msg.c_str()));
        }
};


void SeqMemOptimizer::optimize_mem_plan_dynamic(OperatorNodeBase *opr) {
    mgb_assert(!m_status);
    m_status = Status::ALLOW_FWD_IN2OUT_READONLY;
    opr->mem_plan_fwd_in2out_readonly();
    m_status = 0;
}

void SeqMemOptimizer::optimize_mem_plan() {
    if (!m_graph->options().seq_opt.enable_mem_plan_opt) {
        mgb_log_warn("mem plan optimization disabled");
        // we still run the passes below to check potential errors; actual mem
        // plan optimization is disabled by VarNodeMemManager funcs returning
        // false in fwd test
    }

    OperatorNodeBase *opr = nullptr;
    MGB_TRY {
        m_writable_fwd_mem_plans.clear();
        m_status = Status::ALLOW_FWD_IN2OUT_READONLY;
        OprNodeArray oprs_to_run;
        for (auto i: *m_cur_seq_sys_alloc) {
            opr = i;
            if (is_all_input_static_storage(opr)) {
                // if there are dynamic input vars, opr forwarding may not work
                // property (we have assumed shapes to be available in
                // mem_plan_fwd_in2out_readonly to make subspec)
                opr->mem_plan_fwd_in2out_readonly();
                oprs_to_run.push_back(opr);
            }
        }
        opr = nullptr;
        m_status = Status::ALLOW_FWD_IN2OUT_WRITABLE;
        for (auto i: oprs_to_run) {
            opr = i;
            opr->mem_plan_fwd_in2out_writable();
        }
        m_status = 0;
    } MGB_CATCH(MegBrainError &exc,  {
        if (opr && !exc.extra_info())
            OperatorNodeExcExtraInfo::record(opr, exc);
        throw;
    })
}

bool SeqMemOptimizer::should_static_alloc_var(VarNode *var) {
    if (!m_cur_static_alloc_var->count(var)) {
        return false;
    }

    auto &&chk = var->mem_plan().chunk();
    if (!chk.size()) {
        mgb_assert(var->contain_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE));
        return false;
    }
    if (!chk.mem_alloc_status.is_invalid()) {
        mgb_assert(chk.mem_alloc_status.is_from_owner_var() &&
                   chk.owner_var->dev_tensor().storage().size() >= chk.size());
        return false;
    }
    return true;
}

bool SeqMemOptimizer::plan_chunk_allocation() {
    if (!m_static_mem_usage.valid()) {
        m_static_mem_usage.emplace();
    }

    if (m_graph->options().seq_opt.enable_mem_reuse_alloc) {
        return run_static_mem_alloc();
    }

    mgb_log_warn(
            "static memory optimization disabled, allocating in a naive way");
    auto&& cn2usage = m_static_mem_usage.val();

    // clear so usage can start at zero
    cn2usage.clear();

    for (auto&& opr : *m_cur_seq_sys_alloc) {
        for (auto&& var : opr->output()) {
            if (should_static_alloc_var(var)) {
                auto chunk = &var->mem_plan().chunk();
                if (chunk->owner_var == var) {
                    size_t& usage = cn2usage[var->comp_node()];
                    size_t offset = usage;
                    usage += get_aligned_power2(
                            chunk->size() + var->comp_node().get_mem_padding(),
                            var->comp_node().get_mem_addr_alignment());
                    chunk->mem_alloc_status.set_static_offset(offset);
                }
            }
        }
    }
    return false;
}

bool SeqMemOptimizer::run_static_mem_alloc() {
    // map from chunk pointer to life interval
    // multiple var nodes share the same chunk pointer by readonly memory
    // forwarding
    ThinHashMap<MemAllocPlan::Chunk*, MemChunkLifeInterval> chk2interval;

    // get all memory chunks
    for (size_t idx = 0; idx < m_cur_seq_full->size(); ++ idx) {
        OperatorNodeBase *opr = m_cur_seq_full->at(idx);

        auto &&dep_map = opr->node_prop().dep_map();

        if (in_sys_alloc(opr)) {
            // find all output vars, marking start of chunk life
            for (VarNode *i: opr->output()) {

                if (!should_static_alloc_var(i))
                    continue;

                auto cur_chk = &i->mem_plan().chunk();
                auto insert_rst = chk2interval.insert({cur_chk, {}});

                auto &&dest = insert_rst.first->second;
                if (insert_rst.second) {
                    dest.begin = idx;
                    dest.chunk = cur_chk;
                    dest.comp_node = i->comp_node();
                    mgb_assert(cur_chk->owner_var == i);
                } else {
                    // forwarded from another var
                    mgb_assert(i->comp_node() == dest.comp_node &&
                               cur_chk->owner_var != i);
                }

                if (i->contain_flag(VarNode::Flag::NO_MEM_RECLAIM)) {
                    dest.end = std::numeric_limits<size_t>::max();
                }
            }
        }

        // find all input vars, marking end of chunk life
        for (auto &&dep_entry: dep_map) {
            if (!(OperatorNodeBase::NodeProp::is_device_value_dep(
                            dep_entry.second)))
                continue;

            auto ivar = dep_entry.first;
            auto iter = chk2interval.end();
            if (ivar->mem_plan().valid())
                iter = chk2interval.find(&ivar->mem_plan().chunk());
            if (iter == chk2interval.end()) {
                // some operator may produce statically shaped output even with
                // dynamic input, and we need to allocate them
                mgb_assert(!should_static_alloc_var(ivar));
                continue;
            }

            auto &&dest = iter->second;
            mgb_assert(dest.comp_node == ivar->comp_node());
            dest.end = std::max(dest.end, idx + 1);
        }
    }

    // group memory chunks by comp_node
    CompNode::UnorderedMap<std::vector<MemChunkLifeInterval>> group_by_cn;

    for (auto &&i: chk2interval) {
        if (!i.second.end) {
            // unused output
            i.second.end = i.second.begin + 1;
        }
        mgb_assert(i.second.end > i.second.begin);
        group_by_cn[i.first->owner_var->comp_node()].push_back(i.second);
    }

    {
        // force release memory
        decltype(chk2interval) v;
        chk2interval.swap(v);
    }

    StaticMemAllocLogger::FakeImpl fake_logger;
#if MGB_ENABLE_LOGGING
    StaticMemAllocLogger::LogImpl real_logger;
    StaticMemAllocLogger *logger =
        m_graph->options().log_level ?
        static_cast<StaticMemAllocLogger*>(&real_logger) :
        static_cast<StaticMemAllocLogger*>(&fake_logger);
#else
    StaticMemAllocLogger *logger = &fake_logger;
#endif

    bool ret = false;
    for (auto &&i: group_by_cn) {
        auto cmp = [](
                const MemChunkLifeInterval &a, const MemChunkLifeInterval &b) {
            return a.begin < b.begin || (a.begin == b.begin && a.end < b.end);
        };
        // sort for stable order
        std::sort(i.second.begin(), i.second.end(), cmp);
        ret |= run_static_mem_alloc_on_comp_node(i.first, i.second, *logger);
    }
    logger->flush();

    // trigger event for other comp nodes
    for (auto i : m_all_comp_nodes) {
        if (!group_by_cn.count(i)) {
            bool need_realloc = false;
            m_graph->event().signal_inplace<event::StaticMemAlloc>(
                    &need_realloc, i, static_cast<size_t>(0));
            ret |= need_realloc;
        }
    }

    m_graph->event().signal_inplace<event::StaticMemAlloc>(
            nullptr, CompNode{}, static_cast<size_t>(0));

    return ret;
}

bool SeqMemOptimizer::run_static_mem_alloc_on_comp_node(
        CompNode comp_node,
        const std::vector<MemChunkLifeInterval> &chunks,
        StaticMemAllocLogger &static_mem_alloc_logger) {

    size_t size_ub = 0;

    auto allocator = StaticMemAlloc::make(
            StaticMemAlloc::AllocatorAlgo::PUSHDOWN);
    allocator->alignment(comp_node.get_mem_addr_alignment());
    allocator->padding(comp_node.get_mem_padding());
#if MGB_ENABLE_DEBUG_UTIL
    allocator->dbg_key2varnode = [](StaticMemAlloc::UserKeyType key) {
        return static_cast<const MemChunkLifeInterval*>(key)->chunk->owner_var;
    };
#endif
    ThinHashMap<MemAllocPlan::Chunk*, size_t> chunk2allocatorid;
    for (auto &&chk: chunks) {
        auto id = allocator->add(
                chk.begin, chk.end, chk.chunk->size(), &chk);
        auto ins_rst = chunk2allocatorid.emplace(chk.chunk, id);
        mgb_assert(ins_rst.second);
        size_ub += chk.chunk->size();
    }

    for (auto &&i: m_writable_fwd_mem_plans) {
        auto from_iter = chunk2allocatorid.find(&i.first->chunk()),
             to_iter = chunk2allocatorid.find(&i.second->chunk());

        // ignore mem fwd specs that involve other chunks
        if (from_iter != chunk2allocatorid.end() &&
                to_iter != chunk2allocatorid.end()) {

            allocator->add_overwrite_spec(to_iter->second, from_iter->second,
                    i.first->offset_in_chunk_byte());
        }
    }
    {
        decltype(chunk2allocatorid) v;
        chunk2allocatorid.swap(v);
    }

    allocator->solve();
    size_t size = allocator->tot_alloc(),
           size_lb = allocator->tot_alloc_lower_bound();

    static_mem_alloc_logger.push(comp_node, size, size_lb, size_ub);

    bool should_realloc = false;
    m_graph->event().signal_inplace<event::StaticMemAlloc>(
            &should_realloc, comp_node, size);

    if (!should_realloc) {
        m_static_mem_usage.val()[comp_node] = size;
        for (auto&& chk : chunks) {
            chk.chunk->mem_alloc_status.set_static_offset(
                    allocator->get_start_addr(&chk));
        }
    }

    return should_realloc;
}

void SeqMemOptimizer::reset_opr_seq(const OprNodeArray *seq,
                const OprNodeArray *seq_sys_alloc,
                const VarNodeSet *static_alloc_var,
                SmallVector<CompNode> all_comp_nodes) {
    m_cur_seq_full = seq;
    m_cur_seq_sys_alloc = seq_sys_alloc;
    m_cur_seq_sys_alloc_set = {
        seq_sys_alloc->begin(), seq_sys_alloc->end()};
    m_cur_static_alloc_var = static_alloc_var;
    m_all_comp_nodes = std::move(all_comp_nodes);
    m_static_mem_usage.invalidate();
}

void SeqMemOptimizer::add_writable_fwd_mem_plan_pair(
        MemAllocPlan *from, MemAllocPlan *to) {
    mgb_assert(&from->chunk() != &to->chunk() && from != to);
    m_writable_fwd_mem_plans.emplace_back(from, to);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
