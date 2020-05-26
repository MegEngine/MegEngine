/**
 * \file src/core/impl/graph/var_node_mem_mgr/static_mem_alloc/impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./impl.h"
#include "./interval_move.h"
#include "./best_fit.h"
#include "./pushdown.h"

#include <map>

#if MGB_ENABLE_DEBUG_UTIL
#include "megbrain/graph/var_node.h"
#include "megbrain/graph/operator_node.h"
#include <cstdio>
#include <fstream>
#endif

using namespace mgb;
using namespace cg;

constexpr size_t StaticMemAllocImplHelper::INVALID;

StaticMemAllocImplHelper::Interval*
StaticMemAllocImplHelper::Interval::overwrite_dest_root_path_compression() {
    auto &&ptr = m_overwrite_dest_root;
    if (!ptr)
        return this;
    auto root = ptr->overwrite_dest_root_path_compression();
    if (root != ptr) {
        m_offset_in_overwrite_dest_root += ptr->m_offset_in_overwrite_dest_root;
        ptr = root;
    }
    return root;
}

void StaticMemAllocImplHelper::init_overwrite_dest() {
    for (auto &&spec: m_overwrite_spec) {
        auto src = m_interval_storage.data() + std::get<0>(spec),
             dest = m_interval_storage.data() + std::get<1>(spec);
        // src overwrites a part in dest
        size_t offset = std::get<2>(spec);
        mgb_assert(src->time_begin < dest->time_end);

        auto orig_src = dest->m_overwrite_src;

        // each interval could only be overwritten by one interval, and we
        // prefer the interval with largest size to be overwritter
        if (src->time_begin == dest->time_end - 1 && !src->m_overwrite_dest &&
                (!orig_src || src->size > orig_src->size)) {
            if (orig_src) {
                orig_src->m_overwrite_dest = nullptr;
                orig_src->m_offset_in_overwrite_dest = 0;
            }
            dest->m_overwrite_src = src;
            src->m_overwrite_dest = dest;
            src->m_offset_in_overwrite_dest = offset;
        }
    }

    for (auto &&i: m_interval_storage) {
        if (i.m_overwrite_dest) {
            i.m_overwrite_dest_root = i.m_overwrite_dest;
            i.m_offset_in_overwrite_dest_root = i.m_offset_in_overwrite_dest;
            mgb_assert(i.m_overwrite_dest->m_overwrite_src == &i);
        }
        if (i.m_overwrite_src)
            mgb_assert(i.m_overwrite_src->m_overwrite_dest == &i);
    }

    for (auto &&i: m_interval_storage)
        i.overwrite_dest_root_path_compression();
}

size_t StaticMemAllocImplHelper::add(size_t begin, size_t end, size_t size,
        UserKeyType key) {

    mgb_assert(begin < end);
    auto id = m_interval_storage.size();
    m_interval_storage.push_back({begin, end, size + m_padding, key, id});
    return id;
}

StaticMemAlloc& StaticMemAllocImplHelper::add_overwrite_spec(
        size_t iid_src, size_t iid_dest, size_t offset) {
    auto &&src = m_interval_storage.at(iid_src),
         &&dest = m_interval_storage.at(iid_dest);
    mgb_assert(iid_src != iid_dest);
    mgb_assert(offset + src.size <= dest.size);
    m_overwrite_spec.emplace_back(iid_src, iid_dest, offset);
    return *this;
}

size_t StaticMemAllocImplHelper::get_start_addr(UserKeyType key) const {
    return m_userkey2itrv.at(key)->addr_begin;
}

StaticMemAlloc& StaticMemAllocImplHelper::solve() {
    dbg_dump_interval_list();
    dbg_load_interval_list();
    m_interval.clear();
    m_interval.reserve(m_interval_storage.size());
    m_userkey2itrv.clear();
    for (auto &&i: m_interval_storage) {
        m_interval.push_back(&i);
        auto ist = m_userkey2itrv.insert({i.key, &i});
        mgb_assert(ist.second, "duplicated user key");
    }

    init_overwrite_dest();

    do_solve();

    check_result_and_calc_lower_bound();

    return *this;
}

void StaticMemAllocImplHelper::dbg_dump_interval_list() {
#if MGB_ENABLE_DEBUG_UTIL
    const char *fdir = MGB_GETENV("MGB_DUMP_INTERVAL_LIST_DIR");
    if (!fdir)
        return;
    static int run_id = 0;
    auto fpath = ssprintf("%s/mgb-interval-%d.txt", fdir, run_id ++);
    mgb_log_warn("dump static mem alloc interval list to %s", fpath.c_str());
    FILE *fout = fopen(fpath.c_str(), "w");
    mgb_assert(fout, "failed to open %s", fpath.c_str());

    fprintf(fout, "%zu\n"_fmt, m_interval_storage.size());
    for (auto &&i: m_interval_storage)
        fprintf(fout, "%zu %zu %zu\n", i.time_begin_orig, i.time_end_orig,
                i.size_orig);

    fprintf(fout, "%zu\n"_fmt, m_overwrite_spec.size());
    for (auto &&i: m_overwrite_spec)
        fprintf(fout, "%zu %zu %zu\n"_fmt, std::get<0>(i),
                std::get<1>(i), std::get<2>(i));

    fclose(fout);
#endif
}

void StaticMemAllocImplHelper::dbg_load_interval_list() {
#if MGB_ENABLE_DEBUG_UTIL
    const char *fpath = MGB_GETENV("MGB_LOAD_INTERVAL");
    if (!fpath)
        return;
    unsetenv("MGB_DUMP_INTERVAL_LIST_DIR");
    unsetenv("MGB_LOAD_INTERVAL");
    mgb_log_warn("load interval from %s for debug", fpath);
    std::ifstream fin(fpath);
    mgb_assert(fin.good(), "failed to open %s", fpath);

    m_interval_storage.clear();
    m_overwrite_spec.clear();
    size_t nr_interval;
    fin >> nr_interval;
    for (size_t i = 0; i < nr_interval; ++ i) {
        size_t begin, end, size;
        fin >> begin >> end >> size;
        add(begin, end, size, reinterpret_cast<UserKeyType>(i));
    }

    size_t nr_overwrite;
    fin >> nr_overwrite;
    for (size_t i = 0; i < nr_overwrite; ++ i) {
        size_t s, d, o;
        fin >> s >> d >> o;
        add_overwrite_spec(s, d, o);
    }

    solve();

    printf("allocation result tot_alloc=%zu(%.2fMiB):\n",
            tot_alloc(), tot_alloc() / 1024.0 / 1024.0);
    for (auto &&i: m_interval_storage) {
        printf("id=%zu size=%zu(%.2fMiB) time=[%zu, %zu) addr=[%zu, %zu)\n",
                i.id, i.size_orig, i.size_orig / 1024.0 / 1024,
                i.time_begin_orig, i.time_end_orig,
                i.addr_begin, i.addr_end());
    }
    fflush(stdout);

    mgb_trap();
#endif
}

void StaticMemAllocImplHelper::check_result_and_calc_lower_bound() {
    size_t peak = 0;

    // time => pair(alloc, free)
    using TimeEvent = std::pair<IntervalPtrArray, IntervalPtrArray>;
    std::map<size_t, TimeEvent> time2event;

    for (auto &&i: m_interval_storage) {
        mgb_assert(i.addr_begin != INVALID);
        time2event[i.time_begin_orig].first.push_back(&i);
        time2event[i.time_end_orig].second.push_back(&i);
        update_max(peak, i.addr_end());
        if (i.is_overwrite_root()) {
            // modify size for calc lower bound
            i.size = align(i.size_orig);
            mgb_assert(i.addr_begin == align(i.addr_begin));
        } else {
            auto offset = i.offset_in_overwrite_dest_root();
            i.size = align(offset + i.size_orig) - (
                    offset - (offset & (m_alignment - 1)));
        }
    }
    mgb_assert(peak <= tot_alloc() && align(peak) == align(tot_alloc()));

    // get lower bound
    {
        m_peak_lower_bound = 0;
        size_t usage = 0;
        for (auto &&tpair: time2event) {
            for (auto i: tpair.second.first) {
                if (i->is_overwrite_root())
                    usage += i->size;
            }
            for (auto &&i: tpair.second.second) {
                usage -= i->size;
                if (i->m_overwrite_src) {
                    // this interval is overwritten by another one, so count its
                    // size in current usage
                    usage += i->m_overwrite_src->size;
                }
            }
            update_max(m_peak_lower_bound, usage);
        }
        mgb_assert(!usage);
    }

    print_bottleneck_oprs(time2event);

    // restore time and size; check overwrite addr
    for (auto &&i: m_interval_storage) {
        i.time_begin = i.time_begin_orig;
        i.time_end = i.time_end_orig;
        i.size = i.size_orig;

        if (!i.is_overwrite_root()) {
            mgb_assert(i.overwrite_dest()->addr_begin +
                    i.offset_in_overwrite_dest() == i.addr_begin);
        }
    }

    std::map<size_t, Interval*> cur_allocated;
    IntervalPtrArray id_overwriter;

    auto remove_alloc = [&](Interval *i) {
        auto iter = cur_allocated.find(i->addr_begin);
        mgb_assert(iter != cur_allocated.end() && iter->second == i);
        cur_allocated.erase(iter);

        if (auto s = i->overwrite_src()) {
            auto ins = cur_allocated.insert({s->addr_begin, s});
            mgb_assert(ins.second);
        }
    };

    // check for conflicts
    for (auto &&tpair: time2event) {

        // free and set overwriter addr
        id_overwriter.clear();
        for (auto i: tpair.second.second) {
            if (!i->is_overwrite_root() &&
                    i->time_end_orig == i->overwrite_dest()->time_end_orig &&
                    !i->offset_in_overwrite_dest()) {
                // a overwrites b, a and b share same time end, zero offset
                mgb_assert(i->addr_begin == i->overwrite_dest()->addr_begin);
                id_overwriter.push_back(i);
                continue;
            }
            remove_alloc(i);
        }
        for (auto i: id_overwriter)
            remove_alloc(i);

        // alloc
        for (auto i: tpair.second.first) {
            auto iter = cur_allocated.lower_bound(i->addr_begin);

            if (i->is_overwrite_root()) {
                if (iter != cur_allocated.end()) {
                    mgb_assert(i->addr_end() <= iter->first);
                }
                if (!cur_allocated.empty() && iter != cur_allocated.begin()) {
                    -- iter;
                    mgb_assert(iter->second->addr_end() <= i->addr_begin);
                }
                cur_allocated[i->addr_begin] = i;
            }
        }
    }

    mgb_assert(cur_allocated.empty());
}

template<typename T>
void StaticMemAllocImplHelper::print_bottleneck_oprs(const T& time2event) {
#if MGB_ENABLE_DEBUG_UTIL
    if (!MGB_GETENV("MGB_PRINT_STATIC_ALLOC_BOTTLENECK"))
        return;
    mgb_assert(dbg_key2varnode);

    size_t peak = 0, usage = 0;
    std::unordered_set<UserKeyType> alive, peak_alive;

    for (auto &&tpair: time2event) {
        for (Interval* i: tpair.second.first) {
            if (i->is_overwrite_root()) {
                usage += i->size;
                alive.insert(i->key);
            }
        }
        for (Interval *i: tpair.second.second) {
            usage -= i->size;
            if (i->m_overwrite_src) {
                usage += i->m_overwrite_src->size;
                alive.insert(i->m_overwrite_src->key);
            }
        }
        for (Interval *i: tpair.second.second) {
            alive.erase(i->key);
        }
        if (usage > peak) {
            peak = usage;
            peak_alive = alive;
        }
    }
    mgb_assert(!usage && alive.empty());

    printf("mgb static alloc bottleneck: size=%.3fMiB {\n",
            peak / 1024.0 / 1024);
    using SizeVar = std::tuple<size_t, size_t, VarNode*>;
    std::vector<SizeVar> vars;
    for (auto i: peak_alive) {
        auto var = dbg_key2varnode(i);
        vars.emplace_back(var->mem_plan().chunk().size(), var->id(), var);
    }
    auto cmp = [](const SizeVar &a, const SizeVar &b) {
        auto sza = std::get<0>(a), szb = std::get<0>(b);
        return sza > szb || (sza == szb && std::get<1>(a) < std::get<1>(b));
    };
    std::sort(vars.begin(), vars.end(), cmp);
    for (auto &&i: vars) {
        auto size = std::get<0>(i);
        VarNode* v = std::get<2>(i);
        OperatorNodeBase* o = v->owner_opr();
        printf("  var%zu %s owner=%s{%s} shape=%s alloc_size=%zu\n",
                v->id(), v->cname(), o->cname(), o->dyn_typeinfo()->name,
                v->shape().to_string().c_str(),
                size);
    }
    printf("}\n");
#endif
}

StaticMemAllocImplHelper::~StaticMemAllocImplHelper() noexcept = default;

std::unique_ptr<StaticMemAlloc> StaticMemAlloc::make(AllocatorAlgo algo) {
    switch (algo) {
#if !MGB_BUILD_SLIM_SERVING
        case AllocatorAlgo::INTERVAL_MOVE:
            return std::make_unique<StaticMemAllocIntervalMove>();
        case AllocatorAlgo::BEST_FIT:
            return std::make_unique<StaticMemAllocBestFit>();
#endif
        case AllocatorAlgo::PUSHDOWN:
            return std::make_unique<StaticMemAllocPushdown>();
        default:
            mgb_assert(0, "unknown mem allocator algorithm");
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

