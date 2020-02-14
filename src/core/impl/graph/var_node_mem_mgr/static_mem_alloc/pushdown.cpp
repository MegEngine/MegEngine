/**
 * \file src/core/impl/graph/var_node_mem_mgr/static_mem_alloc/pushdown.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./pushdown.h"
#include "./best_fit_helper.h"

#include "megbrain/utils/metahelper.h"

#include <set>
#include <list>

using namespace mgb;
using namespace cg;

namespace {
    size_t safe_sub(size_t a, size_t b, size_t min_delta = 0) {
        mgb_assert(a >= b + min_delta);
        return a - b;
    }
}

/* ======================== BestfitPrealloc ======================== */

/*!
 * \brief pre-allocate by best-fit
 *
 * Note: different from standard best-fit allocators, this allocator allows
 * already allocated chunks to be increased. More specifically, when trying to
 * find a best fit but no free chunk can hold requested size, then the largest
 * free chunk would be increased, unlike the conventional algorithm that
 * allocates a new chunk of memory from the parent allocator.
 *
 * This step is used to determine the relative order of intervals; it does not
 * handle alignment.
 */
class StaticMemAllocPushdown::BestfitPrealloc {
    public:
        class AllocResult;

        AllocResult alloc(Interval *interval);

        /*!
         * \brief allocate to overwrite an existing interval
         * \param dest interval to be overwritten, which would be freed
         */
        AllocResult alloc_overwrite(
                AllocResult &dest, size_t offset, Interval *interval);

        void free(AllocResult &alloc_rst);

    private:
        struct MemBlock;

        struct FreeBlockBySizeItem;

        std::list<MemBlock> m_mem_block;
        using MemBlockIter = decltype(m_mem_block.begin());
        std::set<FreeBlockBySizeItem> m_free_by_size;
        using FreeBySizeIter = decltype(m_free_by_size.begin());

        struct MemBlock {
            inline MemBlock();
            inline ~MemBlock();

            //! corresponding interval if allocated; otherwise nullptr
            Interval *interval = nullptr;

            /*!
             * size for unallocated intervals; should be accessed only when
             * *interval* is nullptr
             */
            size_t size = 0;

            /*!
             * FreeBlockBySizeItem iter for unallocated intervals; should be
             * accessed only when *interval* is nullptr
             */
            const FreeBySizeIter& fsiter() const {
                return m_fs_iter.get();
            }

            FreeBySizeIter& fsiter() {
                return m_fs_iter.get();
            }

            bool allocated() const {
                return interval;
            }

            private:
                IncompleteObjStorageMock<
                    FreeBySizeIter, std::set<int>::iterator> m_fs_iter;
        };

        struct FreeBlockBySizeItem {
            bool blk_iter_valid;
            size_t size;
            MemBlockIter blk_iter;

            FreeBlockBySizeItem(size_t s, MemBlockIter bi):
                blk_iter_valid{true}, size{s}, blk_iter{bi}
            {}

            static FreeBlockBySizeItem make_for_compare_size(size_t s) {
                FreeBlockBySizeItem ret{s, {}};
                ret.blk_iter_valid = false;
                return ret;
            }

            bool operator < (const FreeBlockBySizeItem &rhs) const {
                return size < rhs.size || (size == rhs.size &&
                        (!blk_iter_valid ||
                         (rhs.blk_iter_valid && &*blk_iter < &*rhs.blk_iter)));

            }
        };

        /*!
         * \brief make an AllocResult at given position
         */
        AllocResult make_alloc_result(MemBlockIter pos, Interval *interval);

        /*!
         * \brief insert a free MemBlock before given position; try to merge
         *      with existing free blocks
         * \param size size of free block; if it is zero, no insertion would be
         * performed
         */
        void insert_free_blk_before(MemBlockIter pos, size_t size);

        /*!
         * \brief insert a free MemBlock after given position
         */
        void insert_free_blk_after(MemBlockIter pos, size_t size) {
            mgb_assert(pos != m_mem_block.end());
            return insert_free_blk_before(++ pos, size);
        }
};
StaticMemAllocPushdown::BestfitPrealloc::MemBlock::~MemBlock() = default;
StaticMemAllocPushdown::BestfitPrealloc::MemBlock::MemBlock() = default;

#define PREALLOC_MEM(ret) \
    ret StaticMemAllocPushdown::BestfitPrealloc

#define PREALLOC_MEM_TR(ret) \
    StaticMemAllocPushdown::BestfitPrealloc::ret \
    StaticMemAllocPushdown::BestfitPrealloc

/*!
 * \brief proxy class to extract information from allocation results
 */
PREALLOC_MEM(class)::AllocResult {
    bool m_valid = false;
    MemBlockIter m_iter;
    Interval *m_prev = nullptr, *m_next = nullptr;

    friend class BestfitPrealloc;

    AllocResult(const MemBlockIter &iter,
            Interval *prev,  Interval *next):
        m_valid(true),
        m_iter(iter), m_prev(prev), m_next(next)
    {}


    public:
        AllocResult() = default;

        /*!
         * \brief get the interval whose address is less than newly
         *      allocated one, or nullptr if the new interval is at beginning
         */
        Interval* prev() const {
            return m_prev;
        }

        /*!
         * \brief get the interval whose address is greater than newly
         *      allocated one, or nullptr if the new interval is at end
         */
        Interval* next() const {
            return m_next;
        }
};

PREALLOC_MEM_TR(AllocResult)::alloc(Interval *interval) {
    if (m_free_by_size.empty()) {
        auto iter = m_mem_block.insert(m_mem_block.end(), MemBlock());
        return make_alloc_result(iter, interval);
    }

    auto iter = m_free_by_size.lower_bound(
            FreeBlockBySizeItem::make_for_compare_size(interval->size));
    if (iter != m_free_by_size.end()) {
        mgb_assert(iter->size >= interval->size);
    } else {
        // get largest block and grow to size
        -- iter;
        mgb_assert(iter->size < interval->size);
        iter->blk_iter->size = interval->size;
    }
    auto blkpos = iter->blk_iter;
    m_free_by_size.erase(iter);

    mgb_assert(!blkpos->allocated());

    auto free_size = safe_sub(blkpos->size, interval->size);
    // mark allocated result before inserting block
    blkpos->interval = interval;
    insert_free_blk_after(blkpos, free_size);
    return make_alloc_result(blkpos, interval);
}

PREALLOC_MEM_TR(AllocResult)::alloc_overwrite(
        AllocResult &dest, size_t offset, Interval *interval) {
    auto iter = dest.m_iter;
    mgb_assert(dest.m_valid && iter->allocated());
    insert_free_blk_before(iter, offset);
    insert_free_blk_after(iter, safe_sub(
                iter->interval->size, offset + interval->size));
    dest.m_valid = false;
    return make_alloc_result(iter, interval);
}

PREALLOC_MEM_TR(AllocResult)::make_alloc_result(
        MemBlockIter pos, Interval *interval) {
    pos->interval = interval;
    pos->fsiter() = {};
    pos->size = 0;

    Interval *iprev = nullptr, *inext = nullptr;
    // find prev allocated interval
    {
        auto p = pos;
        if (!m_mem_block.empty() && p != m_mem_block.begin()) {
            -- p;
            if (p != m_mem_block.begin() && !p->allocated()) {
                -- p;
                mgb_assert(p->allocated(), "found adjacent free blocks");
            }
            if (p->allocated())
                iprev = p->interval;
        }
    }
    // find next allocated interval
    {
        auto p = pos;
        ++ p;
        if (p != m_mem_block.end() && !p->allocated()) {
            ++ p;
            if (p != m_mem_block.end())
                mgb_assert(p->allocated(), "found adjacent free blocks");
        }
        if (p != m_mem_block.end() && p->allocated())
            inext = p->interval;
    }

    return {pos, iprev, inext};
}

PREALLOC_MEM(void)::free(AllocResult &alloc_rst) {
    mgb_assert(alloc_rst.m_valid);
    auto iter = alloc_rst.m_iter;
    mgb_assert(iter->allocated());
    auto size = iter->interval->size;
    auto pos = iter;
    ++ pos;
    m_mem_block.erase(iter);
    alloc_rst.m_valid = false;
    insert_free_blk_before(pos, size);
}

PREALLOC_MEM(void)::insert_free_blk_before(MemBlockIter pos, size_t size) {
    auto rm = [this](MemBlockIter it) {
        mgb_assert(!it->allocated());
        m_free_by_size.erase(it->fsiter());
        m_mem_block.erase(it);
    };

    // merge with next
    {
        auto inext = pos;
        if (inext != m_mem_block.end() && !inext->allocated()) {
            ++ pos;
            size += inext->size;
            rm(inext);
        }
    }

    // merge with prev
    if (!m_mem_block.empty() && pos != m_mem_block.begin()) {
        auto iprev = pos;
        -- iprev;
        if (!iprev->allocated()) {
            size += iprev->size;
            rm(iprev);
        }
    }

    if (!size)
        return;

    auto blk_iter = m_mem_block.insert(pos, MemBlock());
    auto rst_s = m_free_by_size.insert({size, blk_iter});
    mgb_assert(rst_s.second);
    blk_iter->size = size;
    blk_iter->fsiter() = rst_s.first;
}

#undef PREALLOC_MEM
#undef PREALLOC_MEM_TR

/* ======================== StaticMemAllocPushdown ======================== */

void StaticMemAllocPushdown::init_topo_order() {
    BestfitPrealloc prealloc;
    std::vector<BestfitPrealloc::AllocResult> alloc_result(m_interval.size());

    BestFitHelper helper;
    helper.alloc = [&](Interval *p) {
        alloc_result.at(p->id) = prealloc.alloc(p);
    };
    helper.alloc_overwrite = [&](Interval *dest,
            size_t offset, Interval *p) {
        alloc_result.at(p->id) = prealloc.alloc_overwrite(
                alloc_result.at(dest->id), offset, p);
    };
    helper.free = [&](Interval *p) {
        prealloc.free(alloc_result.at(p->id));
    };

    helper.run(m_interval);

    // get topo order
    m_interval_below.clear();
    m_interval_below.resize(m_interval.size());
    for (auto i: m_interval) {
        auto &&rst = alloc_result.at(i->id);

        if (Interval* p = rst.next())
            m_interval_below[p->id].push_back(i);

        if (Interval* p = rst.prev())
            m_interval_below[i->id].push_back(p);
    }
}

size_t StaticMemAllocPushdown::get_interval_addr_end(Interval *interval) {
    if (interval->addr_begin != INVALID)
        return interval->addr_end();

    auto ow_root = interval->is_overwrite_root() ? interval :
        interval->overwrite_dest_root();
    mgb_assert(!ow_root->offset_in_overwrite_dest_root());
    size_t addr = 0;
    for (auto i = ow_root; i; i = i->overwrite_src()) {
        mgb_assert(i == ow_root || i->overwrite_dest_root() == ow_root);
        auto offset = i->offset_in_overwrite_dest_root();
        for (auto j: m_interval_below[i->id]) {
            auto cur = get_interval_addr_end(j);
            if (cur >= offset)
                update_max(addr, cur - offset);
        }
    }

    addr = align(addr);
    for (auto i = ow_root; i; i = i->overwrite_src()) {
        i->addr_begin = addr + i->offset_in_overwrite_dest_root();
    }
    mgb_assert(interval->addr_begin != INVALID);
    update_max(m_peak_usage, align(ow_root->addr_end()));
    return interval->addr_end();
}

void StaticMemAllocPushdown::do_solve() {
    m_peak_usage = 0;

    init_topo_order();

    for (auto i: m_interval)
        get_interval_addr_end(i);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

