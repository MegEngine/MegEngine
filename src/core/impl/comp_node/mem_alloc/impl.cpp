/**
 * \file src/core/impl/comp_node/mem_alloc/impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain_build_config.h"

#include "./impl.h"
#include "megbrain/utils/arith_helper.h"

#include <algorithm>

using namespace mgb;
using namespace mem_alloc;

/* ===================== MemAllocImplHelper ===================== */

MemAllocImplHelper::MemAddr MemAllocImplHelper::do_alloc(
        size_t size, bool allow_from_parent, bool log_stat_on_error) {

    mgb_assert(size);
    m_mutex.lock();

    auto iter = m_free_blk_size.lower_bound(FreeBlock{MemAddr{0, 0}, size});
    if (iter == m_free_blk_size.end()) {
        m_mutex.unlock();
        if (!allow_from_parent) {
            if (log_stat_on_error) {
                print_memory_state();
            }
            mgb_throw(MemAllocError,
                    "out of memory while requesting %zu bytes; you can try "
                    "setting MGB_CUDA_RESERVE_MEMORY to reserve all memory. "
                    "If there are dynamic variables, you can also try enabling "
                    "graph option `enable_grad_var_static_reshape` so "
                    "some gradient variables can be statically allocated",
                    size);
        }
        return alloc_from_parent(size);
    }

    size_t remain = iter->first.size - size;
    auto alloc_addr = iter->first.addr;
    m_free_blk_addr.erase(iter->second.aiter);
    m_free_blk_size.erase(iter);

    if (remain)
        insert_free_unsafe({alloc_addr + size, remain});

    m_mutex.unlock();
    return alloc_addr;
}

void MemAllocImplHelper::merge_free_unsafe(FreeBlock block) {
    auto iter = m_free_blk_addr.lower_bound(block.addr.addr);

    // merge with previous
    if (!block.addr.is_head && iter != m_free_blk_addr.begin()) {
        auto iprev = iter;
        -- iprev;
        if (iprev->first + iprev->second.size == block.addr.addr) {
            block.addr.addr = iprev->first;
            block.addr.is_head = iprev->second.is_head;
            block.size += iprev->second.size;
            m_free_blk_size.erase(iprev->second.siter);
            m_free_blk_addr.erase(iprev);
        }
    }

    // merge with next
    if (iter != m_free_blk_addr.end()) {
        mgb_assert(iter->first >= block.end());
        if (!iter->second.is_head && block.end() == iter->first) {
            block.size += iter->second.size;
            m_free_blk_size.erase(iter->second.siter);
            m_free_blk_addr.erase(iter);
        }
    }

    insert_free_unsafe(block);
}

void MemAllocImplHelper::insert_free_unsafe(const FreeBlock &block) {
    auto rst0 = m_free_blk_size.insert({block, {}});
    auto rst1 = m_free_blk_addr.insert({block.addr.addr, {}});
    mgb_assert(rst0.second & rst1.second);
    rst0.first->second.aiter = rst1.first;
    rst1.first->second.is_head = block.addr.is_head;
    rst1.first->second.size = block.size;
    rst1.first->second.siter = rst0.first;
}

void MemAllocImplHelper::print_memory_state() {
    auto stat = get_free_memory();
    MGB_MARK_USED_VAR(stat);
    mgb_log("device memory allocator stats: %s: "
            "used=%zu free={tot:%zu, min_blk:%zu, max_blk:%zu, nr:%zu}",
            get_name().c_str(), get_used_memory(),
            stat.tot, stat.min, stat.max, stat.nr_blk);
}

FreeMemStat MemAllocImplHelper::get_free_memory_self_unsafe() {
    FreeMemStat stat{0, std::numeric_limits<size_t>::max(), 0, 0};
    for (auto &&i: m_free_blk_size) {
        auto size = i.first.size;
        stat.tot += size;
        stat.min = std::min(stat.min, size);
        stat.max = std::max(stat.max, size);
        ++ stat.nr_blk;
    }
    return stat;
}

FreeMemStat MemAllocImplHelper::get_free_memory() {
    MGB_LOCK_GUARD(m_mutex);
    return get_free_memory_self_unsafe();
}

/* ===================== StreamMemAllocImpl ===================== */
std::string StreamMemAllocImpl::get_name() const {
    return ssprintf("stream allocator %d@%d",
            m_stream_id, m_dev_alloc->device());
}

void* StreamMemAllocImpl::alloc(size_t size) {
    size = get_aligned_power2(size, m_dev_alloc->alignment());
    auto addr = do_alloc(size, true);
    MGB_LOCK_GUARD(m_mutex);
    m_allocated_blocks[addr.addr_ptr()] = {addr.is_head, size};
    return addr.addr_ptr();
}

MemAllocImplHelper::MemAddr StreamMemAllocImpl::alloc_from_parent(size_t size) {
    auto addr = m_dev_alloc->alloc(size);
    MGB_LOCK_GUARD(m_mutex);
    m_allocated_blocks[addr.addr_ptr()] = {addr.is_head, size};
    return addr;
}

void StreamMemAllocImpl::free(void *addr) {
    MGB_LOCK_GUARD(m_mutex);
    auto iter = m_allocated_blocks.find(addr);
    mgb_assert(iter != m_allocated_blocks.end(),
            "releasing bad pointer: %p", addr);
    FreeBlock fb{
        MemAddr{iter->second.is_head, reinterpret_cast<size_t>(addr)},
        iter->second.size};
    m_allocated_blocks.erase(iter);
    merge_free_unsafe(fb);
}

void StreamMemAllocImpl::get_mem_info(size_t& free, size_t& tot) {
    auto&& stat = get_free_memory();
    free = stat.tot;
    auto used = get_used_memory();
    tot = free + used;
}

size_t StreamMemAllocImpl::get_used_memory() {
    MGB_LOCK_GUARD(m_mutex);
    size_t size = 0;
    for (auto &&i: m_allocated_blocks)
        size += i.second.size;
    return size;
}

FreeMemStat StreamMemAllocImpl::get_free_memory_dev() {
    return m_dev_alloc->get_free_memory_dev();
}

/* ===================== DevMemAllocImpl ===================== */

StreamMemAlloc* DevMemAllocImpl::add_stream(StreamKey stream) {
    MGB_LOCK_GUARD(m_mutex);
    auto&& ptr = m_stream_alloc[stream];
    if (!ptr)
        ptr.reset(new StreamMemAllocImpl(this, m_stream_alloc.size() - 1));
    return ptr.get();
}

MemAllocImplHelper::MemAddr DevMemAllocImpl::alloc(size_t size) {
    auto addr = do_alloc(size, true);
    m_used_size += size;
    return addr;
}

MemAllocImplHelper::MemAddr DevMemAllocImpl::alloc_from_parent(size_t size) {
    // pre-allocate to size_upper
    auto &&prconf = prealloc_config();
    auto size_upper = std::max<size_t>(std::max(size, prconf.min_req),
            m_tot_allocated_from_raw * prconf.growth_factor);
    size_upper = std::min(size_upper, size + prconf.max_overhead);
    size_upper = get_aligned_power2(size_upper, prconf.alignment);

    auto ptr = m_raw_allocator->alloc(size_upper);

    if (!ptr && size_upper > size) {
        // failed to allocate; do not pre-allocate and try again
        size_upper = size;
        ptr = m_raw_allocator->alloc(size_upper);
    }

    if (!ptr) {
        // gather free memory from other streams on this device and try again
        auto get = gather_stream_free_blk_and_release_full();
        MGB_MARK_USED_VAR(get);
        mgb_log("could not allocate memory on device %d; "
                "try to gather free blocks from child streams, "
                "got %.2fMiB(%zu bytes).",
                m_device, get / 1024.0 / 1024, get);

        ptr = m_raw_allocator->alloc(size_upper);

        if (!ptr) {
            // sync other devices in the hope that they can release memory on
            // this device; then try again
            auto&& runtime_policy = device_runtime_policy();
            auto callback = [&runtime_policy](CompNode cn) {
                if (cn.device_type() == runtime_policy->device_type()) {
                    int dev = cn.locator().device;
                    runtime_policy->device_synchronize(dev);
                }
            };
            MGB_TRY { CompNode::foreach (callback); }
            MGB_FINALLY({ m_runtime_policy->set_device(m_device); });

            {
                // sleep to wait for async dealloc
                using namespace std::literals;
                std::this_thread::sleep_for(0.2s);
            }
            get = gather_stream_free_blk_and_release_full();
            mgb_log("device %d: sync all device and try to "
                    "allocate again: got %.2fMiB(%zu bytes).",
                    m_device, get / 1024.0 / 1024, get);

            ptr = m_raw_allocator->alloc(size_upper);

            if (!ptr) {
                // try to alloc from newly gathered but unreleased (i.e. thoses
                // that are not full chunks from raw allocator) chunks
                //
                // exception would be thrown from here
                auto t = do_alloc(size, false, true);
                m_used_size += size;
                return t;
            }
        }
    }

    MGB_LOCK_GUARD(m_mutex);
    m_alloc_from_raw[ptr] = size_upper;
    auto ptr_int = reinterpret_cast<size_t>(ptr);
    if (size_upper > size) {
        insert_free_unsafe({
                MemAddr{false, ptr_int + size},
                size_upper - size});
    }
    m_tot_allocated_from_raw += size_upper;
    return {true, ptr_int};
}

size_t DevMemAllocImpl::gather_stream_free_blk_and_release_full() {
    size_t free_size = 0;
    std::vector<void*> to_free_by_raw;

    MGB_LOCK_GUARD(m_mutex);
    auto return_full_free_blk_unsafe = [&](MemAllocImplHelper* alloc) {
        auto&& free_blk_size = alloc->m_free_blk_size;
        auto&& free_blk_addr = alloc->m_free_blk_addr;
        using Iter = decltype(m_free_blk_size.begin());
        for (Iter i = free_blk_size.begin(), inext; i != free_blk_size.end();
                i = inext) {
            inext = i;
            ++ inext;
            auto &&blk = i->first;
            if (blk.addr.is_head) {
                auto riter = m_alloc_from_raw.find(blk.addr.addr_ptr());
                mgb_assert(riter != m_alloc_from_raw.end() &&
                        blk.size <= riter->second);
                if (blk.size == riter->second) {
                    to_free_by_raw.push_back(blk.addr.addr_ptr());
                    free_size += blk.size;
                    auto j = i->second.aiter;
                    free_blk_size.erase(i);
                    free_blk_addr.erase(j);
                    m_alloc_from_raw.erase(riter);
                }
            }
        }
    };

    if (auto child = get_single_child_stream_unsafe()) {
        MGB_LOCK_GUARD(child->m_mutex);
        return_full_free_blk_unsafe(child);
        mgb_assert(free_size <= m_used_size.load());
        m_used_size -= free_size;
    } else {
        size_t gathered_size = 0;
        for (auto &&pair: m_stream_alloc) {
            auto ch = pair.second.get();
            auto &&chmtx = ch->m_mutex;

            MGB_LOCK_GUARD(chmtx);
            for (auto &&i: ch->m_free_blk_size) {
                merge_free_unsafe(i.first);
                gathered_size += i.first.size;
            }
            ch->m_free_blk_addr.clear();
            ch->m_free_blk_size.clear();
        }
        mgb_assert(gathered_size <= m_used_size.load());
        m_used_size -= gathered_size;
    }

    return_full_free_blk_unsafe(this);
    m_tot_allocated_from_raw -= free_size;

    // we have to sync to ensure no kernel on the child stream still uses
    // freed memory
    m_runtime_policy->device_synchronize(m_device);

    for (auto i: to_free_by_raw)
        m_raw_allocator->free(i);

    return free_size;
}

DevMemAllocImpl::DevMemAllocImpl(
        int device, size_t reserve_size,
        const std::shared_ptr<mem_alloc::RawAllocator>& raw_allocator,
        const std::shared_ptr<mem_alloc::DeviceRuntimePolicy>&
                runtime_policy)
        : m_device(device),
          m_raw_allocator(raw_allocator),
          m_runtime_policy(runtime_policy) {
    if (reserve_size) {
        auto ptr = m_raw_allocator->alloc(reserve_size);
        mgb_throw_if(!ptr, MemAllocError,
                "failed to reserve memory for %zu bytes", reserve_size);
        insert_free_unsafe({
                MemAddr{true, reinterpret_cast<size_t>(ptr)},
                reserve_size});

        m_alloc_from_raw[ptr] = reserve_size;
        m_tot_allocated_from_raw += reserve_size;
    }
}

void DevMemAllocImpl::print_memory_state() {
    MemAllocImplHelper::print_memory_state();
    for (auto &&i: m_stream_alloc)
        i.second->print_memory_state();
}

FreeMemStat DevMemAllocImpl::get_free_memory_dev() {
    MGB_LOCK_GUARD(m_mutex);
    auto ret = get_free_memory_self_unsafe();
    for (auto&&i : m_stream_alloc) {
        MGB_LOCK_GUARD(i.second->m_mutex);
        auto cur = i.second->get_free_memory_self_unsafe();
        ret.tot += cur.tot;
        ret.min = std::min(ret.min, cur.min);
        ret.max = std::max(ret.max, cur.max);
        ret.nr_blk += cur.nr_blk;
    }
    return ret;
}

void DevMemAllocImpl::insert_free_unsafe(const FreeBlock &block) {
    if (auto child = get_single_child_stream_unsafe()) {
        {
            MGB_LOCK_GUARD(child->m_mutex);
            child->insert_free_unsafe(block);
        }
        m_used_size += block.size;
    } else {
        MemAllocImplHelper::insert_free_unsafe(block);
    }
}

StreamMemAllocImpl* DevMemAllocImpl::get_single_child_stream_unsafe() {
    if (m_stream_alloc.size() == 1) {
        return m_stream_alloc.begin()->second.get();
    }
    return nullptr;
}

DevMemAllocImpl::~DevMemAllocImpl() {
    for (auto &&i: m_alloc_from_raw)
        m_raw_allocator->free(i.first);
}

std::unique_ptr<SimpleCachingAlloc> SimpleCachingAlloc::make(std::unique_ptr<RawAllocator> raw_alloc) {
    return std::make_unique<SimpleCachingAllocImpl>(std::move(raw_alloc));
}

SimpleCachingAllocImpl::SimpleCachingAllocImpl(std::unique_ptr<RawAllocator> raw_alloc)
        : m_raw_alloc(std::move(raw_alloc)) {}

void* SimpleCachingAllocImpl::alloc(size_t size) {
    size = get_aligned_power2(size, m_alignment);
    auto&& addr = do_alloc(size, true);
    auto ptr = addr.addr_ptr();
    MGB_LOCK_GUARD(m_mutex);
    m_allocated_blocks[ptr] = {addr.is_head, size};
    m_used_size += size;
    return ptr;
}

void SimpleCachingAllocImpl::free(void* ptr) {
    MGB_LOCK_GUARD(m_mutex);
    auto&& iter = m_allocated_blocks.find(ptr);
    mgb_assert(iter != m_allocated_blocks.end(),
            "releasing bad pointer: %p", ptr);
    auto size = iter->second.size;
    FreeBlock fb{MemAddr{iter->second.is_head, reinterpret_cast<size_t>(ptr)}, size};
    m_allocated_blocks.erase(iter);
    merge_free_unsafe(fb);
    m_used_size -= size;
}

SimpleCachingAllocImpl::~SimpleCachingAllocImpl() {
    for (auto&& ptr_size : m_alloc_from_raw) {
        m_raw_alloc->free(ptr_size.first);
    }
}

SimpleCachingAllocImpl::MemAddr SimpleCachingAllocImpl::alloc_from_parent(size_t size) {
    void* ptr = m_raw_alloc->alloc(size);
    m_alloc_from_raw[ptr] = size;
    return {true, reinterpret_cast<size_t>(ptr)};
}

std::string SimpleCachingAllocImpl::get_name() const {
    return "SimpleCachingAllocImpl";
}

size_t SimpleCachingAllocImpl::get_used_memory() {
    return m_used_size;
}

FreeMemStat SimpleCachingAllocImpl::get_free_memory_dev() {
    return get_free_memory();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
