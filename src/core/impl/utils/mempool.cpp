/**
 * \file src/core/impl/utils/mempool.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/mempool.h"
#include "megbrain/common.h"

using namespace mgb;

MemPoolStorage::MemPoolStorage() noexcept = default;
MemPoolStorage::MemPoolStorage(MemPoolStorage &&rhs) noexcept = default;
MemPoolStorage::~MemPoolStorage() noexcept = default;
MemPoolStorage& MemPoolStorage::operator = (
        MemPoolStorage &&rhs) noexcept = default;

void MemPoolStorage::swap(MemPoolStorage &other) {
    m_buf.swap(other.m_buf);
    m_free.swap(other.m_free);
    std::swap(m_disable_freelist, other.m_disable_freelist);
    std::swap(m_cur_buf_pos, other.m_cur_buf_pos);
    std::swap(m_cur_buf_size_bytes, other.m_cur_buf_size_bytes);
}

void *MemPoolStorage::alloc(size_t elem_size) {
    constexpr size_t MAX_BUF_SIZE = 32 * 1024; // max 32 KiB per buf
    if (!m_free.empty()) {
        auto ptr = m_free.back();
        m_free.pop_back();
        return ptr;
    }
    if (m_cur_buf_pos >= m_cur_buf_size_bytes) {
        auto buf_size = m_cur_buf_size_bytes;
        if (!buf_size) {
            buf_size = elem_size * 2;
        }
        buf_size = std::min(buf_size * 2, 2048 * elem_size);
        if (buf_size > MAX_BUF_SIZE) {
            buf_size = std::max<size_t>(
                    MAX_BUF_SIZE - MAX_BUF_SIZE % elem_size,
                    16 * elem_size);
        }
        m_buf.emplace_back(new uint8_t[buf_size]);
        m_cur_buf_pos = 0;
        m_cur_buf_size_bytes = buf_size;
    }
    auto ptr = m_buf.back().get() + m_cur_buf_pos;
    m_cur_buf_pos += elem_size;
    return ptr;
}

void MemPoolStorage::free(void *ptr) {
    if (!m_disable_freelist)
        m_free.push_back(ptr);
}

void MemPoolStorage::reorder_free() {
    std::sort(m_free.begin(), m_free.end());
}

void MemPoolStorage::clear() {
    m_cur_buf_pos = m_cur_buf_size_bytes = 0;
    m_buf.clear();
    m_free.clear();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

