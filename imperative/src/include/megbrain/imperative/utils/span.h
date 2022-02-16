/**
 * \file imperative/src/include/megbrain/imperative/utils/span.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <array>
#include <vector>

#include "megbrain/utils/small_vector.h"

namespace mgb::imperative {

/**
 * \brief wrapper for c-style array
 *
 * \tparam T value type
 */
template <typename T>
class Span {
private:
    const T* m_begin = nullptr;
    const T* m_end = nullptr;

public:
    Span() {}
    Span(const T* begin, const T* end) : m_begin{begin}, m_end{end} {}
    Span(const T* begin, size_t size) : Span(begin, begin + size) {}
    template <typename TContainer>
    Span(const TContainer& container) : Span(container.data(), container.size()) {}
    const T* begin() const { return m_begin; }
    const T* end() const { return m_end; }
    const T* data() const { return m_begin; }
    size_t size() const { return m_end - m_begin; }
    template <typename TContainer>
    TContainer copy_into() {
        return TContainer(m_begin, m_end);
    }
    const T& operator[](size_t idx) const { return m_begin[idx]; }
    const T& at(size_t idx) const { return m_begin[idx]; }
    const T& item() const {
        mgb_assert(
                m_end - m_begin == 1, "size mismatch: %zu vs %zu", (m_end - m_begin),
                (size_t)1);
        return m_begin[0];
    }

    template <size_t N>
    const std::array<T, N>& as_array() {
        mgb_assert(
                m_end - m_begin == N, "size mismatch: %zu vs %zu", (m_end - m_begin),
                N);
        return *reinterpret_cast<const std::array<T, N>*>(m_begin);
    }

    Span sub(size_t begin, size_t length) {
        mgb_assert(begin + length <= m_end - m_begin);
        return {m_begin + begin, length};
    }
};

}  // namespace mgb::imperative
