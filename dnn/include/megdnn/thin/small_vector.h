/**
 * \file dnn/include/megdnn/thin/small_vector.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
//===- llvm/ADT/SmallVector.h - 'Normally small' vectors --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the SmallVector class.
//
//===----------------------------------------------------------------------===//
/**
 * \file include/megdnn/thin/small_vector.h
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \brief thin megdnn function
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 */
#pragma once

#include "megdnn/arch.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>

#include "megdnn/internal/visibility_prologue.h"

namespace megdnn {

class SmallVectorBase {
protected:
    void *m_begin_ptr, *m_end_ptr, *m_capacity_ptr;

    MEGDNN_NORETURN static void on_invalid_at(size_t idx, size_t size);

protected:
    SmallVectorBase(void* first_elm, size_t size)
            : m_begin_ptr(first_elm),
              m_end_ptr(first_elm),
              m_capacity_ptr(static_cast<char*>(first_elm) + size) {}

    void grow_pod(void* first_elm_ptr, size_t min_sz_in_bytes,
                  size_t type_size);

public:
    size_t size_in_bytes() const {
        return size_t(static_cast<char*>(m_end_ptr) -
                      static_cast<char*>(m_begin_ptr));
    }

    size_t capacity_in_bytes() const {
        return size_t(static_cast<char*>(m_capacity_ptr) -
                      static_cast<char*>(m_begin_ptr));
    }

    bool empty() const { return m_begin_ptr == m_end_ptr; }
};
template <typename T, typename = void>
class SmallVectorTemplateCommon : public SmallVectorBase {
private:
    template <typename, unsigned>
    friend struct SmallVectorStorage;

    using U = typename std::aligned_storage<sizeof(T), alignof(T)>::type;

    U m_first_elm;

protected:
    SmallVectorTemplateCommon(size_t size)
            : SmallVectorBase(&m_first_elm, size) {}

    void grow_pod(size_t min_sz_in_bytes, size_t type_size) {
        SmallVectorBase::grow_pod(&m_first_elm, min_sz_in_bytes, type_size);
    }

    bool is_small() {
        return m_begin_ptr == static_cast<const void*>(&m_first_elm);
    }

    void reset_to_small() {
        m_begin_ptr = m_end_ptr = m_capacity_ptr = &m_first_elm;
    }

    void set_end(T* p) { m_end_ptr = p; }

public:
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;

    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;

    size_t capacity() const { return capacity_ptr() - begin(); }

protected:
    iterator capacity_ptr() { return static_cast<iterator>(m_capacity_ptr); }
    const_iterator capacity_ptr() const {
        return static_cast<const_iterator>(m_capacity_ptr);
    }

public:
    // forwarding iterator creation
    iterator begin() { return static_cast<iterator>(m_begin_ptr); }
    const_iterator begin() const {
        return static_cast<const_iterator>(m_begin_ptr);
    }
    const_iterator cbegin() const {
        return static_cast<const_iterator>(m_begin_ptr);
    }

    iterator end() { return static_cast<iterator>(m_end_ptr); }
    const_iterator end() const {
        return static_cast<const_iterator>(m_end_ptr);
    }
    const_iterator cend() const {
        return static_cast<const_iterator>(m_end_ptr);
    }

    reference at(size_type idx) {
        if (idx >= size()) {
            on_invalid_at(idx, size());
        }
        return begin()[idx];
    }
    const_reference at(size_type idx) const {
        if (idx >= size()) {
            on_invalid_at(idx, size());
        }
        return begin()[idx];
    }

    reference operator[](size_type idx) { return begin()[idx]; }
    const_reference operator[](size_type idx) const { return begin()[idx]; }

    reference front() { return begin()[0]; }
    const_reference front() const { return begin()[0]; }

    reference back() { return rbegin()[0]; }
    const_reference back() const { return rbegin()[0]; }

    // reverse iterator creation method.
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const {
        return const_reverse_iterator(end());
    }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const {
        return const_reverse_iterator(begin());
    }

    pointer data() { return pointer(begin()); }
    const_pointer data() const { return const_pointer(begin()); }

    size_type size() const { return end() - begin(); }
    size_type max_size() const {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }

    template <typename in_iter>
    in_iter find(in_iter first, in_iter last, const T& value) const {
        while (first != last) {
            if (*first == value)
                return first;
            ++first;
        }
        return last;
    }
};
template <typename T, bool is_pod>
class SmallVectorTemplateBase : public SmallVectorTemplateCommon<T> {
protected:
    SmallVectorTemplateBase(size_t size) : SmallVectorTemplateCommon<T>(size) {}

    static void destroy_range(T* start, T* end) {
        while (start != end) {
            --end;
            end->~T();
        }
    }

    template <typename It1, typename It2>
    static void uninitialized_move(It1 first, It1 last, It2 dest) {
        std::uninitialized_copy(std::make_move_iterator(first),
                                std::make_move_iterator(last), dest);
    }

    template <typename It1, typename It2>
    static void uninitialized_copy(It1 first, It1 last, It2 dest) {
        std::uninitialized_copy(first, last, dest);
    }

    void grow(size_t min_sz = 0);

public:
    void push_back(const T& _elm) {
        if (megdnn_unlikely(this->m_end_ptr >= this->m_capacity_ptr)) {
            T elm = _elm;
            this->grow();
            new (static_cast<void*>(this->end())) T(std::move(elm));
        } else {
            new (static_cast<void*>(this->end())) T(_elm);
        }
        this->set_end(this->end() + 1);
    }

    void push_back(T&& elm) {
        if (megdnn_unlikely(this->m_end_ptr >= this->m_capacity_ptr)) {
            this->grow();
        }
        new (static_cast<void*>(this->end())) T(std::move(elm));
        this->set_end(this->end() + 1);
    }

    void pop_back() {
        this->set_end(this->end() - 1);
        this->end()->~T();
    }
};
template <typename T, bool is_pod>
void SmallVectorTemplateBase<T, is_pod>::grow(size_t min_sz) {
    size_t cur_capacity = this->capacity();
    size_t cur_sz = this->size();
    size_t new_capacity = (cur_capacity + 2) * 2;
    if (new_capacity < min_sz) {
        new_capacity = min_sz;
    }
    T* elms = static_cast<T*>(malloc(new_capacity * sizeof(T)));

    this->uninitialized_move(this->begin(), this->end(), elms);

    this->destroy_range(this->begin(), this->end());

    if (!this->is_small()) {
        free(this->begin());
    }

    this->m_begin_ptr = elms;
    this->set_end(elms + cur_sz);
    this->m_capacity_ptr = this->begin() + new_capacity;
}

template <typename T>
class SmallVectorTemplateBase<T, true> : public SmallVectorTemplateCommon<T> {
protected:
    SmallVectorTemplateBase(size_t size) : SmallVectorTemplateCommon<T>(size) {}

    static void destroy_range(T*, T*) {}

    template <typename It1, typename It2>
    static void uninitialized_move(It1 first, It1 last, It2 dest) {
        uninitialized_copy(first, last, dest);
    }

    template <typename It1, typename It2>
    static void uninitialized_copy(It1 first, It1 last, It2 dest) {
        std::uninitialized_copy(first, last, dest);
    }

    template <typename T1, typename T2>
    static void uninitialized_copy(
            T1* first, T1* last, T2* dest,
            typename std::enable_if<std::is_same<
                    typename std::remove_const<T1>::type, T2>::value>::type* =
                    nullptr) {
        if (first != last)
            memcpy(dest, first, (last - first) * sizeof(T));
    }

    void grow(size_t min_sz = 0) {
        this->grow_pod(min_sz * sizeof(T), sizeof(T));
    }

public:
    void push_back(const T& _elm) {
        if (megdnn_unlikely(this->m_end_ptr >= this->m_capacity_ptr)) {
            T elm = _elm;
            this->grow();
            memcpy(this->end(), &elm, sizeof(T));
        } else {
            memcpy(this->end(), &_elm, sizeof(T));
        }
        this->set_end(this->end() + 1);
    }

    void pop_back() { this->set_end(this->end() - 1); }
};

/*!
 * \brief the implementation class of SmallVector
 *
 * SmallVector<T, N> can be converted to SmallVectorImpl<T> to erase N
 */
template <typename T>
class SmallVectorImpl
        : public SmallVectorTemplateBase<T, std::is_pod<T>::value> {
    using SuperClass = SmallVectorTemplateBase<T, std::is_pod<T>::value>;

public:
    using iterator = typename SuperClass::iterator;
    using const_iterator = typename SuperClass::const_iterator;
    using size_type = typename SuperClass::size_type;

protected:
    explicit SmallVectorImpl(unsigned n)
            : SmallVectorTemplateBase<T, std::is_pod<T>::value>(n * sizeof(T)) {
    }

public:
    SmallVectorImpl(const SmallVectorImpl&) = delete;

    ~SmallVectorImpl() {
        this->destroy_range(this->begin(), this->end());

        if (!this->is_small())
            free(this->begin());
    }

    void clear() {
        this->destroy_range(this->begin(), this->end());
        this->m_end_ptr = this->m_begin_ptr;
    }

    void resize(size_type n) {
        if (n < this->size()) {
            this->destroy_range(this->begin() + n, this->end());
            this->set_end(this->begin() + n);
        } else if (n > this->size()) {
            if (this->capacity() < n)
                this->grow(n);
            for (auto it = this->end(), end = this->begin() + n; it != end;
                 ++it)
                new (&*it) T();
            this->set_end(this->begin() + n);
        }
    }

    void resize(size_type n, const T& _nv) {
        T nv = _nv;
        if (n < this->size()) {
            this->destroy_range(this->begin() + n, this->end());
            this->set_end(this->begin() + n);
        } else if (n > this->size()) {
            if (this->capacity() < n)
                this->grow(n);
            std::uninitialized_fill(this->end(), this->begin() + n, nv);
            this->set_end(this->begin() + n);
        }
    }

    void reserve(size_type n) {
        if (this->capacity() < n) {
            this->grow(n);
        }
    }

    T pop_back_val() {
        T result = std::move(this->back());
        this->pop_back();
        return result;
    }

    void swap(SmallVectorImpl<T>& rhs);

    /// Add the specified range to the end of the SmallVector.
    template <typename in_iter,
              typename = typename std::enable_if<std::is_convertible<
                      typename std::iterator_traits<in_iter>::iterator_category,
                      std::input_iterator_tag>::value>::type>
    void append(in_iter in_start, in_iter in_end) {
        size_type num_inputs = std::distance(in_start, in_end);
        // Grow allocated space if needed.
        if (num_inputs > size_type(this->capacity_ptr() - this->end()))
            this->grow(this->size() + num_inputs);

        // Copy the new elements over.
        this->uninitialized_copy(in_start, in_end, this->end());
        this->set_end(this->end() + num_inputs);
    }

    /// Add the specified range to the end of the SmallVector.
    void append(size_type num_inputs, const T& _elm) {
        T elm = _elm;
        // Grow allocated space if needed.
        if (num_inputs > size_type(this->capacity_ptr() - this->end()))
            this->grow(this->size() + num_inputs);

        // Copy the new elements over.
        std::uninitialized_fill_n(this->end(), num_inputs, elm);
        this->set_end(this->end() + num_inputs);
    }

    void append(std::initializer_list<T> init_list) {
        append(init_list.begin(), init_list.end());
    }

    // FIXME: Consider assigning over existing elements, rather than clearing &
    // re-initializing them - for all assign(...) variants.

    void assign(size_type num_elms, const T& _elm) {
        T elm = _elm;
        clear();
        if (this->capacity() < num_elms)
            this->grow(num_elms);
        this->set_end(this->begin() + num_elms);
        std::uninitialized_fill(this->begin(), this->end(), elm);
    }

    template <typename in_iter,
              typename = typename std::enable_if<std::is_convertible<
                      typename std::iterator_traits<in_iter>::iterator_category,
                      std::input_iterator_tag>::value>::type>
    void assign(in_iter in_start, in_iter in_end) {
        clear();
        append(in_start, in_end);
    }

    void assign(std::initializer_list<T> init_list) {
        clear();
        append(init_list);
    }

    iterator erase(const_iterator cit) {
        // Just cast away constness because this is a non-const member function.
        iterator it = const_cast<iterator>(cit);
        iterator n = it;
        // Shift all elms down one.
        std::move(it + 1, this->end(), it);
        // Drop the last elm.
        this->pop_back();
        return (n);
    }

    iterator erase(const_iterator c_first, const_iterator c_last) {
        // Just cast away constness because this is a non-const member function.
        iterator first = const_cast<iterator>(c_first);
        iterator last = const_cast<iterator>(c_last);
        iterator n = first;
        // Shift all elms down.
        iterator it = std::move(last, this->end(), first);
        // Drop the last elms.
        this->destroy_range(it, this->end());
        this->set_end(it);
        return (n);
    }

    iterator insert(iterator it, T&& elm) {
        if (it == this->end()) {  // Important special case for empty vector.
            this->push_back(std::move(elm));
            return this->end() - 1;
        }

        if (this->m_end_ptr >= this->m_capacity_ptr) {
            size_t elm_idx = it - this->begin();
            this->grow();
            it = this->begin() + elm_idx;
        }

        new (static_cast<void*>(this->end())) T(std::move(this->back()));
        // Push everything else over.
        std::move_backward(it, this->end() - 1, this->end());
        this->set_end(this->end() + 1);

        // If we just moved the element we're inserting, be sure to update
        // the reference.
        T* elm_ptr = &elm;
        if (it <= elm_ptr && elm_ptr < this->m_end_ptr)
            ++elm_ptr;

        *it = std::move(*elm_ptr);
        return it;
    }

    iterator insert(iterator it, const T& _elm) {
        if (it == this->end()) {  // Important special case for empty vector.
            this->push_back(_elm);
            return this->end() - 1;
        }
        T elm = _elm;
        if (this->m_end_ptr >= this->m_capacity_ptr) {
            size_t elm_idx = it - this->begin();
            this->grow();
            it = this->begin() + elm_idx;
        }
        new (static_cast<void*>(this->end())) T(std::move(this->back()));
        // Push everything else over.
        std::move_backward(it, this->end() - 1, this->end());
        this->set_end(this->end() + 1);

        // If we just moved the element we're inserting, be sure to update
        // the reference.
        const T* elm_ptr = &elm;
        if (it <= elm_ptr && elm_ptr < this->m_end_ptr)
            ++elm_ptr;

        *it = *elm_ptr;
        return it;
    }

    iterator insert(iterator it, size_type num_to_insert, const T& _elm) {
        // Convert iterator to elm# to avoid invalidating iterator
        // when we reserve()
        size_t elm_idx = it - this->begin();

        if (it == this->end()) {  // Important special case for empty vector.
            append(num_to_insert, _elm);
            return this->begin() + elm_idx;
        }

        T elm = _elm;

        // Ensure there is enough space.
        reserve(this->size() + num_to_insert);

        // Uninvalidate the iterator.
        it = this->begin() + elm_idx;

        // If there are more elements between the insertion point and
        // the end of the range than there are being inserted,
        // we can use a simple approach to insertion.
        // Since we already reserved space, we know that this won't
        // reallocate the vector.
        if (size_t(this->end() - it) >= num_to_insert) {
            T* old_end = this->end();
            append(std::move_iterator<iterator>(this->end() - num_to_insert),
                   std::move_iterator<iterator>(this->end()));

            // Copy the existing elements that get replaced.
            std::move_backward(it, old_end - num_to_insert, old_end);

            std::fill_n(it, num_to_insert, elm);
            return it;
        }

        // Otherwise, we're inserting more elements than exist already,
        // and we're not inserting at the end.

        // Move over the elements that we're about to overwrite.
        T* old_end = this->end();
        this->set_end(this->end() + num_to_insert);
        size_t num_overwritten = old_end - it;
        this->uninitialized_move(it, old_end, this->end() - num_overwritten);

        // Replace the overwritten part.
        std::fill_n(it, num_overwritten, elm);

        // Insert the non-overwritten middle part.
        std::uninitialized_fill_n(old_end, num_to_insert - num_overwritten,
                                  elm);
        return it;
    }

    template <
            typename IterType,
            typename = typename std::enable_if<std::is_convertible<
                    typename std::iterator_traits<IterType>::iterator_category,
                    std::input_iterator_tag>::value>::type>
    iterator insert(iterator it, IterType from, IterType to) {
        // Convert iterator to elm# to avoid invalidating iterator
        // when we reserve()
        size_t elm_idx = it - this->begin();

        if (it == this->end()) {  // Important special case for empty vector.
            append(from, to);
            return this->begin() + elm_idx;
        }

        size_t num_to_insert = std::distance(from, to);

        // Ensure there is enough space.
        reserve(this->size() + num_to_insert);

        // Uninvalidate the iterator.
        it = this->begin() + elm_idx;

        // If there are more elements between the insertion point and
        // the end of the range than there are being inserted,
        // we can use a simple approach to insertion.
        // Since we already reserved space, we know that this won't
        // reallocate the vector.
        if (size_t(this->end() - it) >= num_to_insert) {
            T* old_end = this->end();
            append(std::move_iterator<iterator>(this->end() - num_to_insert),
                   std::move_iterator<iterator>(this->end()));

            // Copy the existing elements that get replaced.
            std::move_backward(it, old_end - num_to_insert, old_end);

            std::copy(from, to, it);
            return it;
        }

        // Otherwise, we're inserting more elements than exist already,
        // and we're not inserting at the end.

        // Move over the elements that we're about to overwrite.
        T* old_end = this->end();
        this->set_end(this->end() + num_to_insert);
        size_t num_overwritten = old_end - it;
        this->uninitialized_move(it, old_end, this->end() - num_overwritten);

        // Replace the overwritten part.
        for (T* iter = it; num_overwritten > 0; --num_overwritten) {
            *iter = *from;
            ++iter;
            ++from;
        }

        // Insert the non-overwritten middle part.
        this->uninitialized_copy(from, to, old_end);
        return it;
    }

    void insert(iterator it, std::initializer_list<T> init_list) {
        insert(it, init_list.begin(), init_list.end());
    }

    template <typename... ArgTypes>
    void emplace_back(ArgTypes&&... args) {
        if (megdnn_unlikely(this->m_end_ptr >= this->m_capacity_ptr)) {
            this->grow();
        }
        new (static_cast<void*>(this->end()))
                T(std::forward<ArgTypes>(args)...);
        this->set_end(this->end() + 1);
    }

    SmallVectorImpl& operator=(const SmallVectorImpl& rhs);

    SmallVectorImpl& operator=(SmallVectorImpl&& rhs);

    bool operator==(const SmallVectorImpl<T>& rhs) const {
        if (this->size() != rhs.size())
            return false;
        return std::equal(this->begin(), this->end(), rhs.begin());
    }

    bool operator!=(const SmallVectorImpl<T>& rhs) const {
        return !(*this == rhs);
    }

    bool operator<(const SmallVectorImpl<T>& rhs) const {
        return std::lexicographical_compare(this->begin(), this->end(),
                                            rhs.begin(), rhs.end());
    }
};

template <typename T>
void SmallVectorImpl<T>::swap(SmallVectorImpl<T>& rhs) {
    if (this == &rhs)
        return;

    // We can only avoid copying elements if neither vector is small.
    if (!this->is_small() && !rhs.is_small()) {
        std::swap(this->m_begin_ptr, rhs.m_begin_ptr);
        std::swap(this->m_end_ptr, rhs.m_end_ptr);
        std::swap(this->m_capacity_ptr, rhs.m_capacity_ptr);
        return;
    }
    if (rhs.size() > this->capacity())
        this->grow(rhs.size());
    if (this->size() > rhs.capacity())
        rhs.grow(this->size());

    // Swap the shared elements.
    size_t num_shared = this->size();
    if (num_shared > rhs.size())
        num_shared = rhs.size();
    for (size_type i = 0; i != num_shared; ++i)
        std::swap((*this)[i], rhs[i]);

    // Copy over the extra elms.
    if (this->size() > rhs.size()) {
        size_t elm_diff = this->size() - rhs.size();
        this->uninitialized_move(this->begin() + num_shared, this->end(),
                                 rhs.end());
        rhs.set_end(rhs.end() + elm_diff);
        this->destroy_range(this->begin() + num_shared, this->end());
        this->set_end(this->begin() + num_shared);
    } else if (rhs.size() > this->size()) {
        size_t elm_diff = rhs.size() - this->size();
        this->uninitialized_move(rhs.begin() + num_shared, rhs.end(),
                                 this->end());
        this->set_end(this->end() + elm_diff);
        this->destroy_range(rhs.begin() + num_shared, rhs.end());
        rhs.set_end(rhs.begin() + num_shared);
    }
}

template <typename T>
SmallVectorImpl<T>& SmallVectorImpl<T>::operator=(
        const SmallVectorImpl<T>& rhs) {
    if (this == &rhs)
        return *this;
    size_t rhs_sz = rhs.size();
    size_t cur_sz = this->size();
    if (cur_sz >= rhs_sz) {
        iterator new_end;
        if (rhs_sz) {
            new_end = std::copy(rhs.begin(), rhs.end(), this->begin());
        } else {
            new_end = this->begin();
        }
        this->destroy_range(new_end, this->end());
        this->set_end(new_end);
        return *this;
    }
    if (this->capacity() < rhs_sz) {
        // save time for no copy when growing
        this->destroy_range(this->begin(), this->end());
        this->set_end(this->begin());
        cur_sz = 0;
        this->grow(rhs_sz);
    } else if (cur_sz) {
        std::copy(rhs.begin(), rhs.begin() + cur_sz, this->begin());
    }
    std::uninitialized_copy(rhs.begin() + cur_sz, rhs.end(),
                            this->begin() + cur_sz);
    this->set_end(this->begin() + rhs_sz);
    return *this;
}

template <typename T>
SmallVectorImpl<T>& SmallVectorImpl<T>::operator=(SmallVectorImpl<T>&& rhs) {
    // avoid self assignment
    if (this == &rhs)
        return *this;

    // copy ptr when rhs is small
    if (!rhs.is_small()) {
        this->destroy_range(this->begin(), this->end());
        if (!this->is_small())
            free(this->begin());
        this->m_begin_ptr = rhs.m_begin_ptr;
        this->m_end_ptr = rhs.m_end_ptr;
        this->m_capacity_ptr = rhs.m_capacity_ptr;
        rhs.reset_to_small();
        return *this;
    }

    size_t rhs_sz = rhs.size();
    size_t cur_sz = this->size();
    if (cur_sz >= rhs_sz) {
        iterator new_end = this->begin();
        if (rhs_sz) {
            new_end = std::move(rhs.begin(), rhs.end(), new_end);
        }
        this->destroy_range(new_end, this->end());
        this->set_end(new_end);
        rhs.clear();
        return *this;
    }
    if (this->capacity() < rhs_sz) {
        this->destroy_range(this->begin(), this->end());
        this->set_end(this->begin());
        cur_sz = 0;
        this->grow(rhs_sz);
    } else if (cur_sz) {
        std::move(rhs.begin(), rhs.begin() + cur_sz, this->begin());
    }

    this->uninitialized_move(rhs.begin() + cur_sz, rhs.end(),
                             this->begin() + cur_sz);

    this->set_end(this->begin() + rhs_sz);

    rhs.clear();
    return *this;
}
template <typename T, unsigned N>
struct SmallVectorStorage {
    typename SmallVectorTemplateCommon<T>::U inline_elms[N - 1];
};
template <typename T>
struct SmallVectorStorage<T, 1> {};
template <typename T>
struct SmallVectorStorage<T, 0> {};

/*!
 * \brief This is a 'vector' (really, a variable-sized array), optimized for the
 *      case when the array is small.
 *
 * It contains some number of elements in-place,
 * which allows it to avoid heap allocation when the actual number of elements
 * is below that threshold. This allows normal "small" cases to be fast without
 * losing generality for large inputs.
 *
 * Note that this does not attempt to be exception safe.
 *
 * SmallVector<T, N>& can be converted to SmallVectorImpl<T>& to erase the
 * template param \p N; this is useful for function params.
 *
 * \tparam T emelment type
 * \tparam N number of elements to be stored in the class object
 */
template <typename T, unsigned N = 4>
class SmallVector : public SmallVectorImpl<T> {
    SmallVectorStorage<T, N> m_storage;

public:
    SmallVector() : SmallVectorImpl<T>(N) {}

    explicit SmallVector(size_t size, const T& value = T())
            : SmallVectorImpl<T>(N) {
        this->assign(size, value);
    }

    template <
            typename IterType,
            typename = typename std::enable_if<std::is_convertible<
                    typename std::iterator_traits<IterType>::iterator_category,
                    std::input_iterator_tag>::value>::type>
    SmallVector(IterType first, IterType last) : SmallVectorImpl<T>(N) {
        this->append(first, last);
    }

    SmallVector(std::initializer_list<T> init_list) : SmallVectorImpl<T>(N) {
        this->assign(init_list);
    }

    SmallVector(const SmallVector& rhs) : SmallVectorImpl<T>(N) {
        if (!rhs.empty())
            SmallVectorImpl<T>::operator=(rhs);
    }

    ~SmallVector() {}

    const SmallVector& operator=(const SmallVector& rhs) {
        SmallVectorImpl<T>::operator=(rhs);
        return *this;
    }

    SmallVector(SmallVector&& rhs) : SmallVectorImpl<T>(N) {
        if (!rhs.empty())
            SmallVectorImpl<T>::operator=(std::move(rhs));
    }

    SmallVector(SmallVectorImpl<T>&& rhs) : SmallVectorImpl<T>(N) {
        if (!rhs.empty())
            SmallVectorImpl<T>::operator=(std::move(rhs));
    }

    const SmallVector& operator=(SmallVector&& rhs) {
        SmallVectorImpl<T>::operator=(std::move(rhs));
        return *this;
    }

    const SmallVector& operator=(SmallVectorImpl<T>&& rhs) {
        SmallVectorImpl<T>::operator=(std::move(rhs));
        return *this;
    }

    const SmallVector& operator=(std::initializer_list<T> init_list) {
        this->assign(init_list);
        return *this;
    }
};

template <typename T, unsigned n>
static inline size_t capacity_in_bytes(const SmallVector<T, n>& vec) {
    return vec.capacity_in_bytes();
}

template <typename T>
inline typename SmallVectorImpl<T>::const_iterator find(
        const SmallVectorImpl<T>& vec, const T& value) {
    return vec.find(vec.begin(), vec.end(), value);
}

}  // end namespace megdnn

#include "megdnn/internal/visibility_epilogue.h"

namespace std {

/// Implement std::swap in terms of SmallVector swap.
template <typename T>
inline void swap(megdnn::SmallVectorImpl<T>& lhs,
                 megdnn::SmallVectorImpl<T>& rhs) {
    lhs.swap(rhs);
}

/// Implement std::swap in terms of SmallVector swap.
template <typename T, unsigned N>
inline void swap(megdnn::SmallVector<T, N>& lhs,
                 megdnn::SmallVector<T, N>& rhs) {
    lhs.swap(rhs);
}
}  // end namespace std

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
