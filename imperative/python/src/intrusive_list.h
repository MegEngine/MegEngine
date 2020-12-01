/**
 * \file imperative/python/src/intrusive_list.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/metahelper.h"

namespace mgb::imperative::python::intrusive_list {

// copy policy
struct after_t {};
struct before_t {};
struct disable_t {};

template <typename T> struct Tail;

// invariant: next->prev == this
template <typename T>
struct Head {
    Tail<T>* next;

    Head(Tail<T>* node = nullptr) : next(node) {}
    Head(const Head<T>&) = delete;
    Head<T>& operator=(const Head<T>&) = delete;
    Head(Head<T>&& rhs) : next(rhs.next) {
        rhs.next = nullptr;
        if (next) {
            next->prev = this;
        }
    }
    Head<T>& operator=(Head<T>&& rhs) {
        mgb_assert(!next);
        next = rhs.next;
        rhs.next = nullptr;
        if (next) {
            next->prev = this;
        }
        return *this;
    }
    ~Head() {
        if (next) {
            next->prev = nullptr;
        }
    }
};

// invariant: prev->next == this
template <typename T>
struct Tail {
    Head<T>* prev;

    Tail(Head<T>* node = nullptr) : prev(node) {}
    Tail(const Tail<T>&) = delete;
    Tail<T>& operator=(const Tail<T>&) = delete;
    Tail(Tail<T>&& rhs) : prev(rhs.prev) {
        rhs.prev = nullptr;
        if (prev) {
            prev->next = this;
        }
    }
    Tail<T>& operator=(Tail<T>&& rhs) {
        mgb_assert(!prev);
        prev = rhs.prev;
        rhs.prev = nullptr;
        if (prev) {
            prev->next = this;
        }
        return *this;
    }
    ~Tail() {
        if (prev) {
            prev->next = nullptr;
        }
    }
};

template <typename T, typename policy> struct Node;

template <typename T>
class Iterator {
    T* ptr;

    void inc() {ptr = static_cast<T*>(ptr->Head<T>::next);}
    void dec() {ptr = static_cast<T*>(ptr->Head<T>::prev);}

public:
    Iterator(Head<T>& head) : ptr(static_cast<T*>(head.next)) {}
    Iterator(Tail<T>& tail) : ptr(static_cast<T*>(tail.prev)) {}

    template<typename policy>
    Iterator(Node<T, policy>& node) : ptr(static_cast<T*>(&node)) {}

    T& operator*() {return *static_cast<T*>(ptr);}
    T* operator->() {return static_cast<T*>(ptr);}

    operator bool() {return ptr;}
    bool operator==(const Iterator<T>& rhs) {return ptr == rhs.ptr;}

    Iterator& operator++() {inc(); return *this;}
    Iterator& operator--() {dec(); return *this;}
    Iterator operator++(int) {auto ret = *this; inc(); return ret;}
    Iterator operator--(int) {auto ret = *this; dec(); return ret;}
};

// Node in a doubly linked list. Unlike std::list, nodes are not owned by a container.
// Instead, nodes may join or leave a list freely.
// NOTE: Derived classes have to explicitly declare copy / assignment as default,
//       otherwise the compiler generated version would use the const T& signature,
//       which is deleted.
template <typename T = void, typename policy = disable_t>
struct Node : Tail<std::conditional_t<std::is_same_v<T, void>, Node<T, policy>, T>>,
              Head<std::conditional_t<std::is_same_v<T, void>, Node<T, policy>, T>> {
private:
    using this_t = Node<T, policy>;
    using U = std::conditional_t<std::is_same_v<T, void>, this_t, T>;

public:
    using head_t = Head<U>;
    using tail_t = Tail<U>;
    using head_t::next;
    using tail_t::prev;

    Node() = default;
    Node(const this_t&) = delete;
    this_t& operator=(const this_t&) = delete;

    //! constructed node is inserted after the input node
    Node(after_t, head_t& node) : tail_t(&node), head_t(node.next) {
        node.next = this;
        if (next) {
            next->prev = this;
        }
    }

    //! constructed node is inserted before the input node
    Node(before_t, tail_t& node) : head_t(&node), tail_t(node.prev) {
        node.prev = this;
        if (prev) {
            prev->next = this;
        }
    }

    Node(this_t&& rhs) : tail_t(rhs.prev), head_t(rhs.next) {
        rhs.prev = nullptr;
        rhs.next = nullptr;
        if (prev) {
            prev->next = this;
        }
        if (next) {
            next->prev = this;
        }
    }

    Node& operator=(this_t&& rhs) {
        unlink();
        prev = rhs.prev;
        next = rhs.next;
        rhs.prev = nullptr;
        rhs.next = nullptr;
        if (prev) {
            prev->next = this;
        }
        if (next) {
            next->prev = this;
        }
        return *this;
    }

    template<typename p = policy,
             typename = std::enable_if_t<std::is_same_v<p, before_t> || std::is_same_v<p, after_t>, void>>
    Node(this_t& rhs) : Node(policy{}, rhs) {}

    template<typename p = policy,
             typename = std::enable_if_t<std::is_same_v<p, before_t> || std::is_same_v<p, after_t>, void>>
    this_t& operator=(this_t& rhs) {
        insert(policy{}, rhs);
        return *this;
    }

    void unlink() {
        if (prev) {
            prev->next = next;
        }
        if (next) {
            next->prev = prev;
        }
        prev = nullptr;
        next = nullptr;
    }

    //! this node is unlinked from its list and inserted after the input node
    void insert(after_t, head_t& node) {
        unlink();
        prev = &node;
        next = node.next;
        node.next = this;
        if (next) {
            next->prev = this;
        }
    }

    //! this node is unlinked from its list and inserted before the input node
    void insert(before_t, tail_t& node) {
        unlink();
        next = &node;
        prev = node.prev;
        node.prev = this;
        if (prev) {
            prev->next = this;
        }
    }

    void insert_before(tail_t& node) {insert(before_t{}, node);}
    void insert_after(head_t& node) {insert(after_t{}, node);}

    ~Node() {
        unlink();
    }
};

} // namespace mgb::imperative::python::intrusive_list
