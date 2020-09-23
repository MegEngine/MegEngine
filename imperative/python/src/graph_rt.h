/**
 * \file imperative/python/src/graph_rt.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./helper.h"

#include <memory>
#include <mutex>
#include <future>
#include "megbrain/plugin/opr_footprint.h"
#include "megbrain/graph.h"

template<typename T>
class GraphNodePtr {
    std::shared_ptr<mgb::cg::ComputingGraph> m_graph;
    T* m_node;
public:
    GraphNodePtr(T* node) :
        m_graph(node ? node->owner_graph()->shared_from_this() : nullptr),
        m_node(node) {}
    T* operator->() {return m_node;}
    T& operator*() {return *m_node;}
    operator bool() {return m_node;}
    T* get() {return m_node;}
};

PYBIND11_DECLARE_HOLDER_TYPE(T, GraphNodePtr<T>, true);

class RendezvousBase {
public:
    virtual ~RendezvousBase() = default;
    virtual void set_exception(std::exception_ptr p) = 0;
};

template<typename R>
class Rendezvous: public RendezvousBase {
    std::mutex m_lock;
    int m_read_ahead = 0;
    bool m_drop_next = false;
    std::promise<R> m_promise;
    Rendezvous() = default;
    struct Factory {
        template<typename ...Args>
        static auto make_rendezvous(Args&& ...args) {
            auto ptr = new Rendezvous<R>{std::forward(args)...};
            return std::shared_ptr<Rendezvous<R>>(ptr);
        }
    };
public:
    Rendezvous(const Rendezvous& rhs) = delete;
    Rendezvous(Rendezvous&& rhs) = delete;
    Rendezvous& operator=(const Rendezvous& rhs) = delete;

    template<typename ...Args>
    static auto make(Args&& ...args) {
        return Factory::make_rendezvous(std::forward<Args>(args)...);
    }

    R get() {
        std::future<R> f;
        {
            MGB_LOCK_GUARD(m_lock);
            mgb_assert(m_read_ahead <= 0);
            mgb_assert(m_read_ahead >= -1);
            f = m_promise.get_future();
            if (m_read_ahead == -1) {
                m_promise = {};
            }
            ++m_read_ahead;
        }
        return f.get();
    }

    void drop() {
        MGB_LOCK_GUARD(m_lock);
        mgb_assert(m_read_ahead <= 0);
        mgb_assert(m_read_ahead >= -1);
        if (m_read_ahead == -1) {
            m_promise = {};
        } else {
            m_drop_next = true;
        }
        ++m_read_ahead;
    }

    template<typename T>
    void set(T&& value) {
        MGB_LOCK_GUARD(m_lock);
        mgb_assert(m_read_ahead >= 0);
        mgb_assert(m_read_ahead <= 1);
        if (m_drop_next) {
            m_drop_next = false;
        } else {
            m_promise.set_value(std::forward<T>(value));
        }
        if (m_read_ahead == 1) {
            m_promise = {};
        }
        --m_read_ahead;
    }

    void reset() {
        MGB_LOCK_GUARD(m_lock);
        m_promise = {};
        m_read_ahead = 0;
        m_drop_next = false;
    }

    void set_exception(std::exception_ptr e) {
        if (e) {
            MGB_LOCK_GUARD(m_lock);
            if (m_read_ahead >= 0) {
                mgb_assert(m_read_ahead <= 1);
                if (m_drop_next) {
                    m_drop_next = false;
                } else {
                    m_promise.set_exception(e);
                }
                if (m_read_ahead == 1) {
                    m_promise = {};
                }
                --m_read_ahead;
            } else {
                mgb_assert(m_read_ahead == -1);
                // TODO: maybe exception should be ignored
                // if value was already set ?
                m_promise.set_exception(e);
            }
        }
    }
};

void init_graph_rt(pybind11::module m);
