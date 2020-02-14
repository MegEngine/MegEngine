/**
 * \file src/core/include/megbrain/utils/shared_set.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/utils/metahelper.h"
#include "megbrain/utils/thin/hash_table.h"

namespace mgb {

/*!
 * \brief set that allows fast merging from others and sharing memory if
 *      possible
 *
 * Note: for best performance, ref count is not maintained, so the root
 * SharedSet (i.e. the one that others call merge_from() on) must be alive for
 * others to be valid
 */
template<typename Elem, class Container = ThinHashSet<Elem>>
class SharedSet: public NonCopyableObj {
    bool m_own_ptr = false;
    Container *m_container = nullptr;

    void ensure_own() {
        if (m_own_ptr)
            return;
        if (!m_container)
            m_container = new Container();
        else
            m_container = new Container(*m_container);
        m_own_ptr = true;
    }

    static Container& sentinel_container() {
        static Container ins;
        return ins;
    }

    public:
        SharedSet() = default;

        SharedSet(SharedSet &&rhs) noexcept
        {
            operator=(std::move(rhs));
        }

        SharedSet& operator = (SharedSet &&rhs) noexcept {
            m_own_ptr = rhs.m_own_ptr;
            m_container = rhs.m_container;
            rhs.m_own_ptr = false;
            rhs.m_container = nullptr;
        }

        ~SharedSet() noexcept {
            if (m_own_ptr)
                delete m_container;
        }

        /*!
         * \brief insert an element
         */
        void insert(const Elem &elem) {
            if (m_container && m_container->count(elem))
                return;
            ensure_own();
            m_container->insert(elem);
        }

        /*!
         * \brief insert all elements in another set into this
         */
        void merge_from(const SharedSet &rhs) {
            if (!rhs)
                return;

            if (m_own_ptr) {
                auto pct = m_container;
                for (auto &&i: *rhs.m_container)
                    pct->insert(i);
                return;
            }

            if (!m_container) {
                m_container = rhs.m_container;
                return;
            }

            for (auto &&i: *rhs.m_container) {
                if (!m_container->count(i)) {
                    ensure_own();
                    m_container->insert(i);
                }
            }
        }

        /*!
         * \brief membership test
         */
        bool contain(const Elem &elem) const {
            return m_container && m_container->count(elem);
        }

        operator bool() const {
            return m_container;
        }

        const Container* get() const {
            return m_container;
        }

        decltype(auto) begin() const {
            if (!m_container)
                return sentinel_container().cbegin();

            return m_container->cbegin();
        }

        decltype(auto) end() const {
            if (!m_container)
                return sentinel_container().cend();

            return m_container->cend();
        }
};

} // mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

