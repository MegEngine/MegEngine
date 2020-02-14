/**
 * \file src/core/impl/utils/event.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/event.h"

#include <iterator>

using namespace mgb;

class SyncEventConnecter::ReceiverHandlerImpl {
    SyncEventConnecter *m_par_connector;
    std::weak_ptr<ReceiverMap> m_map;
    Typeinfo* m_map_key;
    ReceiverList::iterator m_hdl_iter;

    public:
        ReceiverHandlerImpl(
                SyncEventConnecter *par_connector,
                Typeinfo* map_key,
                ReceiverList::iterator iter):
            m_par_connector{par_connector},
            m_map{par_connector->m_receiver_map},
            m_map_key{map_key}, m_hdl_iter{iter}
        {}

        ~ReceiverHandlerImpl() {
            auto p = m_map.lock();
            if (p) {
                MGB_LOCK_GUARD(m_par_connector->m_mtx);
                ++ m_par_connector->m_version;
                auto &&seq = p->at(m_map_key);
                seq.erase(m_hdl_iter);
                if (seq.empty()) {
                    p->erase(m_map_key);
                    if (p->empty()) {
                        m_par_connector->m_is_empty = true;
                    }
                }
            }
        }
};


void SyncEventConnecter::ReceiverHandlerImplDeleter::operator()(
        ReceiverHandlerImpl *ptr) {
    delete ptr;
}

SyncEventConnecter::ReceiverHandler SyncEventConnecter::do_register_receiver(
        Typeinfo *type, std::unique_ptr<ReceiverBase> receiver) {
    MGB_LOCK_GUARD(m_mtx);

    ++ m_version;
    m_is_empty = false;
    ReceiverList &list = m_receiver_map->operator[](type);
    list.push_back(std::move(receiver));
    auto iter = std::prev(list.end());
    return ReceiverHandler{new ReceiverHandlerImpl{this, type, iter}};
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

