/**
 * \file src/core/include/megbrain/utils/event.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/common.h"
#include "megbrain/utils/metahelper.h"
#include "megbrain/utils/thin/hash_table.h"

#include <memory>
#include "megbrain/utils/thin/function.h"
#include <list>
#include <utility>

namespace mgb {

/*!
 * \brief synchronized event connector
 *
 * When signaling an event, all the receivers would be invoked before the
 * signaling call returns
 *
 * A signal is represented by its type
 *
 * The insert and erase methods are threadsafe
 */
class SyncEventConnecter: public NonCopyableObj {
    class ReceiverBase {
        public:
            virtual ~ReceiverBase() = default;
    };

    template<typename T>
    class Receiver: public ReceiverBase {
        public:
            thin_function<void(const T&)> callback;

            template<typename Callback>
            Receiver(Callback &&cb):
                callback{std::forward<Callback>(cb)}
            {}
    };

    using ReceiverList = std::list<std::unique_ptr<ReceiverBase>>;
    using ReceiverMap = ThinHashMap<Typeinfo*, ReceiverList>;

    bool m_is_empty = true;
    std::mutex m_mtx;
    //! map from type to receiver; use shared_ptr because it would be kept by
    //! handlers
    std::shared_ptr<ReceiverMap> m_receiver_map =
        std::make_shared<ReceiverMap>();
    size_t m_version = 0;

    public:
        /*!
         * \brief hold resource for a receiver; when destructed, the
         *      corresponding receiver would be removed
         */
        class ReceiverHandlerImpl;
        struct ReceiverHandlerImplDeleter {
            public:
                void operator()(ReceiverHandlerImpl*);
        };
        using ReceiverHandler = std::unique_ptr<
            ReceiverHandlerImpl, ReceiverHandlerImplDeleter>;

        /*!
         * \brief register a receiver to receive events of type T
         * \return receiver hander; if it is destoried, the receiver would be
         *      removed
         */
        template<typename T, typename Callback>
        MGB_WARN_UNUSED_RESULT
        ReceiverHandler register_receiver(Callback &&callback) {
            auto receiver = std::make_unique<Receiver<T>>(
                    std::forward<Callback>(callback));
            return do_register_receiver(T::typeinfo(), std::move(receiver));
        }

        //! register a permanent handler, which could not be un-registered
        template<typename T, typename Callback>
        void register_receiver_permanent(Callback &&callback) {
            auto hdl = register_receiver<T>(std::forward<Callback>(callback));
            m_permanent_handler.push_back(std::move(hdl));
        }

        //! signal an event, giving arguments for constructor of T
        template<typename T, typename ...Args>
        void signal_inplace(Args&& ...args) const {
            if (m_is_empty)
                return;
            auto iter = m_receiver_map->find(T::typeinfo());
            if (iter == m_receiver_map->end())
                return;
            T t_ins{std::forward<Args>(args)...};
            using R = Receiver<T>;
            for (auto &&i: iter->second) {
                static_cast<R*>(i.get())->callback(t_ins);
            }
        }

        //! version of last modification; non-zero if any modification happened
        size_t version() const {
            return m_version;
        }

    private:
        std::vector<ReceiverHandler> m_permanent_handler;

        ReceiverHandler do_register_receiver(
                Typeinfo *type, std::unique_ptr<ReceiverBase> receiver);

};

}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

