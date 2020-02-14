/**
 * \file src/core/include/megbrain/utils/async_worker_impl_0.h
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

#include <thread>
#include <vector>

namespace mgb {

class AsyncWorkerSet final: public NonCopyableObj {
    public:
        using Task = thin_function<void()>;

        void add_worker(const std::string &name, const Task &task);

        void start();

        void wait_all();

        bool empty() const { return !m_task; }

    private:
        Task m_task;
};

class FutureThreadPoolBase : public NonCopyableObj {
    std::vector<std::thread::id> m_ids;
    public:
        FutureThreadPoolBase(const Maybe<std::string>& = None) {}

        const std::vector<std::thread::id>& start(size_t concurrency) {
            m_ids.resize(concurrency, std::this_thread::get_id());
            return m_ids;
        }

        void stop() {
        }
};

template<class R>
class FutureThreadPool final: public FutureThreadPoolBase {
    public:
        using FutureThreadPoolBase::FutureThreadPoolBase;

        class Future {
            friend class FutureThreadPool;
            R m_result;
            public:
                const R& get() const {
                    return m_result;
                }
        };

        template<typename Func, typename ...Args>
        Future launch(Func&& func, Args&&... args) {
            return {func(std::forward<Args>(args)...)};
        }
};
template<>
class FutureThreadPool<void> final: public FutureThreadPoolBase {
    public:
        using FutureThreadPoolBase::FutureThreadPoolBase;

        class Future {
            public:
                void get() const {
                }
        };

        template<typename Func, typename ...Args>
        Future launch(Func&& func, Args&&... args) {
            func(std::forward<Args>(args)...);
            return {};
        }
};

}


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

