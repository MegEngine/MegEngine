#pragma once

#include "megbrain/utils/metahelper.h"

#include <thread>
#include <vector>

namespace mgb {

class AsyncWorkerSet final : public NonCopyableObj {
public:
    using Task = thin_function<void()>;

    void add_worker(const std::string& name, const Task& task);

    void start();

    void wait_all();

    bool empty() const { return !m_task; }

private:
    Task m_task;
};

class FutureThreadPoolBase : public NonCopyableObj {
#if !__DEPLOY_ON_XP_SP2__
    std::vector<std::thread::id> m_ids;
#endif
public:
    FutureThreadPoolBase(const Maybe<std::string>& = None) {}

#if __DEPLOY_ON_XP_SP2__
    size_t start(size_t concurrency) { return concurrency; }
#else
    const std::vector<std::thread::id>& start(size_t concurrency) {
        m_ids.resize(concurrency, std::this_thread::get_id());
        return m_ids;
    }
#endif

    void stop() {}
};

template <class R>
class FutureThreadPool final : public FutureThreadPoolBase {
public:
    using FutureThreadPoolBase::FutureThreadPoolBase;

    class Future {
        friend class FutureThreadPool;
        R m_result;

    public:
        const R& get() const { return m_result; }
    };

    template <typename Func, typename... Args>
    Future launch(Func&& func, Args&&... args) {
        return {func(std::forward<Args>(args)...)};
    }
};
template <>
class FutureThreadPool<void> final : public FutureThreadPoolBase {
public:
    using FutureThreadPoolBase::FutureThreadPoolBase;

    class Future {
    public:
        void get() const {}
    };

    template <typename Func, typename... Args>
    Future launch(Func&& func, Args&&... args) {
        func(std::forward<Args>(args)...);
        return {};
    }
};

}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
