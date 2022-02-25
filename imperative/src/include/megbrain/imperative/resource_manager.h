#pragma once

#include <any>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "megbrain/common.h"
#include "megbrain/utils/metahelper.h"
#include "megbrain/utils/thread.h"

namespace mgb {
namespace imperative {

class ResourceManager : public NonCopyableObj {
protected:
    std::vector<std::any> m_handles;
    std::mutex m_mutex;

private:
    static ResourceManager& get_global();
    static ResourceManager& get_local();

public:
    template <typename T, typename... TArgs>
    static T* create_global(TArgs&&... args) {
        mgb_log_debug("create global resource: %s", typeid(T).name());
        auto instance = std::make_shared<T>(std::forward<TArgs&&>(args)...);
        auto& manager = get_global();
        MGB_LOCK_GUARD(manager.m_mutex);
        manager.m_handles.push_back((std::any)instance);
        return instance.get();
    }

    template <typename T, typename... TArgs>
    static T* create_local(TArgs&&... args) {
        mgb_log_debug("create local resource: %s", typeid(T).name());
        auto instance = std::make_shared<T>(std::forward<TArgs&&>(args)...);
        get_local().m_handles.push_back((std::any)instance);
        return instance.get();
    }

    void clear();

    ~ResourceManager() { clear(); }
};

template <typename T>
class CompNodeDependentResource : public NonCopyableObj {
private:
    std::function<std::unique_ptr<T>()> m_ctor;
    std::unique_ptr<T> m_ptr;
    Spinlock m_spin;

public:
    explicit CompNodeDependentResource(std::function<std::unique_ptr<T>()> ctor)
            : m_ctor(ctor) {}

    T& operator*() {
        if ((!m_ptr) || m_ptr->is_finalized()) {
            m_ptr = m_ctor();
        }
        return *m_ptr;
    }

    T* operator->() {
        if ((!m_ptr) || m_ptr->is_finalized()) {
            m_ptr = m_ctor();
        }
        return m_ptr.get();
    }
};

}  // namespace imperative
}  // namespace mgb
