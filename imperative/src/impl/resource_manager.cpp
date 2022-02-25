#include "megbrain/imperative/resource_manager.h"

#include <thread>
#include <unordered_map>

using namespace mgb;
using namespace imperative;

namespace {

class LocalResourceManager;

std::unordered_map<std::thread::id, std::shared_ptr<LocalResourceManager>>
        local_managers;
std::mutex global_lock;
bool throw_all_resources = false;

class LocalResourceManager final : public ResourceManager {
private:
    std::thread::id m_id;

public:
    LocalResourceManager() : m_id(std::this_thread::get_id()) {}

    std::thread::id id() const { return m_id; }
};

class GlobalResourceManager final : public ResourceManager {
public:
    ~GlobalResourceManager() {
#if MGB_CUDA && defined(WIN32)
        //! FIXME: windows cuda driver shutdown before call atexit function even
        //! register atexit function after init cuda driver! as a workround
        //! recovery resource by OS temporarily, may need remove this after
        //! upgrade cuda runtime
        throw_all_resources = true;
#endif
        MGB_LOCK_GUARD(global_lock);
        local_managers.clear();
    }
};

class LocalResourceManagerRef : public NonCopyableObj {
private:
    std::weak_ptr<LocalResourceManager> m_manager;

public:
    LocalResourceManagerRef() {
        auto manager = std::make_shared<LocalResourceManager>();
        mgb_assert(
                local_managers.insert({manager->id(), manager}).second,
                "duplicated local manager");
        m_manager = manager;
    }

    ~LocalResourceManagerRef() {
        if (auto manager = m_manager.lock()) {
            local_managers.erase(manager->id());
        }
    }

    ResourceManager& operator*() { return *m_manager.lock(); }
};

}  // namespace

void ResourceManager::clear() {
    if (throw_all_resources) {
        new std::vector<std::any>(std::move(m_handles));
    }
    for (auto iter = m_handles.rbegin(); iter != m_handles.rend(); ++iter) {
        (*iter) = {};
    }
}

ResourceManager& ResourceManager::get_global() {
    static GlobalResourceManager sl_manager;
    return sl_manager;
}

ResourceManager& ResourceManager::get_local() {
    thread_local LocalResourceManagerRef tl_manager;
    return *tl_manager;
}
