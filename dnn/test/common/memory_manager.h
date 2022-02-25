#pragma once
#include <cstddef>

#include <mutex>
#include <unordered_map>
#include "megdnn/handle.h"

namespace megdnn {
namespace test {

class MemoryManager {
public:
    MemoryManager() = default;
    virtual ~MemoryManager() = default;
    virtual void* malloc(size_t size) = 0;
    virtual void free(void* ptr) = 0;
};

/**
 * \brief manages mapping from Handle* to MemoryManager*
 *
 * this class is a singleton
 */
class MemoryManagerHolder {
private:
    static MemoryManagerHolder m_instance;
    std::unordered_map<Handle*, std::unique_ptr<MemoryManager>> m_map;
    std::mutex m_map_mutex;

public:
    static MemoryManagerHolder* instance() { return &m_instance; }
    MemoryManager* get(Handle* handle);
    void update(Handle* handle, std::unique_ptr<MemoryManager> memory_manager);
    void clear();
};
/**
 * \brief HandleMemoryManager utilizes megcore device handle in megdnn handle to
 * perform memory operations
 */
class HandleMemoryManager : public MemoryManager {
private:
    Handle* m_handle;

public:
    HandleMemoryManager(Handle* handle);
    void* malloc(size_t size) override;
    void free(void* ptr) override;
};

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
