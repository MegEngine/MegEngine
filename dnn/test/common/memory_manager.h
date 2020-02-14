/**
 * \file dnn/test/common/memory_manager.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cstddef>

#include "megdnn/handle.h"
#include <unordered_map>
#include <mutex>

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
