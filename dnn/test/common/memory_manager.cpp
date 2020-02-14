/**
 * \file dnn/test/common/memory_manager.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./memory_manager.h"

#include "test/common/utils.h"
#include "src/common/utils.h"

namespace {

using namespace megdnn;
using namespace test;

std::unique_ptr<MemoryManager> create_memory_manager_from_handle(Handle *handle)
{
    return make_unique<HandleMemoryManager>(handle);
}

} // anonymous namespace

megdnn::test::MemoryManagerHolder megdnn::test::MemoryManagerHolder::m_instance;

megdnn::test::HandleMemoryManager::HandleMemoryManager(Handle *handle)
    : MemoryManager(), m_handle(handle)
{}

void* megdnn::test::HandleMemoryManager::malloc(size_t size)
{
    auto comp_handle = m_handle->megcore_computing_handle();
    megcoreDeviceHandle_t dev_handle;
    megcore_check(megcoreGetDeviceHandle(comp_handle, &dev_handle));
    void *ptr;
    megcore_check(megcoreMalloc(dev_handle, &ptr, size));
    return ptr;
}

void megdnn::test::HandleMemoryManager::free(void* ptr)
{
    auto comp_handle = m_handle->megcore_computing_handle();
    megcoreDeviceHandle_t dev_handle;
    megcore_check(megcoreGetDeviceHandle(comp_handle, &dev_handle));
    megcore_check(megcoreFree(dev_handle, ptr));
}

megdnn::test::MemoryManager*
megdnn::test::MemoryManagerHolder::get(Handle* handle)
{
    std::lock_guard<std::mutex> lock(m_map_mutex);
    auto i = m_map.find(handle);
    if (i != m_map.end()) {
        // found
        return i->second.get();
    } else {
        // not found. create it
        auto mm = create_memory_manager_from_handle(handle);
        auto res = mm.get();
        m_map.emplace(std::make_pair(handle, std::move(mm)));
        return res;
    }
}

void MemoryManagerHolder::update(Handle* handle,
        std::unique_ptr<MemoryManager> memory_manager)
{
    std::lock_guard<std::mutex> lock(m_map_mutex);
    m_map[handle] = std::move(memory_manager);
}

void MemoryManagerHolder::clear()
{
    std::lock_guard<std::mutex> lock(m_map_mutex);
    m_map.clear();
}

// vim: syntax=cpp.doxygen
