/**
 * \file dnn/test/cpu/fixture.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cpu/fixture.h"

#include "test/common/utils.h"
#include "test/common/memory_manager.h"
#include "test/common/random_state.h"

namespace megdnn {
namespace test {

void CPU::SetUp()
{
    RandomState::reset();
    m_handle = create_cpu_handle(0);
}

void CPU::TearDown()
{
    m_handle.reset();
    MemoryManagerHolder::instance()->clear();
}

void CPU_MULTI_THREADS::SetUp() {
    RandomState::reset();
#if MEGDNN_ENABLE_MULTI_THREADS
    TaskExecutorConfig config;
    size_t nr_threads = std::min<size_t>(get_cpu_count(), 2);
    config.nr_thread = nr_threads;
    m_handle = create_cpu_handle(0, true, &config);
#else
    m_handle = create_cpu_handle(0);
#endif
}

void CPU_MULTI_THREADS::TearDown() {
    m_handle.reset();
    MemoryManagerHolder::instance()->clear();
}

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen
