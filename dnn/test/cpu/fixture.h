/**
 * \file dnn/test/cpu/fixture.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <gtest/gtest.h>

#include "megdnn/handle.h"
#include "test/common/utils.h"

#include <memory>

namespace megdnn {
namespace test {

class CPU : public ::testing::Test {
public:
    void SetUp() override;
    void TearDown() override;

    Handle* handle() { return m_handle.get(); }

protected:
    std::unique_ptr<Handle> m_handle;
};

class CPU_MULTI_THREADS : public ::testing::Test {
public:
    void SetUp() override;
    void TearDown() override;

    Handle* handle() { return m_handle.get(); }

protected:
    std::unique_ptr<Handle> m_handle;
};

class CPU_BENCHMARK_MULTI_THREADS : public ::testing::Test {
    Handle* handle() {
        if (!m_handle) {
            m_handle = create_cpu_handle(0, true, &m_thread_config);
        }
        return m_handle.get();
    }
    void set_thread_config(TaskExecutorConfig&& thread_config) {
        set_thread_config(thread_config);
    }
    void set_thread_config(TaskExecutorConfig& thread_config) {
        m_thread_config = thread_config;
        m_handle.reset(create_cpu_handle(0, true, &m_thread_config).get());
    }
    TaskExecutorConfig get_thread_config() { return m_thread_config; }

protected:
    TaskExecutorConfig m_thread_config;
    std::unique_ptr<Handle> m_handle;
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
