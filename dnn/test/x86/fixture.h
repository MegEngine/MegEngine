/**
 * \file dnn/test/x86/fixture.h
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
#include "test/cpu/fixture.h"

#include <memory>

namespace megdnn {
namespace test {

class X86 : public CPU {
public:
    void TearDown() override;

    Handle* fallback_handle();

private:
    std::unique_ptr<Handle> m_handle, m_fallback_handle;
};

class X86_MULTI_THREADS : public CPU_MULTI_THREADS {};

class X86_BENCHMARK_MULTI_THREADS : public CPU_BENCHMARK_MULTI_THREADS {};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
