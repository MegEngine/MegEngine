/**
 * \file dnn/test/naive/fixture.h
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
#include "test/common/utils.h"

#include <memory>

namespace megdnn {
namespace test {

class NAIVE : public ::testing::Test {
public:
    void SetUp() override;
    void TearDown() override;

    Handle* handle() { return m_handle.get(); }

private:
    std::unique_ptr<Handle> m_handle;
};

class NAIVE_MULTI_THREADS : public ::testing::Test {
public:
    void SetUp() override;
    void TearDown() override;

    Handle* handle() { return m_handle.get(); }

private:
    std::unique_ptr<Handle> m_handle;

};

class NAIVE_BENCHMARK_MULTI_THREADS : public ::testing::Test {};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
