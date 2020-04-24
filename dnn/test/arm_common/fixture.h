/**
 * \file dnn/test/arm_common/fixture.h
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
#include "test/cpu/fixture.h"

namespace megdnn {
namespace test {

class ARM_COMMON : public CPU {};

class ARM_COMMON_MULTI_THREADS : public CPU_MULTI_THREADS {};

class ARM_COMMON_BENCHMARK_MULTI_THREADS : public CPU_BENCHMARK_MULTI_THREADS {
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
