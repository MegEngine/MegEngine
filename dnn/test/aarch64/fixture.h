/**
 * \file dnn/test/aarch64/fixture.h
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
#include "test/arm_common/fixture.h"

#include <memory>

namespace megdnn {
namespace test {

class AARCH64 : public ARM_COMMON {};

class AARCH64_MULTI_THREADS : public ARM_COMMON_MULTI_THREADS {};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
