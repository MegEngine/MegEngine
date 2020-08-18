/**
 * \file dnn/test/x86/cpuinfo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/common/utils.h"
#if defined(MGB_ENABLE_CPUINFO_CHECK) && MGB_ENABLE_CPUINFO
#include <cpuinfo.h>
#include <inttypes.h>
#include "gtest/gtest.h"

namespace megdnn {
namespace test {

TEST(X86_RUNTIME, CPUINFO_XEON6130) {
    ASSERT_TRUE(cpuinfo_initialize());

    int right_cpu =
            strcmp(cpuinfo_get_package(0)->name, "Intel Xeon Gold 6130");

    if (!right_cpu) {
        ASSERT_TRUE(cpuinfo_get_processors());

        ASSERT_TRUE(cpuinfo_has_x86_avx2());

        ASSERT_TRUE(cpuinfo_has_x86_avx512f());

        ASSERT_TRUE(cpuinfo_has_x86_sse4_2());

        ASSERT_TRUE(cpuinfo_has_x86_avx());

    } else {
        printf("detect cpu: %s ,skip test.\n", cpuinfo_get_package(0)->name);
    }
}
}  // namespace test
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
