/**
 * \file dnn/test/arm_common/cpuinfo.cpp
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

TEST(ARM_RUNTIME, CPUINFO_KIRIN980) {
    ASSERT_TRUE(cpuinfo_initialize());

    int right_soc = strcmp(cpuinfo_get_package(0)->name, "HiSilicon Kirin 980");

    if (!right_soc) {
        ASSERT_EQ(8, cpuinfo_get_processors_count());

        ASSERT_TRUE(cpuinfo_get_processors());

        ASSERT_TRUE(cpuinfo_has_arm_neon());

        ASSERT_TRUE(cpuinfo_has_arm_neon_fp16());

        ASSERT_TRUE(cpuinfo_has_arm_neon_fma());

        ASSERT_TRUE(cpuinfo_has_arm_neon_dot());

        for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
            ASSERT_EQ(cpuinfo_get_core(i), cpuinfo_get_processor(i)->core);
        }

        for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
            ASSERT_EQ(cpuinfo_vendor_arm, cpuinfo_get_core(i)->vendor);
        }

        for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
            switch (i) {
                case 0:
                case 1:
                case 2:
                case 3:
                    ASSERT_EQ(cpuinfo_uarch_cortex_a76,
                              cpuinfo_get_core(i)->uarch);
                    break;
                case 4:
                case 5:
                case 6:
                case 7:
                    ASSERT_EQ(cpuinfo_uarch_cortex_a55,
                              cpuinfo_get_core(i)->uarch);
                    break;
            }
        }
    } else {
        printf("detect soc: %s ,skip test.\n", cpuinfo_get_package(0)->name);
    }
}

TEST(ARM_RUNTIME, CPUINFO_SDM8150) {
    ASSERT_TRUE(cpuinfo_initialize());

    int right_soc =
            strcmp(cpuinfo_get_package(0)->name, "Qualcomm Snapdragon 8150");

    if (!right_soc) {
        ASSERT_EQ(8, cpuinfo_get_processors_count());

        ASSERT_TRUE(cpuinfo_get_processors());

        ASSERT_TRUE(cpuinfo_has_arm_neon());

        ASSERT_TRUE(cpuinfo_has_arm_neon_fp16());

        ASSERT_TRUE(cpuinfo_has_arm_neon_fma());

        ASSERT_TRUE(cpuinfo_has_arm_neon_dot());

        for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
            ASSERT_EQ(cpuinfo_get_core(i), cpuinfo_get_processor(i)->core);
        }

        for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
            ASSERT_EQ(cpuinfo_vendor_arm, cpuinfo_get_core(i)->vendor);
        }

        for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
            switch (i) {
                case 0:
                case 1:
                case 2:
                case 3:
                    ASSERT_EQ(cpuinfo_uarch_cortex_a76,
                              cpuinfo_get_core(i)->uarch);
                    break;
                case 4:
                case 5:
                case 6:
                case 7:
                    ASSERT_EQ(cpuinfo_uarch_cortex_a55,
                              cpuinfo_get_core(i)->uarch);
                    break;
            }
        }
    } else {
        printf("detect soc: %s ,skip test.\n", cpuinfo_get_package(0)->name);
    }
}

TEST(ARM_RUNTIME, CPUINFO_SDM660) {
    ASSERT_TRUE(cpuinfo_initialize());

    int right_soc =
            strcmp(cpuinfo_get_package(0)->name, "Qualcomm Snapdragon 660");

    if (!right_soc) {
        ASSERT_EQ(8, cpuinfo_get_processors_count());

        ASSERT_TRUE(cpuinfo_get_processors());

        ASSERT_TRUE(cpuinfo_has_arm_neon());

        ASSERT_TRUE(cpuinfo_has_arm_neon_fp16());

        ASSERT_TRUE(cpuinfo_has_arm_neon_fma());

        ASSERT_FALSE(cpuinfo_has_arm_neon_dot());

        for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
            ASSERT_EQ(cpuinfo_get_core(i), cpuinfo_get_processor(i)->core);
        }

        for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
            ASSERT_EQ(cpuinfo_vendor_arm, cpuinfo_get_core(i)->vendor);
        }

        for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
            switch (i) {
                case 0:
                case 1:
                case 2:
                case 3:
                    ASSERT_EQ(cpuinfo_uarch_cortex_a73,
                              cpuinfo_get_core(i)->uarch);
                    break;
                case 4:
                case 5:
                case 6:
                case 7:
                    ASSERT_EQ(cpuinfo_uarch_cortex_a53,
                              cpuinfo_get_core(i)->uarch);
                    break;
            }
        }
    } else {
        printf("detect soc: %s ,skip test.\n", cpuinfo_get_package(0)->name);
    }
}

}  // namespace test
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
