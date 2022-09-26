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
                    ASSERT_EQ(cpuinfo_uarch_cortex_a76, cpuinfo_get_core(i)->uarch);
                    break;
                case 4:
                case 5:
                case 6:
                case 7:
                    ASSERT_EQ(cpuinfo_uarch_cortex_a55, cpuinfo_get_core(i)->uarch);
                    break;
            }
        }
    } else {
        printf("detect soc: %s ,skip test.\n", cpuinfo_get_package(0)->name);
    }
}

TEST(ARM_RUNTIME, CPUINFO_SDM8150) {
    ASSERT_TRUE(cpuinfo_initialize());

    int right_soc = strcmp(cpuinfo_get_package(0)->name, "Qualcomm Snapdragon 8150");

    if (!right_soc) {
        ASSERT_EQ(8, cpuinfo_get_processors_count());

        ASSERT_TRUE(cpuinfo_get_processors());

        ASSERT_TRUE(cpuinfo_has_arm_neon());

        ASSERT_TRUE(cpuinfo_has_arm_neon_fp16());

        ASSERT_TRUE(cpuinfo_has_arm_neon_fma());

        ASSERT_TRUE(cpuinfo_has_arm_neon_dot());

        ASSERT_FALSE(cpuinfo_has_arm_i8mm());

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
                    ASSERT_EQ(cpuinfo_uarch_cortex_a76, cpuinfo_get_core(i)->uarch);
                    break;
                case 4:
                case 5:
                case 6:
                case 7:
                    ASSERT_EQ(cpuinfo_uarch_cortex_a55, cpuinfo_get_core(i)->uarch);
                    break;
            }
        }
    } else {
        printf("detect soc: %s ,skip test.\n", cpuinfo_get_package(0)->name);
    }
}

TEST(ARM_RUNTIME, CPUINFO_SDM660) {
    ASSERT_TRUE(cpuinfo_initialize());

    int right_soc = strcmp(cpuinfo_get_package(0)->name, "Qualcomm Snapdragon 660");

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
                    ASSERT_EQ(cpuinfo_uarch_cortex_a73, cpuinfo_get_core(i)->uarch);
                    break;
                case 4:
                case 5:
                case 6:
                case 7:
                    ASSERT_EQ(cpuinfo_uarch_cortex_a53, cpuinfo_get_core(i)->uarch);
                    break;
            }
        }
    } else {
        printf("detect soc: %s ,skip test.\n", cpuinfo_get_package(0)->name);
    }
}

TEST(ARM_RUNTIME, CPUINFO_TAISHAN) {
    ASSERT_TRUE(cpuinfo_initialize());

    bool right_soc =
            cpuinfo_get_processors_count() == 96 &&
            cpuinfo_get_processor(0)->core->uarch == cpuinfo_uarch_taishan_v110;
    if (right_soc) {
        ASSERT_TRUE(cpuinfo_get_processors());

        ASSERT_TRUE(cpuinfo_has_arm_neon());

        ASSERT_TRUE(cpuinfo_has_arm_neon_fp16());

        ASSERT_TRUE(cpuinfo_has_arm_neon_fma());

        ASSERT_TRUE(cpuinfo_has_arm_neon_dot());

        ASSERT_FALSE(cpuinfo_has_arm_i8mm());

        for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
            ASSERT_EQ(cpuinfo_get_core(i), cpuinfo_get_processor(i)->core);
        }

        for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
            ASSERT_EQ(cpuinfo_vendor_huawei, cpuinfo_get_core(i)->vendor);
        }

    } else {
        printf("detect soc: %s ,skip test.\n", cpuinfo_get_package(0)->name);
    }
}

TEST(ARM_RUNTIME, CPUINFO_SDM8GEN1) {
    ASSERT_TRUE(cpuinfo_initialize());
    bool right_soc =
            cpuinfo_get_processors_count() == 8 &&
            cpuinfo_get_processor(0)->core->uarch == cpuinfo_uarch_cortex_x2 &&
            cpuinfo_get_processor(1)->core->uarch == cpuinfo_uarch_cortex_a710 &&
            cpuinfo_get_processor(7)->core->uarch == cpuinfo_uarch_cortex_a510;

    if (right_soc) {
        ASSERT_TRUE(cpuinfo_get_processors());

        ASSERT_TRUE(cpuinfo_has_arm_neon());

        ASSERT_TRUE(cpuinfo_has_arm_neon_fp16());

        ASSERT_TRUE(cpuinfo_has_arm_neon_fma());

        ASSERT_TRUE(cpuinfo_has_arm_neon_dot());

        ASSERT_FALSE(cpuinfo_has_arm_sve2());

        ASSERT_TRUE(cpuinfo_has_arm_i8mm());

        for (uint32_t i = 0; i < cpuinfo_get_processors_count(); i++) {
            ASSERT_EQ(cpuinfo_get_core(i), cpuinfo_get_processor(i)->core);
        }

        for (uint32_t i = 0; i < cpuinfo_get_cores_count(); i++) {
            ASSERT_EQ(cpuinfo_vendor_arm, cpuinfo_get_core(i)->vendor);
        }

    } else {
        printf("detect soc: %s ,skip test.\n", cpuinfo_get_package(0)->name);
    }
}

}  // namespace test
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
