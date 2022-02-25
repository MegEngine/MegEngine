#include "hcc_detail/hcc_defs_prologue.h"

#include "./fixture.h"
#include "test/rocm/utils.h"

#include <gtest/gtest.h>
#include "hip_header.h"

void MegcoreROCM::SetUp() {
    hip_check(hipGetDeviceCount(&nr_devices_));
    printf("We have %d GPUs\n", nr_devices_);
}

void MegcoreROCM::TearDown() {
    hip_check(hipDeviceReset());
}

// vim: syntax=cpp.doxygen
