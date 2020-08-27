/**
 * \file dnn/test/rocm/megcore/fixture.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
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
