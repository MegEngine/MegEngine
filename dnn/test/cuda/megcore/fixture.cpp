/**
 * \file dnn/test/cuda/megcore/fixture.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./fixture.h"
#include "test/cuda/utils.h"

#include <gtest/gtest.h>
#include <cuda_runtime_api.h>

void MegcoreCUDA::SetUp()
{
    cuda_check(cudaGetDeviceCount(&nr_devices_));
    printf("We have %d GPUs\n", nr_devices_);
}

void MegcoreCUDA::TearDown()
{
    cuda_check(cudaDeviceReset());
}

// vim: syntax=cpp.doxygen
