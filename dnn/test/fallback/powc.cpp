/**
 * \file dnn/test/fallback/powc.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/powc.h"

#include "test/fallback/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(FALLBACK, POW_C_F32) {
    run_powc_test(handle(), dtype::Float32{});
}

#if !MEGDNN_DISABLE_FLOAT16
TEST_F(FALLBACK, POW_C_F16) {
    run_powc_test(handle(), dtype::Float16{});
}
#endif

// vim: syntax=cpp.doxygen
