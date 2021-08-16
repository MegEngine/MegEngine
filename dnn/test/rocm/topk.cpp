/**
 * \file dnn/test/rocm/topk.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "test/common/topk.h"
#include "test/rocm/fixture.h"

using namespace megdnn;
using namespace test;


/*
 * !!!!!!!!!!!!!!!! IMPORTANT NOTE !!!!!!!!!!!!!!!!
 * The kernels are indepedently developed and tested in the
 * MegDNN/expr/cuda_topk directory. Here we only check some common cases.
 */

TEST_F(ROCM, TOP_K) {
    run_topk_test<dtype::Float32>(handle_rocm());
}
TEST_F(ROCM, TOP_K_I32) {
    run_topk_test<dtype::Int32>(handle_rocm());
}
// #if !MEGDNN_DISABLE_FLOAT16
// TEST_F(ROCM, TOP_K_F16) {
//     run_topk_test<dtype::Float16>(handle_rocm());
// }
// #endif

// vim: syntax=cpp.doxygen
