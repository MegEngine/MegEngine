/**
 * \file dnn/test/cuda/topk.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/topk.h"
#include "test/cuda/fixture.h"

using namespace megdnn;
using namespace test;


/*
 * !!!!!!!!!!!!!!!! IMPORTANT NOTE !!!!!!!!!!!!!!!!
 * The kernels are indepedently developed and tested in the
 * MegDNN/expr/cuda_topk directory. Here we only check some common cases.
 */

TEST_F(CUDA, TOP_K) {
    run_topk_test<dtype::Float32>(handle_cuda());
}
TEST_F(CUDA, TOP_K_I32) {
    run_topk_test<dtype::Int32>(handle_cuda());
}


// vim: syntax=cpp.doxygen
