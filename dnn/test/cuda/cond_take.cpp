/**
 * \file dnn/test/cuda/cond_take.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs.h"
#include "test/cuda/fixture.h"
#include "test/common/checker.h"
#include "test/common/cond_take.h"

using namespace megdnn;
using namespace test;

TEST_F(CUDA, COND_TAKE) {
    auto opr_naive = handle_naive()->create_operator<CondTake>();
    auto opr_cuda = handle_cuda()->create_operator<CondTake>();

    size_t tot_size = 0;
    for (auto &&i: CondTakeTestcase::make()) {
        auto ret_naive = i.run(opr_naive.get()),
             ret_cuda = i.run(opr_cuda.get());
        MEGDNN_ASSERT_TENSOR_EQ(*ret_naive.first, *ret_cuda.first);
        MEGDNN_ASSERT_TENSOR_EQ(*ret_naive.second, *ret_cuda.second);
        tot_size += ret_naive.first->layout.total_nr_elems();
    }
    ASSERT_GT(tot_size, (size_t)0);
}

// vim: syntax=cpp.doxygen
