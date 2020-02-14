/**
 * \file dnn/test/naive/svd.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/naive/fixture.h"

#include "megdnn/oprs/linalg.h"
#include "test/common/checker.h"
#include "test/common/random_state.h"
#include "test/common/svd.h"

using namespace megdnn;
using namespace test;

TEST_F(NAIVE, SINGULAR_VALUE_DECOMPOSITION) {
    auto opr = handle()->create_operator<SVDForward>();
    auto testcases = SVDTestcase::make();
    for (auto& t : testcases) {
        // Not supported, skip for now.
        if (t.m_param.full_matrices)
            continue;

        auto naive_result = t.run(opr.get());
        for (size_t i = 0; i < naive_result.s->layout.total_nr_elems(); i++) {
            EXPECT_GE(naive_result.s->ptr<dt_float32>()[i], 0);
        }
        if (t.m_param.compute_uv) {
            MEGDNN_ASSERT_TENSOR_EQ(*naive_result.recovered_mat, t.m_mat);
        }
    }
}

// vim: syntax=cpp.doxygen
