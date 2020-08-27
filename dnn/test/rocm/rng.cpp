/**
 * \file dnn/test/rocm/rng.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "megdnn/oprs.h"
#include "test/rocm/fixture.h"
#include "test/naive/rng.h"
#include "test/common/tensor.h"

namespace megdnn {

namespace test {

TEST_F(ROCM, UNIFORM_RNG_F32) {
    auto opr = handle_rocm()->create_operator<UniformRNG>();
    SyncedTensor<> t(handle_rocm(), {TensorShape{200000}, dtype::Float32()});
    opr->exec(t.tensornd_dev(), {});

    assert_uniform_correct(t.ptr_mutable_host(),
            t.layout().total_nr_elems());
}

TEST_F(ROCM, GAUSSIAN_RNG_F32) {
    auto opr = handle_rocm()->create_operator<GaussianRNG>();
    opr->param().mean = 0.8;
    opr->param().std = 2.3;
    for (size_t size: {1, 200000, 200001}) {
        TensorLayout ly{{size}, dtype::Float32()};
        Tensor<dt_byte> workspace(handle_rocm(),
                {TensorShape{opr->get_workspace_in_bytes(ly)},
                dtype::Byte()});
        SyncedTensor<> t(handle_rocm(), ly);
        opr->exec(t.tensornd_dev(),
                {workspace.ptr(), workspace.layout().total_nr_elems()});

        auto ptr = t.ptr_mutable_host();

        if (size >= 1000) {
            auto stat = get_mean_var(ptr, size, 0.8f);
            ASSERT_LE(std::abs(stat.first - 0.8), 5e-3);
            ASSERT_LE(std::abs(stat.second - 2.3 * 2.3), 5e-2);
        }
    }
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen



