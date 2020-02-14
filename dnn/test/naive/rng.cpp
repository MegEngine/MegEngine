/**
 * \file dnn/test/naive/rng.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn.h"
#include "test/naive/fixture.h"
#include "test/naive/rng.h"
#include "test/common/tensor.h"

namespace megdnn {

namespace test {

template<typename ctype>
void assert_uniform_correct(const ctype *src, size_t size) {
    for (size_t i = 0; i < size; ++ i) {
        ASSERT_GT(src[i], ctype(0));
        ASSERT_LE(src[i], ctype(1));
    }
    auto stat = get_mean_var(src, size, ctype(0.5));
    ASSERT_LE(std::abs(stat.first - 0.5), 1e-3);
    ASSERT_LE(std::abs(stat.second - 1.0 / 12), 1e-3);
}

namespace {
    template<typename dtype>
    void run_uniform(Handle *handle) {
        auto opr = handle->create_operator<UniformRNG>();
        Tensor<typename DTypeTrait<dtype>::ctype> t(
                handle, {TensorShape{200000}, dtype()});
        opr->exec(t.tensornd(), {});
        assert_uniform_correct(t.ptr(), t.layout().total_nr_elems());
    }

    template<typename dtype>
    void run_gaussian(Handle *handle) {
        using ctype = typename DTypeTrait<dtype>::ctype;
        auto opr = handle->create_operator<GaussianRNG>();
        opr->param().mean = 0.8;
        opr->param().std = 2.3;
        Tensor<ctype> t(handle, {TensorShape{200001}, dtype()});
        opr->exec(t.tensornd(), {});

        auto ptr = t.ptr();
        auto size = t.layout().total_nr_elems();
        for (size_t i = 0; i < size; ++ i) {
            ASSERT_LE(std::abs(ptr[i] - 0.8), ctype(15));
        }
        auto stat = get_mean_var(ptr, size, ctype(0.8));
        ASSERT_LE(std::abs(stat.first - 0.8), 5e-3);
        ASSERT_LE(std::abs(stat.second - 2.3 * 2.3), 5e-2);
    }
}

TEST_F(NAIVE, UNIFORM_RNG_F32) {
    run_uniform<dtype::Float32>(handle());
}

TEST_F(NAIVE, UNIFORM_RNG_F16) {
    MEGDNN_INC_FLOAT16(run_uniform<dtype::Float16>(handle()));
}

TEST_F(NAIVE, GAUSSIAN_RNG_F32) {
    run_gaussian<dtype::Float32>(handle());
}

TEST_F(NAIVE, GAUSSIAN_RNG_F16) {
    MEGDNN_INC_FLOAT16(run_gaussian<dtype::Float16>(handle()));
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen



