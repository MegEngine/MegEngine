/**
 * \file dnn/test/naive/matrix_inverse.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs/linalg.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/naive/fixture.h"

using namespace megdnn;
using namespace test;

namespace {
void run_check(Handle* handle, const size_t B, const size_t N,
               const TensorShape& shp) {
    SyncedTensor<> input(handle, shp), output(handle, input.layout()),
            mul_check(handle, input.layout());

    {
        auto t = input.tensornd_host();
        InvertibleMatrixRNG{}.gen(t);
    }
    auto opr = handle->create_operator<MatrixInverse>();
    auto wk_size = opr->get_workspace_in_bytes(input.layout(), output.layout());
    std::unique_ptr<dt_byte[]> wk_storage{new dt_byte[wk_size]};
    opr->exec(input.tensornd_dev(), output.tensornd_dev(),
              {wk_storage.get(), wk_size});

    auto batch_mul = handle->create_operator<BatchedMatrixMul>();
    auto make_std_tensor = [B, N](SyncedTensor<>& t) {
        auto ret = t.tensornd_dev();
        ret.layout.ndim = 3;
        ret.layout[0] = B;
        ret.layout[1] = ret.layout[2] = N;
        ret.layout.init_contiguous_stride();
        return ret;
    };
    auto batch_mul_inp = make_std_tensor(input);
    auto batch_mul_wk_size = batch_mul->get_workspace_in_bytes(
            batch_mul_inp.layout, batch_mul_inp.layout, batch_mul_inp.layout);
    std::unique_ptr<dt_byte[]> batch_mul_wk{new dt_byte[batch_mul_wk_size]};
    batch_mul->exec(make_std_tensor(output), batch_mul_inp,
                    make_std_tensor(mul_check),
                    {batch_mul_wk.get(), batch_mul_wk_size});

    auto hptr = mul_check.ptr_host();
    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < N; ++k) {
                auto val = hptr[i * N * N + j * N + k];
                if (j == k) {
                    ASSERT_LT(std::abs(val - 1.f), 1e-4) << ssprintf(
                            "%zu,%zu,%zu/%zu,%zu: %g", i, j, k, N, B, val);
                } else {
                    ASSERT_LT(std::abs(val - 0.f), 1e-4) << ssprintf(
                            "%zu,%zu,%zu/%zu,%zu: %g", i, j, k, N, B, val);
                }
            }
        }
    }
}
}  // namespace

TEST_F(NAIVE, MATRIX_INVERSE) {
    run_check(handle(), 2, 1, {1, 2, 1, 1});
    run_check(handle(), 1, 2, {2, 2});
    run_check(handle(), 4, 3, {2, 2, 3, 3});
    run_check(handle(), 4, 23, {4, 23, 23});
    run_check(handle(), 1, 100, {100, 100});
    run_check(handle(), 100, 3, {100, 3, 3});
}

// vim: syntax=cpp.doxygen
