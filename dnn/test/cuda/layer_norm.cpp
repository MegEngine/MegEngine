/**
 * \file dnn/test/cuda/layer_norm.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "test/cuda/fixture.h"

#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, LAYERNORM_FORWARD) {
    using Param = LayerNormForward::Param;
    Param param;
    param.affine = true;
    param.eps = 1e-6;
    param.normalized_dim = 1;
    Checker<LayerNormForward> checker(handle_cuda());
    checker.set_epsilon(1e-2);

    auto run = [&](DType d) {
        for (size_t n_slices : {10, 30})
            for (size_t slice_len : {10, 30}) {
                param.normalized_size = slice_len;
                checker.set_param(param)
                        .set_dtype(0, d)
                        .set_dtype(1, d)
                        .set_dtype(2, d)
                        .set_dtype(3, d)
                        .set_dtype(4, dtype::Float32())
                        .set_dtype(5, dtype::Float32())
                        .execs({{n_slices, slice_len},
                                {slice_len},
                                {slice_len},
                                {n_slices, slice_len},
                                {n_slices},
                                {n_slices}});
            }
    };

    run(dtype::Float32());
    run(dtype::Float16());
    run(dtype::BFloat16());
}

TEST_F(CUDA, LAYERNORM_BACKWARD) {
    using Param = LayerNormBackward::Param;
    Param param;
    param.affine = true;
    param.eps = 1e-6;
    param.normalized_dim = 1;
    Checker<LayerNormBackward> checker(handle_cuda());
    checker.set_epsilon(1e-1);

    auto run = [&](DType d) {
        for (size_t n_slices : {10, 30})
            for (size_t slice_len : {10, 30}) {
                param.normalized_size = slice_len;
                checker.set_param(param)
                        .set_dtype(0, d)
                        .set_dtype(1, d)
                        .set_dtype(2, d)
                        .set_dtype(3, dtype::Float32())
                        .set_dtype(4, dtype::Float32())
                        .set_dtype(5, d)
                        .set_dtype(6, d)
                        .set_dtype(7, d)
                        .execs({{n_slices, slice_len},
                                {n_slices, slice_len},
                                {slice_len},
                                {n_slices},
                                {n_slices},
                                {n_slices, slice_len},
                                {slice_len},
                                {slice_len}});
            }
    };

    run(dtype::Float32());
    run(dtype::Float16());
    run(dtype::BFloat16());
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
