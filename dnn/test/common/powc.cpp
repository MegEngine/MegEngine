/**
 * \file dnn/test/common/powc.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/common/powc.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

void test::run_powc_test(Handle* handle, DType dtype) {
    Checker<PowC> checker{handle};
    checker.set_dtype(0, dtype);

    float dt_val_max;
    if (dtype == dtype::Float32{}) {
        dt_val_max = DTypeTrait<dt_float32>::max();
    } else {
        megdnn_assert(dtype == dtype::Float16{});
        dt_val_max = DTypeTrait<dt_float16>::max();
        checker.set_epsilon(1e-2);
    }

    dt_val_max /= 4;

    for (float exp : {0.f, 1.f / 3.f, 1.f / 3.f + 0.01f, .5f, 1.f, 1.2f, 2.f,
                      3.f, 4.f, 7.f, 8.f}) {
        float rng_max = exp ? std::pow(dt_val_max, std::min(1.f / exp, 1.f))
                            : dt_val_max;
        bool allow_neg;
        {
            auto d = exp - std::floor(exp);
            if (d >= .1f) {
                allow_neg = false;
            } else {
                allow_neg = true;
            }
        }
        UniformFloatRNG rng0{-rng_max, rng_max}, rng1{0.f, rng_max};
        checker.set_rng(0, allow_neg ? &rng0 : &rng1);
        checker.set_param(exp);
        checker.execs({TensorShape{23, 34}, {}});
        if (::testing::Test::HasFailure()) {
            printf("failed for %g\n", exp);
            return;
        }

        UniformFloatNonZeroRNG rng2{1.f / rng_max, dt_val_max};
        UniformFloatRNG rng3{1.f / rng_max, dt_val_max};
        if (allow_neg) {
            checker.set_rng(0, &rng2);
        } else {
            checker.set_rng(0, &rng3);
        }
        checker.set_param(-exp);
        checker.execs({TensorShape{3, 7, 2}, {}});
        if (::testing::Test::HasFailure()) {
            printf("failed for %g\n", -exp);
            return;
        }

        // non contig
        TensorLayout layout{{4, 9}, dtype};
        layout.stride[0] *= 3;
        layout.stride[1] *= 2;
        checker.execl({layout, {}});
        if (::testing::Test::HasFailure()) {
            printf("failed for %g noncontig\n", -exp);
            return;
        }
    }
}

// vim: syntax=cpp.doxygen

