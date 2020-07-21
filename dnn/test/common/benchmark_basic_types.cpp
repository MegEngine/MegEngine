/**
 * \file dnn/test/common/benchmark_basic_types.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megdnn/basic_types.h"
#include "test/common/timer.h"
#include "test/common/utils.h"

#include <gtest/gtest.h>
#include <random>

using namespace megdnn;

namespace {

bool eq_shape0(const TensorShape& a, const TensorShape& b) {
    if (a.ndim != b.ndim)
        return false;
    return std::equal(a.shape, a.shape + a.ndim, b.shape);
}

bool eq_shape1(const TensorShape& a, const TensorShape& b) {
    if (a.ndim == b.ndim) {
        size_t eq = 0;
        switch (a.ndim) {
            case 7:
                eq += a.shape[6] == b.shape[6];
                MEGDNN_FALLTHRU
            case 6:
                eq += a.shape[5] == b.shape[5];
                MEGDNN_FALLTHRU
            case 5:
                eq += a.shape[4] == b.shape[4];
                MEGDNN_FALLTHRU
            case 4:
                eq += a.shape[3] == b.shape[3];
                MEGDNN_FALLTHRU
            case 3:
                eq += a.shape[2] == b.shape[2];
                MEGDNN_FALLTHRU
            case 2:
                eq += a.shape[1] == b.shape[1];
                MEGDNN_FALLTHRU
            case 1:
                eq += a.shape[0] == b.shape[0];
        }
        return eq == a.ndim;
    }
    return false;
}

bool eq_layout0(const TensorLayout& a, const TensorLayout& b) {
    if (!eq_shape0(a, b))
        return false;

    return std::equal(a.stride, a.stride + a.ndim, b.stride);
}

bool eq_layout1(const TensorLayout& a, const TensorLayout& b) {
    auto ax = [](size_t shape0, size_t shape1, ptrdiff_t stride0,
                 ptrdiff_t stride1) {
        return (shape0 == shape1) & ((shape0 == 1) | (stride0 == stride1));
    };
    if (a.ndim == b.ndim) {
        size_t eq = 0;
        switch (a.ndim) {
            case 7:
                eq += ax(a.shape[6], b.shape[6], a.stride[6], b.stride[6]);
                MEGDNN_FALLTHRU
            case 6:
                eq += ax(a.shape[5], b.shape[5], a.stride[5], b.stride[5]);
                MEGDNN_FALLTHRU
            case 5:
                eq += ax(a.shape[4], b.shape[4], a.stride[4], b.stride[4]);
                MEGDNN_FALLTHRU
            case 4:
                eq += ax(a.shape[3], b.shape[3], a.stride[3], b.stride[3]);
                MEGDNN_FALLTHRU
            case 3:
                eq += ax(a.shape[2], b.shape[2], a.stride[2], b.stride[2]);
                MEGDNN_FALLTHRU
            case 2:
                eq += ax(a.shape[1], b.shape[1], a.stride[1], b.stride[1]);
                MEGDNN_FALLTHRU
            case 1:
                eq += ax(a.shape[0], b.shape[0], a.stride[0], b.stride[0]);
        }
        return eq == a.ndim;
    }
    return false;
}

}  // anonymous namespace
// config NR_TEST at small memory device, eg, EV300 etc
static constexpr size_t NR_TEST = 10000;
TEST(BENCHMARK_BASIC_TYPES, EQ_SHAPE) {
    std::mt19937_64 rng;
    static TensorShape s0, s1[NR_TEST];
    auto init = [&rng](TensorShape& ts) {
        for (size_t i = 0; i < ts.ndim; ++i)
            ts.shape[i] = rng();
    };
    s0.ndim = rng() % TensorShape::MAX_NDIM + 1;
    init(s0);
    auto gen = [&](int type) {
        if (type == 0) {
            return s0;
        } else {
            TensorShape ret;
            if (type == 1)
                ret.ndim = s0.ndim;
            else
                ret.ndim = rng() % TensorShape::MAX_NDIM + 1;
            init(ret);
            return ret;
        }
    };
    s0 = gen(false);
    for (size_t i = 0; i < NR_TEST; ++i) {
        s1[i] = gen(rng() % 3);
    }

    int nr_diff = 0;
    test::Timer timer;
    timer.start();
    for (size_t i = 0; i < NR_TEST; ++i)
        nr_diff += eq_shape0(s1[i], s0);
    timer.stop();
    auto time0 = timer.get_time_in_us();

    timer.reset();
    timer.start();
    for (size_t i = 0; i < NR_TEST; ++i)
        nr_diff -= eq_shape1(s1[i], s0);
    timer.stop();
    auto time1 = timer.get_time_in_us();

    printf("time per  eq_shape: %.3fus vs %.3fus; diff=%d\n",
           time0 / double(NR_TEST), time1 / double(NR_TEST), nr_diff);
}

TEST(BENCHMARK_BASIC_TYPES, EQ_LAYOUT) {
    std::mt19937_64 rng;
    static TensorLayout s0, s1[NR_TEST];
    auto init = [&rng](TensorLayout& tl) {
        for (size_t i = 0; i < tl.ndim; ++i) {
            tl.shape[i] = rng();
            tl.stride[i] = rng();
        }
    };
    s0.ndim = rng() % TensorShape::MAX_NDIM + 1;
    init(s0);
    auto gen = [&](int type) {
        if (type == 0) {
            return s0;
        } else {
            TensorLayout ret;
            if (type == 1)
                ret.ndim = s0.ndim;
            else
                ret.ndim = rng() % TensorShape::MAX_NDIM + 1;
            init(ret);
            return ret;
        }
    };
    s0 = gen(false);
    for (size_t i = 0; i < NR_TEST; ++i) {
        s1[i] = gen(rng() % 3);
    }

    int nr_diff = 0;
    test::Timer timer;
    timer.start();
    for (size_t i = 0; i < NR_TEST; ++i)
        nr_diff += eq_layout0(s1[i], s0);
    timer.stop();
    auto time0 = timer.get_time_in_us();

    timer.reset();
    timer.start();
    for (size_t i = 0; i < NR_TEST; ++i)
        nr_diff -= eq_layout1(s1[i], s0);
    timer.stop();
    auto time1 = timer.get_time_in_us();

    printf("time per eq_layout: %.3fus vs %.3fus; diff=%d\n",
           time0 / double(NR_TEST), time1 / double(NR_TEST), nr_diff);
}
// vim: syntax=cpp.doxygen
