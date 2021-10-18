/**
 * \file dnn/test/common/relayout.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs/general.h"

#include "src/common/relayout_helper.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/relayout.h"

using namespace megdnn;
using namespace test;
using namespace megdnn::relayout;
using namespace test::relayout;

namespace {
TestArg generate_transpose_args(
        size_t batch, size_t m, size_t n, size_t c, DType dtype) {
    TestArg arg;
    arg.src = TensorLayout(
            TensorShape{batch, n, m, c},
            {static_cast<std::ptrdiff_t>(n * m * c), static_cast<std::ptrdiff_t>(c),
             static_cast<std::ptrdiff_t>(n * c), 1},
            dtype);
    arg.dst = TensorLayout(TensorShape{batch, n, m, c}, dtype);
    return arg;
}
}  // anonymous namespace

namespace megdnn {
namespace test {
namespace relayout {

void run_test_cv(Handle* handle, size_t CH) {
    std::vector<TestArg> args;

    for (size_t M = 124; M <= 130; ++M) {
        for (size_t N = 124; N <= 130; ++N) {
            args.push_back(generate_transpose_args(1, M, N, CH, dtype::Uint8()));
            args.push_back(generate_transpose_args(1, M, N, CH, dtype::Int32()));
            args.push_back(generate_transpose_args(1, M, N, CH, dtype::Float32()));
            args.push_back(generate_transpose_args(3, M, N, CH, dtype::Float32()));
        }
    }

    Checker<Relayout> checker(handle);

    for (auto&& arg : args) {
        checker.execl({arg.src, arg.dst});
    }
}

#define DEF_TEST(name) \
    template <>        \
    void run_test<name>(Handle * handle)

DEF_TEST(cv) {
    run_test_cv(handle, 1);
}

DEF_TEST(cv_ch3) {
    run_test_cv(handle, 3);
}

DEF_TEST(cv_ch5) {
    run_test_cv(handle, 5);
}

DEF_TEST(broadcast) {
    std::vector<TestArg> args;
    TensorLayout src{{2, 3, 4}, dtype::Float32()}, dst{{2, 3, 4}, dtype::Float32()};

    src.stride[0] = 4;
    src.stride[1] = 0;
    args.emplace_back(src, dst);

    // last stride contiguous
    args.emplace_back(
            TensorLayout({3, 100, 2}, {2, 0, 1}, dtype::Float16()),
            TensorLayout({3, 100, 2}, {200, 2, 1}, dtype::Float16()));
    Checker<Relayout> checker(handle);

    for (auto&& arg : args) {
        checker.execl({arg.src, arg.dst});
    }
}

DEF_TEST(negative) {
    TensorLayout src{{7, 8, 10}, dtype::Float32()}, dst{{7, 8, 10}, dtype::Float32()};

    src.stride[0] *= -1;

    Checker<Relayout> checker(handle);
    checker.execl({src, dst});
}

DEF_TEST(transpose) {
    Checker<Relayout> checker(handle);
    {
        TensorLayout sl({8, 10}, dtype::Int32()), dl({10, 8}, dtype::Int32());
        sl = sl.dimshuffle({1, 0});
        checker.execl({sl, dl});
        checker.execl({dl, sl});
    }
    {
        TensorLayout sl({8, 10, 2}, dtype::Int32()), dl({2, 8, 10}, dtype::Int32());
        sl = sl.dimshuffle({2, 0, 1});
        checker.execl({sl, dl});
        checker.execl({dl, sl});
    }
}

#undef DEF_TEST

}  // namespace relayout
}  // namespace test
}  // namespace megdnn

void test::relayout::run_cv_benchmark(Handle* handle) {
    auto handle_naive = create_cpu_handle(2);
    std::vector<TestArg> args;

    args.push_back(generate_transpose_args(1, 255, 256, 1, dtype::Int32()));
    args.push_back(generate_transpose_args(1, 513, 1025, 3, dtype::Int32()));

    args.push_back(generate_transpose_args(1, 255, 256, 1, dtype::Uint8()));
    args.push_back(generate_transpose_args(1, 513, 1025, 3, dtype::Uint8()));

    args.push_back(generate_transpose_args(1, 255, 256, 3, dtype::Float32()));
    args.push_back(generate_transpose_args(1, 513, 1025, 1, dtype::Float32()));

    args.push_back(generate_transpose_args(2, 987, 573, 6, dtype::Float32()));

    Benchmarker<Relayout> benchmarker(handle);
    Benchmarker<Relayout> benchmarker_naive(handle_naive.get());

    Checker<Relayout> checker(handle);
    benchmarker_naive.set_times(1).set_display(false);
    benchmarker.set_times(1).set_display(false);
    for (auto&& arg : args) {
        checker.execl({arg.src, arg.dst});
        auto t0 = benchmarker.execl({arg.src, arg.dst});
        auto t1 = benchmarker_naive.execl({arg.src, arg.dst});
        double k = arg.dst.span().dist_byte() * 1e3 / (1024 * 1024 * 1024);
        printf("cur=%7.3fms,%5.2fGiB/s naive=%7.3fms,%5.2fGiB/s %s %s\n", t0, k / t0,
               t1, k / t1, arg.dst.TensorShape::to_string().c_str(),
               arg.dst.dtype.name());
    }
}
TEST(RELAYOUT, TRANSPOSE_DET) {
    auto run = [](const TensorShape& shape, const std::vector<size_t>& dimshuffle,
                  bool expect_is_transpose, const TransposeParam& p = {}) {
        TensorLayout src{shape, dtype::Float32{}};
        src = src.dimshuffle(dimshuffle).collapse_contiguous();
        TensorLayout dst{TensorShape{src.total_nr_elems()}, src.dtype};
        TransposeParam p_get;
        bool succ = is_transpose(src, dst, p_get);
        ASSERT_EQ(expect_is_transpose, succ);
        if (succ) {
            ASSERT_EQ(p_get.batch, p.batch);
            ASSERT_EQ(p_get.m, p.m);
            ASSERT_EQ(p_get.n, p.n);
            ASSERT_EQ(p_get.c, p.c);
        }
        // swap m, n
        succ = is_transpose(dst, src, p_get);
        ASSERT_EQ(expect_is_transpose, succ);
        if (succ) {
            ASSERT_EQ(p_get.batch, p.batch);
            ASSERT_EQ(p_get.m, p.n);
            ASSERT_EQ(p_get.n, p.m);
            ASSERT_EQ(p_get.c, p.c);
        }
    };
    run({2, 3}, {1, 0}, true, {1, 2, 3, 1, 0});
    run({2, 3, 5}, {1, 0, 2}, true, {1, 2, 3, 5, 0});
    run({2, 3, 5}, {0, 2, 1}, true, {2, 3, 5, 1, 0});
    run({3, 2, 3, 5}, {0, 2, 1, 3}, true, {3, 2, 3, 5, 0});
    run({3, 2, 3, 5}, {0, 1, 3, 2}, true, {6, 3, 5, 1, 0});
    run({2, 3, 5}, {2, 1, 0}, false);
    run({3, 2, 3, 5}, {3, 2, 1, 0}, false);
}
// vim: syntax=cpp.doxygen
