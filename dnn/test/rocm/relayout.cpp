/**
 * \file dnn/test/rocm/relayout.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "test/rocm/fixture.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/benchmarker.h"
#include "test/common/relayout.h"
#include "test/rocm/benchmarker.h"

using namespace megdnn;
using namespace test;

namespace {
template<typename tag>
class ROCM_RELAYOUT: public ROCM {
};
TYPED_TEST_CASE(ROCM_RELAYOUT, relayout::test_types);
TYPED_TEST(ROCM_RELAYOUT, run) {
    relayout::run_test<TypeParam>(this->handle_rocm());
}
}

TEST_F(ROCM, RELAYOUT_MEMCPY_ASYNC) {
    Checker<Relayout> checker(handle_rocm());
    checker.set_epsilon(1e-3);
    struct Arg {
        TensorLayout src, dst;
        Arg(TensorLayout src, TensorLayout dst) : src(src), dst(dst) {}
    };
    std::vector<Arg> args;
    // test for contig
    args.emplace_back(Arg{{{51200}, {1}, dtype::Float32()},
                          {{51200}, {1}, dtype::Float32()}});

    // test for copy_2d
    args.emplace_back(Arg{{{51200}, {9}, dtype::Float32()},
                          {{51200}, {1}, dtype::Float32()}});

    for (auto&& arg : args) {
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .execl({arg.src, arg.dst});
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(ROCM, RELAYOUT_BENCHMARK) {
    //! benchmark contious layout, such as (a, b, c, d) -> (b, a, c,d)
    //! just change the first two axis
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), true);
    auto benchmarker = ROCMBenchmarker<RelayoutForward>(handle_rocm(),
                                                        handle_naive(false));
    benchmarker.set_display(true);

    auto run = [&](const TensorLayoutArray& layouts) {
        for (auto&& layout : layouts) {
            TensorLayout src = layout.dimshuffle({1, 0, 2});
            TensorLayout dst = layout;
            std::swap(dst.shape[0], dst.shape[1]);
            dst.init_contiguous_stride();
            benchmarker.execl({src, dst});
            auto used = benchmarker.execl({src, dst});
            used = benchmarker.execl({src, dst});
            printf("layout: %s bandwith: %f gbps/s\n",
                   layout.to_string().c_str(),
                   2 * layout.total_nr_elems() * layout.dtype.size() / used *
                           1000 / (1024 * 1024 * 1024));
        }

    };

    TensorLayoutArray layouts = {
            {{12, 23, 2}, dtype::Int32()},
            {{12, 23, 8}, dtype::Int32()},
            {{12, 23, 17}, dtype::Int32()},
            {{12, 23, 64}, dtype::Int32()},
            {{12, 23, 129}, dtype::Int32()},
            {{12, 23, 256}, dtype::Int32()},
            {{12, 23, 1029}, dtype::Int32()},
            {{12, 23, 4096}, dtype::Int32()},
            {{12, 23, 9143}, dtype::Int32()},
            {{12, 23, 18284}, dtype::Int32()},
            {{2, 2, 1000000}, dtype::Int32()},
    };
    run(layouts);

    auto run2 = [&](const TensorLayoutArray& layouts) {
        for (auto&& layout : layouts) {
            TensorLayout src = layout.dimshuffle({0, 2, 1, 3});
            TensorLayout dst = layout;
            std::swap(dst.shape[0], dst.shape[1]);
            dst.init_contiguous_stride();
            benchmarker.execl({src, dst});
            auto used = benchmarker.execl({src, dst});
            used = benchmarker.execl({src, dst});
            printf("layout: %s bandwith: %f gbps/s\n",
                   layout.to_string().c_str(),
                   2 * layout.total_nr_elems() * layout.dtype.size() / used *
                           1000 / (1024 * 1024 * 1024));
        }

    };

    layouts = {
            {{3, 12, 24, 100}, dtype::Int32()},
            {{3, 12, 24, 1029}, dtype::Int32()},
            {{3, 4, 24, 9143}, dtype::Int32()},
            {{3, 4, 24, 18284}, dtype::Int32()},
    };

    run2(layouts);
}

TEST_F(ROCM, RELAYOUT_LAST_CONTIG_BENCHMARK) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), true);
    auto benchmarker = ROCMBenchmarker<RelayoutForward>(handle_rocm(),
                                                        handle_naive(false));
    benchmarker.set_display(true);

    TensorLayout src =
            TensorLayout({5, 5, 100000}, {800000, 100000, 1}, dtype::Float32());
    TensorLayout dst =
            TensorLayout({5, 5, 100000}, {700000, 100000, 1}, dtype::Float32());
    benchmarker.execl({src, dst});
    auto used = benchmarker.execl({src, dst});
    used = benchmarker.execl({src, dst});
    printf("src: %s dst: %s bandwith: %f gbps/s\n", src.to_string().c_str(),
           dst.to_string().c_str(),
           2 * src.total_nr_elems() * src.dtype.size() / used * 1000 /
                   (1024 * 1024 * 1024));
}
#endif

TEST_F(ROCM, RELAYOUT) {
    struct Arg {
        TensorLayout src, dst;
        Arg(TensorLayout src, TensorLayout dst) : src(src), dst(dst) {}
    };
    std::vector<Arg> args;
#if !MEGDNN_DISABLE_FLOAT16
    {
        // contiguous stride
        args.emplace_back(TensorLayout({4, 3, 2}, {2, 8, 1}, dtype::Float16()),
                          TensorLayout({4, 3, 2}, {6, 2, 1}, dtype::Float16()));
        args.emplace_back(TensorLayout({4, 3, 2}, {6, 2, 1}, dtype::Float16()),
                          TensorLayout({4, 3, 2}, {2, 8, 1}, dtype::Float16()));
        args.emplace_back(
                TensorLayout({2, 4, 3, 5}, {60, 5, 20, 1}, dtype::Float16()),
                TensorLayout({2, 4, 3, 5}, {60, 15, 5, 1}, dtype::Float16()));
    }
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Float16()),
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Float16()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Float16()),
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Float16()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Float16()),
            TensorLayout({2, 3, 4, 5}, {180, 60, 15, 3}, dtype::Float16()));
#endif
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Int32()),
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int32()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int32()),
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Int32()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int32()),
            TensorLayout({2, 3, 4, 5}, {180, 60, 15, 3}, dtype::Int32()));
    {
        // 1d
        size_t n = 10000;
        args.emplace_back(TensorLayout({n}, {1}, dtype::Int32()),
                          TensorLayout({n}, {1}, dtype::Int32()));
        args.emplace_back(TensorLayout({n}, {1}, dtype::Int32()),
                          TensorLayout({n}, {2}, dtype::Int32()));
        args.emplace_back(TensorLayout({n}, {2}, dtype::Int32()),
                          TensorLayout({n}, {1}, dtype::Int32()));
        args.emplace_back(TensorLayout({n}, {2}, dtype::Int32()),
                          TensorLayout({n}, {2}, dtype::Int32()));
    }
    {
        // 2d
        size_t m = 200, n = 300, k = 400;
        ptrdiff_t k2 = k * 2;
        args.emplace_back(TensorLayout({m, n}, {k2, 2}, dtype::Int32()),
                          TensorLayout({m, n}, {k2 + 1, 2}, dtype::Int32()));
        args.emplace_back(TensorLayout({m, n}, {2, k2}, dtype::Int32()),
                          TensorLayout({m, n}, {2, k2 + 1}, dtype::Int32()));
        args.emplace_back(TensorLayout({m, n}, {2, k2}, dtype::Int32()),
                          TensorLayout({m, n}, {k2 + 1, 2}, dtype::Int32()));
        args.emplace_back(TensorLayout({m, n}, {k2, 2}, dtype::Int32()),
                          TensorLayout({m, n}, {2, k2 + 1}, dtype::Int32()));
        args.emplace_back(TensorLayout({m, n}, {k2, 1}, dtype::Int32()),
                          TensorLayout({m, n}, {k2 + 1, 1}, dtype::Int32()));
        args.emplace_back(TensorLayout({m, n}, {1, k2}, dtype::Int32()),
                          TensorLayout({m, n}, {1, k2 + 1}, dtype::Int32()));
        args.emplace_back(TensorLayout({m, n}, {1, k2}, dtype::Int32()),
                          TensorLayout({m, n}, {k2 + 1, 1}, dtype::Int32()));
        args.emplace_back(TensorLayout({m, n}, {k2, 1}, dtype::Int32()),
                          TensorLayout({m, n}, {1, k2 + 1}, dtype::Int32()));
    }
    {
        // 3d
        size_t m = 20, n = 30, k = 40;
        ptrdiff_t k2 = k;
        args.emplace_back(
                TensorLayout({m, n, k}, {k2 * k2 * 4, k2 * 3, 2},
                             dtype::Int32()),
                TensorLayout({m, n, k}, {2 * k2 * k2 * k2 * 4, k2 * 3, 2},
                             dtype::Int32()));
    }
    {
        // simplify_layout
        // 234..56
        // 2..3456
        args.emplace_back(
                TensorLayout(
                        {2, 3, 4, 5, 6},
                        {2 * 3 * 4 * 5 * 6, 2 * 4 * 5 * 6, 2 * 5 * 6, 6, 1},
                        dtype::Int32()),
                TensorLayout({2, 3, 4, 5, 6},
                             {4 * 3 * 4 * 5 * 6, 4 * 5 * 6, 5 * 6, 6, 1},
                             dtype::Int32()));
    }

    Checker<Relayout> checker(handle_rocm());
    for (auto&& arg : args) {
        checker.exec(TensorLayoutArray{arg.src, arg.dst});
    }
}

// vim: syntax=cpp.doxygen
