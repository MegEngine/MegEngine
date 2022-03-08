/**
 * \file dnn/test/fallback/elemwise.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/fallback/fixture.h"

#include <ctime>
#include "test/common/checker.h"
#include "test/common/elemwise.h"
#include "test/common/task_record_check.h"
#include "test/common/tensor.h"

using namespace megdnn;
using namespace test;

template <typename tag>
class FALLBACK_ELEMWISE : public FALLBACK {};
TYPED_TEST_CASE(FALLBACK_ELEMWISE, elemwise::test_types);
TYPED_TEST(FALLBACK_ELEMWISE, run) {
    elemwise::run_test<TypeParam>(this->handle());
}
TEST_F(FALLBACK, ELEMWISE_RECORD) {
    TaskRecordChecker<Elemwise> checker{1};
    checker.set_param({Elemwise::Mode::ADD});
    checker.set_dtype(0, dtype::Float32());
    checker.set_dtype(1, dtype::Float32());
    checker.set_dtype(2, dtype::Float32());
    UniformIntRNG rng{-100, 100};
    checker.set_rng(0, &rng);
    checker.set_rng(1, &rng);
    checker.set_rng(2, &rng);
    checker.execs({{10, 10, 32}, {10, 10, 32}, {}});
}


TEST_F(FALLBACK, ELEMWISE_FORWARD_TERNARY) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle());
    checker.set_param(Mode::FUSE_MUL_ADD3);

    auto run = [&] {
        //! nchw44
        checker.execs({{1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.execs({{1, 3, 1, 1, 4}, {2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.execs({{1, 8, 1, 1, 4}, {3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {}});
        checker.execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.execs({{1, 2, 1, 1, 4}, {1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {}});

        //! nchw44
        checker.execs({{1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {}});
        checker.execs({{2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {2, 3, 2, 2, 4}, {}});
        checker.execs({{3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {3, 8, 5, 3, 4}, {}});
        checker.execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.execs({{1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {1, 2, 5, 7, 4}, {}});

        //! nchw88
        checker.execs({{1, 3, 1, 1, 8}, {1, 3, 2, 2, 8}, {1, 3, 1, 1, 8}, {}});
        checker.execs({{1, 3, 1, 1, 8}, {2, 3, 2, 2, 8}, {1, 3, 1, 1, 8}, {}});
        checker.execs({{1, 8, 1, 1, 8}, {3, 8, 5, 3, 8}, {1, 8, 1, 1, 8}, {}});
        checker.execs({{3, 4, 5, 7, 8}, {3, 4, 5, 7, 8}, {3, 4, 5, 7, 8}, {}});
        checker.execs({{1, 2, 1, 1, 8}, {1, 2, 5, 7, 8}, {1, 2, 1, 1, 8}, {}});

        //! nchw88
        checker.execs({{1, 3, 2, 2, 8}, {1, 3, 1, 1, 8}, {1, 3, 2, 2, 8}, {}});
        checker.execs({{2, 3, 2, 2, 8}, {1, 3, 1, 1, 8}, {2, 3, 2, 2, 8}, {}});
        checker.execs({{3, 8, 5, 3, 8}, {1, 8, 1, 1, 8}, {3, 8, 5, 3, 8}, {}});
        checker.execs({{3, 4, 5, 7, 8}, {3, 4, 5, 7, 8}, {3, 4, 5, 7, 8}, {}});
        checker.execs({{1, 2, 5, 7, 8}, {1, 2, 1, 1, 8}, {1, 2, 5, 7, 8}, {}});

        checker.execs({{3, 4, 7}, {3, 4, 7}, {3, 4, 7}, {}});
        checker.execs({{1, 4, 1, 1}, {3, 4, 5, 7}, {1, 4, 1, 1}, {}});
        checker.execs({{1, 4, 1}, {3, 4, 7}, {1, 4, 1}, {}});
        checker.execs({{3, 4, 5, 7}, {3, 4, 5, 7}, {1, 1, 1, 1}, {}});
        checker.execs({{1, 7}, {1, 7}, {1, 7}, {}});
        checker.execs({{1, 2, 1}, {1, 2, 2}, {1, 2, 1}, {}});
        checker.execs({{1, 2, 2}, {1, 2, 2}, {1, 1, 1}, {}});
        checker.execs({{3, 4, 1}, {3, 4, 1}, {3, 4, 1}, {}});
        checker.execs({{3, 4, 5}, {1}, {1}, {}});
        checker.execs({{1}, {3, 4, 5}, {1}, {}});
    };

    // case int
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    checker.set_dtype(2, dtype::Int8());
    run();

    checker.set_dtype(0, dtype::Int16());
    checker.set_dtype(1, dtype::Int16());
    checker.set_dtype(2, dtype::Int16());
    run();

    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());
    checker.set_dtype(2, dtype::Int32());
    run();

    // case float
    UniformFloatRNG rng(1e-5, 7e1);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-5);
    checker.set_dtype(0, dtype::Float32());
    checker.set_dtype(1, dtype::Float32());
    checker.set_dtype(2, dtype::Float32());
    run();
}

TEST_F(FALLBACK, ELEMWISE_FORWARD_NCHW44_INT8_INT16_INT32) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle());

    auto run = [&]() {
        // VEC_BCAST101x not PowOp
        checker.set_param(Mode::ADD).execs({{1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.set_param(Mode::ADD).execs({{2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.set_param(Mode::ADD).execs({{3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {}});
        checker.set_param(Mode::ADD).execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.set_param(Mode::ADD).execs({{1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {}});
        checker.set_param(Mode::RMULH).execs({{1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.set_param(Mode::RMULH).execs({{2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.set_param(Mode::RMULH).execs({{3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {}});
        checker.set_param(Mode::RMULH).execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.set_param(Mode::RMULH).execs({{1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {}});
        // BCAST101x_VEC not PowOp
        checker.set_param(Mode::ADD).execs({{1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {}});
        checker.set_param(Mode::ADD).execs({{1, 3, 1, 1, 4}, {2, 3, 2, 2, 4}, {}});
        checker.set_param(Mode::ADD).execs({{1, 8, 1, 1, 4}, {3, 8, 5, 3, 4}, {}});
        checker.set_param(Mode::ADD).execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.set_param(Mode::ADD).execs({{1, 2, 1, 1, 4}, {1, 2, 5, 7, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{1, 3, 1, 1, 4}, {2, 3, 2, 2, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{1, 8, 1, 1, 4}, {3, 8, 5, 3, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.set_param(Mode::FUSE_ADD_RELU)
                .execs({{1, 2, 1, 1, 4}, {1, 2, 5, 7, 4}, {}});
    };
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    run();
    checker.set_dtype(0, dtype::Int16());
    checker.set_dtype(1, dtype::Int16());
    run();
    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());
    run();
}

TEST_F(FALLBACK, ELEMWISE_FORWARD_NCHW44_FP32) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle());

    UniformFloatRNG rng(1e-5, 7e1);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-5);
    checker.set_dtype(0, dtype::Float32());
    checker.set_dtype(1, dtype::Float32());

    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 3, 1, 1, 4}, {2, 3, 2, 2, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 8, 1, 1, 4}, {3, 8, 5, 3, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 2, 1, 1, 4}, {1, 2, 5, 7, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {}});

    auto run = [&](Mode mode) {
        // VEC_BCAST101x
        checker.set_param(mode).execs({{1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.set_param(mode).execs({{2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.set_param(mode).execs({{3, 8, 5, 3, 4}, {1, 8, 1, 1, 4}, {}});
        checker.set_param(mode).execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.set_param(mode).execs({{1, 2, 5, 7, 4}, {1, 2, 1, 1, 4}, {}});
        // BCAST101x_VEC not powOp
        checker.set_param(mode).execs({{1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {}});
        checker.set_param(mode).execs({{1, 3, 1, 1, 4}, {2, 3, 2, 2, 4}, {}});
        checker.set_param(mode).execs({{1, 8, 1, 1, 4}, {3, 8, 5, 3, 4}, {}});
        checker.set_param(mode).execs({{3, 4, 5, 7, 4}, {3, 4, 5, 7, 4}, {}});
        checker.set_param(mode).execs({{1, 2, 1, 1, 4}, {1, 2, 5, 7, 4}, {}});
    };
    run(Mode::ADD);
    run(Mode::FUSE_ADD_H_SWISH);
    run(Mode::FUSE_ADD_RELU);
    run(Mode::MAX);
    run(Mode::MIN);
    run(Mode::MUL);
    run(Mode::SUB);
    run(Mode::TRUE_DIV);
    run(Mode::POW);
}

TEST_F(FALLBACK, ELEMWISE_FORWARD_NCHW88_FP) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle());

    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 3, 1, 1, 8}, {1, 3, 2, 2, 8}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 3, 1, 1, 8}, {2, 3, 2, 2, 8}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 8, 1, 1, 8}, {3, 8, 5, 3, 8}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{3, 4, 5, 7, 8}, {3, 4, 5, 7, 8}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 2, 1, 1, 8}, {1, 2, 5, 7, 8}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 3, 2, 2, 8}, {1, 3, 1, 1, 8}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{2, 3, 2, 2, 8}, {1, 3, 1, 1, 8}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{3, 8, 5, 3, 8}, {1, 8, 1, 1, 8}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{3, 4, 5, 7, 8}, {3, 4, 5, 7, 8}, {}});
    checker.set_param(Mode::FUSE_ADD_RELU)
            .execs({{1, 2, 5, 7, 8}, {1, 2, 1, 1, 8}, {}});

    auto run = [&](Mode mode) {
        // VEC_BCAST101x
        checker.set_param(mode).execs({{1, 3, 2, 2, 8}, {1, 3, 1, 1, 8}, {}});
        checker.set_param(mode).execs({{2, 3, 2, 2, 8}, {1, 3, 1, 1, 8}, {}});
        checker.set_param(mode).execs({{3, 8, 5, 3, 8}, {1, 8, 1, 1, 8}, {}});
        checker.set_param(mode).execs({{3, 4, 5, 7, 8}, {3, 4, 5, 7, 8}, {}});
        checker.set_param(mode).execs({{1, 2, 5, 7, 8}, {1, 2, 1, 1, 8}, {}});
        // BCAST101x_VEC not powOp
        checker.set_param(mode).execs({{1, 3, 1, 1, 8}, {1, 3, 2, 2, 8}, {}});
        checker.set_param(mode).execs({{1, 3, 1, 1, 8}, {2, 3, 2, 2, 8}, {}});
        checker.set_param(mode).execs({{1, 8, 1, 1, 8}, {3, 8, 5, 3, 8}, {}});
        checker.set_param(mode).execs({{3, 4, 5, 7, 8}, {3, 4, 5, 7, 8}, {}});
        checker.set_param(mode).execs({{1, 2, 1, 1, 8}, {1, 2, 5, 7, 8}, {}});
    };
    auto run_all = [&]() {
        run(Mode::ADD);
        run(Mode::FUSE_ADD_H_SWISH);
        run(Mode::FUSE_ADD_RELU);
        run(Mode::MAX);
        run(Mode::MIN);
        run(Mode::MUL);
        run(Mode::SUB);
        run(Mode::TRUE_DIV);
        run(Mode::POW);
    };

    {
        UniformFloatRNG rng(1e-5, 7e1);
        checker.set_rng(0, &rng);
        checker.set_epsilon(1e-5);
        checker.set_dtype(0, dtype::Float32());
        checker.set_dtype(1, dtype::Float32());
        run_all();
    }
}

TEST_F(FALLBACK, ELEMWISE_FORWARD_N1HW_FP32_BCAST) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle());

    UniformFloatRNG rng(1e-5, 7e1);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-5);
    checker.set_dtype(0, dtype::Float32());
    checker.set_dtype(1, dtype::Float32());

    //! 2 dim
    auto run = [&](Mode mode) {
        // VEC_BCASTX0X
        checker.set_param(mode).execs({{2, 8, 4, 4}, {2, 1, 4, 4}, {}});
        checker.set_param(mode).execs({{4, 21, 78}, {4, 1, 78}, {}});
        // BCASTX0X_VEC
        checker.set_param(mode).execs({{2, 1, 4, 4}, {2, 8, 4, 4}, {}});
        checker.set_param(mode).execs({{4, 1, 78}, {4, 21, 78}, {}});
    };
    run(Mode::ADD);
    run(Mode::MUL);
    run(Mode::SUB);
}

TEST_F(FALLBACK, ELEMWISE_FORWARD_TERNARY_RECORD) {
    using Mode = ElemwiseForward::Param::Mode;
    TaskRecordChecker<ElemwiseForward> checker(0);
    checker.set_param(Mode::FUSE_MUL_ADD3);

    auto run = [&] {
        //! nchw44
        checker.execs({{1, 3, 1, 1, 4}, {1, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});
        checker.execs({{1, 3, 1, 1, 4}, {2, 3, 2, 2, 4}, {1, 3, 1, 1, 4}, {}});

        //! nchw88
        checker.execs({{1, 3, 1, 1, 8}, {1, 3, 2, 2, 8}, {1, 3, 1, 1, 8}, {}});
        checker.execs({{1, 3, 1, 1, 8}, {2, 3, 2, 2, 8}, {1, 3, 1, 1, 8}, {}});

        checker.execs({{3, 4, 7}, {3, 4, 7}, {3, 4, 7}, {}});
        checker.execs({{1, 4, 1, 1}, {3, 4, 5, 7}, {1, 4, 1, 1}, {}});
    };

    // case int
    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());
    checker.set_dtype(2, dtype::Int32());
    run();

    // case float
    UniformFloatRNG rng(1e-5, 7e1);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-5);
    checker.set_dtype(0, dtype::Float32());
    checker.set_dtype(1, dtype::Float32());
    checker.set_dtype(2, dtype::Float32());
    run();
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(FALLBACK, BENCHMARK_ELEMWISE) {
    auto naive_handle = create_cpu_handle(2);
    auto run = [&](const TensorShape& shp0, const TensorShape& shp1) {
        TensorShape shpo;
        Elemwise::deduce_shape({shp0, shp1}, shpo);
        Tensor<> op0(handle(), {shp0, dtype::Float32()}),
                op1(handle(), {shp1, dtype::Float32()}),
                out(handle(), {shpo, dtype::Float32()});
        auto opr_cur = handle()->create_operator<Elemwise>();
        auto opr_naive = naive_handle->create_operator<Elemwise>();
        opr_cur->param() = {Elemwise::Mode::ADD};
        opr_naive->param() = {Elemwise::Mode::ADD};

        auto timeit = [&](Elemwise* opr) {
            opr->exec({op0.tensornd(), op1.tensornd()}, out.tensornd());
            auto start = clock();
            opr->exec({op0.tensornd(), op1.tensornd()}, out.tensornd());
            auto stop = clock();
            return (stop - start) * 1e3 / CLOCKS_PER_SEC;
        };
        auto t0 = timeit(opr_cur.get()), t1 = timeit(opr_naive.get());
        double tot_size_gb_ms =
                (op0.layout().span().dist_byte() + op1.layout().span().dist_byte() +
                 out.layout().span().dist_byte()) /
                1024.0 / 1024.0 / 1024.0 * 1e3;
        printf("%15s+%-15s: fallback=%7.3fms,%5.2fGiB/s "
               "naive=%7.3fms,%5.2fGiB/s\n",
               shp0.to_string().c_str(), shp1.to_string().c_str(), t0,
               tot_size_gb_ms / t0, t1, tot_size_gb_ms / t1);
    };
    // contig
    run({1024, 1024, 32}, {1024, 1024, 32});
    // bcast 101
    run({1024, 1024, 32}, {1, 1024, 1});
    // bcast 01
    run({4096 * 4, 1024}, {4096 * 4, 1});
    // bcast 10
    run({4096 * 4, 1024}, {1, 1024});

    // non-contig, fallback to naive
    run({1024, 1024, 32}, {1024, 1, 32});
}
#endif

// vim: syntax=cpp.doxygen
