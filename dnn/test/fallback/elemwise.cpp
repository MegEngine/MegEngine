/**
 * \file dnn/test/fallback/elemwise.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/fallback/fixture.h"

#include "test/common/checker.h"
#include "test/common/tensor.h"
#include "test/common/elemwise.h"

#include <ctime>

using namespace megdnn;
using namespace test;

template<typename tag>
class FALLBACK_ELEMWISE: public FALLBACK {
};
TYPED_TEST_CASE(FALLBACK_ELEMWISE, elemwise::test_types);
TYPED_TEST(FALLBACK_ELEMWISE, run) {
    elemwise::run_test<TypeParam>(this->handle());
}
#if MEGDNN_WITH_BENCHMARK
TEST_F(FALLBACK, BENCHMARK_ELEMWISE) {
    auto naive_handle = create_cpu_handle(2);
    auto run = [&](const TensorShape &shp0, const TensorShape &shp1) {
        TensorShape shpo;
        Elemwise::deduce_shape({shp0, shp1}, shpo);
        Tensor<> op0(handle(), {shp0, dtype::Float32()}),
                 op1(handle(), {shp1, dtype::Float32()}),
                 out(handle(), {shpo, dtype::Float32()});
        auto opr_cur = handle()->create_operator<Elemwise>();
        auto opr_naive = naive_handle->create_operator<Elemwise>();
        opr_cur->param() = {Elemwise::Mode::ADD};
        opr_naive->param() = {Elemwise::Mode::ADD};

        auto timeit = [&](Elemwise *opr) {
            opr->exec({op0.tensornd(), op1.tensornd()}, out.tensornd());
            auto start = clock();
            opr->exec({op0.tensornd(), op1.tensornd()}, out.tensornd());
            auto stop = clock();
            return (stop - start) * 1e3 / CLOCKS_PER_SEC;
        };
        auto t0 = timeit(opr_cur.get()),
             t1 = timeit(opr_naive.get());
        double tot_size_gb_ms = (
                op0.layout().span().dist_byte() +
                op1.layout().span().dist_byte() +
                out.layout().span().dist_byte()) /
            1024.0 / 1024.0 / 1024.0 * 1e3;
        printf("%15s+%-15s: fallback=%7.3fms,%5.2fGiB/s "
                "naive=%7.3fms,%5.2fGiB/s\n",
                shp0.to_string().c_str(), shp1.to_string().c_str(),
                t0, tot_size_gb_ms / t0, t1, tot_size_gb_ms / t1);
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



