/**
 * \file dnn/test/rocm/indexing_multi_axis_vec.cpp
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
#include "test/rocm/benchmarker.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/indexing_multi_axis_vec.h"
#include "test/common/index.h"

#include <random>

using namespace megdnn;
using namespace test;

namespace {

    class OrderedRNG final: public RNG {
        public:
            void gen(const TensorND &tensor) override {
                auto span = tensor.layout.span();
                if (tensor.layout.dtype == dtype::Float32()) {
                    auto ptr = tensor.ptr<float>() + span.low_elem;
                    for (size_t i = 0, it = span.dist_elem(); i < it; ++ i) {
                        ptr[i] = i;
                    }
                } else {
                    auto ptr = tensor.ptr<int>() + span.low_elem;
                    for (size_t i = 0, it = span.dist_elem(); i < it; ++ i) {
                        ptr[i] = i;
                    }
                }
            }
    };

    template<class Opr>
    void run_check(Handle *handle) {
        // see OprProxyIndexingMultiAxisVecHelper for more details
        // execs() give input, output and index layouts

        Checker<Opr> checker(handle);
        size_t idx_size0, idx_size1;
        OrderedRNG rng_inp;
        IndexRNG rng0{idx_size0, 2}, rng1{idx_size1, 3};
        checker.
            set_dtype(0, dtype::Float32()). // data
            set_dtype(1, dtype::Float32()). // value
            set_dtype(2, dtype::Int32()).   // idx0
            set_dtype(3, dtype::Int32()).   // idx1
            set_rng(0, &rng_inp).
            set_rng(1, &rng_inp).
            set_rng(2, &rng0).
            set_rng(3, &rng1);

        idx_size0 = 23;
        checker.
            set_proxy({{0}}).
            execs({{23}, {100}, {100}}).
            execs({{23, 5}, {100, 5}, {100}});

        idx_size0 = 2;
        idx_size1 = 3;
        checker.
            set_proxy({{0, 1}}).
            execs({{2, 3}, {10}, {10}, {10}}).
            execs({{2, 3, 5}, {10, 5}, {10}, {10}});

        idx_size0 = 4;
        idx_size1 = 6;
        TensorLayout inp_layout{{3, 4, 5, 6}, dtype::Float32()};
        inp_layout.stride[0] *= 8;
        inp_layout.stride[1] *= 2;
        checker.
            set_proxy({{1, 3}}).
            execl({inp_layout,
                    {{7, 3, 5}, dtype::Float32()},
                    {{7}, dtype::Int32()},
                    {{1}, dtype::Int32()},
                    });

        idx_size0 = 4;
        idx_size1 = 5;
        checker.
            set_proxy({{2, 3}}).
            execs({{2, 3, 4, 5, 6, 7}, {2, 3, 10, 6, 7}, {10}, {10}});

        idx_size0 = 4;
        checker.
            set_proxy({{1}}).
            execs({{1, 4}, {1, 1024 * 1024}, {1024 * 1024}});

        if (std::is_same<Opr, IndexingIncrMultiAxisVec>::value) {
            idx_size0 = 4;
            TensorLayout val_layout{{23}, dtype::Float32()};
            val_layout.stride[0] = 0;
            checker.
                set_proxy({{0}}).
                execl({{{4}, dtype::Float32()},
                        val_layout,
                        {{23}, dtype::Int32()}
                        });
        }
    }
}

TEST_F(ROCM, INDEXING_MULTI_AXIS_VEC) {
    run_check<IndexingMultiAxisVec>(handle_rocm());
    Checker<IndexingMultiAxisVec> checker(handle_rocm());
    size_t idx_size0;
    OrderedRNG rng_inp;
    IndexRNG rng0{idx_size0, 2};
    checker.
        set_dtype(0, dtype::Float32()). // data
        set_dtype(1, dtype::Float32()). // value
        set_dtype(2, dtype::Int32()).   // idx0
        set_rng(0, &rng_inp).
        set_rng(1, &rng_inp).
        set_rng(2, &rng0);

    idx_size0 = 20;
    checker.set_proxy({{0}})
        .execl({TensorLayout{{20}, dtype::Float32()},
                TensorLayout{{9}, dtype::Float32()},
                TensorLayout{TensorShape{9}, {-1}, dtype::Int32()}});
}

TEST_F(ROCM, INDEXING_INCR_MULTI_AXIS_VEC) {
    run_check<IndexingIncrMultiAxisVec>(handle_rocm());
}

TEST_F(ROCM, INDEXING_SET_MULTI_AXIS_VEC) {
    Checker<IndexingSetMultiAxisVec> checker(handle_rocm());
    OrderedRNG rng;
    checker.
        set_dtype(0, dtype::Float32()). // data
        set_dtype(1, dtype::Float32()). // value
        set_dtype(2, dtype::Int32()).   // idx0
        set_rng(0, &rng).
        set_rng(1, &rng).
        set_rng(2, &rng);

    checker.
        set_proxy({{1}}).
        execs({{5, 8, 3}, {5, 2, 3}, {2}});
}

TEST_F(ROCM_ERROR_INFO, INDEXING_MULTI_AXIS_VEC) {
    Checker<IndexingMultiAxisVec> checker(handle_rocm());
    UniformIntRNG idx_rng{-5, 5};
    checker.
        set_dtype(0, dtype::Float32()). // data
        set_dtype(1, dtype::Float32()). // value
        set_dtype(2, dtype::Int32()).   // idx
        set_rng(2, &idx_rng);

    bool failed = false;
    ASSERT_EQ(0u, get_error_info().nr_error);
    auto on_fail = [&failed, this]() {
        failed = true;
        auto info = get_error_info();
        ASSERT_GE(info.nr_error, 1u);
        printf("error msg: ");
        printf(info.msg, info.msg_args[0], info.msg_args[1], info.msg_args[2],
                info.msg_args[3]);
        printf("\n");
    };

    checker.
        set_proxy({{0}}).
        execs({{23}, {100}, {100}});

    idx_rng = {-500, 500};
    checker.
        set_expect_exec_fail(on_fail).
        execs({{23}, {100}, {100}});

    ASSERT_TRUE(failed);
}

TEST_F(ROCM, INDEXING_MULTI_AXIS_VEC_BENCHMARK) {
    ROCMBenchmarker<IndexingMultiAxisVec> benchmarker(handle_rocm(), handle_naive(false));
    benchmarker.set_display(true);
    OrderedRNG rng_inp;
    size_t idx_size = 10000;
    IndexRNG rng0{idx_size, 3}, rng1{idx_size, 1};
    benchmarker.
        set_dtype(0, dtype::Float32()).
        set_dtype(1, dtype::Float32()).
        set_dtype(2, dtype::Int32()).
        set_dtype(3, dtype::Int32()).
        set_rng(0, &rng_inp).
        set_rng(1, &rng_inp).
        set_rng(2, &rng0).
        set_rng(3, &rng1).
        set_proxy({{0, 1}});
    auto time_ms = benchmarker.execs({{1000, 1000, 1000}, {1000, 1000}, {1000}, {1000}});
    long io = 2 * 1000 * 1000 * dtype::Float32().size();
    printf("io = %.3f GB, random access bandwidth = %.3f GB/s\n",
           (float)(io / 1e9), (float)(io / (time_ms * 1e6)));
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

