/**
 * \file dnn/test/fallback/relayout.cpp
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
#include "test/common/relayout.h"

#include "megdnn/basic_types.h"

#include <ctime>

using namespace megdnn;
using namespace test;

namespace {
template<typename tag>
class FALLBACK_RELAYOUT: public FALLBACK {
};
TYPED_TEST_CASE(FALLBACK_RELAYOUT, relayout::test_types);
TYPED_TEST(FALLBACK_RELAYOUT, run) {
    relayout::run_test<TypeParam>(this->handle());
}
}
#if MEGDNN_WITH_BENCHMARK
TEST_F(FALLBACK, BENCHMARK_RELAYOUT_CV) {
    relayout::run_cv_benchmark(handle());
}


TEST_F(FALLBACK, BENCHMARK_RELAYOUT) {
    auto naive_handle = create_cpu_handle(2);
    bool verbose = false;

    auto run = [&](
            bool out_cont,
            const TensorLayout &cont_layout,
            const TensorLayout &noncont_layout) {
        megdnn_assert(cont_layout.dtype == dtype::Int32() &&
                noncont_layout.dtype == dtype::Int32() &&
                noncont_layout.span().low_byte == 0);
        auto noncont_storage_size = noncont_layout.span().high_elem;
        Tensor<dt_int32>
            noncont_storage0(handle(),
                    {{noncont_storage_size}, dtype::Int32()}),
            noncont_storage1(handle(),
                    {{noncont_storage_size}, dtype::Int32()}),
            cont_storage0(handle(), cont_layout),
            cont_storage1(handle(), cont_layout);

        auto noncont0 = noncont_storage0.tensornd(),
             noncont1 = noncont_storage1.tensornd();
        noncont0.layout = noncont_layout;
        noncont1.layout = noncont_layout;

        TensorND src, dst0, dst1;
        if (out_cont) {
            src = noncont0;
            dst0 = cont_storage0.tensornd();
            dst1 = cont_storage1.tensornd();
            auto ptr = src.ptr<int>();
            for (size_t i = 0; i < noncont_storage_size; ++ i) {
                ptr[i] = i;
            }
        } else {
            memset(noncont_storage0.ptr(), -1,
                    noncont_storage0.layout().span().dist_byte());
            memset(noncont_storage1.ptr(), -1,
                    noncont_storage1.layout().span().dist_byte());
            src = cont_storage0.tensornd();
            dst0 = noncont0;
            dst1 = noncont1;
            auto ptr = src.ptr<int>();
            for (size_t i = 0, it = src.layout.total_nr_elems(); i < it; ++ i) {
                ptr[i] = i;
            }
        }
        auto opr_cur = handle()->create_operator<Relayout>();
        auto opr_naive = naive_handle->create_operator<Relayout>();

        auto timeit = [&src](Relayout *opr, TensorND out) {
            opr->exec(src, out);
            auto start = clock();
            opr->exec(src, out);
            auto stop = clock();
            return (stop - start) * 1e3 / CLOCKS_PER_SEC;
        };
        auto t1 = timeit(opr_naive.get(), dst1),
             t0 = timeit(opr_cur.get(), dst0);
        double tot_size_gb_ms = cont_layout.total_nr_elems() * sizeof(int) /
            1024.0 / 1024.0 / 1024.0 * 1e3;
        if (verbose) {
            printf("noncont-%zu dir=%d: fallback=%7.3fms,%5.2fGiB/s "
                    "naive=%7.3fms,%5.2fGiB/s\n",
                    noncont_layout.collapse_contiguous().ndim, out_cont,
                    t0, tot_size_gb_ms / t0, t1, tot_size_gb_ms / t1);
        }

        ASSERT_EQ(0, memcmp(dst0.ptr<int>(), dst1.ptr<int>(),
                    dst0.layout.span().dist_byte()));
    };

    auto run_preset = [&](const TensorShape &noncont_shp, int swap, bool sub,
            bool out_cont) {

        TensorLayout noncont_layout(noncont_shp, dtype::Int32());
        if (swap) {
            auto a = swap - 1, b = swap;
            std::swap(noncont_layout.shape[a], noncont_layout.shape[b]);
            std::swap(noncont_layout.stride[a], noncont_layout.stride[b]);
        }

        TensorLayout cont_layout = noncont_layout;
        cont_layout.init_contiguous_stride();

        TensorShape noncont_storage_shp(cont_layout);
        if (sub) {
            ++ noncont_storage_shp[noncont_layout.ndim - 1];
            noncont_layout.init_contiguous_stride(noncont_storage_shp);
            -- noncont_layout.shape[noncont_layout.ndim - 1];
        }

        run(out_cont, cont_layout, noncont_layout);
    };
    for (bool out_cont: {false, true}) {
        verbose = false;
        run_preset({2, 3}, 1, false, out_cont);
        run_preset({2, 2, 2}, 0, true, out_cont);
        {
            // padding-like
            TensorLayout cont{{2, 3, 3}, dtype::Int32()}, noncont = cont;
            noncont.stride[1] = 5;
            noncont.stride[0] = 25;
            run(out_cont, cont, noncont);
        }

        verbose = true;
        run_preset({1234, 5678}, 0, false, out_cont);
        run_preset({256, 256, 256}, 0, true, out_cont);
        run_preset({2, 3, 1024, 1024}, 1, false, out_cont);
        run_preset({1025, 2049}, 1, false, out_cont);
        run_preset({2049, 1025}, 1, false, out_cont);
        run_preset({10, 1024, 1024}, 2, false, out_cont);

        {
            // padding-like
            TensorLayout cont{{60, 60, 60}, dtype::Int32()}, noncont = cont;
            noncont.stride[1] = 63;
            noncont.stride[0] = 63 * 63;
            run(out_cont, cont, noncont);
        }
    }
}
#endif

// vim: syntax=cpp.doxygen
