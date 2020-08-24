/**
 * \file dnn/test/cuda/relayout.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/common/relayout.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/fixture.h"
#include "test/common/rng.h"

using namespace megdnn;
using namespace test;

namespace {
template <typename tag>
class CUDA_RELAYOUT : public CUDA {};
TYPED_TEST_CASE(CUDA_RELAYOUT, relayout::test_types);
TYPED_TEST(CUDA_RELAYOUT, run) {
    relayout::run_test<TypeParam>(this->handle_cuda());
}
}  // namespace

TEST_F(CUDA, RELAYOUT_TRANSPOSE) {
    Checker<Relayout> checker(handle_cuda());
    auto run = [&](size_t batch, size_t m, size_t n, size_t c, DType dtype) {
        checker.set_dtype(0, dtype).set_dtype(1, dtype);
        TensorLayout src = {{batch, m, n, c}, dtype};
        src.init_contiguous_stride();
        TensorLayout dst = {{batch, m, n, c}, dtype};
        dst.stride[0] = m * n * c;
        dst.stride[1] = c;
        dst.stride[2] = m * c;
        dst.stride[3] = 1;
        checker.execl({src, dst});
    };
    run(16, 30, 40, 4, dtype::Int8());
    run(16, 20, 10, 4, dtype::Int8());
    run(1, 30, 20, 1, dtype::Int32());
    run(1, 20, 30, 1, dtype::Int32());
    run(1, 11, 21, 1, dtype::Float32());
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_RELAYOUT_TRANSPOSE) {
    static constexpr size_t RUNS = 1000;
    CUBenchmarker<Relayout> benchmarker(handle_cuda());
    benchmarker.set_times(RUNS);
    auto run = [&](size_t batch, size_t m, size_t n, size_t c, DType dtype) {
        benchmarker.set_dtype(0, dtype).set_dtype(1, dtype);
        TensorLayout src = {{batch, m, n, c}, dtype};
        src.init_contiguous_stride();
        TensorLayout dst = {{batch, m, n, c}, dtype};
        dst.stride[0] = m * n * c;
        dst.stride[1] = c;
        dst.stride[2] = m * c;
        dst.stride[3] = 1;
        auto time_ms =
                benchmarker.execl({src, dst}) / RUNS;
        printf("{%zux%zux%zux%zu}->{%zux%zux%zux%zu} bandwidth: %.2f gbps\n",
               batch, m, n, c, batch, n, m, c,
               2.f * batch * m * n * c * dtype.size() / (1e6 * time_ms));
    };
    run(16, 640, 480, 4, dtype::Int8());
    run(256, 224, 224, 4, dtype::Int8());
    run(1, 256, 224 * 224, 1, dtype::Int32());
    run(1, 256, 7 * 7 * 512, 1, dtype::Int32());
    run(1, 4096, 4096, 1, dtype::Float32());
}

TEST_F(CUDA, BENCHMARK_RELAYOUT) {
    //! benchmark contious layout, such as (a, b, c, d) -> (b, a, c,d)
    //! just change the first two axis
    static constexpr size_t RUNS = 3;
    auto run = [&](const TensorLayoutArray& layouts) {
        Benchmarker<Relayout> benchmarker(handle_cuda());

        benchmarker.set_times(RUNS);
        for (auto&& layout : layouts) {
            TensorLayout src = layout.dimshuffle({1, 0, 2});
            TensorLayout dst = layout;
            std::swap(dst.shape[0], dst.shape[1]);
            dst.init_contiguous_stride();
            auto used = benchmarker.execl({src, dst});
            printf("layout: %s bandwith: %f gbps/s\n",
                   layout.to_string().c_str(),
                   2 * layout.total_nr_elems() * layout.dtype.size() * RUNS /
                           used * 1000 / (1024 * 1024 * 1024));
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
        Benchmarker<Relayout> benchmarker(handle_cuda());

        benchmarker.set_times(RUNS);
        for (auto&& layout : layouts) {
            TensorLayout src = layout.dimshuffle({0, 2, 1, 3});
            TensorLayout dst = layout;
            std::swap(dst.shape[0], dst.shape[1]);
            dst.init_contiguous_stride();
            auto used = benchmarker.execl({src, dst});

            printf("layout: %s bandwith: %f gbps/s\n",
                   layout.to_string().c_str(),
                   2 * layout.total_nr_elems() * layout.dtype.size() * RUNS /
                           used * 1000 / (1024 * 1024 * 1024));
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

TEST_F(CUDA, BENCHMARK_RELAYOUT_LAST_CONTIG) {
    //! src and dst are all get subtensor in channel axis
    static constexpr size_t RUNS = 3;

    Benchmarker<Relayout> benchmarker(handle_cuda());
    benchmarker.set_times(RUNS);
    TensorLayout src =
            TensorLayout({5, 5, 100000}, {800000, 100000, 1}, dtype::Float32());
    TensorLayout dst =
            TensorLayout({5, 5, 100000}, {700000, 100000, 1}, dtype::Float32());
    auto used = benchmarker.execl({src, dst});

    printf("src: %s dst: %s bandwith: %f gbps/s\n", src.to_string().c_str(),
           dst.to_string().c_str(),
           2 * src.total_nr_elems() * src.dtype.size() * RUNS / used * 1000 /
                   (1024 * 1024 * 1024));
}

TEST_F(CUDA, BENCHMARK_RELAYOUT_LAST_NOT_CONTIG) {
    static constexpr size_t RUNS = 3;

    auto run = [&](TensorLayout src, TensorLayout dst) {
        Benchmarker<Relayout> benchmarker(handle_cuda());
        auto&& layout = src;
        benchmarker.set_times(RUNS);
        dst.init_contiguous_stride();
        auto used = benchmarker.execl({src, dst});
        printf("layout: %s bandwith: %f gbps/s\n", layout.to_string().c_str(),
               2 * layout.total_nr_elems() * layout.dtype.size() * RUNS / used *
                       1000 / (1024 * 1024 * 1024));
    };

    run({{16, 128, 128}, {49152, 384, 3}, dtype::Float32()},
        {{16, 128, 128}, {16384, 128, 1}, dtype::Float32()});
}

TEST_F(CUDA, BENCHMARK_RELAYOUT_6) {
    static constexpr size_t RUNS = 3;
    auto run = [&](TensorLayoutArray layouts,
                   std::vector<std::vector<size_t>> permutations) {
        Benchmarker<Relayout> benchmarker(handle_cuda());

        benchmarker.set_times(RUNS);
        int i = 0;
        for (auto&& layout : layouts) {
            auto per = permutations[i];
            TensorLayout src = layout.dimshuffle(per);
            TensorLayout dst = layout;
            std::swap(dst.shape[0], dst.shape[1]);
            dst.init_contiguous_stride();
            auto used = benchmarker.execl({src, dst});
            Checker<Relayout> checker(handle_cuda());
            checker.exec(TensorLayoutArray{src, dst});
            printf("layout: %s bandwith: %f gbps/s\n",
                   layout.to_string().c_str(),
                   2 * layout.total_nr_elems() * layout.dtype.size() * RUNS /
                           used * 1000 / (1024 * 1024 * 1024));
            i++;
        }
    };
    TensorLayoutArray layouts = {
            {{7248, 7248}, dtype::Int32()},
            {{43408, 1216}, dtype::Int32()},
            {{1216, 43408}, dtype::Int32()},
            {{368, 384, 384}, dtype::Int32()},
            {{2144, 64, 384}, dtype::Int32()},
            {{368, 64, 2307}, dtype::Int32()},
            {{384, 384, 355}, dtype::Int32()},
            {{2320, 384, 59}, dtype::Int32()},
            {{384, 2320, 59}, dtype::Int32()},
            {{384, 355, 384}, dtype::Int32()},
            {{2320, 59, 384}, dtype::Int32()},
            {{384, 59, 2320}, dtype::Int32()},
            {{80, 96, 75, 96}, dtype::Int32()},
            {{464, 16, 75, 96}, dtype::Int32()},
            {{80, 16, 75, 582}, dtype::Int32()},
            {{96, 75, 96, 75}, dtype::Int32()},
            {{608, 12, 96, 75}, dtype::Int32()},
            {{96, 12, 608, 75}, dtype::Int32()},
            {{96, 75, 96, 75}, dtype::Int32()},
            {{608, 12, 96, 75}, dtype::Int32()},
            {{96, 12, 608, 75}, dtype::Int32()},
            {{96, 96, 75, 75}, dtype::Int32()},
            {{608, 96, 12, 75}, dtype::Int32()},
            {{96, 608, 12, 75}, dtype::Int32()},
            {{96, 75, 75, 96}, dtype::Int32()},
            {{608, 12, 75, 96}, dtype::Int32()},
            {{96, 12, 75, 608}, dtype::Int32()},
            {{32, 48, 28, 28, 48}, dtype::Int32()},
            {{176, 8, 28, 28, 48}, dtype::Int32()},
            {{32, 8, 28, 28, 298}, dtype::Int32()},
            {{48, 28, 28, 48, 28}, dtype::Int32()},
            {{352, 4, 28, 48, 28}, dtype::Int32()},
            {{48, 4, 28, 352, 28}, dtype::Int32()},
            {{48, 28, 48, 28, 28}, dtype::Int32()},
            {{352, 4, 48, 28, 28}, dtype::Int32()},
            {{48, 4, 352, 28, 28}, dtype::Int32()},
            {{48, 48, 28, 28, 28}, dtype::Int32()},
            {{352, 48, 4, 28, 28}, dtype::Int32()},
            {{48, 352, 4, 28, 28}, dtype::Int32()},
            {{48, 28, 28, 28, 48}, dtype::Int32()},
            {{352, 4, 28, 28, 48}, dtype::Int32()},
            {{48, 4, 28, 28, 352}, dtype::Int32()},
            {{16, 32, 15, 32, 15, 15}, dtype::Int32()},
            {{48, 10, 15, 32, 15, 15}, dtype::Int32()},
            {{16, 10, 15, 103, 15, 15}, dtype::Int32()},
            {{32, 15, 15, 32, 15, 15}, dtype::Int32()},
            {{112, 5, 15, 32, 15, 15}, dtype::Int32()},
            {{32, 5, 15, 112, 15, 15}, dtype::Int32()},
            {{32, 15, 32, 15, 15, 15}, dtype::Int32()},
            {{112, 5, 32, 15, 15, 15}, dtype::Int32()},
            {{32, 5, 112, 15, 15, 15}, dtype::Int32()},
            {{32, 15, 15, 32, 15, 15}, dtype::Int32()},
            {{112, 5, 15, 32, 15, 15}, dtype::Int32()},
            {{32, 5, 15, 112, 15, 15}, dtype::Int32()},
            {{32, 15, 15, 15, 15, 32}, dtype::Int32()},
            {{112, 5, 15, 15, 15, 32}, dtype::Int32()},
            {{32, 5, 15, 15, 15, 112}, dtype::Int32()},
    };

    std::vector<std::vector<size_t>> permutations = {
            std::vector<size_t>{1, 0},
            std::vector<size_t>{1, 0},
            std::vector<size_t>{1, 0},
            std::vector<size_t>{0, 2, 1},
            std::vector<size_t>{0, 2, 1},
            std::vector<size_t>{0, 2, 1},
            std::vector<size_t>{1, 0, 2},
            std::vector<size_t>{1, 0, 2},
            std::vector<size_t>{1, 0, 2},
            std::vector<size_t>{2, 1, 0},
            std::vector<size_t>{2, 1, 0},
            std::vector<size_t>{2, 1, 0},
            std::vector<size_t>{0, 3, 2, 1},
            std::vector<size_t>{0, 3, 2, 1},
            std::vector<size_t>{0, 3, 2, 1},
            std::vector<size_t>{2, 1, 3, 0},
            std::vector<size_t>{2, 1, 3, 0},
            std::vector<size_t>{2, 1, 3, 0},
            std::vector<size_t>{2, 0, 3, 1},
            std::vector<size_t>{2, 0, 3, 1},
            std::vector<size_t>{2, 0, 3, 1},
            std::vector<size_t>{1, 0, 3, 2},
            std::vector<size_t>{1, 0, 3, 2},
            std::vector<size_t>{1, 0, 3, 2},
            std::vector<size_t>{3, 2, 1, 0},
            std::vector<size_t>{3, 2, 1, 0},
            std::vector<size_t>{3, 2, 1, 0},
            std::vector<size_t>{0, 4, 2, 1, 3},
            std::vector<size_t>{0, 4, 2, 1, 3},
            std::vector<size_t>{0, 4, 2, 1, 3},
            std::vector<size_t>{3, 2, 1, 4, 0},
            std::vector<size_t>{3, 2, 1, 4, 0},
            std::vector<size_t>{3, 2, 1, 4, 0},
            std::vector<size_t>{2, 0, 4, 1, 3},
            std::vector<size_t>{2, 0, 4, 1, 3},
            std::vector<size_t>{2, 0, 4, 1, 3},
            std::vector<size_t>{1, 3, 0, 4, 2},
            std::vector<size_t>{1, 3, 0, 4, 2},
            std::vector<size_t>{1, 3, 0, 4, 2},
            std::vector<size_t>{4, 3, 2, 1, 0},
            std::vector<size_t>{4, 3, 2, 1, 0},
            std::vector<size_t>{4, 3, 2, 1, 0},
            std::vector<size_t>{0, 3, 2, 5, 4, 1},
            std::vector<size_t>{0, 3, 2, 5, 4, 1},
            std::vector<size_t>{0, 3, 2, 5, 4, 1},
            std::vector<size_t>{3, 2, 0, 5, 1, 4},
            std::vector<size_t>{3, 2, 0, 5, 1, 4},
            std::vector<size_t>{3, 2, 0, 5, 1, 4},
            std::vector<size_t>{2, 0, 4, 1, 5, 3},
            std::vector<size_t>{2, 0, 4, 1, 5, 3},
            std::vector<size_t>{2, 0, 4, 1, 5, 3},
            std::vector<size_t>{3, 2, 5, 1, 0, 4},
            std::vector<size_t>{3, 2, 5, 1, 0, 4},
            std::vector<size_t>{3, 2, 5, 1, 0, 4},
            std::vector<size_t>{5, 4, 3, 2, 1, 0},
            std::vector<size_t>{5, 4, 3, 2, 1, 0},
            std::vector<size_t>{5, 4, 3, 2, 1, 0}};
    run(layouts, permutations);
}

TEST_F(CUDA, BENCHMARK_RELAYOUT_7) {
    static constexpr size_t RUNS = 3;

    auto isTrivial = [&](std::vector<size_t>& permutation) {
        for (size_t i = 0; i < permutation.size(); i++) {
            if (permutation[i] != i)
                return false;
        }
        return true;
    };
    auto run = [&](TensorLayout layout, std::vector<size_t> per) {
        Benchmarker<Relayout> benchmarker(handle_cuda());

        benchmarker.set_times(RUNS);

        TensorLayout src = layout.dimshuffle(per);
        TensorLayout dst = layout;
        std::swap(dst.shape[0], dst.shape[1]);
        dst.init_contiguous_stride();
        auto used = benchmarker.execl({src, dst});
        Checker<Relayout> checker(handle_cuda());
        checker.exec(TensorLayoutArray{src, dst});
        printf("layout: %s bandwith: %f gbps/s\n", layout.to_string().c_str(),
               2 * layout.total_nr_elems() * layout.dtype.size() * RUNS / used *
                       1000 / (1024 * 1024 * 1024));
    };

    std::vector<size_t> _dim = {5, 3, 2, 4, 35, 33, 37};
    std::vector<size_t> permutation(7);
    // Inverse
    for (size_t r = 0; r < _dim.size(); r++) {
        size_t size = _dim.size();
        permutation[r] = size - 1 - r;
    }
    run({{_dim[0], _dim[1], _dim[2], _dim[3], _dim[4], _dim[5], _dim[6]},
         dtype::Int32()},
        permutation);
    // Random
    for (size_t r = 0; r < _dim.size(); r++)
        permutation[r] = r;
    for (int nsample = 0; nsample < 50; nsample++) {
        COMPAT_RANDOM(_dim.begin(), _dim.end());
        COMPAT_RANDOM(permutation.begin(), permutation.end());
        if (!isTrivial(permutation)) {
            run({{_dim[0], _dim[1], _dim[2], _dim[3], _dim[4], _dim[5],
                  _dim[6]},
                 dtype::Int32()},
                permutation);
        }
    }
}

TEST_F(CUDA, BENCHMARK_RELAYOUT_5) {
    static constexpr size_t RUNS = 10;

    auto isTrivial = [&](std::vector<size_t>& permutation) {
        for (size_t i = 0; i < permutation.size(); i++) {
            if (permutation[i] != i)
                return false;
        }
        return true;
    };
    auto run = [&](TensorLayout layout, std::vector<size_t> per) {
        CUBenchmarker<Relayout> benchmarker(handle_cuda());

        benchmarker.set_times(RUNS);

        TensorLayout src = layout.dimshuffle(per);
        TensorLayout dst = layout;
        // std::swap(dst.shape[0], dst.shape[1]);
        dst.init_contiguous_stride();
        auto used = benchmarker.execl({src, dst});
        Checker<Relayout> checker(handle_cuda());
        checker.exec(TensorLayoutArray{src, dst});
        printf("layout: %s bandwith: %f gbps/s\n", layout.to_string().c_str(),
               2 * layout.total_nr_elems() * layout.dtype.size() * RUNS / used *
                       1000 / (1024 * 1024 * 1024));
    };

    size_t two = 2;
    int ratio = 5;
    int numElemAvg = 1000000 * 200;
    UniformFloatRNG numElem_dist((double)numElemAvg, (double)numElemAvg*0.2);
    for (int rank = 5; rank <= 5; rank++) {
        for (int iter = 0; iter < 20; iter++) {
            int numElem = (int)numElem_dist.gen_single_val();

            std::vector<size_t> dim(rank);
            std::vector<size_t> permutation(rank);
            std::vector<double> dimf(rank);
            double volf = 1.0;
            for (int r = 0; r < rank; r++) {
                permutation[r] = (size_t)r;
                dimf[r] = 1.0 + (double)r * (ratio - 1.0) / (double)(rank - 1);
                volf *= dimf[r];
            }
            // fprintf(stderr, "volf %lf\n", volf);
            double scale = pow((double)numElem / volf, 1.0 / (double)rank);
            // fprintf(stderr, "scale %lf\n", scale);
            int vol = 1;
            for (int r = 0; r < rank; r++) {
                if (r == rank - 1) {
                    dim[r] = ratio * dim[0];
                } else {
                    dim[r] = (size_t)round(dimf[r] * scale);
                }
                dim[r] = std::max(two, dim[r]);
                vol *= dim[r];
            }
            // fprintf(stderr, "dim[0] %lf\n", dim[0]);
            double cur_ratio = (double)dim[rank - 1] / (double)dim[0];
            double vol_re = fabs((double)(vol - numElem) / (double)numElem);
            // Fix dimensions if volume is off by more than 5%
            if (vol_re > 0.05) {
                size_t d = (vol < numElem) ? 1 : -1;
                int r = 1;
                while (vol_re > 0.05 && r < rank) {
                    size_t dim_plus_d = std::max(two, dim[r] + d);
                    vol = (vol / dim[r]) * dim_plus_d;
                    dim[r] = dim_plus_d;
                    vol_re = fabs((double)(vol - numElem) / (double)numElem);
                    r++;
                }
            }
            size_t minDim = *(std::min_element(dim.begin(), dim.end()));
            size_t maxDim = *(std::max_element(dim.begin(), dim.end()));
            cur_ratio = (double)maxDim / (double)minDim;
            printf("vol %d cur_ratio %lf | %lf\n", vol, cur_ratio, vol_re);
            // printVec(dim);

            COMPAT_RANDOM(dim.begin(), dim.end());

            while (isTrivial(permutation)) {
                COMPAT_RANDOM(permutation.begin(), permutation.end());
            }

            run({{dim[0], dim[1], dim[2], dim[3], dim[4]}, dtype::Int32()},
                permutation);
            // if (!bench_tensor<T>(dim, permutation)) return false;
        }
    }
}

TEST_F(CUDA, BENCHMARK_RELAYOUT_NCHW_NCHW4) {
    static constexpr size_t RUNS = 10;

    auto run = [&](TensorLayout layout, std::vector<size_t> per) {
        CUBenchmarker<Relayout> benchmarker(handle_cuda());

        benchmarker.set_times(RUNS);

        TensorLayout src = layout.dimshuffle(per);
        TensorLayout dst = layout;
        dst.init_contiguous_stride();
        auto used = benchmarker.execl({src, dst});
        Checker<Relayout> checker(handle_cuda());
        checker.exec(TensorLayoutArray{src, dst});

        printf("layout: %s bandwith: %f gbps/s\n", layout.to_string().c_str(),
               2 * layout.total_nr_elems() * layout.dtype.size() * RUNS / used *
                       1000 / (1024 * 1024 * 1024));
    };
    UniformIntRNG u(2,100);
    printf("NCHW->NCHW4\n");
    for (int i = 0; i < 20; i++) {
        int d1 = u.gen_single_val();
        int d2 = (u.gen_single_val() / 4 + 1) * 4;
        int d3 = 4;
        // int d4=(u.gen_single_val()/4+1)*4;
        int d4 = (u.gen_single_val());
        int d5 = (u.gen_single_val());
        // int d5=(u.gen_single_val()/4+1)*4;

        // int d5 = (u.gen_single_val())*2+1;
        run({{(size_t)d1, (size_t)d2 / 4, (size_t)d3, (size_t)d4, (size_t)d5},
             {d2 * d3 * d4 * d5 / 4, d3 * d4 * d5, d4 * d5, d5, 1},
             dtype::Int8()},
            {0, 1, 3, 4, 2});
    }
    printf("\n\nNCHW4->NCHW\n");
    for (int i = 0; i < 20; i++) {
        int d1 = u.gen_single_val();
        int d2 = (u.gen_single_val() / 4 + 1) * 4;
        int d3 = u.gen_single_val();
        // int d5=(u.gen_single_val()/4+1)*4;
        int d4 = u.gen_single_val();
        int d5 = 4;
        run({{(size_t)d1, (size_t)d2 / 4, (size_t)d3, (size_t)d4, (size_t)d5},
             {d2 * d3 * d4 * d5 / 4, d3 * d4 * d5, d4 * d5, d5, 1},
             dtype::Int8()},
            {0, 1, 4, 2, 3});
    }
}

TEST_F(CUDA, BENCHMARK_RELAYOUT_NCHW4_NCHW32) {
    static constexpr size_t RUNS = 10;
    auto run = [&](TensorLayout layout, std::vector<size_t> per) {
        CUBenchmarker<Relayout> benchmarker(handle_cuda());

        benchmarker.set_times(RUNS);

        TensorLayout src = layout.dimshuffle(per);
        TensorLayout dst = layout;
        dst.init_contiguous_stride();
        auto used = benchmarker.execl({src, dst});

        Checker<Relayout> checker(handle_cuda());
        checker.exec(TensorLayoutArray{src, dst});

        printf("layout: %s bandwith: %f gbps/s\n", layout.to_string().c_str(),
               2 * layout.total_nr_elems() * layout.dtype.size() * RUNS / used *
                       1000 / (1024 * 1024 * 1024));
    };
    UniformIntRNG u(4,50);
    printf("NCHW4 to NCHW32\n");
    for (int i = 0; i < 20; i++) {
        int d1 = u.gen_single_val();
        int d2 = (u.gen_single_val() / 8 + 1) * 8;
        int d3 = 8;
        int d4 = u.gen_single_val();
        int d5 = u.gen_single_val();
        int d6 = 4;
        run({{(size_t)d1, (size_t)d2 / 8, (size_t)d3, (size_t)d4, (size_t)d5,
              (size_t)d6},
             {d2 * d3 * d4 * d5 * d6 / 8, d3 * d4 * d5 * d6, d4 * d5 * d6,
              d5 * d6, d6, 1},
             dtype::Int8()},
            {0, 1, 3, 4, 2, 5});
    }
    printf("\n\nNCHW32 to NCHW4\n");
    for (int i = 0; i < 20; i++) {
        int d1 = u.gen_single_val();
        int d2 = (u.gen_single_val() / 8 + 1) * 8;
        int d3 = u.gen_single_val();
        int d4 = u.gen_single_val();
        int d5 = 8;
        int d6 = 4;
        run({{(size_t)d1, (size_t)d2 / 8, (size_t)d3, (size_t)d4, (size_t)d5,
              (size_t)d6},
             {d2 * d3 * d4 * d5 * d6 / 8, d3 * d4 * d5 * d6, d4 * d5 * d6,
              d5 * d6, d6, 1},
             dtype::Int8()},
            {0, 1, 4, 2, 3, 5});
    }
}

TEST_F(CUDA, BENCHMARK_LAST_CONTIG_ALIGN_TEST) {
    static constexpr size_t RUNS = 10;

    auto run = [&](TensorLayout layout, std::vector<size_t> per) {
        CUBenchmarker<Relayout> benchmarker(handle_cuda());
        benchmarker.set_times(RUNS);

        TensorLayout src = layout.dimshuffle(per);
        TensorLayout dst = layout;
        // std::swap(dst.shape[0], dst.shape[1]);
        dst.init_contiguous_stride();
        auto used = benchmarker.execl({src, dst});
        Checker<Relayout> checker(handle_cuda());
        checker.exec(TensorLayoutArray{src, dst});
        printf("layout: %s bandwith: %f gbps/s\n", layout.to_string().c_str(),
               2 * layout.total_nr_elems() * layout.dtype.size() * RUNS / used *
                       1000 / (1024 * 1024 * 1024));
    };
    UniformIntRNG u(4,50);
    std::vector<size_t> _dim(6);
    std::vector<size_t> permutation(_dim.size());
    for (size_t r = 0; r < _dim.size(); r++) {
        size_t size = _dim.size();
        permutation[r] = size - 1 - r;
    }
    _dim[0] = u.gen_single_val();
    _dim[1] = u.gen_single_val();
    _dim[2] = u.gen_single_val();
    _dim[3] = u.gen_single_val();
    _dim[4] = u.gen_single_val();
    _dim[5] = (u.gen_single_val() / 4 + 1) * 4;
    run({{_dim[0], _dim[1], _dim[2], _dim[3], _dim[4], _dim[5]}, dtype::Int8()},
        permutation);
    // Random
    for (size_t r = 0; r < _dim.size(); r++)
        permutation[r] = r;
    for (int nsample = 0; nsample < 20; nsample++) {
        COMPAT_RANDOM(_dim.begin(), _dim.end() - 1);

        COMPAT_RANDOM(permutation.begin(), permutation.end() - 1);

        if (nsample < 5)
            _dim[5] = (u.gen_single_val() / 4 + 1) * 4;
        else
            _dim[5] = u.gen_single_val();

        run({{_dim[0], _dim[1], _dim[2], _dim[3], _dim[4], _dim[5]},
             dtype::Int8()},
            permutation);
    }
}
#endif

TEST_F(CUDA, RELAYOUT) {
    struct Arg {
        TensorLayout src, dst;
        Arg(TensorLayout src, TensorLayout dst) : src(src), dst(dst) {}
    };
    std::vector<Arg> args;
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
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Int32()),
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int32()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int32()),
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Int32()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int32()),
            TensorLayout({2, 3, 4, 5}, {180, 60, 15, 3}, dtype::Int32()));
    args.emplace_back(
            TensorLayout({16, 128, 128}, {49152, 384, 3}, dtype::Float32()),
            TensorLayout({16, 128, 128}, {16384, 128, 1}, dtype::Float32()));

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

    Checker<Relayout> checker(handle_cuda());
    for (auto&& arg : args) {
        checker.exec(TensorLayoutArray{arg.src, arg.dst});
    }
}

TEST_F(CUDA, TRANSPOSE_INT8) {
    auto run = [&](TensorLayout layout, std::vector<size_t> per) {
        TensorLayout src = layout.dimshuffle(per);
        TensorLayout dst = layout;
        dst.init_contiguous_stride();

        Checker<Relayout> checker(handle_cuda());
        checker.exec(TensorLayoutArray{src, dst});
    };
    //! for last contig(NCHW4<->NCHW32)
    run({{5, 8, 4, 3, 8}, dtype::Int8()}, {1, 3, 0, 2, 4});
    run({{5, 8, 4, 3, 5}, dtype::Int8()}, {1, 3, 0, 2, 4});
    run({{5, 8, 4, 3, 64}, dtype::Int8()}, {1, 3, 0, 2, 4});
    //! for last no contig(NCHW->NCHW4)
    run({{7, 4, 32}, dtype::Int8()}, {2, 0, 1});
    run({{7, 4, 64}, dtype::Int8()}, {2, 0, 1});
    run({{7, 4, 7}, dtype::Int8()}, {2, 0, 1});
    //! for copy
    run({{2, 3, 4, 5, 6},
         {2 * 3 * 4 * 5 * 6, 2 * 4 * 5 * 6, 2 * 5 * 6, 6, 1},
         dtype::Int8()},
        {0, 1, 2, 3, 4});
}

TEST_F(CUDA, RELAYOUT_INT8) {
    struct Arg {
        TensorLayout src, dst;
        Arg(TensorLayout src, TensorLayout dst) : src(src), dst(dst) {}
    };
    std::vector<Arg> args;
    {
        // contiguous stride
        args.emplace_back(TensorLayout({4, 3, 2}, {2, 8, 1}, dtype::Int8()),
                          TensorLayout({4, 3, 2}, {6, 2, 1}, dtype::Int8()));
        args.emplace_back(TensorLayout({4, 3, 2}, {6, 2, 1}, dtype::Int8()),
                          TensorLayout({4, 3, 2}, {2, 8, 1}, dtype::Int8()));
        args.emplace_back(
                TensorLayout({2, 4, 3, 5}, {60, 5, 20, 1}, dtype::Int8()),
                TensorLayout({2, 4, 3, 5}, {60, 15, 5, 1}, dtype::Int8()));
    }
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Int8()),
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int8()),
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int8()),
            TensorLayout({2, 3, 4, 5}, {180, 60, 15, 3}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Int8()),
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int8()),
            TensorLayout({2, 3, 4, 5}, {60, 20, 5, 1}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({2, 3, 4, 5}, {120, 40, 10, 2}, dtype::Int8()),
            TensorLayout({2, 3, 4, 5}, {180, 60, 15, 3}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({16, 128, 128}, {49152, 384, 3}, dtype::Int8()),
            TensorLayout({16, 128, 128}, {16384, 128, 1}, dtype::Int8()));

    {
        // 1d
        size_t n = 10000;
        args.emplace_back(TensorLayout({n}, {1}, dtype::Int8()),
                          TensorLayout({n}, {1}, dtype::Int8()));
        args.emplace_back(TensorLayout({n}, {1}, dtype::Int8()),
                          TensorLayout({n}, {2}, dtype::Int8()));
        args.emplace_back(TensorLayout({n}, {2}, dtype::Int8()),
                          TensorLayout({n}, {1}, dtype::Int8()));
        args.emplace_back(TensorLayout({n}, {2}, dtype::Int8()),
                          TensorLayout({n}, {2}, dtype::Int8()));
    }
    {
        // 2d
        size_t m = 200, n = 300, k = 400;
        ptrdiff_t k2 = k * 2;
        args.emplace_back(TensorLayout({m, n}, {k2, 2}, dtype::Int8()),
                          TensorLayout({m, n}, {k2 + 1, 2}, dtype::Int8()));
        args.emplace_back(TensorLayout({m, n}, {2, k2}, dtype::Int8()),
                          TensorLayout({m, n}, {2, k2 + 1}, dtype::Int8()));
        args.emplace_back(TensorLayout({m, n}, {2, k2}, dtype::Int8()),
                          TensorLayout({m, n}, {k2 + 1, 2}, dtype::Int8()));
        args.emplace_back(TensorLayout({m, n}, {k2, 2}, dtype::Int8()),
                          TensorLayout({m, n}, {2, k2 + 1}, dtype::Int8()));
        args.emplace_back(TensorLayout({m, n}, {k2, 1}, dtype::Int8()),
                          TensorLayout({m, n}, {k2 + 1, 1}, dtype::Int8()));
        args.emplace_back(TensorLayout({m, n}, {1, k2}, dtype::Int8()),
                          TensorLayout({m, n}, {1, k2 + 1}, dtype::Int8()));
        args.emplace_back(TensorLayout({m, n}, {1, k2}, dtype::Int8()),
                          TensorLayout({m, n}, {k2 + 1, 1}, dtype::Int8()));
        args.emplace_back(TensorLayout({m, n}, {k2, 1}, dtype::Int8()),
                          TensorLayout({m, n}, {1, k2 + 1}, dtype::Int8()));
    }
    {
        // 3d
        size_t m = 20, n = 30, k = 40;
        ptrdiff_t k2 = k;
        args.emplace_back(
                TensorLayout({m, n, k}, {k2 * k2 * 4, k2 * 3, 2},
                             dtype::Int8()),
                TensorLayout({m, n, k}, {2 * k2 * k2 * k2 * 4, k2 * 3, 2},
                             dtype::Int8()));
    }
    {
        // simplify_layout
        // 234..56
        // 2..3456
        args.emplace_back(
                TensorLayout(
                        {2, 3, 4, 5, 6},
                        {2 * 3 * 4 * 5 * 6, 2 * 4 * 5 * 6, 2 * 5 * 6, 6, 1},
                        dtype::Int8()),
                TensorLayout({2, 3, 4, 5, 6},
                             {4 * 3 * 4 * 5 * 6, 4 * 5 * 6, 5 * 6, 6, 1},
                             dtype::Int8()));

        args.emplace_back(
                TensorLayout(
                        {2, 3, 4, 5, 6},
                        {4 * 3 * 4 * 5 * 6, 4 * 4 * 5 * 6, 2 * 5 * 6, 6, 1},
                        dtype::Int8()),
                TensorLayout({2, 3, 4, 5, 6},
                             {4 * 3 * 4 * 5 * 6, 4 * 5 * 6, 5 * 6, 6, 1},
                             dtype::Int8()));
    }

    Checker<Relayout> checker(handle_cuda());
    for (auto&& arg : args) {
        checker.exec(TensorLayoutArray{arg.src, arg.dst});
    }
}

TEST_F(CUDA, RELAYOUT_TEST) {
    struct Arg {
        TensorLayout src, dst;
        Arg(TensorLayout src, TensorLayout dst) : src(src), dst(dst) {}
    };
    std::vector<Arg> args;
    //! dst contig
    args.emplace_back(TensorLayout({5, 32, 9}, {288, 1, 32}, dtype::Int8()),
                      TensorLayout({5, 9, 32}, {288, 32, 1}, dtype::Int8()));
    args.emplace_back(TensorLayout({5, 9, 32}, {288, 1, 9}, dtype::Int8()),
                      TensorLayout({5, 32, 9}, {288, 9, 1}, dtype::Int8()));

    args.emplace_back(TensorLayout({5, 4, 9}, {36, 1, 4}, dtype::Int8()),
                      TensorLayout({5, 9, 4}, {36, 4, 1}, dtype::Int8()));
    args.emplace_back(TensorLayout({5, 9, 4}, {36, 1, 9}, dtype::Int8()),
                      TensorLayout({5, 4, 9}, {36, 9, 1}, dtype::Int8()));

    args.emplace_back(TensorLayout({5, 32, 4}, {128, 1, 32}, dtype::Int8()),
                      TensorLayout({5, 4, 32}, {128, 32, 1}, dtype::Int8()));
    args.emplace_back(TensorLayout({5, 4, 32}, {128, 1, 4}, dtype::Int8()),
                      TensorLayout({5, 32, 4}, {128, 4, 1}, dtype::Int8()));

    args.emplace_back(TensorLayout({5, 7, 5}, {35, 1, 7}, dtype::Int8()),
                      TensorLayout({5, 5, 7}, {35, 7, 1}, dtype::Int8()));
    args.emplace_back(TensorLayout({5, 5, 7}, {35, 1, 5}, dtype::Int8()),
                      TensorLayout({5, 7, 5}, {35, 5, 1}, dtype::Int8()));
    //! src contig
    args.emplace_back(TensorLayout({5, 9, 32}, {288, 32, 1}, dtype::Int8()),
                      TensorLayout({5, 32, 9}, {288, 1, 32}, dtype::Int8()));
    args.emplace_back(TensorLayout({5, 32, 9}, {288, 9, 1}, dtype::Int8()),
                      TensorLayout({5, 9, 32}, {288, 1, 9}, dtype::Int8()));

    args.emplace_back(TensorLayout({5, 9, 4}, {36, 4, 1}, dtype::Int8()),
                      TensorLayout({5, 4, 9}, {36, 1, 4}, dtype::Int8()));
    args.emplace_back(TensorLayout({5, 4, 9}, {36, 9, 1}, dtype::Int8()),
                      TensorLayout({5, 9, 4}, {36, 1, 9}, dtype::Int8()));

    args.emplace_back(TensorLayout({5, 4, 32}, {128, 32, 1}, dtype::Int8()),
                      TensorLayout({5, 32, 4}, {128, 1, 32}, dtype::Int8()));
    args.emplace_back(TensorLayout({5, 32, 4}, {128, 4, 1}, dtype::Int8()),
                      TensorLayout({5, 4, 32}, {128, 1, 4}, dtype::Int8()));

    args.emplace_back(TensorLayout({5, 5, 7}, {35, 7, 1}, dtype::Int8()),
                      TensorLayout({5, 7, 5}, {35, 1, 7}, dtype::Int8()));
    args.emplace_back(TensorLayout({5, 7, 5}, {35, 5, 1}, dtype::Int8()),
                      TensorLayout({5, 5, 7}, {35, 1, 5}, dtype::Int8()));
    //! cross
    args.emplace_back(
            TensorLayout({5, 9, 32}, {288 * 4, 32 * 3, 1}, dtype::Int8()),
            TensorLayout({5, 32, 9}, {288 * 4, 1, 32 * 3}, dtype::Int8()));
    args.emplace_back(
            TensorLayout({5, 32, 9}, {288 * 3, 9 * 2, 1}, dtype::Int8()),
            TensorLayout({5, 9, 32}, {288 * 3, 1, 9 * 2}, dtype::Int8()));

    args.emplace_back(
            TensorLayout({5, 9, 4}, {36 * 10, 4 * 7, 1}, dtype::Int8()),
            TensorLayout({5, 4, 9}, {36 * 10, 1, 4 * 7}, dtype::Int8()));

    Checker<Relayout> checker(handle_cuda());
    for (auto&& arg : args) {
        checker.exec(TensorLayoutArray{arg.src, arg.dst});
    }
}
// vim: syntax=cpp.doxygen
