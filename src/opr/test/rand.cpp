/**
 * \file src/opr/test/rand.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/rand.h"
#include "megbrain/opr/io.h"
#include "megbrain/test/helper.h"
#include "megbrain/utils/arith_helper.h"

#include <cmath>

using namespace mgb;

namespace {
    struct BasicStat {
        double mean, std, min, max;

        static BasicStat make(const float *ptr, size_t size,
                double mean_expect = 0) {
            double sum = 0, sum2 = 0,
                   min = std::numeric_limits<double>::max(),
                   max = std::numeric_limits<double>::lowest();
            for (size_t i = 0; i < size; ++ i) {
                double cur = ptr[i];
                min = std::min(min, cur);
                max = std::max(max, cur);
                cur -= mean_expect;
                sum += cur;
                sum2 += cur * cur;
            }

            double mean = sum / size + mean_expect,
                   std = sqrt((sum2 - sum * sum / size) / (size - 1));
            return {mean, std, min, max};
        }
    };

    void check_reproducibility(
            thin_function<SymbolVar(SymbolVar, uint64_t seed)> make) {
        auto graph = ComputingGraph::make();
        constexpr size_t SIZE = 123;

        // out[func][opr][run]
        HostTensorND out[2][2][2];

        auto run = [&](int fid) {
            SymbolVar
                o0 = make(cg::var_from_tensor_shape(*graph,
                            {CompNode::load("xpu0")}, "shp0", {SIZE}), 0),
                o1 = make(cg::var_from_tensor_shape(*graph,
                            {CompNode::load("xpu0")}, "shp0", {SIZE}), 1);
            HostTensorND host_o0, host_o1;
            auto func = graph->compile({
                    make_callback_copy(o0, host_o0),
                    make_callback_copy(o1, host_o1)});
            for (int i = 0; i < 2; ++ i) {
                func->execute();
                out[fid][0][i].copy_from(host_o0);
                out[fid][1][i].copy_from(host_o1);
            }
        };
        run(0);
        run(1);

        for (int i = 0; i < 2; ++ i) {
            for (int j = 0; j < 2; ++ j)
                MGB_ASSERT_TENSOR_EQ(out[0][i][j], out[1][i][j]);
        }

        auto max_diff = [&](int off0, int off1) {
            float diff = 0;
            auto p0 = out[0][off0 / 2][off0 % 2].ptr<float>(),
                 p1 = out[0][off1 / 2][off1 % 2].ptr<float>();
            for (size_t i = 0; i < SIZE; ++ i) {
                update_max(diff, std::abs(p0[i] - p1[i]));
            }
            return diff;
        };

        for (int i = 0; i < 4; ++ i) {
            for (int j = i + 1; j < 4; ++ j)
                ASSERT_GT(max_diff(i, j), 0.3) << i << " " << j;
        }
    }

} // anonymous namespace

TEST(TestOprRand, Uniform) {
    static constexpr size_t M = 128, N = 64;
    auto graph = ComputingGraph::make();
    SymbolVar dev_out = opr::UniformRNG::make(
            *graph, {M, N}, {CompNode::load("xpu0")});

    HostTensorND host_out;
    auto func = graph->compile({make_callback_copy(dev_out, host_out)});

    func->execute();

    ASSERT_EQ(host_out.shape(), TensorShape({M, N}));
    auto stat = BasicStat::make(host_out.ptr<float>(), M * N, 0.5);
    ASSERT_LT(fabs(stat.mean - 0.5), 0.01);
    ASSERT_LT(fabs(stat.std - sqrt(1 / 12.0)), 0.1);
    ASSERT_GT(stat.min, 0);
    ASSERT_LE(stat.max, 1);
}

TEST(TestOprRand, Gaussian) {
    static constexpr size_t SIZE = 123451;
    constexpr float MEAN = 1, STD = 2;
    auto graph = ComputingGraph::make();
    auto y = opr::GaussianRNG::make(
            SymbolVar::make_scalar(int(SIZE), *graph, {CompNode::load("xpu0")}),
            {23, MEAN, STD});

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    func->execute();

    ASSERT_EQ(TensorShape({SIZE}), host_y.shape());
    auto stat = BasicStat::make(host_y.ptr<float>(), SIZE, MEAN);
    ASSERT_LT(fabs(stat.mean - MEAN), 0.01);
    ASSERT_LT(fabs(stat.std - STD), 0.1);
}

TEST(TestOprRand, UniformReprod) {
    check_reproducibility([](SymbolVar shp, uint64_t seed) {
        return opr::UniformRNG::make(shp, {seed});
    });
}

TEST(TestOprRand, GaussianReprod) {
    check_reproducibility([](SymbolVar shp, uint64_t seed) {
        return opr::GaussianRNG::make(shp, {seed});
    });
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

