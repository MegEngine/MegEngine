/**
 * \file src/opr/test/rand.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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

    static BasicStat make(const float* ptr, size_t size,
                          double mean_expect = 0) {
        double sum = 0, sum2 = 0, min = std::numeric_limits<double>::max(),
               max = std::numeric_limits<double>::lowest();
        for (size_t i = 0; i < size; ++i) {
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

void check_reproducibility(std::shared_ptr<ComputingGraph> graph, size_t size,
                           thin_function<SymbolVar(uint64_t seed)> make) {
    // out[func][opr][run]
    HostTensorND out[2][2][2];

    auto run = [&](int fid) {
        SymbolVar o0 = make(0), o1 = make(1);
        HostTensorND host_o0, host_o1;
        auto func = graph->compile({make_callback_copy(o0, host_o0),
                                    make_callback_copy(o1, host_o1)});
        for (int i = 0; i < 2; ++i) {
            func->execute();
            out[fid][0][i].copy_from(host_o0);
            out[fid][1][i].copy_from(host_o1);
        }
    };
    run(0);
    run(1);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j)
            MGB_ASSERT_TENSOR_EQ(out[0][i][j], out[1][i][j]);
    }

    auto max_diff = [&](int off0, int off1) {
        float diff = 0;
        auto p0 = out[0][off0 / 2][off0 % 2].ptr<float>(),
             p1 = out[0][off1 / 2][off1 % 2].ptr<float>();
        for (size_t i = 0; i < size; ++i) {
            update_max(diff, std::abs(p0[i] - p1[i]));
        }
        return diff;
    };

    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j)
            ASSERT_GT(max_diff(i, j), 0.3) << i << " " << j;
    }
}

}  // anonymous namespace

TEST(TestOprRand, Uniform) {
    static constexpr size_t M = 128, N = 64;
    auto graph = ComputingGraph::make();

    SymbolVar dev_out = opr::UniformRNG::make(
            *graph, {M, N}, {CompNode::load("xpu0")}, {23, DTypeEnum::Float32});

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
            {23, MEAN, STD, DTypeEnum::Float32});

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    func->execute();

    ASSERT_EQ(TensorShape({SIZE}), host_y.shape());
    auto stat = BasicStat::make(host_y.ptr<float>(), SIZE, MEAN);
    ASSERT_LT(fabs(stat.mean - MEAN), 0.01);
    ASSERT_LT(fabs(stat.std - STD), 0.1);
}

TEST(TestOprRand, Gamma) {
    std::shared_ptr<HostTensorND> shape_host(new HostTensorND{
            CompNode::load("xpux"), TensorShape{2000000*5}, dtype::Float32()});
    std::shared_ptr<HostTensorND> scale_host(new HostTensorND{
            CompNode::load("xpux"), TensorShape{2000000*5}, dtype::Float32()});
    auto shape_ptr = shape_host->ptr<float>();
    auto scale_ptr = scale_host->ptr<float>();
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 2000000; ++j) {
            shape_ptr[i * 2000000 + j] = 2 * 0.3 * i + 0.5;
            scale_ptr[i * 2000000 + j] = i * 0.3 + 0.5;
        }
    }
    auto graph = ComputingGraph::make();
    auto shape_sym = opr::Host2DeviceCopy::make(*graph, shape_host);
    auto scale_sym = opr::Host2DeviceCopy::make(*graph, scale_host);
    auto y = opr::GammaRNG::make(shape_sym, scale_sym, {10});

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    func->execute();

    ASSERT_EQ(TensorShape({2000000*5}), host_y.shape());
    for (int i = 0; i < 5; ++i) {
        float a = 2 * 0.3 * i + 0.5, b = i * 0.3 + 0.5;
        float mean = a * b;
        float std = a * (b * b);
        auto stat = BasicStat::make(host_y.ptr<float>() + 2000000 * i,
                                     2000000, mean);
        ASSERT_LT(fabs(stat.mean - mean), 0.01);
        ASSERT_LT(fabs(stat.std - sqrt(std)), 0.01);
    }
}

TEST(TestOprRand, Poisson) {
    std::shared_ptr<HostTensorND> lam_host(new HostTensorND{
            CompNode::load("xpux"), TensorShape{200000*5}, dtype::Float32()});
    auto lam_ptr = lam_host->ptr<float>();
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 200000; ++j) {
            lam_ptr[i * 200000 + j] = i + 1;
        }
    }
    auto graph = ComputingGraph::make();
    auto lam_sym = opr::Host2DeviceCopy::make(*graph, lam_host);
    auto y = opr::PoissonRNG::make(lam_sym, {10});

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    func->execute();

    ASSERT_EQ(TensorShape({200000*5}), host_y.shape());
    for (int i = 0; i < 5; ++i) {
        float lambda = i + 1;
        auto stat = BasicStat::make(host_y.ptr<float>() + 200000 * i, 
                                    200000,lambda);
        ASSERT_LT(fabs(stat.mean - lambda), 0.01);
        ASSERT_LT(fabs(stat.std - sqrt(lambda)), 0.1);
    }
}

TEST(TestOprRand, Beta) {
    std::shared_ptr<HostTensorND> alpha_host(new HostTensorND{
            CompNode::load("xpux"), TensorShape{200000*5}, dtype::Float32()});
    std::shared_ptr<HostTensorND> beta_host(new HostTensorND{
            CompNode::load("xpux"), TensorShape{200000*5}, dtype::Float32()});
    auto alpha_ptr = alpha_host->ptr<float>();
    auto beta_ptr = beta_host->ptr<float>();
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 200000; ++j) {
            alpha_ptr[i * 200000 + j] = 0.3 * i + 0.1;
            beta_ptr[i * 200000 + j] = 2 * i * 0.3 + 0.1;
        }
    }
    auto graph = ComputingGraph::make();
    auto alpha_sym = opr::Host2DeviceCopy::make(*graph, alpha_host);
    auto beta_sym = opr::Host2DeviceCopy::make(*graph, beta_host);
    auto y = opr::BetaRNG::make(alpha_sym,beta_sym, {10});

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    func->execute();

    ASSERT_EQ(TensorShape({200000*5}), host_y.shape());
    for (int i = 0; i < 5; ++i) {
        float a = 0.3 * i + 0.1, b = 2 * i * 0.3 + 0.1;
        float mean = a / (a + b);
        float std = a * b / ((a + b) * (a + b) * (a + b + 1));
        auto stat = BasicStat::make(host_y.ptr<float>() + 200000 * i,
                                    200000, mean);
        ASSERT_LT(fabs(stat.mean - mean), 0.01);
        ASSERT_LT(fabs(stat.std - sqrt(std)), 0.01);
    }
}

TEST(TestOprRand, PermutationRNG) {
    static constexpr size_t SIZE = 123451;
    auto graph = ComputingGraph::make();
    auto y = opr::PermutationRNG::make(
            SymbolVar::make_scalar(int(SIZE), *graph, {CompNode::load("xpu0")}),
            {23, DTypeEnum::Int32});
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});

    func->execute();

    ASSERT_EQ(TensorShape({SIZE}), host_y.shape());
    auto ptr = host_y.ptr<int32_t>();
    std::vector<int32_t> res(SIZE);
    int not_same = 0;
    for (size_t i = 0; i < SIZE; ++i) {
        if ((ptr[i] - int32_t(i)) >= 1) not_same++;
        res[i] = ptr[i];
    }
    ASSERT_GT(not_same, 5000);
    std::sort(res.begin(), res.end());
    for (size_t i = 0; i < SIZE; ++i) {
        ASSERT_LE(std::abs(res[i] - int32_t(i)), 1e-8);
    }
}

TEST(TestOprRand, EmptyShape) {
    auto test_uniform = []() {
        static constexpr size_t M = 128, N = 0;
        auto graph = ComputingGraph::make();
        SymbolVar dev_out = opr::UniformRNG::make(
                *graph, {M, N}, {CompNode::load("xpu0")}, {23, DTypeEnum::Float32});
        HostTensorND host_out;
        auto func = graph->compile({make_callback_copy(dev_out, host_out)});
        func->execute();
        ASSERT_EQ(host_out.shape(), TensorShape({M, N}));

    };
    auto test_gaussian = []() {
        size_t SIZE = 0;
        constexpr float MEAN = 1, STD = 2;
        auto graph = ComputingGraph::make();
        auto y = opr::GaussianRNG::make(
                SymbolVar::make_scalar(int(SIZE), *graph, {CompNode::load("xpu0")}),
                {23, MEAN, STD, DTypeEnum::Float32});
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        ASSERT_EQ(TensorShape({SIZE}), host_y.shape());
    };
    auto test_gamma = []() {
        std::shared_ptr<HostTensorND> shape_host(new HostTensorND{
                CompNode::load("xpux"), TensorShape{10, 0}, dtype::Float32()});
        std::shared_ptr<HostTensorND> scale_host(new HostTensorND{
                CompNode::load("xpux"), TensorShape{10, 0}, dtype::Float32()});
        auto graph = ComputingGraph::make();
        auto shape_sym = opr::Host2DeviceCopy::make(*graph, shape_host);
        auto scale_sym = opr::Host2DeviceCopy::make(*graph, scale_host);

        auto y = opr::GammaRNG::make(shape_sym, scale_sym, {10});
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        ASSERT_EQ(TensorShape({10, 0}), host_y.shape());
    };
    auto test_poisson = []() {
        std::shared_ptr<HostTensorND> lam_host(new HostTensorND{
                CompNode::load("xpux"), TensorShape{10, 0}, dtype::Float32()});
        auto graph = ComputingGraph::make();
        auto lam_sym = opr::Host2DeviceCopy::make(*graph, lam_host);
        auto y = opr::PoissonRNG::make(lam_sym, {10});

        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        ASSERT_EQ(TensorShape({10, 0}), host_y.shape());
    };
    auto test_beta = []() {
        std::shared_ptr<HostTensorND> alpha_host(new HostTensorND{
                CompNode::load("xpux"), TensorShape{10, 0}, dtype::Float32()});
        std::shared_ptr<HostTensorND> beta_host(new HostTensorND{
                CompNode::load("xpux"), TensorShape{10, 0}, dtype::Float32()});
        auto graph = ComputingGraph::make();
        auto alpha_sym = opr::Host2DeviceCopy::make(*graph, alpha_host);
        auto beta_sym = opr::Host2DeviceCopy::make(*graph, beta_host);
        auto y = opr::BetaRNG::make(alpha_sym,beta_sym, {10});

        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        ASSERT_EQ(TensorShape({10, 0}), host_y.shape());
    };
    auto test_permutation = []() {
        static constexpr size_t SIZE = 0;
        auto graph = ComputingGraph::make();
        auto y = opr::PermutationRNG::make(
                SymbolVar::make_scalar(int(SIZE), *graph, {CompNode::load("xpu0")}),
                {23, DTypeEnum::Int32});
        HostTensorND host_y;
        auto func = graph->compile({make_callback_copy(y, host_y)});
        func->execute();
        ASSERT_EQ(TensorShape({SIZE}), host_y.shape());
    };
    test_uniform();
    test_gaussian();
    test_gamma();
    test_poisson();
    test_beta();
    test_permutation();
}


TEST(TestOprRand, ShuffleForward) {
    auto run = [&](TensorShape shape) {
        std::shared_ptr<HostTensorND> src_host(new HostTensorND{
                CompNode::load("xpux"), shape, dtype::Float32()});
        auto sptr = src_host->ptr<dt_float32>();
        auto size = shape.total_nr_elems();
        for (size_t i = 0; i < size; ++i) {
            sptr[i] = i;
        }
        auto graph = ComputingGraph::make();
        auto src_sym = opr::Host2DeviceCopy::make(*graph, src_host);
        auto rec = opr::ShuffleRNG::make(src_sym, {10});
        HostTensorND host_y, host_index;
        auto func = graph->compile({make_callback_copy(rec[0], host_y),
                                    make_callback_copy(rec[1], host_index)});
        func->execute();
        auto dptr = host_y.ptr<dt_float32>();
        auto iptr = host_index.ptr<dt_int32>();

        size_t len = shape[0];
        size_t step = size / len;
        for (size_t i = 0; i < len; ++i) {
            for (size_t j = 0; j < step; ++j) {
                assert(dptr[i * step + j] == sptr[iptr[i] * step + j]);
            }
        }
    };
    run({10});
    run({6, 3});
    run({1, 1});
}

TEST(TestOprRand, UniformReprod) {
    static constexpr size_t SIZE = 123;
    auto graph = ComputingGraph::make();
    auto shp = cg::var_from_tensor_shape(*graph, {CompNode::load("xpu0")},
                                         "shp0", {SIZE});
    check_reproducibility(graph, SIZE, [&shp](uint64_t seed) {
        return opr::UniformRNG::make(shp, {seed});
    });
}

TEST(TestOprRand, GaussianReprod) {
    static constexpr size_t SIZE = 123;
    auto graph = ComputingGraph::make();
    auto shp = cg::var_from_tensor_shape(*graph, {CompNode::load("xpu0")},
                                         "shp0", {SIZE});
    check_reproducibility(graph, SIZE, [&shp](uint64_t seed) {
        return opr::GaussianRNG::make(shp, {seed});
    });
}

TEST(TestOprRand, GammaReprod) {
    static constexpr size_t SIZE = 123;
    std::shared_ptr<HostTensorND> shape_host(new HostTensorND{
            CompNode::load("xpux"), TensorShape{SIZE}, dtype::Float32()});
    std::shared_ptr<HostTensorND> scale_host(new HostTensorND{
            CompNode::load("xpux"), TensorShape{SIZE}, dtype::Float32()});
    auto shape_ptr = shape_host->ptr<float>();
    auto scale_ptr = scale_host->ptr<float>();
    for (size_t i = 0; i < SIZE; ++i){
        shape_ptr[i] = 0.5;
        scale_ptr[i] = 1.2;
    }
    auto graph = ComputingGraph::make();
    auto shape_sym = opr::Host2DeviceCopy::make(*graph, shape_host);
    auto scale_sym = opr::Host2DeviceCopy::make(*graph, scale_host);
    check_reproducibility(graph, SIZE, [&shape_sym,&scale_sym](uint64_t seed) {
        return opr::GammaRNG::make(shape_sym, scale_sym, {seed});
    });
}

TEST(TestOprRand, PoissonReprod) {
    static constexpr size_t SIZE = 123;
    std::shared_ptr<HostTensorND> lam_host(new HostTensorND{
            CompNode::load("xpux"), TensorShape{SIZE}, dtype::Float32()});
    auto lam_ptr = lam_host->ptr<float>();
    for (size_t i = 0; i < SIZE; ++i)
        lam_ptr[i] = 2;
    auto graph = ComputingGraph::make();
    auto lam_sym = opr::Host2DeviceCopy::make(*graph, lam_host);
    check_reproducibility(graph, SIZE, [&lam_sym](uint64_t seed) {
        return opr::PoissonRNG::make(lam_sym, {seed});
    });
}

TEST(TestOprRand, BetaReprod) {
    static constexpr size_t SIZE = 123;
    std::shared_ptr<HostTensorND> alpha_host(new HostTensorND{
            CompNode::load("xpux"), TensorShape{SIZE}, dtype::Float32()});
    std::shared_ptr<HostTensorND> beta_host(new HostTensorND{
            CompNode::load("xpux"), TensorShape{SIZE}, dtype::Float32()});
    auto alpha_ptr = alpha_host->ptr<float>();
    auto beta_ptr = beta_host->ptr<float>();
    for (size_t i = 0; i < SIZE; ++i){
        alpha_ptr[i] = 0.5;
        beta_ptr[i] = 1.2;
    }
    auto graph = ComputingGraph::make();
    auto alpha_sym = opr::Host2DeviceCopy::make(*graph, alpha_host);
    auto beta_sym = opr::Host2DeviceCopy::make(*graph, beta_host);
    check_reproducibility(graph, SIZE, [&alpha_sym,&beta_sym](uint64_t seed) {
        return opr::BetaRNG::make(alpha_sym, beta_sym, {seed});
    });
}

TEST(TestOprRand, PermutationReprod) {
    static constexpr size_t SIZE = 123;
    auto graph = ComputingGraph::make();
    auto shp = cg::var_from_tensor_shape(*graph, {CompNode::load("xpu0")},
                                         "shp0", {SIZE});
    check_reproducibility(graph, SIZE, [&shp](uint64_t seed) {
        return opr::PermutationRNG::make(shp, {seed, DTypeEnum::Float32});
    });
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
