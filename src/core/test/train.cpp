/**
 * \file src/core/test/train.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/tensor_manip.h"

#include "megbrain/test/helper.h"

#include <cmath>
#include <random>

using namespace mgb;

namespace {

class TestTrain: public ::testing::Test {
    protected:
        std::vector<CompNode> cns = load_multiple_xpus(3);

        static constexpr size_t DIM = 1024, NR_DATA = DIM * 2;

        std::shared_ptr<HostTensorND> host_w_truth, host_data, host_w;
        HostTensorND host_y;

        opr::AddUpdate::SharedScalar learning_rate;
        std::shared_ptr<ComputingGraph> graph;

        void SetUp() override;
        void do_train(SymbolVar dev_w_updated, const char *type);
};

float expected_err = -1;

}

void TestTrain::SetUp() {
    // generate data and ground truth

    static std::default_random_engine::result_type
        seed0 = next_rand_seed(), seed1 = next_rand_seed();

    host_y.comp_node(cns[0]).
        dtype(dtype::Float32()).
        resize({NR_DATA});
    graph = ComputingGraph::make();
    learning_rate = std::make_shared<DTypeScalar>(.0f);

    HostTensorGenerator<dtype::Float32, RandomDistribution::GAUSSIAN> gen{
        0, 1, seed0};
    host_w_truth = gen({DIM}, cns[0]);
    host_data = gen({NR_DATA, DIM}, cns[0]);
    std::default_random_engine engine{seed1};
    std::normal_distribution<float> noise{0, 0.01};
    for (size_t y = 0; y < NR_DATA; y ++) {
        float sum = noise(engine);
        auto p0 = host_w_truth->ptr<float>(), p1 = host_data->ptr<float>({y});
        for (size_t x = 0; x < DIM; x ++)
            sum += p0[x] * p1[x];
        host_y.ptr<float>()[y] = sum;
    }
    host_w = gen({DIM}, cns[0]);
}

void TestTrain::do_train(SymbolVar dev_w_updated, const char *type) {

    int iter = 0;
    float err;
    auto update_err = [&]() {
        err = 0;
        auto p0 = host_w->ptr<float>(), p1 = host_w_truth->ptr<float>();
        for (size_t i = 0; i < DIM; i ++) {
            auto d = p0[i] - p1[i];
            err += d * d;
        }
        err = sqrt(err / DIM);
        mgb_log("iter %d: lr=%.2e err=%.5f",
                iter, learning_rate->get<float>(), err);
    };
    auto copy_w = [&](DeviceTensorND &data) {
        if (iter % 20 == 0) {
            host_w->comp_node(data.comp_node());
            host_w->copy_from_fixlayout(data).sync();
            update_err();
        }
    };
    auto func = graph->compile({{dev_w_updated, copy_w}});

    func->to_json()->writeto_fpath(
            output_file(ssprintf("train-%s.json", type)));

    learning_rate->set<float>(-0.3 / NR_DATA);
    update_err();
    ASSERT_GE(err, 1);
    while (iter < 100) {
        iter ++;
        func->execute();
    }

    ASSERT_LE(err, 1e-3);
    if (expected_err == -1) {
        expected_err = err;
    } else {
        MGB_ASSERT_FLOAT_EQ(err, expected_err);
    }
}

TEST_F(TestTrain, SimpleLinearRegression) {
    SymbolVar
        dev_w = opr::SharedDeviceTensor::make(*graph, *host_w, {"w"}),
        dev_data = opr::Host2DeviceCopy::make(*graph, host_data, {"X"}),
        dev_y_target = opr::SharedDeviceTensor::make(*graph, host_y, {"y_t"}),
        dev_y = opr::MatrixMul::make(
                dev_data, dev_w.reshape({DIM, 1})).reshape(
                {NR_DATA}).rename("y"),
        delta = (dev_y - dev_y_target).rename("delta"),
        loss = opr::Dot::make(delta, delta, {"loss"}),
        grad = cg::grad(loss, dev_w).rename("grad"),
        dev_w_updated = opr::AddUpdate::make(dev_w, grad, {1, learning_rate});

    do_train(dev_w_updated, "simple-lr");
}

TEST_F(TestTrain, MultiCardLinearRegression) {
    SymbolVar
        dev_data_all = opr::Host2DeviceCopy::make(*graph, host_data, {"X_all"}),
        dev_w_all = opr::SharedDeviceTensor::make(*graph, *host_w, {"w_all"}),
        dev_y_target = opr::SharedDeviceTensor::make(*graph, host_y, {"y_t"});

    OperatorNodeConfig split_conf;
    split_conf.comp_node_arr({cns[1], cns[2]});

    auto dev_data_splitted = opr::Split::make(
            dev_data_all, opr::Split::Options::make_average(1, 2), split_conf),
         dev_w_splitted = opr::Split::make(
            dev_w_all, opr::Split::Options::make_average(0, 2), split_conf);

    auto fprop = [&](size_t idx) {
        SymbolVar
            dev_data = dev_data_splitted[idx],
            dev_w = dev_w_splitted[idx],
            dev_y = opr::MatrixMul::make(
                    dev_data, dev_w.reshape({DIM / 2, 1})).reshape(
                    {NR_DATA}).rename(ssprintf("y_%zu", idx)),
            dev_y_gpu0 = opr::Copy::make(dev_y, {
                    ssprintf("y_%zu_gpu0", idx), cns[0]});
        return dev_y_gpu0;
    };

    SymbolVar
        dev_y = fprop(0) + fprop(1),
        delta = (dev_y - dev_y_target).rename("delta"),
        loss = opr::Dot::make(delta, delta, {"loss"}),
        grad = cg::grad(loss, dev_w_all).rename("grad"),
        dev_w_updated = opr::AddUpdate::make(dev_w_all, grad,
                {1.f, learning_rate});

    do_train(dev_w_updated, "multi-card-lr");
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

