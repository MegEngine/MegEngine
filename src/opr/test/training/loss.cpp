/**
 * \file src/opr/test/training/loss.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 */

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/indexing.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/tensor.h"
#include "megbrain/test/helper.h"

#include "megbrain/opr/training/loss.h"

using namespace mgb;
using namespace loss;

namespace {
class Device2HostCallback {
public:
    Device2HostCallback(std::shared_ptr<HostTensorND> host) : m_host{host} {}
    void operator()(const DeviceTensorND& device) { m_host->copy_from(device).sync(); }

private:
    std::shared_ptr<HostTensorND> m_host;
};

class CrossEntropyTest : public ::testing::Test {
private:
    /* data */
    std::shared_ptr<HostTensorND> pred, label, truth, loss;
    TensorShape pred_shape = {2, 10};
    TensorShape label_shape = {2};
    TensorShape truth_shape = {1};
    std::vector<float> pred_values = {
            -0.22847f, -0.65020f, -0.42470f, 1.32903f,  -0.58377f, -0.15881f, -0.23134f,
            -0.36147f, -1.05848f, -0.23285f, 0.32360f,  -0.36430f, -0.03172f, 1.18970f,
            -0.23465f, -0.16139f, -0.22942f, -0.22538f, -0.68029f, -0.41004f};
    std::vector<int> label_values = {5, 3};
    std::vector<float> truth_values = {1.8120441};

    CompNode node = CompNode::load("cpu0");

    std::shared_ptr<cg::ComputingGraph> graph;

    CrossEntropyLoss cross_entropy_loss;

public:
    std::unique_ptr<cg::AsyncExecutable> func;

    void setup();
    void build_model(float label_smooth = .0f);
    void verify();
    template <typename T>
    void assign_value(std::shared_ptr<HostTensorND> tensor, std::vector<T> value);
};
}  // namespace

void CrossEntropyTest::setup() {
    pred = std::make_shared<HostTensorND>(node, pred_shape, dtype::Float32());
    label = std::make_shared<HostTensorND>(node, label_shape, dtype::Int32());
    truth = std::make_shared<HostTensorND>(node, truth_shape, dtype::Float32());
    loss = std::make_shared<HostTensorND>(node, truth_shape, dtype::Float32());

    assign_value<float>(pred, pred_values);
    assign_value<int>(label, label_values);
    assign_value<float>(truth, truth_values);
}

template <typename T>
void CrossEntropyTest::assign_value(
        std::shared_ptr<HostTensorND> tensor, std::vector<T> values) {
    ASSERT_EQ(values.size(), tensor->shape().total_nr_elems());
    auto ptr = tensor->ptr<T>();
    for (size_t i = 0, it = tensor->shape().total_nr_elems(); i < it; i += 1) {
        ptr[i] = values.at(i);
    }
}

void CrossEntropyTest::build_model(float label_smooth) {
    graph = cg::ComputingGraph::make();

    SymbolVar symbol_pred = opr::SharedDeviceTensor::make(*graph, *pred);
    SymbolVar symbol_label = opr::SharedDeviceTensor::make(*graph, *label);

    SymbolVar symbol_loss = cross_entropy_loss(symbol_pred, symbol_label);

    cg::ComputingGraph::OutputSpec spec;
    spec.push_back({symbol_loss, Device2HostCallback(loss)});
    func = graph->compile(spec);
}

void CrossEntropyTest::verify() {
    func->execute().wait();
    ASSERT_NEAR(loss->ptr<float>()[0], truth->ptr<float>()[0], 0.001f);
}

TEST_F(CrossEntropyTest, CrossEntropy) {
    setup();
    build_model();
    verify();
}