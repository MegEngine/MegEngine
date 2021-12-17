/**
 * \file src/opr/test/training/optimizer.cpp
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

#include "megbrain/opr/training/optimizer.h"
#include "megbrain/opr/training/utils.h"

using namespace mgb;
using namespace optimizer;

namespace {

class Device2HostCallback {
public:
    Device2HostCallback(std::shared_ptr<HostTensorND> host) : m_host{host} {}
    void operator()(const DeviceTensorND& device) { m_host->copy_from(device).sync(); }

private:
    std::shared_ptr<HostTensorND> m_host;
};

template <typename T>
void assign_value(std::shared_ptr<HostTensorND>& tensor, std::vector<T>& values) {
    ASSERT_EQ(values.size(), tensor->layout().total_nr_elems());
    auto ptr = tensor->ptr<T>();
    for (size_t i = 0, it = tensor->layout().total_nr_elems(); i < it; i += 1) {
        ptr[i] = values.at(i);
    }
}

class OptimizerTest : public ::testing::Test {
public:
    void verify(
            std::shared_ptr<IOptimizer> optimizer, std::shared_ptr<HostTensorND> weight,
            std::shared_ptr<HostTensorND> grad, std::shared_ptr<HostTensorND> truth,
            int execute_times);

protected:
    std::shared_ptr<IOptimizer> optimizer;
    std::shared_ptr<cg::ComputingGraph> graph;
};

void OptimizerTest::verify(
        std::shared_ptr<IOptimizer> optimizer, std::shared_ptr<HostTensorND> weight,
        std::shared_ptr<HostTensorND> grad, std::shared_ptr<HostTensorND> truth,
        int execute_times) {
    graph = cg::ComputingGraph::make();
    SymbolVar symbol_weight = opr::SharedDeviceTensor::make(*graph, *weight);
    SymbolVar symbol_grad = opr::SharedDeviceTensor::make(*graph, *grad);

    cg::ComputingGraph::OutputSpec spec;
    spec.push_back(
            {optimizer->make(symbol_weight, symbol_grad, graph),
             Device2HostCallback(weight)});
    auto func = graph->compile(spec);
    for (int i = 0; i < execute_times; i++) {
        func->execute();
    }
    auto weight_ptr = weight->ptr<float>();
    auto truth_ptr = truth->ptr<float>();
    for (size_t i = 0, it = weight->shape().total_nr_elems(); i < it; i += 1) {
        ASSERT_NEAR(weight_ptr[i], truth_ptr[i], 0.001f);
    }
}

}  // namespace

TEST_F(OptimizerTest, SGD) {
    auto weight = TensorGen::constant({1}, 0.30542f);
    auto grad = TensorGen::constant({1}, -1.81453f);
    auto truth = TensorGen::constant({1}, 1.04673f);
    int execute_times = 10;
    std::shared_ptr<SGD> sgd = std::make_shared<SGD>(0.01f, 5e-2f, 0.9f);

    verify(sgd, weight, grad, truth, execute_times);
}

TEST_F(OptimizerTest, AdamTest) {
    auto weight = TensorGen::constant({1}, 1.62957f);
    auto grad = TensorGen::constant({1}, 1.02605f);
    auto truth = TensorGen::constant({1}, 1.52969f);
    int execute_times = 10;
    std::shared_ptr<Adam> adam = std::make_shared<Adam>(0.01f, 0.9f);

    verify(adam, weight, grad, truth, execute_times);
}
