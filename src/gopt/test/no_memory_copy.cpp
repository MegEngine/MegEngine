/**
 * \file src/gopt/test/no_memory_copy.cpp
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include <memory>
#include "./network.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/test/helper.h"

using namespace mgb;

struct TestGraph {
    CompNode m_cn;
    HostTensorGenerator<> m_gen;
    HostTensorGenerator<dtype::Int32> m_gen_int;
    std::unique_ptr<Network> m_network;
    SymbolVar m_out_var;
    std::shared_ptr<HostTensorND> input_tensor;
    std::shared_ptr<HostTensorND> input_tensor2;

    TestGraph() {
        m_cn = CompNode::load("cpu0");
        m_network = std::make_unique<Network>(m_cn);
    }

    void create_graph() {
        input_tensor = m_gen({1, 3, 32, 32}, m_cn);
        auto input = opr::Host2DeviceCopy::make(*m_network->graph, input_tensor, m_cn)
                             .rename("input");
        auto f = m_network->add_conv(
                input, 4, {3, 3}, dtype::Float32(), true, {2, 2}, {0, 0});
        f = m_network->add_elemwise(
                {f}, dtype::Float32(), opr::Elemwise::Param::Mode::EXP);
        f = m_network->add_conv(f, 8, {3, 3}, dtype::Float32(), true, {1, 1}, {1, 1});
        m_out_var = m_network->add_pooling(f, {2, 2}, {2, 2});
    }

    void create_graph_with_subtensor_forward() {
        input_tensor = m_gen({2, 3, 32, 32}, m_cn);
        auto input = opr::Host2DeviceCopy::make(*m_network->graph, input_tensor, m_cn)
                             .rename("input");

        auto cv = [&](int v) {
            auto rst = input.make_scalar(v);
            return rst;
        };

        using Ad = opr::Subtensor::AxisIndexer;
        auto sub =
                opr::Subtensor::make(input, {Ad::make_interval(0, cv(1), cv(2), None)});

        auto f = m_network->add_conv(
                sub, 4, {3, 3}, dtype::Float32(), true, {2, 2}, {0, 0});
        f = m_network->add_elemwise(
                {f}, dtype::Float32(), opr::Elemwise::Param::Mode::EXP);
        f = m_network->add_conv(f, 8, {3, 3}, dtype::Float32(), true, {1, 1}, {1, 1});
        m_out_var = m_network->add_pooling(f, {2, 2}, {2, 2});
    }

    void create_graph_with_subtensor_relayout() {
        input_tensor = m_gen({2, 3, 32, 40}, m_cn);
        auto input = opr::Host2DeviceCopy::make(*m_network->graph, input_tensor, m_cn)
                             .rename("input");

        auto cv = [&](int v) {
            auto rst = input.make_scalar(v);
            return rst;
        };

        using Ad = opr::Subtensor::AxisIndexer;
        auto sub = opr::Subtensor::make(
                input, {Ad::make_interval(0, cv(1), cv(2), None),
                        Ad::make_interval(3, cv(0), cv(32), None)});

        auto f = m_network->add_conv(
                sub, 4, {3, 3}, dtype::Float32(), true, {2, 2}, {0, 0});
        f = m_network->add_elemwise(
                {f}, dtype::Float32(), opr::Elemwise::Param::Mode::EXP);
        f = m_network->add_conv(f, 8, {3, 3}, dtype::Float32(), true, {1, 1}, {1, 1});
        m_out_var = m_network->add_pooling(f, {2, 2}, {2, 2});
    }

    void create_graph_with_setsubtensor() {
        input_tensor = m_gen({1, 3, 32, 32}, m_cn);
        input_tensor2 = m_gen({1, 1, 32, 32}, m_cn);
        auto input = opr::Host2DeviceCopy::make(*m_network->graph, input_tensor, m_cn)
                             .rename("input");

        auto input_sub =
                opr::Host2DeviceCopy::make(*m_network->graph, input_tensor2, m_cn)
                        .rename("input2");

        auto cv = [&](int v) {
            auto rst = input.make_scalar(v);
            return rst;
        };

        using Ad = opr::Subtensor::AxisIndexer;
        input = opr::SetSubtensor::make(
                input, input_sub, {Ad::make_interval(1, cv(1), cv(2), None)});

        auto f = m_network->add_conv(
                input, 4, {3, 3}, dtype::Float32(), true, {2, 2}, {0, 0});
        f = m_network->add_elemwise(
                {f}, dtype::Float32(), opr::Elemwise::Param::Mode::EXP);
        f = m_network->add_conv(f, 8, {3, 3}, dtype::Float32(), true, {1, 1}, {1, 1});
        m_out_var = m_network->add_pooling(f, {2, 2}, {2, 2});
    }

    std::unique_ptr<cg::AsyncExecutable> compile_without_copy() {
        return m_network->graph->compile({{m_out_var, nullptr}});
    }

    std::unique_ptr<cg::AsyncExecutable> compile_with_copy(HostTensorND& host) {
        auto cb = [&host](const DeviceTensorND& dv) mutable { host.copy_from(dv); };
        return m_network->graph->compile({{m_out_var, std::move(cb)}});
    }
};

namespace {
void test_basic_input_no_copy(bool record) {
    auto test_graph = TestGraph();
    auto compute_graph = test_graph.m_network->graph;
    compute_graph->options().comp_node_seq_record_level = record;
    test_graph.create_graph();
    HostTensorND out, out_pre;
    auto func = test_graph.compile_with_copy(out);
    size_t times = 10;
    for (size_t i = 0; i < times; i++) {
        if (i % 2 == 0) {
            auto input_tensor = test_graph.input_tensor;
            auto layout = input_tensor->layout();
            size_t length = layout.total_nr_elems();
            auto storage = TensorStorage<HostTensorStorageTrait>(test_graph.m_cn);
            storage.ensure_size(length * sizeof(float));
            float* ptr = storage.ptr()->as<float>();
            for (size_t d = 0; d < length; d++) {
                ptr[d] = i;
            }
            if (record) {
                input_tensor->only_reset_raw_storage(storage);
            } else {
                input_tensor->reset(storage, layout);
            }
        }
        func->execute();
        func->wait();
        if (i % 2 != 0) {
            MGB_ASSERT_TENSOR_EQ(out, out_pre);
        }
        out_pre.copy_from(out).sync();
    }
}
}  // namespace

TEST(TestNoCopy, InputNoCopyPtrEQ) {
    test_basic_input_no_copy(0);
}

TEST(TestNoCopy, IONoCopyPtrEQ) {
    auto test_graph = TestGraph();
    auto compute_graph = test_graph.m_network->graph;
    compute_graph->options().force_output_use_user_specified_memory = true;
    test_graph.create_graph();
    auto func = test_graph.compile_without_copy();
    auto&& outvar = func->get_output_vars()[0];
    DeviceTensorND dv0(test_graph.m_cn, {1, 8, 7, 7});
    DeviceTensorND dv1(test_graph.m_cn, {1, 8, 7, 7});
    size_t times = 10;
    for (size_t i = 0; i < times; i++) {
        auto input_tensor = test_graph.input_tensor;
        auto layout = input_tensor->layout();
        size_t length = layout.total_nr_elems();
        auto storage = TensorStorage<HostTensorStorageTrait>(test_graph.m_cn);
        storage.ensure_size(length * sizeof(float));
        float* ptr = storage.ptr()->as<float>();
        for (size_t d = 0; d < length; d++) {
            ptr[d] = i;
        }
        input_tensor->reset(storage, layout);
        if (i % 2 == 0) {
            outvar->init_mem_plan(&dv0);
            outvar->reset_dev_tensor_from_tensor(dv0);
        } else {
            outvar->init_mem_plan(&dv1);
            outvar->reset_dev_tensor_from_tensor(dv1);
        }

        func->execute();
        func->wait();
        auto out = func->get_output_vars()[0]->dev_tensor().ptr<float>();

        if (i % 2 == 0) {
            ASSERT_EQ(dv0.ptr<float>(), out);
        } else {
            ASSERT_EQ(dv1.ptr<float>(), out);
        }
    }
}

TEST(TestNoCopy, IONoCopyCorrect) {
    auto test_graph = TestGraph();
    auto compute_graph = test_graph.m_network->graph;
    compute_graph->options().force_output_use_user_specified_memory = true;
    test_graph.create_graph();
    HostTensorND truth;
    auto func = test_graph.compile_without_copy();
    //! because the output tensor not assign user memory, so it will wrong
    ASSERT_THROW(func->execute(), MegBrainError);
    auto&& outvar = func->get_output_vars()[0];
    size_t times = 10;
    for (size_t i = 0; i < times; i++) {
        auto input_tensor = test_graph.input_tensor;
        auto layout = input_tensor->layout();
        size_t length = layout.total_nr_elems();
        auto storage = TensorStorage<HostTensorStorageTrait>(test_graph.m_cn);
        storage.ensure_size(length * sizeof(float));
        float* ptr = storage.ptr()->as<float>();
        for (size_t d = 0; d < length; d++) {
            ptr[d] = i / 5 + 3;
        }
        input_tensor->reset(storage, layout);
        DeviceTensorND dv(test_graph.m_cn, {1, 8, 7, 7});
        outvar->init_mem_plan(&dv);
        outvar->reset_dev_tensor_from_tensor(dv);

        func->execute();
        func->wait();
        if (i % 5 == 0) {
            truth.copy_from(func->get_output_vars()[0]->dev_tensor()).sync();
            continue;
        }
        HostTensorND to_check;
        to_check.copy_from(func->get_output_vars()[0]->dev_tensor()).sync();
        MGB_ASSERT_TENSOR_EQ(to_check, truth);
    }
}

TEST(TestNoCopy, InputNoCopyRecord) {
    test_basic_input_no_copy(1);
}

TEST(TestNoCopy, IONoCopyRecord) {
    auto test_graph = TestGraph();
    auto compute_graph = test_graph.m_network->graph;
    compute_graph->options().force_output_use_user_specified_memory = true;
    compute_graph->options().comp_node_seq_record_level = 1;
    test_graph.create_graph();
    HostTensorND truth;
    auto func = test_graph.compile_without_copy();
    auto&& outvar = func->get_output_vars()[0];
    DeviceTensorND tmp(test_graph.m_cn, {1, 8, 7, 7});
    outvar->init_mem_plan(&tmp);
    size_t times = 10;
    for (size_t i = 0; i < times; i++) {
        auto input_tensor = test_graph.input_tensor;
        auto layout = input_tensor->layout();
        size_t length = layout.total_nr_elems();
        auto storage = TensorStorage<HostTensorStorageTrait>(test_graph.m_cn);
        storage.ensure_size(length * sizeof(float));
        float* ptr = storage.ptr()->as<float>();
        for (size_t d = 0; d < length; d++) {
            ptr[d] = i / 5 + 3;
        }
        input_tensor->only_reset_raw_storage(storage);
        DeviceTensorND dv(test_graph.m_cn, {1, 8, 7, 7});
        dv.raw_ptr();

        auto& dev_tensor = outvar->mutable_dev_tensor();
        dev_tensor.only_reset_raw_storage(dv.storage());

        func->execute();
        func->wait();
        if (i % 5 == 0) {
            truth.copy_from(dv).sync();
            continue;
        }
        HostTensorND to_check;
        to_check.copy_from(dv).sync();
        MGB_ASSERT_TENSOR_EQ(to_check, truth);
    }
}

namespace {
void test_subtensor_record(int level) {
    auto test_graph = TestGraph();
    auto compute_graph = test_graph.m_network->graph;
    compute_graph->options().force_output_use_user_specified_memory = true;
    compute_graph->options().comp_node_seq_record_level = 1;
    if (level == 2) {
        test_graph.create_graph_with_setsubtensor();
    } else if (level == 1) {
        test_graph.create_graph_with_subtensor_forward();
    } else {
        test_graph.create_graph_with_subtensor_relayout();
    }
    HostTensorND truth;
    auto func = test_graph.compile_without_copy();
    auto&& outvar = func->get_output_vars()[0];
    DeviceTensorND tmp(test_graph.m_cn, {1, 8, 7, 7});
    outvar->init_mem_plan(&tmp);
    size_t times = 10;
    for (size_t i = 0; i < times; i++) {
        auto input_tensor = test_graph.input_tensor;
        auto layout = input_tensor->layout();
        size_t length = layout.total_nr_elems();
        auto storage = TensorStorage<HostTensorStorageTrait>(test_graph.m_cn);
        storage.ensure_size(length * sizeof(float));
        float* ptr = storage.ptr()->as<float>();
        for (size_t d = 0; d < length; d++) {
            ptr[d] = i / 5 + 3;
        }
        input_tensor->only_reset_raw_storage(storage);
        DeviceTensorND dv(test_graph.m_cn, {1, 8, 7, 7});
        dv.raw_ptr();

        auto& dev_tensor = outvar->mutable_dev_tensor();
        dev_tensor.only_reset_raw_storage(dv.storage());

        func->execute();
        func->wait();
        if (i % 5 == 0) {
            truth.copy_from(dv).sync();
            continue;
        }
        HostTensorND to_check;
        to_check.copy_from(dv).sync();
        MGB_ASSERT_TENSOR_EQ(to_check, truth);
    }
}
}  // namespace

TEST(TestNoCopy, IONoCopyRecordSubTensor) {
    test_subtensor_record(0);
}

TEST(TestNoCopy, IONoCopyRecordSubTensorRelayout) {
    test_subtensor_record(1);
}

//! TODO: the test should fix compnode memory copy, which now not record reference
//! ptr, when support it, the test will pass
/*TEST(TestNoCopy, IONoCopyRecordSetSubTensor) {
    test_subtensor_record(2);
}*/

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
