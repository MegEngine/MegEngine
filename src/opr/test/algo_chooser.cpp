/**
 * \file src/opr/test/algo_chooser.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/comp_node_env.h"

#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/helper.h"
#include "megbrain/test/megdnn_helper.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/opr/tensor_manip.h"
#include "megdnn/oprs/base.h"
#include "megdnn/dtype.h"

#include <cmath>
#include <random>
#include <utility>

using namespace mgb;

namespace {

#if MGB_CUDA
#if MGB_ENABLE_FASTRUN
template <typename MgbOpr, int arith>
struct GraphMaker;

template <typename MgbOpr>
struct GraphMaker<MgbOpr, 2> {
    SymbolVar operator()(const std::array<cg::SymbolVar, 2>& inputs,
                          typename MgbOpr::Param& param,
                          typename MgbOpr::ExecutionPolicy& policy) {
        return MgbOpr::make(inputs[0], inputs[1], param, policy);
    }
};

template <>
struct GraphMaker<opr::ConvolutionBackwardData, 2> {
    SymbolVar operator()(
            const std::array<cg::SymbolVar, 2>& inputs,
            opr::ConvolutionBackwardData::Param& param,
            opr::ConvolutionBackwardData::ExecutionPolicy& policy) {
        return opr::ConvolutionBackwardData::make_deconv(inputs[0], inputs[1],
                                                         param, policy);
    }
};

template <>
struct GraphMaker<opr::Convolution3DBackwardData, 2> {
    SymbolVar operator()(
            const std::array<cg::SymbolVar, 2>& inputs,
            opr::Convolution3DBackwardData::Param& param,
            opr::Convolution3DBackwardData::ExecutionPolicy& policy) {
        return opr::Convolution3DBackwardData::make_deconv(inputs[0], inputs[1],
                                                           param, policy);
    }
};

template <typename MgbOpr>
struct GraphMaker<MgbOpr, 3> {
    SymbolVar operator()(const std::array<cg::SymbolVar, 3>& inputs,
                          typename MgbOpr::Param& param,
                          typename MgbOpr::ExecutionPolicy& policy) {
        return MgbOpr::make(inputs[0], inputs[1], inputs[2], param, policy, {});
    }
};

template <typename MgbOpr>
struct GraphMaker<MgbOpr, 4> {
    SymbolVar operator()(const std::array<cg::SymbolVar, 4>& inputs,
                         typename MgbOpr::Param& param,
                         typename MgbOpr::ExecutionPolicy& policy) {
        return MgbOpr::make(inputs[0], inputs[1], inputs[2], inputs[3], param,
                            policy, {});
    }
};

template <typename MgbOpr>
struct GraphMaker<MgbOpr, 5> {
    SymbolVar operator()(const std::array<cg::SymbolVar, 5>& inputs,
                         typename MgbOpr::Param& param,
                         typename MgbOpr::ExecutionPolicy& policy) {
        return MgbOpr::make(inputs[0], inputs[1], inputs[2], inputs[3],
                            inputs[4], param, policy, {});
    }
};

template <typename MgbOpr, int arith, typename dtype = dtype::Float32>
void test_fastrun_opr(std::array<TensorShape, arith> inps0,
                      std::array<TensorShape, arith> inps1,
                      size_t expect_nr_cache_set_inp0 = 0,
                      size_t expect_nr_cache_set_inp1 = 0,
                      typename MgbOpr::Param param = {}) {
    using Policy = opr::Convolution::ExecutionPolicy;
    using S = Policy::Strategy;
    using InputGenerator = std::function<void(HostTensorND & dest)>;
    using ShapeInpArray = std::array<TensorShape, arith>;
    using CacheMem = std::pair<const void*, size_t>;
    auto on_get = [](const std::string&, const void*, size_t, const void*,
                     size_t) {};

    std::vector<std::pair<CacheMem, CacheMem>> cache_set_history;
    auto on_set = [&cache_set_history](const std::string&, const void* key,
                                       size_t key_size, const void* val,
                                       size_t val_size) {
        cache_set_history.emplace_back(std::make_pair(key, key_size),
                                       std::make_pair(val, val_size));
    };

    PersistentCacheHook cache_hook{on_get, on_set};

    CompNode comp_node = CompNode::load("xpu0");
    GraphMaker<MgbOpr, arith> graph_maker;
    auto run = [&param, &comp_node, &graph_maker](
                       const std::shared_ptr<cg::ComputingGraph>& graph,
                       const ShapeInpArray& shapes) {
        std::array<InputGenerator, arith> inputs_generator;
        std::array<std::shared_ptr<HostTensorND>, arith> inputs;
        for (size_t i = 0; i < arith; ++i) {
            inputs[i] = std::make_shared<HostTensorND>(comp_node,
                                                       dtype());
        }
        HostTensorGenerator<dtype> gen_host;
        for (size_t i = 0; i < arith; ++i) {
            inputs[i]->resize(shapes[i]);
            *inputs[i] = *gen_host(inputs[i]->shape(), comp_node);
            mgb_assert(inputs[i]->shape().eq_shape(shapes[i]));
        }
        std::array<cg::SymbolVar, arith> sym_in;
        for (size_t i = 0; i < arith; ++i) {
            // to trigger graph trans
            sym_in[i] = opr::Host2DeviceCopy::make(*graph, inputs[i],
                                                   ssprintf("inp%zu", i));
        }
        Policy policy;
        policy.strategy = S::PROFILE;
        auto out = graph_maker(sym_in, param, policy);

        std::unique_ptr<cg::AsyncExecutable> func =
                graph->compile({{out, {}}});
        func->execute();
    };

    std::shared_ptr<cg::ComputingGraph> fastrun_ignore_batchsize_graph =
            ComputingGraph::make();
    fastrun_ignore_batchsize_graph->options()
            .fast_run_config.shared_batch_size = 20;
    run(fastrun_ignore_batchsize_graph, inps0);
    size_t nr_set_inp0 = cache_set_history.size();
    if (expect_nr_cache_set_inp0) {
        ASSERT_EQ(cache_set_history.size(), expect_nr_cache_set_inp0);
    }
    run(fastrun_ignore_batchsize_graph, inps1);
    size_t nr_set_total = expect_nr_cache_set_inp1 + nr_set_inp0;
    ASSERT_EQ(cache_set_history.size(), nr_set_total);
}

TEST(TestOprDNN, FastrunIgnoreBatchSizeConvolution) {
    REQUIRE_GPU(1);
    test_fastrun_opr<opr::Convolution, 2>(
            {TensorShape{12, 3, 36, 36}, TensorShape{4, 3, 3, 3}},
            {TensorShape{1, 3, 36, 36}, TensorShape{4, 3, 3, 3}});

    test_fastrun_opr<opr::ConvolutionBackwardData, 2>(
            {TensorShape{12, 4, 23, 29}, TensorShape{4, 5, 3, 2}},
            {TensorShape{2, 4, 23, 29}, TensorShape{4, 5, 3, 2}});

    test_fastrun_opr<opr::ConvolutionBackwardFilter, 3>(
            {TensorShape{12, 4, 23, 29}, TensorShape{12, 5, 21, 28},
             TensorShape{5, 4, 3, 2}},
            {TensorShape{2, 4, 23, 29}, TensorShape{2, 5, 21, 28},
             TensorShape{5, 4, 3, 2}});
}

TEST(TestOprDNN, FastrunIgnoreBatchSizeConvBias) {
    REQUIRE_GPU(1);
    test_fastrun_opr<opr::ConvBias, 3>(
            {TensorShape{20, 16, 50, 50}, TensorShape{24, 16, 3, 3},
             TensorShape{1, 24, 1, 1}},
            {TensorShape{1, 16, 50, 50}, TensorShape{24, 16, 3, 3},
             TensorShape{1, 24, 1, 1}});
}

TEST(TestOprDNN, FastrunIgnoreBatchSizeConvolution3D) {
    REQUIRE_GPU(1);
    test_fastrun_opr<opr::Convolution3D, 2>(
            {TensorShape{8, 4, 12, 13, 14}, TensorShape{4, 4, 3, 3, 3}},
            {TensorShape{3, 4, 12, 13, 14}, TensorShape{4, 4, 3, 3, 3}});

    test_fastrun_opr<opr::Convolution3DBackwardData, 2>(
            {TensorShape{14, 5, 12, 12, 16}, TensorShape{5, 5, 3, 3, 3}},
            {TensorShape{4, 5, 12, 12, 16}, TensorShape{5, 5, 3, 3, 3}});

    test_fastrun_opr<opr::Convolution3DBackwardFilter, 3>(
            {TensorShape{64, 16, 18, 18, 18}, TensorShape{64, 16, 18, 18, 18},
             TensorShape{16, 16, 1, 1, 1}},
            {TensorShape{4, 16, 18, 18, 18}, TensorShape{4, 16, 18, 18, 18},
             TensorShape{16, 16, 1, 1, 1}});
}

TEST(TestOprDNN, FastrunIgnoreBatchSizeLocalShare) {
    REQUIRE_GPU(1);
    opr::LocalShare::Param local_share_param;
    local_share_param.mode = opr::LocalShare::Param::Mode::CROSS_CORRELATION;
    local_share_param.pad_h = local_share_param.pad_w = 1;
    local_share_param.stride_h = local_share_param.stride_w = 1;
    local_share_param.spatial_groups_h = local_share_param.spatial_groups_w = 2;
    test_fastrun_opr<opr::LocalShareForward, 2>(
            {TensorShape{32, 2, 23, 23}, TensorShape{2, 2, 2, 2, 2, 7}},
            {TensorShape{3, 2, 23, 23}, TensorShape{2, 2, 2, 2, 2, 7}}, 0, 0,
            local_share_param);

    test_fastrun_opr<opr::LocalShareBackwardData, 3>(
            {TensorShape{3, 3, 128, 1, 1, 128}, TensorShape{32, 128, 24, 24},
             TensorShape{32, 128, 24, 24}},
            {TensorShape{3, 3, 128, 1, 1, 128}, TensorShape{2, 128, 24, 24},
             TensorShape{2, 128, 24, 24}});

    test_fastrun_opr<opr::LocalShareBackwardFilter, 3>(
            {TensorShape{12, 3, 36, 36}, TensorShape{12, 4, 35, 35},
             TensorShape{3, 3, 3, 3, 3, 4}},
            {TensorShape{4, 3, 36, 36}, TensorShape{4, 4, 35, 35},
             TensorShape{3, 3, 3, 3, 3, 4}});
}

TEST(TestOprDNN, FastrunIgnoreBatchSizeDeformableConv) {
    REQUIRE_GPU(1);
    test_fastrun_opr<opr::DeformableConvForward, 4>(
            {TensorShape{12, 6, 20, 20}, TensorShape{6, 6, 3, 3},
             TensorShape{12, 18, 18, 18}, TensorShape{12, 9, 18, 18}},
            {TensorShape{4, 6, 20, 20}, TensorShape{6, 6, 3, 3},
             TensorShape{4, 18, 18, 18}, TensorShape{4, 9, 18, 18}});

    test_fastrun_opr<opr::DeformableConvBackwardData, 5>(
            {TensorShape{12, 6, 20, 20}, TensorShape{6, 6, 3, 3},
             TensorShape{12, 18, 18, 18}, TensorShape{12, 9, 18, 18},
             TensorShape{12, 6, 18, 18}},
            {TensorShape{4, 6, 20, 20},
             TensorShape{6, 6, 3, 3},
             TensorShape{4, 18, 18, 18},
             TensorShape{4, 9, 18, 18},
             TensorShape{4, 6, 18, 18}});

    test_fastrun_opr<opr::DeformableConvBackwardFilter, 5>(
            {TensorShape{12, 6, 20, 20}, TensorShape{6, 6, 3, 3},
             TensorShape{12, 18, 18, 18}, TensorShape{12, 9, 18, 18},
             TensorShape{12, 6, 18, 18}},
            {TensorShape{4, 6, 20, 20}, TensorShape{6, 6, 3, 3},
             TensorShape{4, 18, 18, 18}, TensorShape{4, 9, 18, 18},
             TensorShape{4, 6, 18, 18}});
}

TEST(TestOprDNN, FastrunIgnoreBatchSizeMatrixMul) {
    REQUIRE_GPU(1);
    //! fastrun_shared_batch_size == 20
    //! {20(12), 12(1)}, {12(12), 20(1)} -> {20(12), 20(1)} origin
    //! {12(10), 20(1)}, {12(12), 20(1)} -> {20(12), 20(1)} transA
    //! {12(10), 20(1)}, {20(12), 12(1)} -> {20(12), 20(1)} transA, transB
    //! {20(12), 12(1)}, {20(12), 12(1)} -> {20(12), 20(1)} transB
    //!
    //! {20(12), 12(1)}, {12(12), 20(1)} -> {20(12), 20(1)} origin duplicate
    //! {12(4), 20(1)}, {12(12), 20(1)} -> {20(12), 20(1)} transA
    //! {12(4), 20(1)}, {20(12), 12(1)} -> {20(12), 20(1)} transA, transB
    //! {20(12), 12(1)}, {20(12), 12(1)} -> {20(12), 20(1)} transB duplicate
    test_fastrun_opr<opr::MatrixMul, 2>(
            {TensorShape{10, 12}, TensorShape{12, 12}},
            {TensorShape{4, 12}, TensorShape{12, 12}}, 4, 2);
}

TEST(TestOprDNN, FastrunIgnoreBatchSizeBatchedMatrixMul) {
    REQUIRE_GPU(1);

    //! fastrun_shared_batch_size == 20
    //! {20(48), 6(8), 8(1)}, {20(32), 8(4), 4(1)} -> {20(24), 6(4), 4(1)} origin
    //! {20(48), 8(6), 6(1)}, {20(32), 8(4), 4(1)} -> {20(24), 6(4), 4(1)} transA
    //! {20(48), 8(6), 6(1)}, {20(32), 4(8), 8(1)} -> {20(24), 6(4), 4(1)} transA, transB
    //! {20(48), 6(8), 8(1)}, {20(32), 4(8), 8(1)} -> {20(24), 6(4), 4(1)} transB
    //!
    //! {20(48), 6(8), 8(1)}, {20(32), 8(4), 4(1)} -> {20(24), 6(4), 4(1)} origin duplicate
    //! {20(48), 8(6), 6(1)}, {20(32), 8(4), 4(1)} -> {20(24), 6(4), 4(1)} transA duplicate
    //! {20(48), 8(6), 6(1)}, {20(32), 4(8), 8(1)} -> {20(24), 6(4), 4(1)} transA, transB duplicate
    //! {20(48), 6(8), 8(1)}, {20(32), 4(8), 8(1)} -> {20(24), 6(4), 4(1)} transB duplicate
    test_fastrun_opr<opr::BatchedMatrixMul, 2>(
            {TensorShape{12, 6, 8}, TensorShape{12, 8, 4}},
            {TensorShape{4, 6, 8}, TensorShape{4, 8, 4}});
}

template <typename MgbOpr>
void test_no_profiling_on_shape_change(const TensorShapeArray& inps0,
                                       const TensorShapeArray& inps1) {
    using Policy = typename MgbOpr::ExecutionPolicy;

    int nr_set = 0;
    auto on_get = [](const std::string&, const void*, size_t, const void*,
                     size_t) {};
    auto on_set = [&nr_set](const std::string&, const void*, size_t,
                            const void*, size_t) { nr_set++; };
    PersistentCacheHook cache_hook{on_get, on_set};

    auto cn = CompNode::load("xpu0");
    auto run = [&cn](const TensorShapeArray& shapes) {
        auto graph = ComputingGraph::make();
        graph->options().no_profiling_on_shape_change = true;

        HostTensorGenerator<> gen;
        auto host_a = gen(shapes[0], cn);
        auto host_b = gen(shapes[1], cn);
        HostTensorND host_out;
        auto a = opr::Host2DeviceCopy::make(*graph, host_a),
             b = opr::Host2DeviceCopy::make(*graph, host_b);

        Policy policy;
        policy.strategy = Policy::Strategy::PROFILE;
        auto out = MgbOpr::make(a, b, {}, policy, {});

        std::unique_ptr<cg::AsyncExecutable> func = graph->compile({{out, {}}});
        func->execute();
    };

    run(inps0);
    int nr = nr_set;
    ASSERT_GT(nr, 0);
    run(inps1);
    ASSERT_EQ(nr, nr_set);
}

TEST(TestOprDNN, FastrunNoProfilingOnShapeChange) {
    REQUIRE_GPU(1);

    test_no_profiling_on_shape_change<opr::Convolution>(
            {{12, 3, 36, 36}, {4, 3, 3, 3}}, {{32, 3, 28, 28}, {4, 3, 3, 3}});

    test_no_profiling_on_shape_change<opr::MatrixMul>({{20, 30}, {30, 40}},
                                                      {{30, 40}, {40, 60}});
}

#endif // MGB_ENABLE_FASTRUN
#endif // MGB_CUDA

}  // anonymous namespace

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
