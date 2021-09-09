/**
 * \file src/gopt/test/layout_transform_pass.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./network.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/gopt/global_layout_transform.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/serialization/serializer.h"

using namespace mgb;
using namespace gopt;
using namespace serialization;

#if MGB_CUDA
namespace {
//! find first the operator of specific type; raise exception if not found
template <typename T>
T& find_opr(SymbolVar endpoint) {
    T* found = nullptr;
    auto cb = [&found](cg::OperatorNodeBase* opr) {
        if (!found && opr->same_type<T>()) {
            found = &opr->cast_final_safe<T>();
        }
    };
    cg::DepOprIter{cb}.add(endpoint.node()->owner_opr());
    mgb_assert(found, "not found opr from %s", endpoint.node()->name().c_str());
    return *found;
}

template <typename T>
size_t find_opr_num(SymbolVar endpoint) {
    size_t opr_num = 0;
    auto cb = [&opr_num](cg::OperatorNodeBase* opr) {
        if (opr->same_type<T>()) {
            opr_num++;
        }
    };
    cg::DepOprIter{cb}.add(endpoint.node()->owner_opr());
    return opr_num;
}
}  // namespace

TEST(TestLayoutTransform, Resnet18_QS8) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 75) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 75);
        return;
    }
    Network network(cn);
    /// batch size = 1 reduce test time
    auto output = make_resnet18(network, 16, dtype::QuantizedS8{1.f});
    using S = opr::mixin::AlgoChooserHelper::ExecutionPolicy::Strategy;
    S strategy = S::PROFILE;
    gopt::modify_opr_algo_strategy_inplace({{output}}, strategy);

    HostTensorND t1;
    auto func1 = network.graph->compile({make_callback_copy(output, t1)});
    func1->execute();

    using OprFormat = LayoutTransformContext::OprFormat;
    using OprList = LayoutTransformContext::OprList;
    using Target = LayoutTransformContext::Target;
    using ReformatAttribute = LayoutTransformContext::ReformatAttribute;
    using Attribute = LayoutTransformContext::Attribute;
    OprList opr_list = {
            opr::ConvBiasForward::typeinfo(),
            opr::ElemwiseMultiType::typeinfo(),
            opr::Elemwise::typeinfo(),
            opr::TypeCvt::typeinfo(),
            opr::PoolingForward::typeinfo(),
            opr::WarpPerspectiveForward::typeinfo(),
    };
    SmallVector<TensorFormats> available_tensor_formats = {
            TensorFormats::NCHW, TensorFormats::NHWC, TensorFormats::NCHWc4,
            TensorFormats::NCHWc32, TensorFormats::CHWNc4};
    Attribute attribute = {OprFormat::NCHW, TensorFormats::NCHW, Target::UNSPEC,
                           ReformatAttribute::AUTO_PADDING_NHWC};
    auto ctx = std::make_unique<LayoutTransformContext>(
            std::move(opr_list), std::move(available_tensor_formats),
            attribute);
    ctx->add_opr_config(opr::ConvBiasForward::typeinfo(),
                        {OprFormat::NCHW4, OprFormat::NCHW32, OprFormat::CHWN4,
                         OprFormat::NHWC})
            .add_opr_config(opr::PoolingForward::typeinfo(),
                            {OprFormat::NCHW4, OprFormat::NCHW32,
                             OprFormat::NHWC, OprFormat::CHWN4});
    auto profiler = ProfilerBase::make_profiler();
    std::unique_ptr<SolverBase> solver{
            new DynamicProgrammingSolver(std::move(profiler))};
    auto new_output = gopt::GraphOptimizer{}
                              .add_pass<FuseConvBiasNonlinPass>()
                              .add_pass<FuseConvBiasZPass>()
                              .add_pass<LayoutTransformPass>(std::move(ctx),
                                                             std::move(solver))
                              .add_pass<ShuffleShuffleRemovePass>()
                              .add_pass(FuseNCHW4Int8Preprocess::make())
                              .add_pass<FoldingConvBiasDimshufflePass>()
                              .add_pass<ParamFusePass>()
                              .add_pass<ParamMergePass>()
                              .apply({{output}})
                              .endpoint_vars();
    auto new_out_var = new_output[0];
    /// check global layout transform pass
    auto nr_dimshuffle = find_opr_num<opr::Dimshuffle>(new_out_var);
    ASSERT_EQ(nr_dimshuffle, 3u);
    /// check pass fuse conv bias with z
    auto nr_elemwise_mult_type =
            find_opr_num<opr::ElemwiseMultiType>(new_out_var);
    ASSERT_EQ(nr_elemwise_mult_type, 4u);
    /// 21 convolutions, 21 weights and 21 bias, total 42 parameters
    const auto& param_merge =
            find_opr<opr::MultipleDeviceTensorHolder>(new_out_var);
    ASSERT_EQ(param_merge.output().size(), 42u);
    /// check first conv format
    const auto& first_conv = find_opr<opr::ConvBiasForward>(new_out_var);
    const auto& cast = first_conv.cast_final_safe<opr::ConvBiasForward>();
    ASSERT_EQ(cast.param().format, opr::ConvBias::Param::Format::NCHW4);

    GraphProfiler gprof{network.graph.get()};
    HostTensorND t2;
    auto func2 = network.graph->compile({make_callback_copy(new_out_var, t2)});
    func2->execute();
    gprof.to_json_full(func2.get())
            ->writeto_fpath(output_file("resnet18_qs8.json"));
    /// check correct
    MGB_ASSERT_TENSOR_EQ(t1, t2);
}

TEST(TestLayoutTransform, Resnet18_QS4) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 75) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 75);
        return;
    }
    Network network(cn);
    auto output = make_resnet18(network, 16, dtype::QuantizedS4{1.f});
    using S = opr::mixin::AlgoChooserHelper::ExecutionPolicy::Strategy;
    S strategy = S::PROFILE;
    gopt::modify_opr_algo_strategy_inplace({{output}}, strategy);

    HostTensorND t1;
    auto func1 = network.graph->compile({make_callback_copy(output, t1)});
    func1->execute();

    using OprFormat = LayoutTransformContext::OprFormat;
    using OprList = LayoutTransformContext::OprList;
    using Attribute = LayoutTransformContext::Attribute;
    using Target = LayoutTransformContext::Target;
    using ReformatAttribute = LayoutTransformContext::ReformatAttribute;
    OprList opr_list = {
            opr::ConvBiasForward::typeinfo(),
            opr::ElemwiseMultiType::typeinfo(),
            opr::Elemwise::typeinfo(),
            opr::TypeCvt::typeinfo(),
            opr::PoolingForward::typeinfo(),
            opr::WarpPerspectiveForward::typeinfo(),
    };
    SmallVector<TensorFormats> available_tensor_formats = {
            TensorFormats::NCHW,    TensorFormats::NHWC,
            TensorFormats::NCHWc4,  TensorFormats::NCHWc32,
            TensorFormats::NCHWc64, TensorFormats::CHWNc4};
    Attribute attribute = {OprFormat::NCHW, TensorFormats::NCHW, Target::UNSPEC,
                           ReformatAttribute::AUTO_PADDING_NHWC};
    auto ctx = std::make_unique<LayoutTransformContext>(
            std::move(opr_list), std::move(available_tensor_formats),
            attribute);
    ctx->add_opr_config(opr::ConvBiasForward::typeinfo(),
                        {OprFormat::NCHW4, OprFormat::NCHW32, OprFormat::CHWN4,
                         OprFormat::NHWC, OprFormat::NCHW64})
            .add_opr_config(
                    opr::PoolingForward::typeinfo(),
                    {OprFormat::NCHW4, OprFormat::NCHW32, OprFormat::NCHW64,
                     OprFormat::NHWC, OprFormat::CHWN4});
    auto profiler = ProfilerBase::make_profiler();
    std::unique_ptr<SolverBase> solver{
            new DynamicProgrammingSolver(std::move(profiler))};
    auto new_output = gopt::GraphOptimizer{}
                              .add_pass<FuseConvBiasNonlinPass>()
                              .add_pass<FuseConvBiasZPass>()
                              .add_pass<LayoutTransformPass>(std::move(ctx),
                                                             std::move(solver))
                              .add_pass<ShuffleShuffleRemovePass>()
                              .add_pass(FuseNCHW4Int8Preprocess::make())
                              .add_pass<FoldingConvBiasDimshufflePass>()
                              .add_pass<ParamFusePass>()
                              .add_pass<ParamMergePass>()
                              .apply({{output}})
                              .endpoint_vars();
    auto new_out_var = new_output[0];
    /// check global layout transform pass
    auto nr_dimshuffle = find_opr_num<opr::Dimshuffle>(new_out_var);
    ASSERT_EQ(nr_dimshuffle, 3u);
    /// check pass fuse conv bias with z
    auto nr_elemwise_mult_type =
            find_opr_num<opr::ElemwiseMultiType>(new_out_var);
    ASSERT_EQ(nr_elemwise_mult_type, 4u);
    /// 21 convolutions, 21 weights and 21 bias, total 42 parameters
    const auto& param_merge =
            find_opr<opr::MultipleDeviceTensorHolder>(new_out_var);
    ASSERT_EQ(param_merge.output().size(), 42u);
    /// check first conv format
    const auto& first_conv = find_opr<opr::ConvBiasForward>(new_out_var);
    const auto& cast = first_conv.cast_final_safe<opr::ConvBiasForward>();
    ASSERT_EQ(cast.param().format, opr::ConvBias::Param::Format::NHWC);

    GraphProfiler gprof{network.graph.get()};
    HostTensorND t2;
    auto func2 = network.graph->compile({make_callback_copy(new_out_var, t2)});
    func2->execute();
    gprof.to_json_full(func2.get())
            ->writeto_fpath(output_file("resnet18_qs4.json"));
    MGB_ASSERT_TENSOR_EQ(t1, t2);
}

TEST(TestLayoutTransform, Resnet18_NCHW64) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 75) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 75);
        return;
    }
    Network network(cn);
    auto output = make_resnet18(network, 64, dtype::QuantizedS4{1.f});
    using S = opr::mixin::AlgoChooserHelper::ExecutionPolicy::Strategy;
    S strategy = S::PROFILE;
    gopt::modify_opr_algo_strategy_inplace({{output}}, strategy);

    HostTensorND t1;
    auto func1 = network.graph->compile({make_callback_copy(output, t1)});
    func1->execute();

    SymbolVar new_out_var;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_nchw64();
    unpack_vector(gopt::optimize_for_inference({output}, options), new_out_var);

    GraphProfiler gprof{network.graph.get()};
    HostTensorND t2;
    auto func2 = network.graph->compile({make_callback_copy(new_out_var, t2)});
    func2->execute();
    gprof.to_json_full(func2.get())
            ->writeto_fpath(output_file("resnet18_nchw64.json"));
    MGB_ASSERT_TENSOR_EQ(t1, t2);
}

TEST(TestLayoutTransform, Detection_QS8) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 75) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 75);
        return;
    }
    Network network(cn);
    auto outputs = make_det(network, 16, dtype::QuantizedS8{1.f});
    using S = opr::mixin::AlgoChooserHelper::ExecutionPolicy::Strategy;
    S strategy = S::PROFILE;
    gopt::modify_opr_algo_strategy_inplace({outputs}, strategy);

    using OprFormat = LayoutTransformContext::OprFormat;
    using OprList = LayoutTransformContext::OprList;
    using Attribute = LayoutTransformContext::Attribute;
    using Target = LayoutTransformContext::Target;
    using ReformatAttribute = LayoutTransformContext::ReformatAttribute;
    OprList opr_list = {
            opr::ConvBiasForward::typeinfo(),
            opr::ElemwiseMultiType::typeinfo(),
            opr::Elemwise::typeinfo(),
            opr::TypeCvt::typeinfo(),
            opr::PoolingForward::typeinfo(),
            opr::WarpPerspectiveForward::typeinfo(),
    };
    SmallVector<TensorFormats> available_tensor_formats = {
            TensorFormats::NCHW,    TensorFormats::NHWC,
            TensorFormats::NCHWc4,  TensorFormats::NCHWc32,
            TensorFormats::NCHWc64, TensorFormats::CHWNc4};
    Attribute attribute = {OprFormat::NCHW, TensorFormats::NCHW, Target::UNSPEC,
                           ReformatAttribute::AUTO_PADDING_NHWC};
    auto ctx = std::make_unique<LayoutTransformContext>(
            std::move(opr_list), std::move(available_tensor_formats),
            attribute);
    ctx->add_opr_config(opr::ConvBiasForward::typeinfo(),
                        {OprFormat::NCHW4, OprFormat::NCHW32, OprFormat::CHWN4,
                         OprFormat::NHWC, OprFormat::NCHW64})
            .add_opr_config(
                    opr::PoolingForward::typeinfo(),
                    {OprFormat::NCHW4, OprFormat::NCHW32, OprFormat::NCHW64,
                     OprFormat::NHWC, OprFormat::CHWN4});
    auto profiler = ProfilerBase::make_profiler();
    std::unique_ptr<SolverBase> solver{
            new DynamicProgrammingSolver(std::move(profiler))};
    auto new_outputs = gopt::GraphOptimizer{}
                               .add_pass<FuseConvBiasNonlinPass>()
                               .add_pass<FuseConvBiasZPass>()
                               .add_pass<LayoutTransformPass>(std::move(ctx),
                                                              std::move(solver))
                               .add_pass<ShuffleShuffleRemovePass>()
                               .add_pass(FuseNCHW4Int8Preprocess::make())
                               .add_pass<FoldingConvBiasDimshufflePass>()
                               .add_pass<ParamFusePass>()
                               .add_pass<ParamMergePass>()
                               .apply({{outputs}})
                               .endpoint_vars();

    GraphProfiler gprof{network.graph.get()};
    using OutputSpecItem = cg::ComputingGraph::OutputSpecItem;
    std::vector<OutputSpecItem> output_spec;
    for (const auto& i : new_outputs) {
        output_spec.emplace_back(OutputSpecItem{i, {}});
    }
    auto func = network.graph->compile(output_spec);
    func->execute();
    gprof.to_json_full(func.get())->writeto_fpath(output_file("det_qs8.json"));
}

TEST(TestLayoutTransform, Detection_QS4) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 75) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 75);
        return;
    }
    Network network(cn);
    auto outputs = make_det(network, 16, dtype::QuantizedS4{1.f});
    using S = opr::mixin::AlgoChooserHelper::ExecutionPolicy::Strategy;
    S strategy = S::PROFILE;
    gopt::modify_opr_algo_strategy_inplace({outputs}, strategy);

    using OprFormat = LayoutTransformContext::OprFormat;
    using OprList = LayoutTransformContext::OprList;
    using ReformatAttribute = LayoutTransformContext::ReformatAttribute;
    using Attribute = LayoutTransformContext::Attribute;
    using Target = LayoutTransformContext::Target;
    OprList opr_list = {
            opr::ConvBiasForward::typeinfo(),
            opr::ElemwiseMultiType::typeinfo(),
            opr::Elemwise::typeinfo(),
            opr::TypeCvt::typeinfo(),
            opr::PoolingForward::typeinfo(),
            opr::WarpPerspectiveForward::typeinfo(),
    };
    SmallVector<TensorFormats> available_tensor_formats = {
            TensorFormats::NCHW,    TensorFormats::NHWC,
            TensorFormats::NCHWc4,  TensorFormats::NCHWc32,
            TensorFormats::NCHWc64, TensorFormats::CHWNc4};
    Attribute attribute = {OprFormat::NCHW, TensorFormats::NCHW, Target::UNSPEC,
                           ReformatAttribute::AUTO_PADDING_NHWC};
    auto ctx = std::make_unique<LayoutTransformContext>(
            std::move(opr_list), std::move(available_tensor_formats),
            attribute);
    ctx->add_opr_config(opr::ConvBiasForward::typeinfo(),
                        {OprFormat::NCHW4, OprFormat::NCHW32, OprFormat::CHWN4,
                         OprFormat::NHWC, OprFormat::NCHW64})
            .add_opr_config(
                    opr::PoolingForward::typeinfo(),
                    {OprFormat::NCHW4, OprFormat::NCHW32, OprFormat::NCHW64,
                     OprFormat::NHWC, OprFormat::CHWN4});
    auto profiler = ProfilerBase::make_profiler();
    std::unique_ptr<SolverBase> solver{
            new DynamicProgrammingSolver(std::move(profiler))};
    auto new_outputs = gopt::GraphOptimizer{}
                               .add_pass<FuseConvBiasNonlinPass>()
                               .add_pass<FuseConvBiasZPass>()
                               .add_pass<LayoutTransformPass>(std::move(ctx),
                                                              std::move(solver))
                               .add_pass<ShuffleShuffleRemovePass>()
                               .add_pass(FuseNCHW4Int8Preprocess::make())
                               .add_pass<FoldingConvBiasDimshufflePass>()
                               .add_pass<ParamFusePass>()
                               .add_pass<ParamMergePass>()
                               .apply({{outputs}})
                               .endpoint_vars();

    GraphProfiler gprof{network.graph.get()};
    using OutputSpecItem = cg::ComputingGraph::OutputSpecItem;
    std::vector<OutputSpecItem> output_spec;
    for (const auto& i : new_outputs) {
        output_spec.emplace_back(OutputSpecItem{i, {}});
    }
    auto func = network.graph->compile(output_spec);
    func->execute();
    gprof.to_json_full(func.get())->writeto_fpath(output_file("det_qs4.json"));
}

/*!
 * test the performance of the solver when network is wide.
 */
TEST(TestLayoutTransform, Wide) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    Network network(cn);
    auto data = network.add_var("data", {16, 3, 64, 64});
    auto f = network.add_conv(data, 16, {3, 3}, dtype::Float32(), true, {2, 2},
                              {1, 1});
    f = network.add_conv(f, 16, {3, 3}, dtype::Float32(), true, {2, 2}, {1, 1});
    f = network.add_conv(f, 16, {3, 3}, dtype::Float32(), true, {2, 2}, {1, 1});
    SymbolVarArray stages;
    for (size_t i = 0; i < 8; ++i) {
        f = f * f + f;
        stages.push_back(f);
    }
    auto y = stages[0];
    for (size_t i = 1; i < stages.size(); ++i) {
        y = y + stages[i];
    }

    using S = opr::mixin::AlgoChooserHelper::ExecutionPolicy::Strategy;
    S strategy = S::PROFILE;
    gopt::modify_opr_algo_strategy_inplace({y}, strategy);

    using OprFormat = LayoutTransformContext::OprFormat;
    using OprList = LayoutTransformContext::OprList;
    using ReformatAttribute = LayoutTransformContext::ReformatAttribute;
    using Attribute = LayoutTransformContext::Attribute;
    using Target = LayoutTransformContext::Target;
    OprList opr_list = {
            opr::ConvBiasForward::typeinfo(),
            opr::Elemwise::typeinfo(),
    };
    SmallVector<TensorFormats> available_tensor_formats = {TensorFormats::NCHW,
                                                           TensorFormats::NHWC};
    Attribute attribute = {OprFormat::NCHW, TensorFormats::NCHW, Target::UNSPEC,
                           ReformatAttribute::DEFAULT};
    auto ctx = std::make_unique<LayoutTransformContext>(
            std::move(opr_list), std::move(available_tensor_formats),
            attribute);
    ctx->add_opr_config(opr::ConvBiasForward::typeinfo(),
                        {OprFormat::NCHW, OprFormat::NHWC});
    auto profiler = ProfilerBase::make_profiler();
    std::unique_ptr<SolverBase> solver{
            new DynamicProgrammingSolver(std::move(profiler))};
    auto v = gopt::GraphOptimizer{}
                     .add_pass<FuseConvBiasNonlinPass>()
                     .add_pass<FuseConvBiasZPass>()
                     .add_pass<LayoutTransformPass>(std::move(ctx),
                                                    std::move(solver))
                     .add_pass<ShuffleShuffleRemovePass>()
                     .add_pass<ParamFusePass>()
                     .add_pass<ParamMergePass>()
                     .apply({{y}})
                     .endpoint_vars();
    const auto& sym_o = v[0];
    GraphProfiler gprof{network.graph.get()};
    auto func = network.graph->compile({{sym_o, {}}});
    func->execute();
    gprof.to_json_full(func.get())->writeto_fpath(output_file("wide.json"));
    /// check global layout transform pass, no dimshuffle
    auto nr_dimshuffle = find_opr_num<opr::Dimshuffle>(sym_o);
    ASSERT_EQ(nr_dimshuffle, 0u);
    auto nr_param_merge = find_opr_num<opr::MultipleDeviceTensorHolder>(sym_o);
    ASSERT_EQ(nr_param_merge, 1u);
    /// check first conv format
    const auto& first_conv = find_opr<opr::ConvBiasForward>(sym_o);
    const auto& cast = first_conv.cast_final_safe<opr::ConvBiasForward>();
    ASSERT_EQ(cast.param().format, opr::ConvBias::Param::Format::NCHW);
}

TEST(TestLayoutTransform, ElemwiseMultiType) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    Network network(cn);
    auto x = network.add_var("x", {64, 64, 1, 2});
    auto y = network.add_var("y", {64, 64, 1, 2});
    x = network.add_type_cvt(x, dtype::QuantizedS4{1.f});
    y = network.add_type_cvt(y, dtype::QuantizedS4{1.f});
    auto x_ = network.add_type_cvt(x, dtype::Float32());
    auto y_ = network.add_type_cvt(y, dtype::Float32());
    auto z = network.add_elemwise({x_, y_}, dtype::Float32(),
                                  opr::Elemwise::Mode::FUSE_ADD_RELU);
    z = network.add_type_cvt(z, dtype::QuantizedS4{1.f});
    z = network.add_type_cvt(z, dtype::Float32());
    auto z2 = network.add_elemwise({x, y}, dtype::QuantizedS4{1.f},
                                   opr::Elemwise::Mode::FUSE_ADD_RELU);
    z2 = network.add_type_cvt(z2, dtype::Float32());
    HostTensorND t1;
    auto func1 = network.graph->compile({make_callback_copy(z, t1)});
    func1->execute();

    HostTensorND t3;
    auto func3 = network.graph->compile({make_callback_copy(z2, t3)});
    func3->execute();

    auto alter_x = opr::RelayoutFormat::make(
            x, megdnn::param::RelayoutFormat::Mode::NCHW_NCHW64);
    auto alter_y = opr::RelayoutFormat::make(
            y, megdnn::param::RelayoutFormat::Mode::NCHW_NCHW64);
    auto alter_z =
            network.add_elemwise({alter_x, alter_y}, dtype::QuantizedS4{1.f},
                                 opr::Elemwise::Mode::FUSE_ADD_RELU);
    alter_z = opr::RelayoutFormat::make(
            alter_z, megdnn::param::RelayoutFormat::Mode::NCHW64_NCHW);
    alter_z = network.add_type_cvt(alter_z, dtype::Float32());
    HostTensorND t2;
    auto func2 = network.graph->compile({make_callback_copy(alter_z, t2)});
    func2->execute();
    // MGB_ASSERT_TENSOR_EQ(t1, t3);
    MGB_ASSERT_TENSOR_EQ(t2, t3);
}

TEST(TestLayoutTransform, DetectionHead) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    REQUIRE_CUDA_COMPUTE_CAPABILITY_EQ(7, 5);

    constexpr size_t N = 16, C = 3, H = 736, W = 1280;
    HostTensorGenerator<dtype::Uint8> gen;

    auto graph = ComputingGraph::make();
    auto h2d = opr::Host2DeviceCopy::make(*graph, gen({N, C, H, W}, cn));
    auto data = opr::TypeCvt::make(h2d, dtype::Float32());
    auto sub_128 = data + (-128);
    auto x = opr::TypeCvt::make(sub_128, dtype::QuantizedS8(1.f));
    auto mkcvar = [&](const char* name, const TensorShape& shp,
                      const DType& dtype) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                        .rename(name),
                dtype);
    };
    auto w = mkcvar("w", {16, 3, 3, 3}, dtype::QuantizedS8(1.f));
    auto b = mkcvar("b", {1, 16, 1, 1}, dtype::QuantizedS32(1.f));
    opr::ConvBias::Param param;
    param.format = opr::ConvBias::Param::Format::NCHW;
    param.nonlineMode = opr::ConvBias::Param::NonlineMode::RELU;
    param.stride_h = param.stride_w = 2;
    param.pad_h = param.pad_w = 1;
    auto conv_1 = opr::ConvBias::make(
            x, w, b, param, {}, OperatorNodeConfig(dtype::QuantizedS8(1.f)));
    conv_1 = opr::TypeCvt::make(
            conv_1, dtype::Quantized4Asymm(1.f, static_cast<uint8_t>(8)));
    auto w1 = mkcvar("w1", {16, 16, 3, 3}, dtype::QuantizedS4(1.f));
    auto b1 = mkcvar("b1", {1, 16, 1, 1}, dtype::QuantizedS32(1.f));
    auto y = opr::ConvBias::make(conv_1, w1, b1, param, {},
                                 OperatorNodeConfig(dtype::Quantized4Asymm(
                                         1.f, static_cast<uint8_t>(8))));

    using S = opr::mixin::AlgoChooserHelper::ExecutionPolicy::Strategy;
    S strategy = S::PROFILE;
    gopt::modify_opr_algo_strategy_inplace({y}, strategy);

    using OprFormat = LayoutTransformContext::OprFormat;
    using OprList = LayoutTransformContext::OprList;
    using Attribute = LayoutTransformContext::Attribute;
    using Target = LayoutTransformContext::Target;
    OprList opr_list = {
            opr::ConvBiasForward::typeinfo(),
            opr::ConvolutionForward::typeinfo(),
            opr::ConvolutionBackwardData::typeinfo(),
            opr::ElemwiseMultiType::typeinfo(),
            opr::Elemwise::typeinfo(),
            opr::TypeCvt::typeinfo(),
            opr::PoolingForward::typeinfo(),
            opr::WarpPerspectiveForward::typeinfo(),
    };
    SmallVector<TensorFormats> available_tensor_formats = {
            TensorFormats::NCHW,    TensorFormats::NHWC,
            TensorFormats::NCHWc4,  TensorFormats::NCHWc32,
            TensorFormats::NCHWc64, TensorFormats::CHWNc4};
    Attribute attribute = {OprFormat::NCHW, TensorFormats::NCHW,
                           Target::UNSPEC};
    auto ctx = std::make_unique<LayoutTransformContext>(
            std::move(opr_list), std::move(available_tensor_formats),
            attribute);
    ctx->add_opr_config(
               opr::ConvBiasForward::typeinfo(),
               {OprFormat::NCHW, OprFormat::NHWC, OprFormat::NCHW4,
                OprFormat::NCHW32, OprFormat::NCHW64, OprFormat::CHWN4})
            .add_opr_config(opr::ConvolutionForward::typeinfo(),
                            {OprFormat::NCHW, OprFormat::NCHW4})
            .add_opr_config(opr::ConvolutionBackwardData::typeinfo(),
                            {OprFormat::NCHW, OprFormat::NCHW4})
            .add_opr_config(
                    opr::PoolingForward::typeinfo(),
                    {OprFormat::NCHW4, OprFormat::NCHW32, OprFormat::NHWC,
                     OprFormat::NCHW64, OprFormat::CHWN4})
            .add_opr_config(
                    opr::WarpPerspectiveForward::typeinfo(),
                    {OprFormat::NHWC, OprFormat::NCHW4, OprFormat::NCHW64});

    auto profiler = ProfilerBase::make_profiler();
    std::unique_ptr<SolverBase> solver{
            new DynamicProgrammingSolver(std::move(profiler))};
    auto new_out_vars = gopt::GraphOptimizer{}
                                .add_pass<LayoutTransformPass>(
                                        std::move(ctx), std::move(solver))
                                .add_pass<ShuffleShuffleRemovePass>()
                                .add_pass(FuseNCHW4Int8Preprocess::make())
                                .add_pass<FoldingConvBiasDimshufflePass>()
                                .add_pass<ParamFusePass>()
                                .add_pass<ParamMergePass>()
                                .apply(SymbolVarArray{y})
                                .endpoint_vars();
    const auto& v = new_out_vars[0];
    using OutputSpecItem = cg::ComputingGraph::OutputSpecItem;
    std::vector<OutputSpecItem> outs;
    for (const auto& i : new_out_vars) {
        outs.emplace_back(OutputSpecItem{i, {}});
    }
    GraphProfiler gprof{graph.get()};
    auto func = graph->compile(outs);
    func->execute();
    gprof.to_json_full(func.get())->writeto_fpath(output_file("det_head.json"));
    /// check reformat
    auto nr_reformat = find_opr_num<opr::RelayoutFormat>(v);
    ASSERT_EQ(nr_reformat, 2u);
    /// check dimshuffle
    auto nr_dimshuffle = find_opr_num<opr::Dimshuffle>(v);
    ASSERT_EQ(nr_dimshuffle, 0u);
    /// check conv_bias
    auto nr_conv = find_opr_num<opr::ConvBiasForward>(v);
    ASSERT_EQ(nr_conv, 2u);
    /// check first conv format
    const auto& first_conv = find_opr<opr::ConvBiasForward>(v);
    const auto& cast = first_conv.cast_final_safe<opr::ConvBiasForward>();
    ASSERT_EQ(cast.param().format, opr::ConvBias::Param::Format::NCHW4_NHWC);
}

#endif

TEST(TestLayoutTransform, CanonicalizeLayoutTransform) {
    constexpr size_t N = 64, C = 64, H = 1, W = 1;
    auto cn = CompNode::load("xpu0");
    Network network(cn);
    auto x = network.add_var("x", {N, C / 4, H, W, 4});
    x = network.add_type_cvt(x, dtype::QuantizedS4{1.f});
    using NamedTensorShape = megdnn::NamedTensorShape;
    auto src = NamedTensorShape::make_named_tensor_shape(
            NamedTensorShape::Format::NCHW4);
    auto dst = NamedTensorShape::make_named_tensor_shape(
            NamedTensorShape::Format::NHWC);
    auto [builder, _] = gopt::ReformatEmitter(src, dst).emit();
    MGB_MARK_USED_VAR(_);
    x = SymbolVar(builder({x.node()}));
    x = opr::Reshape::make(x, {N, H, W, C});
    x = network.add_type_cvt(x, dtype::Float32());

    SymbolVar another_x;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ShuffleShuffleRemovePass>()
                          .apply({{x}})
                          .endpoint_vars(),
                  another_x);
    const auto& astype = find_opr<opr::TypeCvt>(x);
    EXPECT_TRUE(astype.input(0)->owner_opr()->dyn_typeinfo() ==
                opr::Host2DeviceCopy::typeinfo());
    const auto& another_astype = find_opr<opr::TypeCvt>(another_x);
    EXPECT_TRUE(another_astype.input(0)->owner_opr()->dyn_typeinfo() ==
                opr::Reshape::typeinfo());

    HostTensorND t1;
    auto func1 = network.graph->compile({make_callback_copy(x, t1)});
    func1->execute();

    HostTensorND t2;
    auto func2 = network.graph->compile({make_callback_copy(another_x, t2)});
    func2->execute();
    MGB_ASSERT_TENSOR_EQ(t1, t2);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
