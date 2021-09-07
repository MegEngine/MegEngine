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

#include "./helper.h"
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
TEST(TestLayoutTransform, Feature) {
    auto inp_file = InputFile::make_fs("./feat.mdl");

    auto format = GraphLoader::identify_graph_dump_format(*inp_file);
    ASSERT_TRUE(format.valid());
    auto loader = GraphLoader::make(std::move(inp_file), format.val());

    GraphLoader::LoadConfig load_config;
    load_config.comp_graph = ComputingGraph::make();
    auto&& graph_opt = load_config.comp_graph->options();
    graph_opt.graph_opt.enable_fuse_conv_bias_nonlinearity();
    graph_opt.graph_opt.enable_fuse_conv_bias_with_z();
    auto ret = loader->load(load_config, false);

    using S = opr::mixin::AlgoChooserHelper::ExecutionPolicy::Strategy;
    S strategy = S::PROFILE;
    gopt::modify_opr_algo_strategy_inplace({ret.output_var_list}, strategy);

    using OprFormat = LayoutTransformContext::OprFormat;
    using OprList = LayoutTransformContext::OprList;
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
            TensorFormats::NCHWc4, TensorFormats::NCHWc32,
            TensorFormats::CHWNc4};
    Attribute attribute = {OprFormat::NCHW4, TensorFormats::NCHWc4,
                           ReformatAttribute::DEFAULT};
    auto ctx = std::make_unique<LayoutTransformContext>(
            std::move(opr_list), std::move(available_tensor_formats),
            attribute);
    ctx->add_opr_config(opr::ConvBiasForward::typeinfo(),
                        {OprFormat::NCHW4, OprFormat::NCHW32, OprFormat::CHWN4})
            .add_opr_config(
                    opr::PoolingForward::typeinfo(),
                    {OprFormat::NCHW4, OprFormat::NCHW32, OprFormat::CHWN4})
            .add_opr_config(opr::WarpPerspectiveForward::typeinfo(),
                            OprFormat::NCHW4);
    auto profiler = ProfilerBase::make_profiler();
    auto filter = [](const GraphPartition& partition) {
        auto has_nchw4_conv = false;
        for (auto&& opr : partition.all_oprs()) {
            if (opr->dyn_typeinfo() == opr::ConvBiasForward::typeinfo()) {
                auto& conv = opr->cast_final_safe<opr::ConvBiasForward>();
                if (conv.param().format ==
                    LayoutTransformContext::OprFormat::NCHW4) {
                    has_nchw4_conv = true;
                    break;
                }
            }
        }
        return has_nchw4_conv;
    };
    std::unique_ptr<SolverBase> solver{new DynamicProgrammingSolver(
            std::move(profiler), std::move(filter))};
    auto new_out_vars = gopt::GraphOptimizer{}
                                .add_pass<FuseConvBiasNonlinPass>()
                                .add_pass<FuseConvBiasZPass>()
                                .add_pass<LayoutTransformPass>(
                                        std::move(ctx), std::move(solver))
                                .add_pass<ShuffleShuffleRemovePass>()
                                .add_pass(FuseNCHW4Int8Preprocess::make())
                                .add_pass<FoldingConvBiasDimshufflePass>()
                                .add_pass<ParamFusePass>()
                                .add_pass<ParamMergePass>()
                                .apply(ret.output_var_list)
                                .endpoint_vars();
    auto dumper = GraphDumper::make(OutputFile::make_fs("model_opt.mgb"));
    dumper->dump({new_out_vars});
}

TEST(TestLayoutTransform, Detection) {
    auto inp_file = InputFile::make_fs("./det.mdl");
    static const char* magic = "mgbteset0";
    size_t skip_size = sizeof(magic) + sizeof(uint32_t);
    char skip[skip_size];
    inp_file->read(skip, skip_size);

    auto format = GraphLoader::identify_graph_dump_format(*inp_file);
    ASSERT_TRUE(format.valid());
    auto loader = GraphLoader::make(std::move(inp_file), format.val());

    GraphLoader::LoadConfig load_config;
    load_config.comp_graph = ComputingGraph::make();
    auto&& graph_opt = load_config.comp_graph->options();
    graph_opt.graph_opt.enable_fuse_conv_bias_nonlinearity();
    graph_opt.graph_opt.enable_fuse_conv_bias_with_z();
    auto ret = loader->load(load_config, false);

    using S = opr::mixin::AlgoChooserHelper::ExecutionPolicy::Strategy;
    S strategy = S::PROFILE;
    gopt::modify_opr_algo_strategy_inplace({ret.output_var_list}, strategy);

    using OprFormat = LayoutTransformContext::OprFormat;
    using OprList = LayoutTransformContext::OprList;
    using ReformatAttribute = LayoutTransformContext::ReformatAttribute;
    using Attribute = LayoutTransformContext::Attribute;
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
                           ReformatAttribute::DEFAULT};
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
                                .apply(ret.output_var_list)
                                .endpoint_vars();
    using OutputSpecItem = cg::ComputingGraph::OutputSpecItem;
    std::vector<OutputSpecItem> outs(new_out_vars.size());
    for (size_t i = 0; i < new_out_vars.size(); ++i) {
        auto cb = [](DeviceTensorND& /* d */) {};
        outs[i] = std::make_pair(new_out_vars[i], cb);
    }
    GraphProfiler gprof{load_config.comp_graph.get()};
    auto func = load_config.comp_graph->compile(outs);
    for (size_t i = 0; i < 10; ++i)
        func->execute();
    func->wait();
    gprof.to_json_full(func.get())->writeto_fpath(output_file("det.json"));
}

TEST(TestLayoutTransform, DetectionHead) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    REQUIRE_CUDA_COMPUTE_CAPABILITY_EQ(7, 5);

    constexpr size_t N = 16, C = 3, H = 768, W = 1280;
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
    using ReformatAttribute = LayoutTransformContext::ReformatAttribute;
    using Attribute = LayoutTransformContext::Attribute;
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
                           ReformatAttribute::DEFAULT};
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
    using OutputSpecItem = cg::ComputingGraph::OutputSpecItem;
    std::vector<OutputSpecItem> outs(new_out_vars.size());
    for (size_t i = 0; i < new_out_vars.size(); ++i) {
        auto cb = [](DeviceTensorND& /* d */) {};
        outs[i] = std::make_pair(new_out_vars[i], cb);
    }
    GraphProfiler gprof{graph.get()};
    auto func = graph->compile(outs);
    for (size_t i = 0; i < 10; ++i)
        func->execute();
    func->wait();
    gprof.to_json_full(func.get())->writeto_fpath(output_file("det_head.json"));
}

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
