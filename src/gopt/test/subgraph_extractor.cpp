/**
 * \file src/gopt/test/subgraph_extractor.cpp
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

#include "megbrain/gopt/subgraph_extractor.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/internal/identical_fwd.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/serialization/serializer.h"

using namespace mgb;
using namespace gopt;
using namespace serialization;

namespace {
// clang-format off
MGB_DEFINE_OPR_CLASS(MultipleInputOutput,
                     cg::SingleCNOperatorNodeBase) // {
public:
    MultipleInputOutput(const VarNodeArray& inputs, const OperatorNodeConfig& config);

    static SymbolVarArray make(const SymbolVarArray& inputs, const OperatorNodeConfig& config = {});
private:
    void scn_do_execute() override {  }
    void init_output_static_infer_desc() override {  }
};
// clang-format on

MGB_DYN_TYPE_OBJ_FINAL_IMPL(MultipleInputOutput);

MultipleInputOutput::MultipleInputOutput(
        const VarNodeArray& inputs, const OperatorNodeConfig& config)
        : Super(inputs[0]->owner_graph(), config, "multiple_input_output", inputs) {
    for (auto&& i : inputs)
        add_input({i});
    if (inputs.size() == 1) {
        add_output(None);
    } else {
        for (size_t i = 0; i < inputs.size(); ++i)
            add_output(ssprintf("o%zu", i));
    }
    cg::add_workspace_output(this);
}

SymbolVarArray MultipleInputOutput::make(
        const SymbolVarArray& inputs, const OperatorNodeConfig& config) {
    auto src = cg::to_var_node_array(inputs);
    auto multiple_io = std::make_unique<MultipleInputOutput>(src, config);
    auto ret = cg::to_symbol_var_array(
            src[0]->owner_graph()->insert_opr(std::move(multiple_io))->output());
    ret.pop_back();
    return ret;
}
}  // namespace

TEST(TestSubGraphExtractor, MultipleOutputs) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();

    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };

    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp)).rename(name);
    };

    graph->options().graph_opt_level = 0;
    auto x = mkvar("x", {8, 8, 8, 8}), w1 = mkcvar("w1", {4, 8, 3, 3});
    auto y = mkvar("y", {1, 8, 1, 1});
    auto add = x + y;

    opr::Convolution::Param param;
    param.pad_h = param.pad_w = 1;
    auto c1 = opr::Convolution::make(add, w1, param);
    auto w2 = mkcvar("w2", {8, 4, 3, 3});
    auto c2 = opr::ConvolutionBackwardData::make(w2, add, param, {}, {});
    auto sym_var_arr = MultipleInputOutput::make({c1, c2});
    auto z = sym_var_arr[1];
    z = z + (-128);

    using OprList = SubGraphExtractor::OprList;
    static const OprList opr_list = {
            opr::ConvolutionForward::typeinfo(),
            opr::Elemwise::typeinfo(),
            opr::TypeCvt::typeinfo(),
            MultipleInputOutput::typeinfo(),
    };
    SubGraphExtractor extractor(opr_list);
    auto partitions = extractor.extract({z});
    ASSERT_EQ(partitions.size(), 1u);
    // outputs: sym_var_arr[0], z, add
    ASSERT_EQ(partitions[0].output().size(), 3u);
    ASSERT_TRUE(partitions[0].output().count(add.node()) > 0);
    ASSERT_TRUE(partitions[0].output().count(z.node()) > 0);
    ASSERT_TRUE(partitions[0].output().count(sym_var_arr[0].node()) > 0);
    ASSERT_TRUE(partitions[0].output().count(sym_var_arr[1].node()) == 0);
    // inputs: x, y, w1, c2, (-128)
    ASSERT_EQ(partitions[0].input().size(), 5u);
    ASSERT_TRUE(partitions[0].input().count(x.node()) > 0);
    ASSERT_TRUE(partitions[0].input().count(c2.node()) > 0);
    // opr: (x + y) conv1 multi_io, (z - 128)
    ASSERT_EQ(partitions[0].opr_set().size(), 4u);
    ASSERT_TRUE(partitions[0].opr_set().count(add.node()->owner_opr()) > 0);
    ASSERT_TRUE(partitions[0].opr_set().count(c1.node()->owner_opr()) > 0);
    ASSERT_TRUE(partitions[0].opr_set().count(sym_var_arr[0].node()->owner_opr()) > 0);
    ASSERT_TRUE(partitions[0].opr_set().count(z.node()->owner_opr()) > 0);
}

TEST(TestSubGraphExtractor, MultipleReaders) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();

    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };

    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp)).rename(name);
    };

    graph->options().graph_opt_level = 0;
    auto x = mkvar("x", {8, 8, 8, 8}), w1 = mkcvar("w1", {4, 8, 3, 3});
    auto y = mkvar("y", {1, 8, 1, 1});
    auto add = x + y;

    opr::Convolution::Param param;
    param.pad_h = param.pad_w = 1;
    auto c1 = opr::Convolution::make(add, w1, param);
    auto w2 = mkcvar("w2", {8, 4, 3, 3});
    auto c2 = opr::ConvolutionBackwardData::make(w2, add, param, {}, {});
    auto z = c1 + c2;

    using OprList = SubGraphExtractor::OprList;
    static const OprList opr_list = {
            opr::ConvolutionForward::typeinfo(),
            opr::Elemwise::typeinfo(),
            opr::TypeCvt::typeinfo(),
    };
    SubGraphExtractor extractor(opr_list);
    auto partitions = extractor.extract({z});
    ASSERT_EQ(partitions.size(), 1u);
    ASSERT_EQ(partitions[0].output().size(), 2u);
    ASSERT_TRUE(partitions[0].output().count(add.node()) > 0);
    ASSERT_TRUE(partitions[0].output().count(z.node()) > 0);
    ASSERT_EQ(partitions[0].input().size(), 4u);
    ASSERT_TRUE(partitions[0].input().count(x.node()) > 0);
    partitions[0].to_json()->writeto_fpath(
            output_file("TestSubGraphExtractor.MultipleReaders.json"));
}

TEST(TestSubGraphExtractor, Complicated) {
    const size_t N = 16, C = 3, H = 768, W = 1280;
    HostTensorGenerator<dtype::Uint8> gen;
    auto graph = ComputingGraph::make();
    /* h2d
        |
        v
       astype(f32)
        |
       add(-128)
        |
        v
       astype(q8)
        |
        v
       conv1
        |
        v
       astype(u4)
          |
         / \
      conv2 conv3 -> astype(q32) -> output
         \ /
         qadd
          |
          v
        astype(q8)
          / \
      deconv conv4
          \ /
         concat -> output */
    auto h2d = opr::Host2DeviceCopy::make(*graph, gen({N, C, H, W}));
    auto data = opr::TypeCvt::make(h2d, dtype::Float32());
    auto sub_128 = data + (-128);
    auto x = opr::TypeCvt::make(sub_128, dtype::QuantizedS8(1.f));
    auto mkcvar = [&](const char* name, const TensorShape& shp, const DType& dtype) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *gen(shp)).rename(name), dtype);
    };
    auto w1 = mkcvar("w1", {16, 3, 3, 3}, dtype::QuantizedS8(1.f));
    auto b1 = mkcvar("b1", {1, 16, 1, 1}, dtype::QuantizedS32(1.f));
    opr::ConvBias::Param param;
    param.stride_h = param.stride_w = 2;
    param.pad_h = param.pad_w = 1;
    auto conv1 = opr::ConvBias::make(
            x, w1, b1, param, {}, OperatorNodeConfig(dtype::QuantizedS8(1.f)));
    conv1 = opr::TypeCvt::make(
            conv1, dtype::Quantized4Asymm(1.f, static_cast<uint8_t>(8)));
    auto w2 = mkcvar("w2", {16, 16, 3, 3}, dtype::QuantizedS4(1.f));
    auto b2 = mkcvar("b2", {1, 16, 1, 1}, dtype::QuantizedS32(1.f));
    auto conv2 = opr::ConvBias::make(
            conv1, w2, b2, param, {},
            OperatorNodeConfig(dtype::Quantized4Asymm(1.f, static_cast<uint8_t>(8))));
    param.pad_h = param.pad_w = 0;
    auto w3 = mkcvar("w3", {16, 16, 1, 1}, dtype::QuantizedS4(1.f));
    auto b3 = mkcvar("b3", {1, 16, 1, 1}, dtype::QuantizedS32(1.f));
    auto conv3 = opr::ConvBias::make(
            conv1, w3, b3, param, {},
            OperatorNodeConfig(dtype::Quantized4Asymm(1.f, static_cast<uint8_t>(8))));
    auto conv3f = opr::TypeCvt::make(conv3, dtype::Float32());
    auto qadd = opr::ElemwiseMultiType::make(
            {conv2, conv3}, {opr::ElemwiseMultiType::Mode::QADD},
            OperatorNodeConfig(dtype::Quantized4Asymm(1.f, static_cast<uint8_t>(8))));
    auto q8 = opr::TypeCvt::make(qadd, dtype::QuantizedS8(1.f));

    auto w4 = mkcvar("w4", {16, 16, 3, 3}, dtype::QuantizedS8(1.f));
    param.stride_h = param.stride_w = 1;
    param.pad_h = param.pad_w = 1;
    auto conv4 = opr::ConvBiasForward::make(
            q8, w4, param, {}, OperatorNodeConfig(dtype::QuantizedS8(1.f)));
    conv4 = opr::TypeCvt::make(conv4, dtype::Float32());

    opr::Convolution::Param conv_param;
    conv_param.stride_h = param.stride_w = 1;
    conv_param.pad_h = param.pad_w = 0;
    auto w5 = mkcvar("w4", {16, 16, 1, 1}, dtype::QuantizedS8(1.f));
    auto deconv = opr::ConvolutionBackwardData::make(
            w5, q8, conv_param, {}, OperatorNodeConfig(dtype::QuantizedS8(1.f)));
    deconv = opr::TypeCvt::make(deconv, dtype::Float32());
    auto z = opr::Concat::make({conv4, deconv}, 1);

    using OprList = SubGraphExtractor::OprList;
    static const OprList opr_list = {
            opr::ConvBiasForward::typeinfo(),
            opr::ConvolutionForward::typeinfo(),
            opr::ConvolutionBackwardData::typeinfo(),
            opr::ElemwiseMultiType::typeinfo(),
            opr::Elemwise::typeinfo(),
            opr::TypeCvt::typeinfo(),
            opr::PoolingForward::typeinfo(),
            opr::WarpPerspectiveForward::typeinfo(),
    };
    SubGraphExtractor extractor(opr_list);
    auto partitions = extractor.extract({conv3f.node(), z.node()});
    ASSERT_EQ(partitions.size(), 1u);
    const char* prefix = "TestSubGraphExtractor.Complicated";
    partitions[0].to_json()->writeto_fpath(
            output_file(ssprintf("%s.json", prefix).c_str()));
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
