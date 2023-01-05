#include "megbrain/graph/cg.h"
#include "megbrain/opr/dnn/local.h"

#include "megbrain/gopt/basic_arith.h"
#include "megbrain/gopt/gtrans.h"
#include "megbrain/gopt/inference.h"

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/adaptive_pooling.h"
#include "megbrain/opr/dnn/batch_norm.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/opr/tensor_gen.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"

#include "helper.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/test/helper.h"

#include "megdnn/tensor_format.h"

#include <random>
#include <vector>

using namespace mgb;

namespace {
//! find first the operator of specific type; raise exception if not found
template <typename T>
T* find_opr(SymbolVar endpoint) {
    T* found = nullptr;
    auto cb = [&found](cg::OperatorNodeBase* opr) {
        if (!found && opr->same_type<T>()) {
            found = &opr->cast_final_safe<T>();
        }
    };
    cg::DepOprIter{cb}.add(endpoint.node()->owner_opr());
    mgb_assert(found, "not found opr from %s", endpoint.node()->name().c_str());
    return found;
}

template <typename T>
T* find_opr(SymbolVar endpoint, const std::string& node_name) {
    T* found = nullptr;
    auto cb = [&found, &node_name](cg::OperatorNodeBase* opr) {
        if (!found && opr->same_type<T>() && opr->name() == node_name) {
            found = &opr->cast_final_safe<T>();
        }
    };
    cg::DepOprIter{cb}.add(endpoint.node()->owner_opr());
    mgb_assert(
            found, "not found opr %s from %s", node_name.c_str(),
            endpoint.node()->name().c_str());
    return found;
}
}  // namespace

TEST(TestGoptInference, ChannelPaddingNCHW44) {
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn)).rename(name);
    };

    auto host_x = gen({1, 3, 8, 8}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    //! Hybrid nchw44 mode
    opr::ConvBias::Param param_conv;
    param_conv.pad_h = param_conv.pad_w = 1;
    auto w1 = mkcvar("w1", {8, 3, 3, 3}), b1 = mkcvar("w1", {1, 8, 1, 1}),
         conv1 = opr::ConvBias::make(
                 x, w1, b1, param_conv, {}, OperatorNodeConfig("conv1"));

    auto w2 = mkcvar("w2", {6, 8, 3, 3}), b2 = mkcvar("b2", {1, 6, 1, 1}),
         conv2 = opr::ConvBias::make(
                 conv1, w2, b2, param_conv, {}, OperatorNodeConfig("conv2"));
    auto w3 = mkcvar("w3", {3, 6, 3, 3}), b3 = mkcvar("b3", {1, 3, 1, 1}),
         conv3 = opr::ConvBias::make(
                 conv2, w3, b3, param_conv, {}, OperatorNodeConfig("conv3"));

    opr::Convolution::Param param_convolution;
    param_convolution.sparse = opr::Convolution::Param::Sparse::GROUP;
    //! channel wise convolution
    auto w4 = mkcvar("w4", {3, 1, 1, 1, 1}),
         conv4 = opr::Convolution::make(
                 conv3, w4, param_convolution, {}, OperatorNodeConfig("conv4"));

    param_convolution.sparse = opr::Convolution::Param::Sparse::DENSE;
    auto w5 = mkcvar("w5", {6, 3, 1, 1}),
         conv5 = opr::Convolution::make(
                 conv4, w5, param_convolution, {}, OperatorNodeConfig("conv5"));

    //! group convolution
    param_convolution.sparse = opr::Convolution::Param::Sparse::GROUP;
    auto w6 = mkcvar("w6", {2, 4, 3, 1, 1}),
         conv6 = opr::Convolution::make(
                 conv5, w6, param_convolution, {}, OperatorNodeConfig("conv6"));

    param_convolution.sparse = opr::Convolution::Param::Sparse::DENSE;
    auto w7 = mkcvar("w7", {3, 8, 1, 1}),
         y = opr::Convolution::make(
                 conv6, w7, param_convolution, {}, OperatorNodeConfig("conv7"));

    SymbolVar y_opt;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_fuse_conv_bias_nonlinearity();
    options.enable_nchw44();
    unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    auto conv1_opt = find_opr<opr::ConvBias>(y_opt, "conv1");
    auto conv2_opt = find_opr<opr::ConvBias>(y_opt, "conv2");
    auto conv3_opt = find_opr<opr::ConvBias>(y_opt, "conv3");
    auto conv4_opt = find_opr<opr::Convolution>(y_opt, "conv4");
    auto conv6_opt = find_opr<opr::Convolution>(y_opt, "conv6");
    //! do not padding input tensor
    ASSERT_EQ(conv1_opt->input(0)->shape()[1], 3);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44, conv1_opt->param().format);
    //! output tensor padding input tensor
    ASSERT_EQ(conv2_opt->input(1)->shape()[0], 2);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44, conv2_opt->param().format);
    ASSERT_EQ(conv3_opt->input(1)->shape()[0], 1);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44, conv3_opt->param().format);

    ASSERT_EQ(conv4_opt->input(1)->shape()[0], 1);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44, conv4_opt->param().format);
    ASSERT_EQ(conv6_opt->input(0)->shape()[1], 6);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW, conv6_opt->param().format);

    //! the dst tensor channel must stay unchange
    ASSERT_EQ(y_opt.node()->shape()[1], 3);
    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(output_file("TestGoptInference.ChannelPaddingNCHW44.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile(
            {make_callback_copy(y, host_y), make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-2);

    //! test change the input shape
    *host_x = *gen({2, 3, 32, 32}, cn);
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-2);
}

TEST(TestGoptInference, ChannelPaddingSubtensor) {
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn)).rename(name);
    };

    auto host_x = gen({1, 3, 8, 8}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    //! Hybrid nchw44 mode
    opr::ConvBias::Param param_conv;
    param_conv.pad_h = param_conv.pad_w = 1;
    auto w1 = mkcvar("w1", {8, 3, 3, 3}), b1 = mkcvar("w1", {1, 8, 1, 1}),
         conv1 = opr::ConvBias::make(
                 x, w1, b1, param_conv, {}, OperatorNodeConfig("conv1"));

    auto w2 = mkcvar("w2", {6, 8, 1, 1}),
         conv2 = opr::Convolution::make(conv1, w2, {}, {}, OperatorNodeConfig("conv2"));
    using AIdx = opr::indexing::AxisIndexer;
    auto sub0 = opr::Subtensor::make(
            conv2,
            {AIdx::make_interval(
                    2, conv2.make_scalar(1), conv2.make_scalar(4),
                    conv2.make_scalar(1))},
            OperatorNodeConfig("sub0"));
    auto sub1 = opr::Subtensor::make(
            conv2,
            {AIdx::make_interval(
                     1, conv2.make_scalar(1), conv2.make_scalar(2),
                     conv2.make_scalar(1)),
             AIdx::make_interval(
                     2, conv2.make_scalar(1), conv2.make_scalar(4),
                     conv2.make_scalar(1))},
            OperatorNodeConfig("sub1"));
    auto sub2 = opr::Subtensor::make(
            conv2,
            {AIdx::make_interval(1, conv2.make_scalar(5), {}, {}),
             AIdx::make_interval(
                     2, conv2.make_scalar(1), conv2.make_scalar(4),
                     conv2.make_scalar(1))},
            OperatorNodeConfig("sub2"));
    auto y_sub = sub0 + sub1 + sub2;

    SymbolVar y_pad;
    unpack_vector(
            gopt::GraphOptimizer{}
                    .add_pass(gopt::PaddingChannelPass::make(
                            cg::GraphCommonOptimizeOptions::LayoutTransform::NCHW44,
                            true))
                    .apply({{y_sub}})
                    .endpoint_vars(),
            y_pad);
    auto conv1_opt = find_opr<opr::ConvBias>(y_pad, "conv1");
    auto conv2_opt = find_opr<opr::Convolution>(y_pad, "conv2");
    auto sub0_opt = find_opr<opr::Subtensor>(y_pad, "sub0");
    auto sub1_opt = find_opr<opr::Subtensor>(y_pad, "sub1");
    auto sub2_opt = find_opr<opr::Subtensor>(y_pad, "sub2");
    //! do not padding input tensor
    ASSERT_EQ(conv1_opt->input(0)->shape()[1], 3);
    //! output tensor padding input tensor
    ASSERT_EQ(conv2_opt->input(1)->shape()[0], 8);
    ASSERT_EQ(conv2_opt->output(0)->shape()[1], 8);

    //! sub0 do not perform on channel dim, so no need to add subtensor
    ASSERT_EQ(sub0_opt->input(0)->shape()[1], 8);
    //! sub1 perform on channel dim, but end is specific, so no need to add subtensor
    ASSERT_EQ(sub1_opt->input(0)->shape()[1], 8);
    //! sub1 perform on channel dim, and end is default, so need to add subtensor
    ASSERT_EQ(sub2_opt->input(0)->shape()[1], 6);

    //! the dst tensor channel must stay unchange
    ASSERT_EQ(y_pad.node()->shape()[1], 6);
    graph->compile({{y_pad, {}}})
            ->to_json()
            ->writeto_fpath(
                    output_file("TestGoptInference.ChannelPaddingSubtensor.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile(
            {make_callback_copy(y_sub, host_y), make_callback_copy(y_pad, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-2);

    //! test change the input shape
    *host_x = *gen({2, 3, 32, 32}, cn);
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-2);
}

TEST(TestGoptInference, ChannelPaddingReduce) {
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn)).rename(name);
    };

    auto host_x = gen({1, 3, 8, 8}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    //! Hybrid nchw44 mode
    opr::ConvBias::Param param_conv;
    param_conv.pad_h = param_conv.pad_w = 1;
    auto w1 = mkcvar("w1", {8, 3, 3, 3}), b1 = mkcvar("w1", {1, 8, 1, 1}),
         conv1 = opr::ConvBias::make(
                 x, w1, b1, param_conv, {}, OperatorNodeConfig("conv1"));

    auto w2 = mkcvar("w2", {6, 8, 1, 1}),
         conv2 = opr::Convolution::make(conv1, w2, {}, {}, OperatorNodeConfig("conv2"));
    auto reduce0 = opr::Reduce::make(
            conv2, {opr::Reduce::Mode::MAX, 1}, {}, OperatorNodeConfig("reduce0"));
    auto reduce1 = opr::Reduce::make(
            conv2, {opr::Reduce::Mode::MAX, 2}, {}, OperatorNodeConfig("reduce1"));
    auto y_reduce = reduce0 + reduce1;

    SymbolVar y_pad;
    unpack_vector(
            gopt::GraphOptimizer{}
                    .add_pass(gopt::PaddingChannelPass::make(
                            cg::GraphCommonOptimizeOptions::LayoutTransform::NCHW44,
                            true))
                    .apply({{y_reduce}})
                    .endpoint_vars(),
            y_pad);
    auto conv1_opt = find_opr<opr::ConvBias>(y_pad, "conv1");
    auto conv2_opt = find_opr<opr::Convolution>(y_pad, "conv2");
    auto reduce0_opt = find_opr<opr::Reduce>(y_pad, "reduce0");
    auto reduce1_opt = find_opr<opr::Reduce>(y_pad, "reduce1");
    //! do not padding input tensor
    ASSERT_EQ(conv1_opt->input(0)->shape()[1], 3);
    //! output tensor padding input tensor
    ASSERT_EQ(conv2_opt->input(1)->shape()[0], 8);
    ASSERT_EQ(conv2_opt->output(0)->shape()[1], 8);

    //! reduce0 perform on channel dim, so need to add subtensor
    ASSERT_EQ(reduce0_opt->input(0)->shape()[1], 6);
    //! reduce1 don't perform on channel dim, so no need to add subtensor
    ASSERT_EQ(reduce1_opt->input(0)->shape()[1], 8);

    //! the dst tensor channel must stay unchange
    ASSERT_EQ(y_pad.node()->shape()[1], 6);
    graph->compile({{y_pad, {}}})
            ->to_json()
            ->writeto_fpath(output_file("TestGoptInference.ChannelPaddingReduce.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile(
            {make_callback_copy(y_reduce, host_y),
             make_callback_copy(y_pad, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-2);

    //! test change the input shape
    *host_x = *gen({2, 3, 32, 32}, cn);
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-2);
}

TEST(TestGoptInference, ChannelPaddingMisc) {
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn)).rename(name);
    };

    auto host_x = gen({1, 3, 8, 8}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    //! Hybrid nchw44 mode
    opr::ConvBias::Param param_conv;
    param_conv.pad_h = param_conv.pad_w = 1;
    auto w1 = mkcvar("w1", {8, 3, 3, 3}), b1 = mkcvar("w1", {1, 8, 1, 1}),
         conv1 = opr::ConvBias::make(
                 x, w1, b1, param_conv, {}, OperatorNodeConfig("conv1"));

    auto w2 = mkcvar("w2", {6, 8, 1, 1}),
         conv2 = opr::Convolution::make(conv1, w2, {}, {}, OperatorNodeConfig("conv2"));
    auto elem0 = conv2 + 1;
    auto concat = opr::Concat::make({elem0, conv2}, 1, OperatorNodeConfig("concat"));

    SymbolVar y_pad;
    unpack_vector(
            gopt::GraphOptimizer{}
                    .add_pass(gopt::PaddingChannelPass::make(
                            cg::GraphCommonOptimizeOptions::LayoutTransform::NCHW44,
                            true))
                    .apply({{concat}})
                    .endpoint_vars(),
            y_pad);
    auto conv1_opt = find_opr<opr::ConvBias>(y_pad, "conv1");
    auto conv2_opt = find_opr<opr::Convolution>(y_pad, "conv2");
    auto elemwise0_opt = find_opr<opr::Elemwise>(y_pad);
    auto concat_opt = find_opr<opr::Concat>(y_pad, "concat");
    //! do not padding input tensor
    ASSERT_EQ(conv1_opt->input(0)->shape()[1], 3);
    //! output tensor padding input tensor
    ASSERT_EQ(conv2_opt->input(1)->shape()[0], 8);
    ASSERT_EQ(conv2_opt->output(0)->shape()[1], 8);

    ASSERT_EQ(elemwise0_opt->output(0)->shape()[1], 8);
    ASSERT_EQ(concat_opt->input(0)->shape()[1], 6);
    ASSERT_EQ(concat_opt->input(1)->shape()[1], 6);

    //! the dst tensor channel must stay unchange
    ASSERT_EQ(y_pad.node()->shape()[1], 12);
    graph->compile({{y_pad, {}}})
            ->to_json()
            ->writeto_fpath(output_file("TestGoptInference.ChannelPaddingMisc.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile(
            {make_callback_copy(concat, host_y),
             make_callback_copy(y_pad, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-2);

    //! test change the input shape
    *host_x = *gen({2, 3, 32, 32}, cn);
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-2);
}

TEST(TestGoptInference, ChannelPaddingMoreOp) {
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn)).rename(name);
    };

    auto host_x = gen({2, 3, 8, 8}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    opr::Convolution::Param param;
    param.pad_h = param.pad_w = 1;
    auto w1 = mkcvar("w1", {6, 3, 3, 3}), conv = opr::Convolution::make(x, w1, param);
    auto shape_of = opr::GetVarShape::make(conv);
    auto subtensor = opr::Subtensor::make(
            shape_of, {opr::Subtensor::AxisIndexer::make_interval(
                              0, x.make_scalar(2), None, x.make_scalar(1))});

    opr::Resize::Param param_resize;
    param_resize.format = opr::Resize::Param::Format::NCHW;
    auto resize = opr::ResizeForward::make(conv, subtensor * 2, param_resize);
    auto mat = mkcvar("mat", {2, 3, 3}),
         warp = opr::WarpPerspectiveForward::make(
                 resize, mat, nullptr, cg::var_from_tensor_shape(x, {4, 4}));

    auto b = mkvar("b", {1, 6, 1, 1}),
         elem = opr::Elemwise::make({warp + b}, opr::Elemwise::Param::Mode::RELU);
    param.pad_h = param.pad_w = 1;
    auto w2 = mkcvar("w2", {7, 6, 3, 3}), y = opr::Convolution::make(elem, w2, param),
         z = opr::AxisAddRemove::make(y, {opr::AxisAddRemove::AxisDesc::make_add(0)});

    SymbolVar y_pad, z_pad;
    unpack_vector(
            gopt::GraphOptimizer{}
                    .add_pass(gopt::PaddingChannelPass::make(
                            cg::GraphCommonOptimizeOptions::LayoutTransform::NCHW44,
                            true))
                    .apply({{y}})
                    .endpoint_vars(),
            y_pad);
    unpack_vector(
            gopt::GraphOptimizer{}
                    .add_pass(gopt::PaddingChannelPass::make(
                            cg::GraphCommonOptimizeOptions::LayoutTransform::NCHW44,
                            true))
                    .apply({{z}})
                    .endpoint_vars(),
            z_pad);

    graph->compile({{y_pad, {}}})
            ->to_json()
            ->writeto_fpath(output_file("TestGoptInference.ChannelPaddingMoreOp.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile(
            {make_callback_copy(y, host_y), make_callback_copy(y_pad, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);

    *host_x = *gen({2, 3, 16, 16}, cn);
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}

TEST(TestGoptInference, DynamicShape) {
    HostTensorGenerator<> gen;
    HostTensorGenerator<dtype::Int32> gen_int;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn)).rename(name);
    };

    auto host_x = gen({1, 4, 8, 8}, cn);
    auto host_dy = gen_int({1}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    auto dy = opr::Host2DeviceCopy::make(*graph, host_dy);
    int32_t* start = host_dy->ptr<int32_t>();
    start[0] = 0;
    dy = opr::MarkDynamicVar::make(dy);
    using AIdx = opr::indexing::AxisIndexer;
    auto sub = opr::Subtensor::make(
            x, {AIdx::make_interval(1, dy, x.make_scalar(4), x.make_scalar(1))},
            OperatorNodeConfig("sub"));

    //! Hybrid nchw44 mode
    opr::ConvBias::Param param_conv;
    param_conv.pad_h = param_conv.pad_w = 1;
    auto w1 = mkcvar("w1", {3, 4, 3, 3}), b1 = mkcvar("w1", {1, 3, 1, 1}),
         conv1 = opr::ConvBias::make(
                 sub, w1, b1, param_conv, {}, OperatorNodeConfig("conv1"));

    auto w2 = mkcvar("w2", {4, 3, 1, 1}),
         y = opr::Convolution::make(conv1, w2, {}, {}, OperatorNodeConfig("conv2"));

    SymbolVar y_pad;
    unpack_vector(
            gopt::GraphOptimizer{}
                    .add_pass(gopt::PaddingChannelPass::make(
                            cg::GraphCommonOptimizeOptions::LayoutTransform::NCHW44,
                            true))
                    .apply({{y}})
                    .endpoint_vars(),
            y_pad);
    auto conv1_opt = find_opr<opr::ConvBias>(y_pad, "conv1");
    auto conv2_opt = find_opr<opr::Convolution>(y_pad, "conv2");
    //! do not padding input tensor
    ASSERT_EQ(conv1_opt->input(0)->shape().ndim, 0);
    //! output tensor padding input tensor
    ASSERT_EQ(conv2_opt->input(0)->shape().ndim, 0);

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile(
            {make_callback_copy(y, host_y), make_callback_copy(y_pad, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-2);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
