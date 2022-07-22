#include "megbrain/opr/dnn/local.h"
#include "megbrain/test/helper.h"

#include "megbrain/gopt/basic_arith.h"
#include "megbrain/gopt/gtrans.h"
#include "megbrain/gopt/inference.h"

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/adaptive_pooling.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/opr/tensor_gen.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"

#include "./helper.h"
#include "megbrain/comp_node_env.h"

#include "megdnn/tensor_format.h"

#include <random>
#include <vector>

using namespace mgb;

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
T& find_opr(SymbolVar endpoint, const std::string& node_name) {
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

TEST(TestGoptOldModel, FoldingGlobalPooling) {
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn)).rename(name);
    };

    auto host_x = gen({2, 3, 16, 16}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    opr::Convolution::Param param_conv;
    param_conv.stride_h = param_conv.stride_w = 1;
    param_conv.pad_h = param_conv.pad_w = 1;
    auto w1 = mkcvar("w1", {8, 3, 3, 3});
    auto conv1 =
            opr::Convolution::make(x, w1, param_conv, {}, OperatorNodeConfig("conv1"));
    auto conv_n = opr::GetVarShape::make(conv1, 0);
    auto conv_c = opr::GetVarShape::make(conv1, 1);
    auto conv_h = opr::GetVarShape::make(conv1, 2);
    auto conv_w = opr::GetVarShape::make(conv1, 3);
    auto hxw = conv_h * conv_w;
    auto reshape_shape = opr::Concat::make({conv_n, conv_c, hxw}, 0);

    auto reshape1 = opr::Reshape::make(conv1, reshape_shape);

    opr::Reduce::Param param_reduce;
    param_reduce.axis = 2;
    param_reduce.mode = opr::Reduce::Mode::SUM;
    auto reduce = opr::Reduce::make(reshape1, param_reduce);
    auto reduce_remove_axis = opr::AxisAddRemove::make(
            reduce, {opr::AxisAddRemove::AxisDesc::make_remove(2)});
    auto hw_count = opr::GetVarShape::make(reshape1, 2);

    auto fp32_hw_count = opr::TypeCvt::make(hw_count, dtype::Float32());
    auto true_div = reduce_remove_axis / fp32_hw_count;
    auto y = opr::AxisAddRemove::make(
            true_div, {opr::AxisAddRemove::AxisDesc::make_add(2),
                       opr::AxisAddRemove::AxisDesc::make_add(3)});

    SymbolVar y_opt = y;
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.fuse_grain = true;
        unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    }
    ASSERT_EQ(
            opr::AdaptivePooling::Param::Mode::AVERAGE,
            find_opr<opr::AdaptivePooling>(y_opt).param().mode);

    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(output_file("TestGoptOldModel.FoldingGlobalPooling.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile(
            {make_callback_copy(y, host_y), make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}

TEST(TestGoptOldModel, FoldingGlobalPooling2) {
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn)).rename(name);
    };

    auto host_x = gen({2, 3, 16, 16}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    opr::Convolution::Param param_conv;
    param_conv.stride_h = param_conv.stride_w = 1;
    param_conv.pad_h = param_conv.pad_w = 1;
    auto w1 = mkcvar("w1", {8, 3, 3, 3});
    auto conv1 =
            opr::Convolution::make(x, w1, param_conv, {}, OperatorNodeConfig("conv1"));
    auto conv_n = opr::GetVarShape::make(conv1, 0);
    auto conv_c = opr::GetVarShape::make(conv1, 1);
    auto conv_h = opr::GetVarShape::make(conv1, 2);
    auto conv_w = opr::GetVarShape::make(conv1, 3);
    auto hxw = conv_h * conv_w;
    auto reshape_shape = opr::Concat::make({conv_n, conv_c, hxw}, 0);

    auto reshape1 = opr::Reshape::make(conv1, reshape_shape);

    opr::Reduce::Param param_reduce;
    param_reduce.axis = 2;
    param_reduce.mode = opr::Reduce::Mode::SUM;
    auto reduce = opr::Reduce::make(reshape1, param_reduce);
    auto reduce_remove_axis = opr::AxisAddRemove::make(
            reduce, {opr::AxisAddRemove::AxisDesc::make_remove(2)});
    auto hw_count = opr::GetVarShape::make(reshape1, 2);

    auto fp32_hw_count = opr::TypeCvt::make(hw_count, dtype::Float32());
    auto true_div = reduce_remove_axis / fp32_hw_count;
    auto y = opr::Dimshuffle::make(true_div, {0, 1, -1, -1});

    SymbolVar y_opt = y;
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.fuse_grain = true;
        unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    }
    ASSERT_EQ(
            opr::AdaptivePooling::Param::Mode::AVERAGE,
            find_opr<opr::AdaptivePooling>(y_opt).param().mode);

    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(output_file("TestGoptOldModel.FoldingGlobalPooling2.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile(
            {make_callback_copy(y, host_y), make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}

TEST(TestGoptOldModel, FoldingReduceMean) {
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn)).rename(name);
    };

    auto host_x = gen({2, 3, 16, 16}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    opr::Convolution::Param param_conv;
    param_conv.stride_h = param_conv.stride_w = 1;
    param_conv.pad_h = param_conv.pad_w = 1;
    auto w1 = mkcvar("w1", {8, 3, 3, 3});
    auto conv1 =
            opr::Convolution::make(x, w1, param_conv, {}, OperatorNodeConfig("conv1"));
    auto conv_n = opr::GetVarShape::make(conv1, 0);
    auto conv_c = opr::GetVarShape::make(conv1, 1);
    auto conv_h = opr::GetVarShape::make(conv1, 2);
    auto conv_w = opr::GetVarShape::make(conv1, 3);
    auto hxw = conv_h * conv_w;
    auto reshape_shape = opr::Concat::make({conv_n, conv_c, hxw}, 0);

    auto reshape1 = opr::Reshape::make(conv1, reshape_shape);

    opr::Reduce::Param param_reduce;
    param_reduce.axis = 2;
    param_reduce.mode = opr::Reduce::Mode::SUM;
    auto reduce = opr::Reduce::make(reshape1, param_reduce);
    auto hw_count = opr::GetVarShape::make(reshape1, 2);

    auto y = reduce / hw_count;

    SymbolVar y_opt = y;
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.fuse_grain = true;
        unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    }
    ASSERT_EQ(
            opr::Reduce::Param::Mode::MEAN, find_opr<opr::Reduce>(y_opt).param().mode);

    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(output_file("TestGoptOldModel.FoldingReduceMean.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile(
            {make_callback_copy(y, host_y), make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);

    *host_x = *gen({2, 3, 16, 16}, cn);
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}
