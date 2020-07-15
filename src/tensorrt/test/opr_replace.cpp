/**
 * \file src/tensorrt/test/opr_replace.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/comp_node_env.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/helper.h"
#include "megbrain/test/megdnn_helper.h"

#if MGB_ENABLE_TENSOR_RT

#include <random>
#include "megbrain/gopt/basic_arith.h"
#include "megbrain/gopt/gtrans.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/tensorrt/opr_replace.h"
#include "megbrain/tensorrt/tensorrt_opr.h"
#include "megbrain/tensorrt/tensorrt_engine_cache.h"
#include "./helper.h"

#define NV_TENSOR_RT_VERSION                                  \
    ((NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + \
     NV_TENSORRT_PATCH)  // major, minor, patch

using namespace mgb;
using namespace opr;
using namespace nvinfer1;
using namespace tensorrt;

TEST(TestTensorRTReplace, Basic) {
    REQUIRE_GPU(1);
    HostTensorGenerator<> gen;
    std::shared_ptr<HostTensorND> host_x, host_w, host_b, host_w_full, host_o;
    std::shared_ptr<ComputingGraph> graph;

    host_x = gen({5, 8, 28, 28});
    host_w = gen({2, 16, 4, 3, 3});
    host_b = gen({1, 32, 1, 1});
    host_w_full = gen({32 * 13 * 13, 10});
    host_o = gen({1, 10});

    graph = ComputingGraph::make();
    using ConvParam = megdnn::Convolution::Param;
    ConvParam conv_param1;
    conv_param1.sparse = ConvParam::Sparse::GROUP;
    auto x = Host2DeviceCopy::make(*graph, host_x),
         w_conv1 = SharedDeviceTensor::make(*graph, *host_w),
         b_conv1 = SharedDeviceTensor::make(*graph, *host_b),
         f_conv1 = opr::Convolution::make(x, w_conv1, conv_param1),
         y_conv1 = f_conv1 + b_conv1;
    using PoolParam = megdnn::Pooling::Param;
    PoolParam pool_param1;
    pool_param1.mode = PoolParam::Mode::MAX;
    pool_param1.window_h = 2;
    pool_param1.window_w = 2;
    auto y_reshape = y_conv1.reshape({5, 32, 26, 26});
    auto y_pooling1 = opr::Pooling::make(y_reshape, pool_param1);
    auto w_full1 = SharedDeviceTensor::make(*graph, *host_w_full);
    auto x_full1 = y_pooling1.reshape({5, 32 * 13 * 13});
    auto y_full1 = opr::MatrixMul::make(x_full1, w_full1);
    auto o = SharedDeviceTensor::make(*graph, *host_o);
    auto out = y_full1 + o;

    SymbolVar out_trt;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::TensorRTReplacePass>()
                          .apply({{out}})
                          .endpoint_vars(),
                  out_trt);
    HostTensorND host_z1;
    HostTensorND host_z2;

    ASSERT_NE(out_trt.node(), out.node());
    auto func = graph->compile({make_callback_copy(out, host_z1),
                                make_callback_copy(out_trt, host_z2)});
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-3);
}

TEST(TensorRTReplacePass, MatrixMul) {
    REQUIRE_GPU(1);
    HostTensorGenerator<> gen;
    auto host_a = gen({5, 8});
    auto host_b = gen({8, 10});
    auto host_c = gen({10, 8});
    auto host_x0 = gen({5, 8});
    auto graph = ComputingGraph::make();
    auto a = Host2DeviceCopy::make(*graph, host_a);
    auto b = SharedDeviceTensor::make(*graph, *host_b);
    auto c = Host2DeviceCopy::make(*graph, host_c);
    auto x0 = Host2DeviceCopy::make(*graph, host_x0);
    auto t = opr::MatrixMul::make(a, b);
    auto y = opr::MatrixMul::make(t, c) + x0;
    SymbolVar y_trt;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::TensorRTReplacePass>()
                          .apply({{y}})
                          .endpoint_vars(),
                  y_trt);
    HostTensorND host_z1;
    HostTensorND host_z2;

    ASSERT_NE(y_trt.node(), y.node());
    auto func = graph->compile({make_callback_copy(y, host_z1),
                                make_callback_copy(y_trt, host_z2)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-3);
}

TEST(TensorRTReplacePass, Elemwise) {
    REQUIRE_GPU(1);
    HostTensorGenerator<> gen;
    auto host_a = gen({5, 8, 28});
    auto host_b = gen({5, 1, 28});
    auto host_c = gen({5, 8, 28});
    auto graph = ComputingGraph::make();
    auto a = Host2DeviceCopy::make(*graph, host_a);
    auto b = Host2DeviceCopy::make(*graph, host_b);
    auto c = Host2DeviceCopy::make(*graph, host_c);
    auto y = a + b + c;
    SymbolVar y_trt;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::TensorRTReplacePass>()
                          .apply({{y}})
                          .endpoint_vars(),
                  y_trt);
    HostTensorND host_z1;
    HostTensorND host_z2;

    ASSERT_NE(y_trt.node(), y.node());
    auto func = graph->compile({make_callback_copy(y, host_z1),
                                make_callback_copy(y_trt, host_z2)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-3);
}

TEST(TestTensorRTReplace, ConcatBasic) {
    REQUIRE_GPU(1);
    HostTensorGenerator<> gen;
    std::shared_ptr<HostTensorND> host_x0, host_x1, host_x2, host_w, host_b,
            host_w_full;
    std::shared_ptr<ComputingGraph> graph;

    host_x0 = gen({5, 3, 20, 28});
    host_x1 = gen({5, 4, 20, 28});
    host_x2 = gen({5, 5, 20, 28});

    host_w = gen({32, 12, 3, 3});
    host_b = gen({1, 12, 1, 1});

    graph = ComputingGraph::make();
    auto x0 = Host2DeviceCopy::make(*graph, host_x0),
         x1 = Host2DeviceCopy::make(*graph, host_x1),
         x2 = Host2DeviceCopy::make(*graph, host_x2),
         x = opr::Concat::make({x0, x1, x2}, 1),
         b = SharedDeviceTensor::make(*graph, *host_b),
         y = x + b;

    SymbolVar y_trt;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::TensorRTReplacePass>()
                          .apply({{y}})
                          .endpoint_vars(),
                  y_trt);
    HostTensorND host_z1;
    HostTensorND host_z2;
    auto func = graph->compile({make_callback_copy(y, host_z1),
                                make_callback_copy(y_trt, host_z2)});
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-4);
}

TEST(TestTensorRTReplace, ElemAddFusion) {
    REQUIRE_GPU(1);
    HostTensorGenerator<> gen;
    std::shared_ptr<HostTensorND> host_x0, host_x1, host_w, host_b, host_w_full;
    std::shared_ptr<ComputingGraph> graph;

    host_x0 = gen({5, 23, 28, 28});
    host_x1 = gen({5, 23, 28, 28});

    host_w = gen({32, 23, 3, 3});
    host_b = gen({1, 32, 1, 1});

    graph = ComputingGraph::make();
    auto x0 = Host2DeviceCopy::make(*graph, host_x0),
         x1 = Host2DeviceCopy::make(*graph, host_x1),
         w = SharedDeviceTensor::make(*graph, *host_w),
         y1 = opr::Convolution::make(x0, w), y2 = opr::Convolution::make(x1, w),

         b = SharedDeviceTensor::make(*graph, *host_b), y3 = y1 + y2,
         y = y3 + b;

    SymbolVar y_trt;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::TensorRTReplacePass>()
                          .apply({{y}})
                          .endpoint_vars(),
                  y_trt);
    HostTensorND host_z1;
    HostTensorND host_z2;
    auto func = graph->compile({make_callback_copy(y, host_z1),
                                make_callback_copy(y_trt, host_z2)});
    func->execute();

    cg::OperatorNodeBase* trt_opr = y_trt.node()->owner_opr();
    ASSERT_EQ(3u, trt_opr->cast_final_safe<opr::TensorRTOpr>()
                          .trt_manager()
                          .iobuf_size());
    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-4);
}

TEST(TestTensorRTReplace, BatchedMatrixMulBasic) {
    REQUIRE_GPU(1);
    HostTensorGenerator<> gen;
    std::shared_ptr<HostTensorND> host_x0, host_x1, host_w, host_b, host_w_full;
    bool transA, transB;
    std::shared_ptr<ComputingGraph> graph;

    host_x0 = gen({3, 14, 28});
    host_x1 = gen({3, 28, 35});
    transA = false;
    transB = false;

    graph = ComputingGraph::make();
    auto param = opr::BatchedMatrixMul::Param{transA, transB};
    auto x0 = Host2DeviceCopy::make(*graph, host_x0),
         x1 = Host2DeviceCopy::make(*graph, host_x1),
         y0 = opr::BatchedMatrixMul::make(x0, x1, param), y = y0;

    SymbolVar y_trt;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::TensorRTReplacePass>()
                          .apply({{y}})
                          .endpoint_vars(),
                  y_trt);
    HostTensorND host_z1;
    HostTensorND host_z2;
    auto func = graph->compile({make_callback_copy(y, host_z1),
                                make_callback_copy(y_trt, host_z2)});
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(host_z1, host_z2, 1e-4);
}

TEST(TestTensorRTReplace, Detection) {
    REQUIRE_GPU(1);
    HostTensorGenerator<> gen;
    std::shared_ptr<ComputingGraph> graph;

    auto&& host_x1 = gen({16, 8, 14, 14});
    auto&& host_x2 = gen({16, 8, 14, 14});
    auto&& host_x3 = gen({16, 8, 14, 14});
    auto&& host_w1 = gen({32, 8, 3, 3});
    auto&& host_w2 = gen({32, 8, 3, 3});
    auto&& host_w3 = gen({32, 8, 3, 3});

    graph = ComputingGraph::make();
    using ConvParam = megdnn::Convolution::Param;
    ConvParam conv_param;
    conv_param.stride_h = conv_param.stride_w = 1;
    conv_param.pad_h = conv_param.pad_w = 1;
    auto x1 = Host2DeviceCopy::make(*graph, host_x1),
         x2 = Host2DeviceCopy::make(*graph, host_x2),
         x3 = Host2DeviceCopy::make(*graph, host_x3);
    auto w11 = SharedDeviceTensor::make(*graph, *host_w1),
         y11 = opr::Convolution::make(x1, w11, conv_param);

    auto w12 = SharedDeviceTensor::make(*graph, *host_w2),
         y12 = opr::Convolution::make(x1, w12, conv_param);

    auto w13 = SharedDeviceTensor::make(*graph, *host_w3),
         y13 = opr::Convolution::make(x2, w13, conv_param);

    auto w21 = SharedDeviceTensor::make(*graph, *host_w1),
         y21 = opr::Convolution::make(x2, w21, conv_param);

    auto w22 = SharedDeviceTensor::make(*graph, *host_w2),
         y22 = opr::Convolution::make(x2, w22, conv_param);

    auto w23 = SharedDeviceTensor::make(*graph, *host_w3),
         y23 = opr::Convolution::make(x2, w23, conv_param);

    auto w31 = SharedDeviceTensor::make(*graph, *host_w1),
         y31 = opr::Convolution::make(x3, w31, conv_param);

    auto w32 = SharedDeviceTensor::make(*graph, *host_w2),
         y32 = opr::Convolution::make(x3, w32, conv_param);

    auto w33 = SharedDeviceTensor::make(*graph, *host_w3),
         y33 = opr::Convolution::make(x3, w33, conv_param);

    SymbolVar sym_y11, sym_y12, sym_y13, sym_y21, sym_y22, sym_y23, sym_y31,
            sym_y32, sym_y33;
    unpack_vector(
            gopt::GraphOptimizer{}
                    .add_pass<gopt::TensorRTReplacePass>()
                    .apply({{y11, y12, y13, y21, y22, y23, y31, y32, y33}})
                    .endpoint_vars(),
            sym_y11, sym_y12, sym_y13, sym_y21, sym_y22, sym_y23, sym_y31,
            sym_y32, sym_y33);

    HostTensorND host_y11, host_y12, host_y13, host_y21, host_y22, host_y23,
            host_y31, host_y32, host_y33;

    graph->options().graph_opt.tensorrt = false;
    auto func = graph->compile({make_callback_copy(sym_y11, host_y11),
                                make_callback_copy(sym_y21, host_y21),
                                make_callback_copy(sym_y31, host_y31),
                                make_callback_copy(sym_y12, host_y12),
                                make_callback_copy(sym_y22, host_y22),
                                make_callback_copy(sym_y32, host_y32),
                                make_callback_copy(sym_y13, host_y13),
                                make_callback_copy(sym_y23, host_y23),
                                make_callback_copy(sym_y33, host_y33)});
    func->execute();
}

TEST(TestTensorRTReplace, AllOpr) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    std::vector<std::pair<const char*, thin_function<void()>>> tasks;

    static auto itrans_none = [](SymbolVar* data, size_t size) {};
    static auto itrans_pos = [](SymbolVar* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            data[i] = opr::abs(data[i]) + float(0.1f + 0.23f * i);
        }
    };
    static auto itrans_clip1 = [](SymbolVar* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            data[i] = opr::max(opr::min(data[i], data[i].make_scalar_dt(0.9f)),
                               data[i].make_scalar_dt(-0.9f));
        }
    };
    static auto itrans_gt0 = [](SymbolVar* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            data[i] = opr::max(data[i], data[i].make_scalar_dt(0.1f));
        }
    };
    static auto itrans_ne0 = [](SymbolVar* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            auto mask = opr::abs(data[i]) < 0.1f;
            data[i] = data[i] * (1.f - mask) + mask * (data[i] + 1.f);
        }
    };
    MGB_MARK_USED_VAR(itrans_ne0);
    MGB_MARK_USED_VAR(itrans_clip1);

#define DO_CHK_ELEM(_mode, _arity, _itrans, _shps...)                         \
    tasks.emplace_back(#_mode, [cn]() {                                       \
        TrtReplaceChecker chk{_arity,                                         \
                              [](SymbolVarArray inps) -> SymbolVar {          \
                                  itrans_##_itrans(inps.data(), inps.size()); \
                                  return opr::Elemwise::make(                 \
                                          inps, opr::Elemwise::Mode::_mode);  \
                              },                                              \
                              cn};                                            \
        for (int i = 0; i < _arity; ++i) {                                    \
            chk.set_dtype(i, dtype::Float32());                               \
        }                                                                     \
        chk.run({_shps});                                                     \
    })
#define CHECK_ELEM1(_mode, _itrans) \
    DO_CHK_ELEM(_mode, 1, _itrans, TensorShape{9, 12, 7})
#define CHECK_ELEM2(_mode, _itrans)                       \
    DO_CHK_ELEM(_mode, 2, _itrans, TensorShape{9, 12, 7}, \
                TensorShape{9, 1, 7});                    \
    DO_CHK_ELEM(_mode, 2, _itrans, TensorShape{9, 12, 7}, \
                TensorShape{9, 12, 7});                   \
    DO_CHK_ELEM(_mode, 2, _itrans, TensorShape{9, 12, 7}, TensorShape{9, 1, 1});
    CHECK_ELEM1(RELU, none);
    CHECK_ELEM1(TANH, none);
    CHECK_ELEM1(EXP, none);
    CHECK_ELEM1(LOG, gt0);
    CHECK_ELEM1(ABS, none);
#if NV_TENSOR_RT_VERSION >= 5105
    CHECK_ELEM1(SIN, none);
    CHECK_ELEM1(COS, none);
    CHECK_ELEM1(ASIN, clip1);
    CHECK_ELEM1(ACOS, clip1);
    CHECK_ELEM1(CEIL, none);
    CHECK_ELEM1(FLOOR, none);
#endif
    CHECK_ELEM1(SIGMOID, none);

    CHECK_ELEM2(MUL, none);
    CHECK_ELEM2(ADD, none);
    CHECK_ELEM2(MIN, none);
    CHECK_ELEM2(MAX, none);
    CHECK_ELEM2(SUB, none);
    CHECK_ELEM2(TRUE_DIV, none);
    CHECK_ELEM2(POW, pos);

    CHECK_ELEM2(FUSE_ADD_RELU, none);
    CHECK_ELEM2(FUSE_ADD_SIGMOID, none);
    CHECK_ELEM2(FUSE_ADD_TANH, none);

#undef CHECK_ELEM1
#undef CHECK_ELEM2
#undef DO_CHK_ELEM
    auto conv_test = [&]() {
        tasks.emplace_back("dense_conv", [cn]() {
            TrtReplaceChecker checker{
                    2,
                    [](const SymbolVarArray& inp) -> SymbolVar {
                        using Param = opr::Convolution::Param;
                        Param param;
                        param.pad_h = param.pad_w = 1;
                        param.stride_h = param.stride_w = 2;

                        return opr::Convolution::make(inp[0], inp[1], param);
                    },
                    cn};
            checker.set_const_var(1);
            checker.run({TensorShape{16, 3, 28, 28}, TensorShape{16, 3, 3, 3}});

        });
    };
    auto grouped_conv_test = [&]() {
        tasks.emplace_back("grouped_conv", [cn]() {
            TrtReplaceChecker checker{
                    2,
                    [](const SymbolVarArray& inp) -> SymbolVar {
                        using Param = opr::Convolution::Param;
                        Param param;
                        param.pad_h = param.pad_w = 1;
                        param.stride_h = param.stride_w = 2;
                        param.sparse = Param::Sparse::GROUP;
                        return opr::Convolution::make(inp[0], inp[1], param);
                    },
                    cn};
            checker.set_const_var(1);
            checker.run(
                    {TensorShape{16, 8, 28, 28}, TensorShape{4, 4, 2, 3, 3}});

        });
    };
    conv_test();
    grouped_conv_test();
    auto dilated_conv_test = [&]() {
        tasks.emplace_back("dilated_conv", [cn]() {
            TrtReplaceChecker checker{
                    2,
                    [](const SymbolVarArray& inp) -> SymbolVar {
                        using Param = opr::Convolution::Param;
                        Param param;
                        param.pad_h = param.pad_w = 1;
                        param.stride_h = param.stride_w = 2;
                        param.dilate_h = param.dilate_w = 2;
                        return opr::Convolution::make(inp[0], inp[1], param);
                    },
                    cn};
            checker.set_const_var(1);
            checker.run({TensorShape{16, 3, 28, 28}, TensorShape{16, 3, 3, 3}});
        });
    };
    dilated_conv_test();
    using PoolingMode = opr::Pooling::Param::Mode;
    auto pooling_test = [&](const char* name, PoolingMode mode) {
        tasks.emplace_back(name, [cn, mode]() {
            TrtReplaceChecker checker{
                    1,
                    [mode](const SymbolVarArray& inp) -> SymbolVar {
                        using Param = opr::Pooling::Param;
                        Param param;
                        param.pad_h = param.pad_w = 1;
                        param.stride_h = param.stride_w = 2;
                        param.window_h = param.window_w = 2;
                        param.mode = mode;
                        return opr::Pooling::make(inp[0], param);
                    },
                    cn};
            checker.run({TensorShape{16, 3, 28, 28}});
        });
    };
    pooling_test("pooling_avg", PoolingMode::AVERAGE);
    pooling_test("pooling_max", PoolingMode::MAX);
    pooling_test("pooling_avg_count_exclude_padding",
                 PoolingMode::AVERAGE_COUNT_EXCLUDE_PADDING);

    auto deconv_test = [&](const char* name) {
        tasks.emplace_back("deconv", [cn]() {
            TrtReplaceChecker checker{
                    2,
                    [](const SymbolVarArray& inp) -> SymbolVar {
                        using Param = opr::ConvolutionBackwardData::Param;
                        Param param;
                        param.pad_h = param.pad_w = 1;
                        param.stride_h = param.stride_w = 2;
                        return opr::ConvolutionBackwardData::make_deconv(
                                inp[0], inp[1], param);
                    },
                    cn};
            checker.set_const_var(1);
            checker.run(
                    {TensorShape{16, 16, 14, 14}, TensorShape{16, 3, 3, 3}});

        });
    };
    deconv_test("deconv");

    auto matmul_test = [&](const char* name, bool transA, bool transB) {
        tasks.emplace_back(name, [cn, transA, transB]() {
            TrtReplaceChecker checker{
                    2,
                    [transA, transB](const SymbolVarArray& inps) -> SymbolVar {
                        using Param = opr::MatrixMul::Param;
                        Param param{transA, transB};
                        SymbolVar mat_a = inps[0], mat_b = inps[1];
                        if (transA) {
                            mat_a = opr::Dimshuffle::make(inps[0], {1, 0});
                        }
                        if (transB) {
                            mat_b = opr::Dimshuffle::make(inps[1], {1, 0});
                        }
                        return opr::MatrixMul::make(mat_a, mat_b, param);
                    },
                    cn};
            checker.run({TensorShape{12, 24}, TensorShape{24, 35}});
        });
    };
    matmul_test("matmul_nn", false, false);
    matmul_test("matmul_nt", false, true);
    matmul_test("matmul_tn", true, false);
    matmul_test("matmul_tt", true, true);

    auto batched_matmul = [&]() {
        tasks.emplace_back("batched_matmul", [cn]() {
            TrtReplaceChecker checker{
                    2,
                    [](const SymbolVarArray& inps) -> SymbolVar {
                        using Param = opr::MatrixMul::Param;
                        Param param{false, false};
                        return opr::BatchedMatrixMul::make(inps[0], inps[1],
                                                           param);
                    },
                    cn};
            checker.run({TensorShape{3, 12, 24}, TensorShape{3, 24, 35}});
        });

    };
    batched_matmul();

    auto concat = [&]() {
        tasks.emplace_back("concat", [cn]() {
            TrtReplaceChecker checker{
                    3,
                    [](const SymbolVarArray& inps) -> SymbolVar {
                        return opr::Concat::make(inps, 1);
                    },
                    cn};
            checker.run({TensorShape{5, 3, 20, 28}, TensorShape{5, 4, 20, 28},
                         TensorShape{5, 5, 20, 28}});
        });
    };
    concat();

    auto conv_bias_test = [&]() {
        tasks.emplace_back("dense_conv_bias", [cn]() {
            TrtReplaceChecker checker{
                    4,
                    [](const SymbolVarArray& inp) -> SymbolVar {
                        using Param = opr::ConvBias::Param;
                        Param param;
                        param.pad_h = param.pad_w = 1;
                        param.stride_h = param.stride_w = 2;
                        param.nonlineMode =
                                opr::ConvBias::Param::NonlineMode::RELU;
                        return opr::ConvBias::make(inp[0], inp[1], inp[2],
                                                   inp[3], param);
                    },
                    cn};
            checker.set_const_var(1);
            checker.set_const_var(2);
            checker.run({TensorShape{16, 4, 28, 28}, TensorShape{16, 4, 3, 3},
                         TensorShape{1, 16, 1, 1},
                         TensorShape{16, 16, 14, 14}});
        });
        tasks.emplace_back("grouped_conv_bias", [cn]() {
            TrtReplaceChecker checker{
                    4,
                    [](const SymbolVarArray& inp) -> SymbolVar {
                        using Param = opr::ConvBias::Param;
                        Param param;
                        param.pad_h = param.pad_w = 1;
                        param.stride_h = param.stride_w = 2;
                        param.nonlineMode =
                                opr::ConvBias::Param::NonlineMode::RELU;
                        param.sparse = opr::ConvBias::Param::Sparse::GROUP;
                        return opr::ConvBias::make(inp[0], inp[1], inp[2],
                                                   inp[3], param);
                    },
                    cn};
            checker.set_const_var(1);
            checker.set_const_var(2);
            checker.run({TensorShape{16, 16, 28, 28},
                         TensorShape{4, 4, 4, 3, 3}, TensorShape{1, 16, 1, 1},
                         TensorShape{16, 16, 14, 14}});
        });
        tasks.emplace_back("dilation_conv_bias", [cn]() {
            TrtReplaceChecker checker{
                    3,
                    [](const SymbolVarArray& inp) -> SymbolVar {
                        using Param = opr::ConvBias::Param;
                        Param param;
                        param.pad_h = param.pad_w = 1;
                        param.stride_h = param.stride_w = 2;
                        param.dilate_h = param.dilate_w = 2;
                        param.nonlineMode =
                                opr::ConvBias::Param::NonlineMode::RELU;
                        return opr::ConvBias::make(inp[0], inp[1], inp[2],
                                                   param);
                    },
                    cn};
            checker.set_const_var(1);
            checker.set_const_var(2);
            checker.run({TensorShape{16, 4, 28, 28}, TensorShape{16, 4, 3, 3},
                         TensorShape{1, 16, 1, 1}});
        });
    };
    conv_bias_test();

    for (auto&& task : tasks) {
        task.second();
    }
}

TEST(TestTensorRTReplace, PowC) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    TrtReplaceChecker checker{1,
                              [](const SymbolVarArray& inp) -> SymbolVar {
                                  using Param = opr::PowC::Param;
                                  Param param;
                                  param.exp = 2.0;
                                  return opr::PowC::make(inp[0], param);
                              },
                              cn};
    checker.run({TensorShape{32, 3, 28, 28}});
}

TEST(TestTensorRTReplace, AllOprQuantized) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 61) {
        printf("This testcase ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 61);
        return;
    }

    std::vector<std::pair<const char*, thin_function<void()>>> tasks;
    //! Changing the random number generator will cause accuracy problem.
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> rng{
            1.2f, 127 * 1.2f};

    static auto itrans_gt_val = [](SymbolVar* data, size_t size, float val) {
        for (size_t i = 0; i < size; ++i) {
            data[i] = opr::max(data[i], data[i].make_scalar_dt(val));
        }
    };

#define DO_CHK_ELEM(_mode, _arity, _src_dtype, _dst_dtype, _shps...)   \
    tasks.emplace_back(#_mode, [cn, &rng]() {                          \
        TrtReplaceChecker chk{                                         \
                _arity,                                                \
                [](SymbolVarArray inps) -> SymbolVar {                 \
                    auto elem = opr::ElemwiseMultiType::make(          \
                            inps, opr::ElemwiseMultiType::Mode::_mode, \
                            OperatorNodeConfig{_dst_dtype});           \
                    return opr::TypeCvt::make(elem, dtype::Float32()); \
                },                                                     \
                cn};                                                   \
        for (int i = 0; i < _arity; ++i) {                             \
            chk.set_dtype(i, _src_dtype);                              \
        }                                                              \
        for (int i = 0; i < _arity; ++i) {                             \
            chk.set_rng_gen(i, &rng);                                  \
        }                                                              \
        chk.run({_shps});                                              \
    })
#define CHECK_ELEM(_mode)                                                     \
    DO_CHK_ELEM(_mode, 2, dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f}, \
                TensorShape{9, 12, 5, 7}, TensorShape{9, 1, 5, 7});           \
    DO_CHK_ELEM(_mode, 2, dtype::QuantizedS8{1.2f}, dtype::QuantizedS8{1.3f}, \
                TensorShape{9, 12, 5, 7}, TensorShape{1, 12, 1, 1});
    CHECK_ELEM(QADD);
    CHECK_ELEM(QFUSE_ADD_RELU);

    auto conv_test = [&]() {
        tasks.emplace_back("dense_conv", [cn, &rng]() {
            TrtReplaceChecker checker{
                    4,
                    [](const SymbolVarArray& inp) -> SymbolVar {
                        using Param = opr::ConvBias::Param;
                        Param param;
                        param.pad_h = param.pad_w = 1;
                        param.stride_h = param.stride_w = 2;
                        param.format = opr::ConvBias::Param::Format::NCHW4;
                        param.nonlineMode =
                                opr::ConvBias::Param::NonlineMode::RELU;
                        auto y = opr::ConvBias::make(
                                inp[0], inp[1], inp[2], inp[3], param, {},
                                OperatorNodeConfig{dtype::QuantizedS8{1.3f}});
                        return opr::TypeCvt::make(y, dtype::Float32());
                    },
                    cn};
            checker.set_const_var(1);
            checker.set_const_var(2);
            checker.set_dtype(0, dtype::QuantizedS8{1.2f})
                    .set_dtype(1, dtype::QuantizedS8{1.3f})
                    .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
                    .set_dtype(3, dtype::QuantizedS8{1.2f});
            for (int i = 0; i < 4; ++i) {
                checker.set_rng_gen(i, &rng);
            }
            checker.run({TensorShape{16, 1, 28, 28, 4},
                         TensorShape{16, 1, 3, 3, 4},
                         TensorShape{1, 4, 1, 1, 4},
                         TensorShape{16, 4, 14, 14, 4}});
        });
        tasks.emplace_back("grouped_conv", [cn, &rng]() {
            TrtReplaceChecker checker{
                    4,
                    [](const SymbolVarArray& inp) -> SymbolVar {
                        using Param = opr::ConvBias::Param;
                        Param param;
                        param.pad_h = param.pad_w = 1;
                        param.stride_h = param.stride_w = 2;
                        param.format = opr::ConvBias::Param::Format::NCHW4;
                        param.nonlineMode =
                                opr::ConvBias::Param::NonlineMode::RELU;
                        param.sparse = opr::ConvBias::Param::Sparse::GROUP;
                        auto y = opr::ConvBias::make(
                                inp[0], inp[1], inp[2], inp[3], param, {},
                                OperatorNodeConfig{dtype::QuantizedS8{1.3f}});
                        return opr::TypeCvt::make(y, dtype::Float32());
                    },
                    cn};
            checker.set_const_var(1);
            checker.set_const_var(2);
            checker.set_dtype(0, dtype::QuantizedS8{1.2f})
                    .set_dtype(1, dtype::QuantizedS8{1.3f})
                    .set_dtype(2, dtype::QuantizedS32{1.2f * 1.3f})
                    .set_dtype(3, dtype::QuantizedS8{1.2f});
            for (int i = 0; i < 4; ++i) {
                checker.set_rng_gen(i, &rng);
            }
            checker.run({TensorShape{16, 4, 28, 28, 4},
                         TensorShape{4, 4, 1, 3, 3, 4},
                         TensorShape{1, 4, 1, 1, 4},
                         TensorShape{16, 4, 14, 14, 4}});
        });
        // quantized conv bias does not support dilation conv in megdnn
#if 0
        tasks.emplace_back("dilation_conv", [cn, &rng]() {
            TrtReplaceChecker checker{
                    3,
                    [](const SymbolVarArray& inp) -> SymbolVar {
                        using Param = opr::ConvBias::Param;
                        Param param;
                        param.pad_h = param.pad_w = 1;
                        param.stride_h = param.stride_w = 2;
                        param.dilate_h = param.dilate_w = 2;
                        param.format = opr::ConvBias::Param::Format::NCHW4;
                        param.nonlineMode =
                                opr::ConvBias::Param::NonlineMode::RELU;
                        auto y = opr::ConvBias::make(
                                inp[0], inp[1], inp[2], param, {},
                                OperatorNodeConfig{dtype::QuantizedS8{1.3f}});
                        return opr::TypeCvt::make(y, dtype::Float32());
                    },
                    cn};
            checker.set_const_var(1);
            checker.set_const_var(2);
            checker.set_dtype(0, dtype::QuantizedS8{1.2f})
                    .set_dtype(1, dtype::QuantizedS8{1.2f})
                    .set_dtype(2, dtype::QuantizedS32{1.44f});
            for (int i = 0; i < 3; ++i) {
                checker.set_rng_gen(i, &rng);
            }
            checker.run({TensorShape{16, 1, 28, 28, 4},
                         TensorShape{16, 1, 3, 3, 4},
                         TensorShape{1, 4, 1, 1, 4}});
        });
#endif
    };
    conv_test();

    using PoolingMode = opr::Pooling::Param::Mode;
    auto pooling_test = [&](const char* name, PoolingMode mode) {
        tasks.emplace_back(name, [cn, mode, &rng]() {
            TrtReplaceChecker checker{
                    1,
                    [mode](SymbolVarArray inp) -> SymbolVar {
                        itrans_gt_val(inp.data(), inp.size(), 40 * 1.2f);
                        using Param = opr::Pooling::Param;
                        Param param;
                        param.pad_h = param.pad_w = 1;
                        param.stride_h = param.stride_w = 2;
                        param.window_h = param.window_w = 2;
                        param.mode = mode;
                        param.format = opr::Pooling::Param::Format::NCHW4;
                        auto y = opr::Pooling::make(inp[0], param);
                        return opr::TypeCvt::make(y, dtype::Float32());
                    },
                    cn};
            for (int i = 0; i < 1; ++i) {
                checker.set_rng_gen(i, &rng);
            }
            checker.set_dtype(0, dtype::QuantizedS8{1.2f});
            //! pooling in tensorrt has rounding precision issue, so we should
            //! change epsilon from 1e-5 to 1e-1
            checker.set_epsilon(1e-1);
            checker.run({TensorShape{16, 1, 28, 28, 4}});
        });
    };
    pooling_test("pooling_avg", PoolingMode::AVERAGE);
    pooling_test("pooling_max", PoolingMode::MAX);
    pooling_test("pooling_avg_count_exclude_padding",
                 PoolingMode::AVERAGE_COUNT_EXCLUDE_PADDING);

    for (auto&& task : tasks) {
        task.second();
    }
}

TEST(TestTensorRTReplace, FloatInt8MixPrecision) {
    REQUIRE_GPU(1);
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{
            1.2f, 127 * 1.2f};
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 61) {
        printf("This testcase ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 61);
        return;
    }

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp,
                     const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name),
                dtype);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp,
                      const DType& dtype) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                        .rename(name),
                dtype);
    };

    auto x = mkvar("x", {32, 1, 28, 28, 4}, dtype::QuantizedS8(2.5f)),
         w = mkcvar("w", {16, 1, 3, 3, 4}, dtype::QuantizedS8(2.5f)),
         b = mkcvar("b", {1, 4, 1, 1, 4}, dtype::QuantizedS32(6.25f)),
         z = mkvar("z", {32, 4, 28, 28, 4}, dtype::QuantizedS8(2.5f));
    opr::ConvBias::Param conv_param;
    conv_param.format = opr::ConvBias::Param::Format::NCHW4;
    conv_param.stride_h = conv_param.stride_w = 1;
    conv_param.pad_h = conv_param.pad_w = 1;
    auto y = opr::ConvBias::make(x, w, b, z, conv_param, {},
                                 OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    opr::Pooling::Param pool_param;
    pool_param.format = opr::Pooling::Param::Format::NCHW4;
    pool_param.stride_h = pool_param.stride_w = 2;
    pool_param.window_h = pool_param.window_w = 2;
    pool_param.pad_h = pool_param.pad_w = 0;
    pool_param.mode = opr::Pooling::Param::Mode::AVERAGE;
    auto y1 = opr::Pooling::make(y, pool_param);

    auto w1 = mkcvar("w1", {32, 4, 3, 3, 4}, dtype::QuantizedS8{2.5f}),
         b1 = mkcvar("b1", {1, 8, 1, 1, 4}, dtype::QuantizedS32{6.25f});
    conv_param.stride_h = conv_param.stride_w = 2;
    conv_param.pad_h = conv_param.pad_w = 1;
    auto y2 = opr::ConvBias::make(y1, w1, b1, conv_param, {},
                                  OperatorNodeConfig{dtype::QuantizedS8{2.5f}});

    auto w2 = mkcvar("w2", {32, 8, 1, 1, 4}, dtype::QuantizedS8{2.5f}),
         b2 = mkcvar("b2", {1, 8, 1, 1, 4}, dtype::QuantizedS32{6.25f});
    conv_param.stride_h = conv_param.stride_w = 1;
    conv_param.pad_h = conv_param.pad_w = 0;
    auto y3 = opr::ConvBias::make(y2, w2, b2, conv_param, {},
                                  OperatorNodeConfig{dtype::QuantizedS8{2.5f}});

    auto y4 = opr::ElemwiseMultiType::make(
            {y2, y3}, {opr::ElemwiseMultiType::Param::Mode::QFUSE_ADD_RELU},
            OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    auto y5 = opr::TypeCvt::make(y4, dtype::Float32());
    auto y6 = y5.reshape({32, 7 * 7 * 32});
    auto w6 = mkcvar("w6", {7 * 7 * 32, 10}, dtype::Float32());
    auto o = opr::MatrixMul::make(y6, w6);
    o = opr::Elemwise::make({o}, {opr::Elemwise::Mode::RELU});

    auto y7 = opr::TypeCvt::make(y2, dtype::Float32());
    auto f = mkvar("f", {32, 1, 7, 1, 4}, dtype::Float32());
    auto o1 = y7 + f * f + 1.f;
    o1 = opr::Elemwise::make({o1}, {opr::Elemwise::Mode::RELU});

    SymbolVar trt_o, trt_o1;
    SymbolVar mgb_o, mgb_o1;

    ComputingGraph::Options opt;
    opt.graph_opt_level = 0;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ExpandFusedArithPass>()
                          .add_pass<gopt::TensorRTReplacePass>()
                          .add_pass<gopt::ArithFusePass>()
                          .apply({{o, o1}})
                          .endpoint_vars(),
                  trt_o, trt_o1);

    opt.graph_opt_level = 0;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_preset_passes(true, nullptr, &opt)
                          .apply({{o, o1}})
                          .endpoint_vars(),
                  mgb_o, mgb_o1);

    size_t nr_trt_opr = 0;
    cg::DepOprIter iter{[&nr_trt_opr](cg::OperatorNodeBase* opr) {
        if (opr->same_type<TensorRTOpr>()) {
            ++nr_trt_opr;
        }
    }};
    iter.add(trt_o.node());
    iter.add(trt_o1.node());
    mgb_assert(nr_trt_opr == 3);

    ComputingGraph::OutputSpec outspec(4);
    SmallVector<HostTensorND> outputs(4);
    outspec[0] = make_callback_copy(trt_o, outputs[0], false);
    outspec[1] = make_callback_copy(trt_o1, outputs[1], false);
    outspec[2] = make_callback_copy(mgb_o, outputs[2], false);
    outspec[3] = make_callback_copy(mgb_o1, outputs[3], false);
    graph->options().graph_opt.tensorrt = false;
    auto func = graph->compile(outspec);
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(outputs[0], outputs[2], 1e-4);
    MGB_ASSERT_TENSOR_NEAR(outputs[1], outputs[3], 1e-4);
}

TEST(TestTensorRTReplace, Int8Inference) {
    REQUIRE_GPU(1);
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{
            1.2f, 127 * 1.2f};
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 61) {
        printf("This testcase ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 61);
        return;
    }

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp,
                     const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name),
                dtype);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp,
                      const DType& dtype) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                        .rename(name),
                dtype);
    };

    auto x = mkvar("x", {32, 1, 28, 28, 4}, dtype::QuantizedS8(2.5f)),
         w = mkcvar("w", {16, 1, 3, 3, 4}, dtype::QuantizedS8(2.5f)),
         b = mkcvar("b", {1, 4, 1, 1, 4}, dtype::QuantizedS32(6.25f)),
         z = mkvar("z", {32, 4, 28, 28, 4}, dtype::QuantizedS8(2.5f));
    opr::ConvBias::Param conv_param;
    conv_param.format = opr::ConvBias::Param::Format::NCHW4;
    conv_param.stride_h = conv_param.stride_w = 1;
    conv_param.pad_h = conv_param.pad_w = 1;
    auto y = opr::ConvBias::make(x, w, b, z, conv_param, {},
                                 OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    opr::Pooling::Param pool_param;
    pool_param.format = opr::Pooling::Param::Format::NCHW4;
    pool_param.stride_h = pool_param.stride_w = 2;
    pool_param.window_h = pool_param.window_w = 2;
    pool_param.pad_h = pool_param.pad_w = 0;
    pool_param.mode = opr::Pooling::Param::Mode::AVERAGE;
    auto y1 = opr::Pooling::make(y, pool_param);

    auto w1 = mkcvar("w1", {32, 4, 3, 3, 4}, dtype::QuantizedS8{2.5f}),
         b1 = mkcvar("b1", {1, 8, 1, 1, 4}, dtype::QuantizedS32{6.25f});
    conv_param.stride_h = conv_param.stride_w = 2;
    conv_param.pad_h = conv_param.pad_w = 1;
    auto y2 = opr::ConvBias::make(y1, w1, b1, conv_param, {},
                                  OperatorNodeConfig{dtype::QuantizedS8{2.5f}});

    auto w2 = mkcvar("w2", {32, 8, 1, 1, 4}, dtype::QuantizedS8{2.5f}),
         b2 = mkcvar("b2", {1, 8, 1, 1, 4}, dtype::QuantizedS32{6.25f});
    conv_param.stride_h = conv_param.stride_w = 1;
    conv_param.pad_h = conv_param.pad_w = 0;
    auto y3 = opr::ConvBias::make(y2, w2, b2, conv_param, {},
                                  OperatorNodeConfig{dtype::QuantizedS8{2.5f}});

    auto y4 = opr::ElemwiseMultiType::make(
            {y2, y3}, {opr::ElemwiseMultiType::Param::Mode::QFUSE_ADD_RELU},
            OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    auto y5 = opr::TypeCvt::make(y4, dtype::Float32());
    auto y6 = y5.reshape({32, 7 * 7 * 32});
    auto w6 = mkcvar("w6", {7 * 7 * 32, 10}, dtype::Float32());
    auto o = opr::MatrixMul::make(y6, w6);
    o = opr::Elemwise::make({o}, {opr::Elemwise::Mode::RELU});

    auto y7 = mkvar("y7", {32, 8, 7, 7, 4}, dtype::QuantizedS8{2.5f});
    auto f = mkvar("f", {32, 1, 7, 1, 4}, dtype::QuantizedS8{2.4f});
    auto o1 = opr::ElemwiseMultiType::make(
            {y7, f}, {opr::ElemwiseMultiType::Mode::QFUSE_ADD_RELU},
            OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    o1 = opr::TypeCvt::make({o1}, dtype::Float32());

    SymbolVar trt_o, trt_o1;
    SymbolVar mgb_o, mgb_o1;

    ComputingGraph::Options opt;
    opt.graph_opt_level = 0;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ExpandFusedArithPass>()
                          .add_pass<gopt::TensorRTReplacePass>()
                          .add_pass<gopt::ArithFusePass>()
                          .apply({{o, o1}})
                          .endpoint_vars(),
                  trt_o, trt_o1);

    opt.graph_opt_level = 0;
    unpack_vector(gopt::GraphOptimizer{}.apply({{o, o1}}).endpoint_vars(),
                  mgb_o, mgb_o1);

    size_t nr_trt_opr = 0;
    {
        cg::DepOprIter iter{[&nr_trt_opr](cg::OperatorNodeBase* opr) {
            if (opr->same_type<TensorRTOpr>()) {
                ++nr_trt_opr;
            }
        }};
        iter.add(trt_o.node());
        iter.add(trt_o1.node());
        mgb_assert(nr_trt_opr == 2);
    }

#if NV_TENSOR_RT_VERSION < 6001
    size_t nr_dimshuffle = 0;
    {
        cg::DepOprIter iter{[&nr_dimshuffle](cg::OperatorNodeBase* opr) {
            if (opr->same_type<Dimshuffle>()) {
                ++nr_dimshuffle;
            }
        }};
        iter.add(trt_o.node());
        iter.add(trt_o1.node());
        mgb_assert(nr_dimshuffle == 3);
    }
#endif

    ComputingGraph::OutputSpec outspec(4);
    SmallVector<HostTensorND> outputs(4);
    outspec[0] = make_callback_copy(trt_o, outputs[0], false);
    outspec[1] = make_callback_copy(trt_o1, outputs[1], false);
    outspec[2] = make_callback_copy(mgb_o, outputs[2], false);
    outspec[3] = make_callback_copy(mgb_o1, outputs[3], false);
    graph->options().graph_opt.tensorrt = false;
    auto func = graph->compile(outspec);
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(outputs[0], outputs[2], 1e-4);
    MGB_ASSERT_TENSOR_NEAR(outputs[1], outputs[3], 1e-4);
}

// copied from jit test case, to check visit complexity
TEST(TestTensorRTReplace, CheckComplexity) {
    REQUIRE_GPU(1);
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{0.01f,
                                                                         0.02f};
    auto cn = CompNode::load("gpu0");

    auto host_x = gen({2, 2, 2, 2}, cn);
    auto make_dst = [&](ComputingGraph& graph) {
        auto x = opr::Host2DeviceCopy::make(graph, host_x);
        auto y = x;
        for (int i = 0; i < 32; ++i) {
            y = y * y + y;
        }
        return y;
    };
    HostTensorND host_y1, host_y2;

    auto g0 = ComputingGraph::make();
    g0->options().graph_opt_level = 0;
    g0->options().graph_opt.tensorrt = false;
    auto f0 = g0->compile({make_callback_copy(make_dst(*g0), host_y1)});

    auto g1 = ComputingGraph::make();
    g1->options().graph_opt_level = 2;
    g1->options().graph_opt.tensorrt = true;
    auto f1 = g1->compile({make_callback_copy(make_dst(*g1), host_y2)});

    auto find_trt_oprs = [](cg::AsyncExecutable& func) {
        SmallVector<TensorRTOpr*> res;
        auto cb = [&res](cg::OperatorNodeBase* opr) {
            if (opr->same_type<TensorRTOpr>()) {
                auto ptr = &(opr->cast_final_safe<TensorRTOpr>());
                res.push_back(ptr);
            }
            return true;
        };
        func.iter_opr_seq(cb);
        return res;
    };
    EXPECT_FALSE(find_trt_oprs(*f1).empty());
    f1->execute();
    f0->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y1, host_y2, 1e-5);

    ASSERT_EQ(1u, find_trt_oprs(*f1).size());

    auto find_elem_oprs = [](cg::AsyncExecutable& func) {
        SmallVector<opr::Elemwise*> res;
        auto cb = [&res](cg::OperatorNodeBase* opr) {
            if (opr->same_type<opr::Elemwise>()) {
                auto ptr = &(opr->cast_final_safe<opr::Elemwise>());
                res.push_back(ptr);
            }
            return true;
        };
        func.iter_opr_seq(cb);
        return res;
    };
    ASSERT_TRUE(find_elem_oprs(*f1).empty());
}

TEST(TestTensorRTReplace, BroadcastScalar) {
    REQUIRE_GPU(1);
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{
            63 * 1.2f, 127 * 1.2f};
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 61) {
        printf("This testcase ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 61);
        return;
    }

    auto host_scalar1 = gen({1}, cn), host_scalar2 = gen({1}, cn),
         host_x = gen({32, 4, 28, 28, 4}, cn);
    auto make_dst = [&](ComputingGraph& graph) {
        auto mkvar = [&](const char* name,
                         const std::shared_ptr<HostTensorND>& host_ts,
                         const DType& dtype) {
            return opr::TypeCvt::make(
                    opr::Host2DeviceCopy::make(graph, host_ts).rename(name),
                    dtype);
        };
        auto scalar1 = mkvar("scalar1", host_scalar1, dtype::QuantizedS8{2.5f}),
             scalar2 = mkvar("scalar2", host_scalar2, dtype::QuantizedS8{2.6f}),
             x = mkvar("x", host_x, dtype::QuantizedS8{2.6f});
        auto scalar = opr::ElemwiseMultiType::make(
                     {scalar1, scalar2}, {opr::ElemwiseMultiType::Mode::QADD},
                     OperatorNodeConfig{dtype::QuantizedS8{2.5f}}),
             y = opr::ElemwiseMultiType::make(
                     {x, scalar},
                     {opr::ElemwiseMultiType::Mode::QFUSE_ADD_RELU},
                     OperatorNodeConfig{dtype::QuantizedS8{2.7f}});

        y = opr::TypeCvt::make(y, dtype::Float32());
        return y;
    };

    HostTensorND host_y1, host_y2;

    auto g0 = ComputingGraph::make();
    g0->options().graph_opt_level = 0;
    g0->options().graph_opt.tensorrt = false;
    auto f0 = g0->compile({make_callback_copy(make_dst(*g0), host_y1)});

    auto g1 = ComputingGraph::make();
    g1->options().graph_opt_level = 0;
    g1->options().graph_opt.tensorrt = true;
    auto f1 = g1->compile({make_callback_copy(make_dst(*g1), host_y2)});

    auto find_trt_oprs = [](cg::AsyncExecutable& func) {
        SmallVector<TensorRTOpr*> res;
        auto cb = [&res](cg::OperatorNodeBase* opr) {
            if (opr->same_type<TensorRTOpr>()) {
                auto ptr = &(opr->cast_final_safe<TensorRTOpr>());
                res.push_back(ptr);
            }
            return true;
        };
        func.iter_opr_seq(cb);
        return res;
    };
    EXPECT_TRUE(find_trt_oprs(*f1).empty());
    EXPECT_TRUE(find_trt_oprs(*f0).empty());
    f1->execute();
    f0->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y1, host_y2, 1e-1);

    ASSERT_EQ(0u, find_trt_oprs(*f1).size());

    auto find_elem_oprs = [](cg::AsyncExecutable& func) {
        SmallVector<opr::ElemwiseMultiType*> res;
        auto cb = [&res](cg::OperatorNodeBase* opr) {
            if (opr->same_type<opr::ElemwiseMultiType>()) {
                auto ptr = &(opr->cast_final_safe<opr::ElemwiseMultiType>());
                res.push_back(ptr);
            }
            return true;
        };
        func.iter_opr_seq(cb);
        return res;
    };
    ASSERT_EQ(2u, find_elem_oprs(*f1).size());
}

TEST(TestTensorRTReplace, MixedTensorFormat) {
    REQUIRE_GPU(1);
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{
            1.2f, 127 * 1.2f};
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 61) {
        printf("This testcase ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 61);
        return;
    }

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp,
                     const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name),
                dtype);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp,
                      const DType& dtype) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                        .rename(name),
                dtype);
    };

    auto x = mkvar("x", {32, 1, 28, 28, 4}, dtype::QuantizedS8(2.5f)),
         w = mkcvar("w", {16, 1, 3, 3, 4}, dtype::QuantizedS8(2.5f)),
         b = mkcvar("b", {1, 4, 1, 1, 4}, dtype::QuantizedS32(6.25f)),
         z = mkvar("z", {32, 4, 28, 28, 4}, dtype::QuantizedS8(2.5f));
    opr::ConvBias::Param conv_param;
    conv_param.format = opr::ConvBias::Param::Format::NCHW4;
    conv_param.stride_h = conv_param.stride_w = 1;
    conv_param.pad_h = conv_param.pad_w = 1;
    auto y = opr::ConvBias::make(x, w, b, z, conv_param, {},
                                 OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    auto o = opr::TypeCvt::make(y, dtype::Float32());

    auto f = mkvar("f", {32, 1, 28, 28, 4}, dtype::QuantizedS8{2.5f});
    auto o1 = opr::ElemwiseMultiType::make(
            {x, f}, {opr::ElemwiseMultiType::Mode::QFUSE_ADD_RELU},
            OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    auto scalar_1 = mkcvar("scalar_1", {1}, dtype::QuantizedS8{2.5f});
    o1 = opr::ElemwiseMultiType::make(
            {o1, scalar_1}, {opr::ElemwiseMultiType::Mode::QADD},
            OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    o1 = opr::TypeCvt::make(o1, dtype::Float32());

    SymbolVar trt_o, trt_o1;
    SymbolVar mgb_o, mgb_o1;

    ComputingGraph::Options opt;
    opt.graph_opt_level = 0;
    opt.graph_opt.tensorrt = true;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ExpandFusedArithPass>()
                          .add_pass<gopt::TensorRTReplacePass>()
                          .add_pass<gopt::ArithFusePass>()
                          .apply({{o, o1}})
                          .endpoint_vars(),
                  trt_o, trt_o1);

    opt.graph_opt_level = 0;
    opt.graph_opt.tensorrt = false;
    unpack_vector(gopt::GraphOptimizer{}
                          .apply({{o, o1}})
                          .endpoint_vars(),
                  mgb_o, mgb_o1);

    size_t nr_trt_opr = 0;
    cg::DepOprIter iter{[&nr_trt_opr](cg::OperatorNodeBase* opr) {
        if (opr->same_type<TensorRTOpr>()) {
            ++nr_trt_opr;
        }
    }};
    iter.add(trt_o.node());
    iter.add(trt_o1.node());
    mgb_assert(nr_trt_opr == 1);

    ComputingGraph::OutputSpec outspec(4);
    SmallVector<HostTensorND> outputs(4);
    outspec[0] = make_callback_copy(trt_o, outputs[0], false);
    outspec[1] = make_callback_copy(trt_o1, outputs[1], false);
    outspec[2] = make_callback_copy(mgb_o, outputs[2], false);
    outspec[3] = make_callback_copy(mgb_o1, outputs[3], false);
    graph->options().graph_opt.tensorrt = false;
    auto func = graph->compile(outspec);
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(outputs[0], outputs[2], 1e-4);
    MGB_ASSERT_TENSOR_NEAR(outputs[1], outputs[3], 1e-4);
}

TEST(TensorRTReplacePass, WideNetwork) {
    /*  x1--|
     *      +--o0
     *  x0--|
     *      +--o1
     *  x-y-|
     *      +--o2
     *  x2--|
     *      +--o3
     *  x3--|
     *  y is a conv in nchw4 layout
     */

    REQUIRE_GPU(1);
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{
            1.2f, 127 * 1.2f};
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 61) {
        printf("This testcase ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 61);
        return;
    }

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp,
                     const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name),
                dtype);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp,
                      const DType& dtype) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                        .rename(name),
                dtype);
    };

    auto add = [&](SymbolVar a, SymbolVar b) {
        return opr::ElemwiseMultiType::make({a, b},
            {opr::ElemwiseMultiType::Mode::QADD},
            OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    };

    auto x = mkvar("x", {32, 1, 28, 28, 4}, dtype::QuantizedS8(2.5f)),
         w = mkcvar("w", {16, 1, 3, 3, 4}, dtype::QuantizedS8(2.5f)),
         b = mkcvar("b", {1, 4, 1, 1, 4}, dtype::QuantizedS32(6.25f)),
         z = mkvar("z", {32, 4, 28, 28, 4}, dtype::QuantizedS8(2.5f));
    opr::ConvBias::Param conv_param;
    conv_param.format = opr::ConvBias::Param::Format::NCHW4;
    conv_param.stride_h = conv_param.stride_w = 1;
    conv_param.pad_h = conv_param.pad_w = 1;
    auto y = opr::ConvBias::make(x, w, b, z, conv_param, {},
                                 OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    auto x0 = mkvar("x0", {32, 4, 28, 28, 4}, dtype::QuantizedS8{2.5f}),
         x1 = mkvar("x1", {32, 4, 28, 28, 4}, dtype::QuantizedS8{2.5f}),
         x2 = mkvar("x2", {32, 4, 28, 28, 4}, dtype::QuantizedS8{2.5f}),
         x3 = mkvar("x2", {32, 4, 28, 28, 4}, dtype::QuantizedS8{2.5f});
    auto o0 = opr::TypeCvt::make(add(x0, x1), dtype::Float32()),
         o1 = opr::TypeCvt::make(add(y, x0), dtype::Float32()),
         o2 = opr::TypeCvt::make(add(y, x2), dtype::Float32()),
         o3 = opr::TypeCvt::make(add(x2, x3), dtype::Float32());

    ComputingGraph::Options opt;
    opt.graph_opt_level = 0;
    opt.graph_opt.tensorrt = true;
    auto trt_o = gopt::GraphOptimizer{}
            .add_preset_passes(true, nullptr, &opt)
            .apply({{o0, o1, o2, o3}})
            .endpoint_vars();

    opt.graph_opt_level = 0;
    opt.graph_opt.tensorrt = false;
    auto mgb_o = gopt::GraphOptimizer{}
            .add_preset_passes(true, nullptr, &opt)
            .apply({{o0, o1, o2, o3}})
            .endpoint_vars();

    ComputingGraph::OutputSpec outspec(8);
    SmallVector<HostTensorND> outputs(8);
    for (size_t i = 0; i < 4; ++ i) {
        outspec[i] = make_callback_copy(trt_o[i], outputs[i], false);
        outspec[i + 4] = make_callback_copy(mgb_o[i], outputs[i + 4], false);
    }
    auto func = graph->compile(outspec);
    func->execute();

    for (size_t i = 0; i < 4; ++ i) {
        MGB_ASSERT_TENSOR_NEAR(outputs[i], outputs[i + 4], 1e-4);
    }
}

#if NV_TENSOR_RT_VERSION < 6001
TEST(TensorRTReplacePass, ShuffleRemove) {
    REQUIRE_GPU(1);
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{
            1.2f, 127 * 1.2f};
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 61) {
        printf("This testcase ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 61);
        return;
    }

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp,
                     const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name),
                dtype);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp,
                      const DType& dtype) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                        .rename(name),
                dtype);
    };

    auto nchw2nchw4 = [](SymbolVar x) {
        auto xshp = opr::GetVarShape::make(x);

        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp = opr::Concat::make(
                {sub(0), sub(1) / 4, cv(4), sub(2), sub(3)}, 0);
        auto y0 = opr::Reshape::make(x, tshp);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 4, 2});
        return y1;
    };

    auto nchw42nchw = [](SymbolVar x) {
        auto xshp = opr::GetVarShape::make(x);

        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp = opr::Concat::make({sub(0), sub(1) * 4, sub(2), sub(3)}, 0);
        auto y0 = opr::Dimshuffle::make(x, {0, 1, 4, 2, 3});
        auto y1 = opr::Reshape::make(y0, tshp);
        return y1;
    };

    auto x = mkvar("x", {32, 4, 28, 28}, dtype::QuantizedS8(2.5f)),
         w = mkcvar("w", {16, 4, 3, 3}, dtype::QuantizedS8(2.5f)),
         b = mkcvar("b", {1, 16, 1, 1}, dtype::QuantizedS32(6.25f)),
         z1 = mkvar("z", {32, 16, 28, 28}, dtype::QuantizedS8(2.5f));
    x = nchw2nchw4(x), w = nchw2nchw4(w), b = nchw2nchw4(b);
    auto z = nchw2nchw4(z1);
    opr::ConvBias::Param conv_param;
    conv_param.format = opr::ConvBias::Param::Format::NCHW4;
    conv_param.stride_h = conv_param.stride_w = 1;
    conv_param.pad_h = conv_param.pad_w = 1;
    auto y = opr::ConvBias::make(x, w, b, z, conv_param, {},
                                 OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    opr::Pooling::Param pool_param;
    pool_param.format = opr::Pooling::Param::Format::NCHW4;
    pool_param.stride_h = pool_param.stride_w = 2;
    pool_param.window_h = pool_param.window_w = 2;
    pool_param.pad_h = pool_param.pad_w = 0;
    pool_param.mode = opr::Pooling::Param::Mode::AVERAGE;
    auto y1 = opr::Pooling::make(y, pool_param);
    y1 = nchw42nchw(y1);
    y1 = opr::TypeCvt::make(y1, dtype::Float32());

    auto y2 = mkvar("y2", {1, 16, 1, 1}, dtype::QuantizedS8{2.5f});
    auto y3 = opr::ElemwiseMultiType::make(
            {z1, y2}, {opr::ElemwiseMultiType::Mode::QADD},
            OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    y3 = opr::TypeCvt::make(y3, dtype::Float32());

    SymbolVar trt_y1, trt_y3;
    SymbolVar mgb_y1, mgb_y3;

    ComputingGraph::Options opt;
    opt.graph_opt_level = 0;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ExpandFusedArithPass>()
                          .add_pass<gopt::TensorRTReplacePass>()
                          .add_pass<gopt::ArithFusePass>()
                          .add_pass<gopt::ShuffleShuffleRemovePass>()
                          .apply({{y1, y3}})
                          .endpoint_vars(),
                  trt_y1, trt_y3);
    trt_y1 = opr::TypeCvt::make(trt_y1, dtype::QuantizedS8{2.5f}),
    trt_y1 = opr::TypeCvt::make(trt_y1, dtype::Float32());
    trt_y3 = opr::TypeCvt::make(trt_y3, dtype::QuantizedS8{2.5f}),
    trt_y3 = opr::TypeCvt::make(trt_y3, dtype::Float32());

    opt.graph_opt_level = 0;
    unpack_vector(gopt::GraphOptimizer{}
                          .apply({{y1, y3}})
                          .endpoint_vars(),
                  mgb_y1, mgb_y3);

    size_t nr_trt_opr = 0;
    cg::DepOprIter iter{[&nr_trt_opr](cg::OperatorNodeBase* opr) {
        if (opr->same_type<TensorRTOpr>()) {
            ++nr_trt_opr;
        }
    }};
    iter.add(trt_y1.node());
    iter.add(trt_y3.node());
    mgb_assert(nr_trt_opr == 1);

    {
        size_t nr_shuffle_opr = 0;
        cg::DepOprIter iter{[&nr_shuffle_opr](cg::OperatorNodeBase* opr) {
            if (opr->same_type<opr::Dimshuffle>()) {
                ++nr_shuffle_opr;
            }
        }};
        iter.add(trt_y1.node());
        iter.add(trt_y3.node());
        mgb_assert(nr_shuffle_opr == 0);
    }

    ComputingGraph::OutputSpec outspec(4);
    SmallVector<HostTensorND> outputs(4);
    outspec[0] = make_callback_copy(trt_y1, outputs[0], false);
    outspec[1] = make_callback_copy(trt_y3, outputs[1], false);
    outspec[2] = make_callback_copy(mgb_y1, outputs[2], false);
    outspec[3] = make_callback_copy(mgb_y3, outputs[3], false);
    graph->options().graph_opt.tensorrt = false;
    auto func = graph->compile(outspec);
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(outputs[0], outputs[2], 1e-4);
    MGB_ASSERT_TENSOR_NEAR(outputs[1], outputs[3], 1e-4);
}

TEST(TestShuffleShuffleRemove, NCHW2NCHW42NCHW) {
    REQUIRE_GPU(1);
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{
            -127.f * 1.2f, 127 * 1.2f};
    auto cn = CompNode::load("gpu0");

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp,
                     const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name),
                dtype);
    };

    auto nchw2nchw4 = [](SymbolVar x) {
        auto xshp = opr::GetVarShape::make(x);

        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp = opr::Concat::make(
                {sub(0), sub(1) / 4, cv(4), sub(2), sub(3)}, 0);
        auto y0 = opr::Reshape::make(x, tshp);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 4, 2});
        return y1;
    };

    auto nchw42nchw = [](SymbolVar x) {
        auto xshp = opr::GetVarShape::make(x);

        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp = opr::Concat::make({sub(0), sub(1) * 4, sub(2), sub(3)}, 0);
        auto y0 = opr::Dimshuffle::make(x, {0, 1, 4, 2, 3});
        auto y1 = opr::Reshape::make(y0, tshp);
        return y1;
    };

    auto x = mkvar("x", {32, 4, 28, 28}, dtype::QuantizedS8{2.5f});
    x = nchw2nchw4(x), x = nchw42nchw(x);
    x = opr::TypeCvt::make(x, dtype::Float32());
    SymbolVar o, o_remove;

    ComputingGraph::Options opt;
    opt.graph_opt_level = 0;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ShuffleShuffleRemovePass>()
                          .apply({{x}})
                          .endpoint_vars(),
                  o_remove);

    {
        size_t nr_shuffle_opr = 0;
        cg::DepOprIter iter{[&nr_shuffle_opr](cg::OperatorNodeBase* opr) {
            if (opr->same_type<opr::Dimshuffle>()) {
                ++nr_shuffle_opr;
            }
        }};
        iter.add(o_remove.node());
        mgb_assert(nr_shuffle_opr == 0);
    }
    {
        size_t nr_type_cvt_opr = 0;
        cg::DepOprIter iter{[&nr_type_cvt_opr](cg::OperatorNodeBase* opr) {
            if (opr->same_type<opr::TypeCvt>()) {
                ++nr_type_cvt_opr;
            }
        }};
        iter.add(o_remove.node());
        mgb_assert(nr_type_cvt_opr == 2);
    }

    opt.graph_opt_level = 0;
    unpack_vector(gopt::GraphOptimizer{}
                          .apply({{x}})
                          .endpoint_vars(),
                  o);

    HostTensorND h_o, h_o_remove;
    graph->options().graph_opt.tensorrt = false;
    auto func = graph->compile({make_callback_copy(o, h_o, false),
                                make_callback_copy(o_remove, h_o_remove)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(h_o, h_o_remove, 1e-4);
}
#endif

TEST(TestShuffleShuffleRemove, NCHW2NCHW42NCHW32) {
    REQUIRE_GPU(1);
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{
            -127.f * 1.2f, 127 * 1.2f};
    auto cn = CompNode::load("gpu0");

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp,
                     const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name),
                dtype);
    };

    auto nchw2nchw4 = [](SymbolVar x) {
        auto xshp = opr::GetVarShape::make(x);

        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp = opr::Concat::make(
                {sub(0), sub(1) / 4, cv(4), sub(2), sub(3)}, 0);
        auto y0 = opr::Reshape::make(x, tshp);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 4, 2});
        return y1;
    };

    auto nchw42nchw32 = [](SymbolVar x) {
        auto xshp = opr::GetVarShape::make(x);

        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                     {sub(0), sub(1) / 8, cv(8), sub(2), sub(3), sub(4)}, 0),
             tshp1 = opr::Concat::make(
                     {sub(0), sub(1) / 8, sub(2), sub(3), sub(4) * 8}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 4, 2, 5});
        auto y2 = opr::Reshape::make(y1, tshp1);

        return y2;
    };

    auto nchw322nchw4 = [](SymbolVar x) {
        auto xshp = opr::GetVarShape::make(x);

        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                     {sub(0), sub(1), sub(2), sub(3), cv(8), sub(4) / 8}, 0),
             tshp1 = opr::Concat::make(
                     {sub(0), sub(1) * 8, sub(2), sub(3), sub(4) / 8}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 4, 2, 3, 5});
        auto y2 = opr::Reshape::make(y1, tshp1);

        return y2;
    };

    auto x = mkvar("x", {32, 32, 28, 28}, dtype::QuantizedS8{2.5f});
    x = nchw2nchw4(x), x = nchw42nchw32(x), x = nchw322nchw4(x),
    x = opr::TypeCvt::make(x, dtype::Float32());
    SymbolVar o, o_remove;

    ComputingGraph::Options opt;
    opt.graph_opt_level = 0;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ShuffleShuffleRemovePass>()
                          .apply({{x}})
                          .endpoint_vars(),
                  o_remove);

    {
        size_t nr_shuffle_opr = 0;
        cg::DepOprIter iter{[&nr_shuffle_opr](cg::OperatorNodeBase* opr) {
            if (opr->same_type<opr::Dimshuffle>()) {
                ++nr_shuffle_opr;
            }
        }};
        iter.add(o_remove.node());
        mgb_assert(nr_shuffle_opr == 1);
    }
    {
        size_t nr_type_cvt_opr = 0;
        cg::DepOprIter iter{[&nr_type_cvt_opr](cg::OperatorNodeBase* opr) {
            if (opr->same_type<opr::TypeCvt>()) {
                ++nr_type_cvt_opr;
            }
        }};
        iter.add(o_remove.node());
        mgb_assert(nr_type_cvt_opr == 2);
    }

    opt.graph_opt_level = 0;
    unpack_vector(gopt::GraphOptimizer{}
                          .apply({{x}})
                          .endpoint_vars(),
                  o);

    HostTensorND h_o, h_o_remove;
    graph->options().graph_opt.tensorrt = false;
    auto func = graph->compile({make_callback_copy(o, h_o, false),
                                make_callback_copy(o_remove, h_o_remove)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(h_o, h_o_remove, 1e-4);
}

TEST(TensorRTReplacePass, EngineCache) {
    REQUIRE_GPU(1);
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{
            1.2f, 127 * 1.2f};
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 61) {
        printf("This testcase ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 61);
        return;
    }

    TensorRTEngineCache::enable_engine_cache(true);
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp,
                     const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name),
                dtype);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp,
                      const DType& dtype) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                        .rename(name),
                dtype);
    };

    auto nchw2nchw4 = [](SymbolVar x) {
        auto xshp = opr::GetVarShape::make(x);

        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp = opr::Concat::make(
                {sub(0), sub(1) / 4, cv(4), sub(2), sub(3)}, 0);
        auto y0 = opr::Reshape::make(x, tshp);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 4, 2});
        return y1;
    };

    auto nchw42nchw = [](SymbolVar x) {
        auto xshp = opr::GetVarShape::make(x);

        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp = opr::Concat::make({sub(0), sub(1) * 4, sub(2), sub(3)}, 0);
        auto y0 = opr::Dimshuffle::make(x, {0, 1, 4, 2, 3});
        auto y1 = opr::Reshape::make(y0, tshp);
        return y1;
    };

    auto x = mkvar("x", {32, 4, 28, 28}, dtype::QuantizedS8(2.5f)),
         w = mkcvar("w", {16, 4, 3, 3}, dtype::QuantizedS8(2.5f)),
         b = mkcvar("b", {1, 16, 1, 1}, dtype::QuantizedS32(6.25f)),
         z1 = mkvar("z", {32, 16, 28, 28}, dtype::QuantizedS8(2.5f));
    x = nchw2nchw4(x), w = nchw2nchw4(w), b = nchw2nchw4(b);
    auto z = nchw2nchw4(z1);
    opr::ConvBias::Param conv_param;
    conv_param.format = opr::ConvBias::Param::Format::NCHW4;
    conv_param.stride_h = conv_param.stride_w = 1;
    conv_param.pad_h = conv_param.pad_w = 1;
    auto y = opr::ConvBias::make(x, w, b, z, conv_param, {},
                                 OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    y = nchw42nchw(y);
    y = opr::TypeCvt::make(y, dtype::Float32());

    SymbolVar trt_y;
    SymbolVar mgb_y;

    ComputingGraph::Options opt;
    opt.graph_opt_level = 0;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ExpandFusedArithPass>()
                          .add_pass<gopt::TensorRTReplacePass>()
                          .add_pass<gopt::ArithFusePass>()
                          .add_pass<gopt::ShuffleShuffleRemovePass>()
                          .apply({{y}})
                          .endpoint_vars(),
                  trt_y);
    trt_y = opr::TypeCvt::make(trt_y, dtype::QuantizedS8{2.5f}),
    trt_y = opr::TypeCvt::make(trt_y, dtype::Float32());

    opt.graph_opt_level = 0;
    unpack_vector(gopt::GraphOptimizer{}.apply({{y}}).endpoint_vars(), mgb_y);

    ComputingGraph::OutputSpec outspec(2);
    SmallVector<HostTensorND> outputs(2);
    outspec[0] = make_callback_copy(trt_y, outputs[0], false);
    outspec[1] = make_callback_copy(mgb_y, outputs[1], false);
    graph->options().graph_opt.tensorrt = false;
    auto func = graph->compile(outspec);
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(outputs[0], outputs[1], 1e-4);
    TensorRTEngineCache::disable_engine_cache();
}

TEST(TestTensorRTReplace, FuseConvAdd) {
    REQUIRE_GPU(1);
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{-3.f,
                                                                         3.f};
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp,
                     const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name),
                dtype);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp,
                      const DType& dtype) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *gen(shp))
                        .rename(name),
                dtype);
    };

    auto x = mkvar("x", {32, 4, 28, 28}, dtype::Float32()),
         w = mkcvar("w", {16, 4, 3, 3}, dtype::Float32()),
         b = mkcvar("b", {1, 16, 1, 1}, dtype::Float32());
    opr::Convolution::Param param;
    param.format = opr::Convolution::Param::Format::NCHW;
    param.stride_h = param.stride_w = 1;
    param.pad_h = param.pad_w = 1;
    auto y = opr::Convolution::make(x, w, param);

    auto nchw2nchw4 = [](SymbolVar x) {
        auto xshp = opr::GetVarShape::make(x);

        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp = opr::Concat::make(
                {sub(0), sub(1) / 4, cv(4), sub(2), sub(3)}, 0);
        auto y0 = opr::Reshape::make(x, tshp);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 4, 2});
        return y1;
    };
    auto y1 = nchw2nchw4(y);
    y = y + b;

    SymbolVar trt_y, trt_y1;
    SymbolVar mgb_y, mgb_y1;

    ComputingGraph::Options opt;
    opt.graph_opt_level = 0;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ExpandFusedArithPass>()
                          .add_pass<gopt::TensorRTReplacePass>()
                          .add_pass<gopt::ArithFusePass>()
                          .apply({{y, y1}})
                          .endpoint_vars(),
                  trt_y, trt_y1);

    opt.graph_opt_level = 0;
    unpack_vector(gopt::GraphOptimizer{}.apply({{y, y1}}).endpoint_vars(),
                  mgb_y, mgb_y1);

    ComputingGraph::OutputSpec outspec(4);
    SmallVector<HostTensorND> outputs(4);
    outspec[0] = make_callback_copy(trt_y, outputs[0], false);
    outspec[1] = make_callback_copy(trt_y1, outputs[1], false);
    outspec[2] = make_callback_copy(mgb_y, outputs[2], false);
    outspec[3] = make_callback_copy(mgb_y1, outputs[3], false);
    graph->options().graph_opt.tensorrt = false;
    auto func = graph->compile(outspec);
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(outputs[0], outputs[2], 1e-3);
    MGB_ASSERT_TENSOR_NEAR(outputs[1], outputs[3], 1e-3);
}

TEST(TestTensorRTReplace, FuseConvAddNchw2nchw4) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    REQUIRE_CUDA_COMPUTE_CAPABILITY(6, 1);

    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen{
            1.2f, 127 * 127};
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp,
                     const DType& dtype) {
        return opr::TypeCvt::make(
                opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name),
                dtype);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp,
                      const DType& dtype) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *gen(shp))
                        .rename(name),
                dtype);
    };

    auto x = mkvar("x", {32, 4, 28, 28}, dtype::QuantizedS8(2.5f)),
         w = mkcvar("w", {16, 4, 3, 3}, dtype::QuantizedS8(2.5f)),
         b = mkcvar("b", {1, 16, 1, 1}, dtype::QuantizedS32(6.25f));
    opr::ConvBias::Param param;
    param.format = opr::ConvBias::Param::Format::NCHW;
    param.stride_h = param.stride_w = 1;
    param.pad_h = param.pad_w = 1;
    auto y = opr::ConvBias::make(x, w, b, param, {},
                                 OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    auto z = opr::TypeCvt::make(y, dtype::Float32());

    SymbolVar trt_z;
    SymbolVar mgb_z;

    ComputingGraph::Options opt;
    opt.graph_opt_level = 0;
    unpack_vector(
            gopt::GraphOptimizer{}
                    .add_pass<gopt::FuseConvBiasNonlinPass>()
                    .add_pass(gopt::EnableNCHW4Pass::make_nchw4_converter())
                    .add_pass<gopt::ExpandFusedArithPass>()
                    .add_pass<gopt::TensorRTReplacePass>()
                    .add_pass<gopt::ArithFusePass>()
                    .apply({{z}})
                    .endpoint_vars(),
            trt_z);

    opt.graph_opt_level = 0;
    unpack_vector(gopt::GraphOptimizer{}.apply({{z}}).endpoint_vars(),
                  mgb_z);

    ComputingGraph::OutputSpec outspec(2);
    SmallVector<HostTensorND> outputs(2);
    outspec[0] = make_callback_copy(trt_z, outputs[0], false);
    outspec[1] = make_callback_copy(mgb_z, outputs[1], false);
    graph->options().graph_opt.tensorrt = false;
    auto func = graph->compile(outspec);
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(outputs[0], outputs[1], 1e-3);
}

#endif  // MGB_ENABLE_TENSOR_RT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
