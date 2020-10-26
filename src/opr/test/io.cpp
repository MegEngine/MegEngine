/**
 * \file src/opr/test/io.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/serialization/opr_shallow_copy.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/comp_node_env.h"

#include "megdnn/basic_types.h"
#include "megdnn/tensor_format.h"

#include <unordered_set>

using namespace mgb;

namespace {
class NaiveMegDNNHandleScope {
    int m_orig_level;
public:
    NaiveMegDNNHandleScope()
            : m_orig_level{MegDNNHandle::exchange_default_dbg_level(2)} {
        CompNode::finalize();
    }
    ~NaiveMegDNNHandleScope() {
        auto set = MegDNNHandle::exchange_default_dbg_level(m_orig_level);
        mgb_assert(set == 2);
        CompNode::finalize();
    }
};

}  // namespace

TEST(TestOprIO, H2D) {
    HostTensorGenerator<> gen;
    auto t0 = gen({123, 456});
    HostTensorND t1(CompNode::load("xpu0"));

    auto graph = ComputingGraph::make();
    SymbolVar y = opr::Host2DeviceCopy::make(*graph, t0);

    bool executed = false;
    auto func = graph->compile({{
            y, [&](DeviceTensorND &v) {
                executed = true;
                t1.copy_from(v);
            }}});
    func->execute();
    ASSERT_TRUE(executed);
    MGB_ASSERT_TENSOR_EQ(*t0, t1.sync());
}

TEST(TestOprIO, H2DCopyShallow) {
    HostTensorGenerator<> gen;
    auto t0 = gen({123, 456});
    HostTensorND t1(CompNode::load("xpu0"));

    auto graph = ComputingGraph::make();
    SymbolVar y_expected =
            opr::Host2DeviceCopy::make_no_value_infer(*graph, t0);

    auto h2d_opr = y_expected.node()->owner_opr();
    SymbolVar y_get = {serialization::copy_opr_shallow(
            *h2d_opr, {}, h2d_opr->config())->output(0)};

    ASSERT_EQ(y_expected, y_get);
}

TEST(TestOprIO, H2DFwd) {
    HostTensorGenerator<> gen;
    auto cn0 = CompNode::load("cpu0"),
         cn1 = CompNode::load("cpu1");
    auto t0 = gen({1}, cn0);
    auto graph = ComputingGraph::make();
    SymbolVar y = opr::Host2DeviceCopy::make(*graph, t0, cn1);
    ASSERT_EQ(y.node()->comp_node(), cn1);

    bool executed = false;
    auto cb = [&](DeviceTensorND &dv) {
        executed = true;
        ASSERT_EQ(t0->raw_ptr(), dv.raw_ptr());
        ASSERT_EQ(t0->layout(), dv.layout());
    };
    auto func = graph->compile({{y, cb}});

    std::vector<HostTensorND> saved;
    for (int i = 0; i < 2; ++ i) {
        // check on multiple pointers
        saved.push_back(*gen({23, 4}, cn0));
        *t0 = saved.back();
        executed = false;
        func->execute();
        ASSERT_TRUE(executed);
    }

    // non-contig
    *t0 = (*t0)[{{2, 4}, {1, 3}}];
    HostTensorND host_y;
    graph->compile({make_callback_copy(y, host_y)})->execute();
    MGB_ASSERT_TENSOR_EQ(*t0, host_y);
}

TEST(TestOprIO, H2DCrossDev) {
    REQUIRE_GPU(1);

    HostTensorGenerator<> gen;
    CompNode cn[2] = {CompNode::load("cpu0"), CompNode::load("gpu0")};
    auto graph = ComputingGraph::make();
    for (int dev = 0; dev < 2; ++ dev) {
        auto t0 = gen({23}, cn[dev]);
        SymbolVar y = opr::Host2DeviceCopy::make(*graph, t0, {cn[!dev]});
        ASSERT_EQ(cn[!dev], y.node()->comp_node());
        HostTensorND t1;
        graph->compile({make_callback_copy(y, t1)})->execute();
        MGB_ASSERT_TENSOR_EQ(*t0, t1);
    }
}

TEST(TestOprIO, ImmutableTensor) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1});
    auto y_expected = host_x->ptr<float>()[0] + 1;
    auto graph = ComputingGraph::make();
    auto x = opr::ImmutableTensor::make(*graph, *host_x);
    EXPECT_THROW(opr::AddUpdate::make(x, opr::Host2DeviceCopy::make(*graph,
                    host_x)), MegBrainError);
    host_x->ptr<float>()[0] ++;
    auto y = x + 1;
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->execute();
    ASSERT_EQ(y_expected, host_y.ptr<float>()[0]);
    host_y.ptr<float>()[0] ++;
    host_x->ptr<float>()[0] ++;
    func->execute();
    ASSERT_EQ(y_expected, host_y.ptr<float>()[0]);

}

TEST(TestOprIO, ImmutableTensorHostvalue) {
    HostTensorGenerator<> gen;
    TensorShape shape({2, 3});
    auto host_x = gen(shape);
    auto graph = ComputingGraph::make();
    auto x = opr::ImmutableTensor::make(*graph, *host_x);
    auto y = x.node()->owner_opr()
                     ->cast_final_safe<opr::ImmutableTensor>()
                     .host_value();
    for (size_t i = 0; i < shape.total_nr_elems(); ++i) {
        ASSERT_EQ(host_x->ptr<float>()[i], y.ptr<float>()[i]);
    }
}

TEST(TestOprIO, ImmutableTensorHostvalueGPU) {
    REQUIRE_GPU(1);
    auto gpu_cn = CompNode::load("gpu0");
    HostTensorGenerator<> gen;
    TensorShape shape({2, 3});
    auto host_x = gen(shape);
    auto graph = ComputingGraph::make();
    auto x = opr::ImmutableTensor::make(*graph, *host_x, {gpu_cn});
    auto y = x.node()->owner_opr()
                     ->cast_final_safe<opr::ImmutableTensor>()
                     .host_value();
    for (size_t i = 0; i < shape.total_nr_elems(); ++i) {
        ASSERT_EQ(host_x->ptr<float>()[i], y.ptr<float>()[i]);
    }
}

TEST(TestOprIO, ImmutableTensorLarge) {
    HostTensorGenerator<> gen;
    auto host_x = gen({1025});
    auto graph = ComputingGraph::make();
    auto a = opr::ImmutableTensor::make(*graph, *host_x),
         b = opr::ImmutableTensor::make(*graph, *host_x),
         y = a + b;
    ASSERT_NE(a.node(), b.node());
    ASSERT_NE(
            &a.node()->owner_opr()->cast_final_safe<
                opr::ImmutableTensor>().value(),
            &b.node()->owner_opr()->cast_final_safe<
                opr::ImmutableTensor>().value());
    HostTensorND host_x_val;
    host_x_val.copy_from(*host_x);
    host_x->copy_from(*gen({1024}));

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->execute();

    auto px = host_x_val.ptr<float>(), py = host_y.ptr<float>();
    for (size_t i = 0; i < 1025; ++ i) {
        MGB_ASSERT_FLOAT_EQ(px[i] * 2, py[i]);
    }
}

TEST(TestOprIO, ImmutableTensorEmpty) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_x = gen({1, 9, 1, 9, 8, 1, 0});
    auto x = opr::ImmutableTensor::make(*graph, *host_x);
    HostTensorND host_x2;
    auto func = graph->compile({make_callback_copy(x, host_x2)});
    func->execute();
    ASSERT_TRUE(host_x2.shape().is_empty());
}

TEST(TestOprIO, SharedDeviceTensor) {
    HostTensorGenerator<> gen;
    auto hv = gen({123});
    auto dv = std::make_shared<DeviceTensorND>();
    DeviceTensorND dv0, dv1;
    dv0.copy_from(*hv);
    dv1.copy_from(*hv);

    {
        auto graph = ComputingGraph::make();
        *dv = dv0;
        SymbolVar x = opr::SharedDeviceTensor::make(*graph, dv);
        HostTensorND host_x;
        auto func = graph->compile({make_callback_copy(x, host_x)});
        func->execute().wait();
        MGB_ASSERT_TENSOR_EQ(*hv, host_x);

        // disallow ptr change
        *dv = dv1;
        ASSERT_THROW(func->execute().wait(), MegBrainError);

        // check that old ptr works
        *dv = dv0;
        func->execute().wait();
        MGB_ASSERT_TENSOR_EQ(*hv, host_x);

        // disallow shape change
        dv->resize({23});
        ASSERT_THROW(func->execute().wait(), MegBrainError);
    }

    {
        auto graph = ComputingGraph::make();
        *dv = dv0;
        SymbolVar x = opr::VolatileSharedDeviceTensor::make(*graph, dv);
        HostTensorND host_x;
        auto func = graph->compile({make_callback_copy(x, host_x)});
        func->execute().wait();
        MGB_ASSERT_TENSOR_EQ(*hv, host_x);

        // allow ptr change
        host_x.resize({});
        *dv = dv1;
        func->execute().wait();
        MGB_ASSERT_TENSOR_EQ(*hv, host_x);

        // allow shape change
        *hv = *gen({23});
        dv->copy_from(*hv);
        func->execute().wait();
        MGB_ASSERT_TENSOR_EQ(*hv, host_x);
    }
}

TEST(TestOprIO, SharedDeviceTensorWithFormat) {
    CompNode cn = CompNode::load("xpu0");
    HostTensorGenerator<> gen;
    auto hv = gen({1, 1, 1, 1, 4});

    auto layout =
            TensorLayout(TensorShape{1, 1, 1, 1, 4}, dtype::Float32{},
                         megdnn::Image2DPack4TensorFormat::make_raw(2, 64));
    auto dv = std::make_shared<DeviceTensorND>(cn, layout);

    DeviceTensorND dv0(cn, layout), dv1(cn, layout);

    EXPECT_NO_THROW(dv0.copy_from_fixlayout(*hv).sync());
    EXPECT_NO_THROW(dv1.copy_from_fixlayout(*hv).sync());

    {
        auto graph = ComputingGraph::make();
        *dv = dv0;
        SymbolVar x = opr::SharedDeviceTensorWithFormat::make(*graph, dv);
        HostTensorND host_x;
        auto func = graph->compile({make_callback_copy(x, host_x)});
        func->execute().wait();
        MGB_ASSERT_TENSOR_EQ(*hv, host_x);

        // disallow ptr change
        *dv = dv1;
        ASSERT_THROW(func->execute().wait(), MegBrainError);

        // check that old ptr works
        *dv = dv0;
        func->execute().wait();
        MGB_ASSERT_TENSOR_EQ(*hv, host_x);

        // disallow shape change
        dv->resize({1, 1, 1, 4});
        ASSERT_THROW(func->execute().wait(), MegBrainError);
    }
}


TEST(TestOprIO, ImmutableTensorDeDup) {
    auto cn = CompNode::load("xpu0");

    auto make_hv = [&](const std::vector<dt_int32> &val) {
        HostTensorND ret(cn, dtype::Int32());
        ret.resize({val.size()});
        memcpy(ret.raw_ptr(), val.data(), sizeof(int) * val.size());
        return ret;
    };
    auto as_opr = [](SymbolVar var) {
        return &var.node()->owner_opr()->
            cast_final_safe<opr::ImmutableTensor>();
    };

    auto make_opr = [&](ComputingGraph &g, const HostTensorND &val) {
        return as_opr(opr::ImmutableTensor::make(g, val));
    };

    auto g0 = ComputingGraph::make(), g1 = ComputingGraph::make();
    auto hv_chg = make_hv({3});
    auto op0 = make_opr(*g0, make_hv({2})),
         op1 = make_opr(*g0, make_hv({2})),
         op2 = as_opr(SymbolVar{op0->output(0)}.make_scalar(2)),
         op3 = make_opr(*g0, make_hv({2, 3})),
         op4 = make_opr(*g1, make_hv({2})),
         op5 = make_opr(*g1, make_hv({2, 3})),
         op6 = make_opr(*g1, make_hv({2, 3, 4})),
         op7 = make_opr(*g1, hv_chg);
    hv_chg.ptr<dt_int32>()[0] = 2;
    auto op8 = make_opr(*g1, hv_chg);
    ASSERT_EQ(op0, op1);
    ASSERT_EQ(op0, op2);

    auto vptr = [](opr::ImmutableTensor *op) {
        return &op->value();
    };

    ASSERT_NE(op0, op3);
    ASSERT_NE(vptr(op0), vptr(op3));

    ASSERT_NE(op3, op5);
    ASSERT_EQ(vptr(op3), vptr(op5));
    ASSERT_NE(op0, op4);
    ASSERT_EQ(vptr(op0), vptr(op4));

    ASSERT_NE(op5, op6);
    ASSERT_NE(vptr(op5), vptr(op6));

    ASSERT_NE(op4, op7);
    ASSERT_EQ(op4, op8);
}

TEST(TestOprIO, D2DCopy) {
    auto cns = load_multiple_xpus(2);
    constexpr size_t SIZE = 23;
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE}, cns[0]);
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
         y0 = (x + 2).rename("y0"),
         y1 = (opr::Copy::make(x, {cns[1]}) + 2).rename("y1");
    HostTensorND y_expected, host_y0, host_y1;
    y_expected.copy_from(*host_x);
    for (size_t i = 0; i < SIZE; ++ i)
        y_expected.ptr<float>()[i] = host_x->ptr<float>()[i] + 2;

    auto func = graph->compile({make_callback_copy(y0, host_y0),
            make_callback_copy(y1, host_y1)});
    func->execute();
    func->to_json()->writeto_fpath(output_file("TestOprIO.D2DCopy.json"));

    ASSERT_NE(y0.node()->prev_dev_ptr(), x.node()->prev_dev_ptr());
    MGB_ASSERT_TENSOR_EQ(y_expected, host_y0);
    MGB_ASSERT_TENSOR_EQ(y_expected, host_y1);
}

TEST(TestOprIO, D2DNonContig) {
    REQUIRE_GPU(2);
    CompNode cns[2] = {CompNode::load("gpu0"), CompNode::load("gpu1")};
    HostTensorGenerator<> gen;
    auto host_x = gen({6, 5, 4, 3});
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x, {}, {cns[0]})
                     .rename("x"),
         _x = opr::Dimshuffle::make(x, {3, 0, 1, 2}, {}),
         y = opr::Copy::make(_x, {cns[1]}),
         cpu_y = opr::Copy::make(_x, {CompNode::load("cpu0")});
    HostTensorND host_y, except_y;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(cpu_y, except_y)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_y, except_y);
}

TEST(TestOprIO, MultipleDeviceTensorHolder) {
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen0;
    HostTensorGenerator<dtype::Int32> gen1;
    auto host_v0 = gen0({2, 3}, cns[0]), host_v1 = gen1({3, 4}, cns[1]);
    auto make_dv = [](const HostTensorND& hv) {
        auto ret = std::make_shared<DeviceTensorND>();
        ret->copy_from(hv);
        return ret;
    };
    auto dev_v0 = make_dv(*host_v0), dev_v1 = make_dv(*host_v1);
    auto graph = ComputingGraph::make();
    SymbolVar var0, var1;
    unpack_vector(
            opr::MultipleDeviceTensorHolder::make(*graph, {dev_v0, dev_v1}),
            var0, var1);
    {
        // dedup
        SymbolVar x, y;
        unpack_vector(
                opr::MultipleDeviceTensorHolder::make(*graph, {dev_v0, dev_v1}),
                x, y);
        ASSERT_EQ(var0, x);
        ASSERT_EQ(var1, y);
    }
    {
        // no dedup
        SymbolVar x, y;
        unpack_vector(
                opr::MultipleDeviceTensorHolder::make(*graph, {dev_v0, dev_v0}),
                x, y);
        ASSERT_NE(var0.node(), x.node());
        ASSERT_NE(var1.node(), y.node());
    }

    HostTensorND got_v0, got_v1;
    auto func = graph->compile({make_callback_copy(var0, got_v0),
                                make_callback_copy(var1, got_v1)});
    func->execute();
    ASSERT_EQ(dtype::Float32{}, got_v0.dtype());
    ASSERT_EQ(cns[0], got_v0.comp_node());
    ASSERT_EQ(dtype::Int32{}, got_v1.dtype());
    ASSERT_EQ(cns[1], got_v1.comp_node());
    MGB_ASSERT_TENSOR_EQ(got_v0, *host_v0);
    MGB_ASSERT_TENSOR_EQ(got_v1, *host_v1);
}

TEST(TestOprIO, MultipleDeviceTensorWithFormatHolder) {
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen0;
    HostTensorGenerator<dtype::Int32> gen1;
    auto host_v0 = gen0({2, 3, 8}, cns[0]), host_v1 = gen1({3, 4, 8}, cns[1]);
    auto make_dv = [](const HostTensorND& hv) {
        TensorLayout layout{hv.layout(), hv.layout().dtype,
                            megdnn::Image2DPack4TensorFormat::make_raw(2, 64)};
        auto ret = std::make_shared<DeviceTensorND>(hv.comp_node(), layout);
        ret->copy_from_fixlayout(hv);
        return ret;
    };
    auto dev_v0 = make_dv(*host_v0), dev_v1 = make_dv(*host_v1);
    auto graph = ComputingGraph::make();
    SymbolVar var0, var1;
    unpack_vector(opr::MultipleDeviceTensorWithFormatHolder::make(
                          *graph, {dev_v0, dev_v1}),
                  var0, var1);
    {
        // dedup
        SymbolVar x, y;
        unpack_vector(opr::MultipleDeviceTensorWithFormatHolder::make(
                              *graph, {dev_v0, dev_v1}),
                      x, y);
        ASSERT_EQ(var0, x);
        ASSERT_EQ(var1, y);
    }
    {
        // no dedup
        SymbolVar x, y;
        unpack_vector(opr::MultipleDeviceTensorWithFormatHolder::make(
                              *graph, {dev_v0, dev_v0}),
                      x, y);
        ASSERT_NE(var0.node(), x.node());
        ASSERT_NE(var1.node(), y.node());
    }

    HostTensorND got_v0, got_v1;
    auto func = graph->compile({make_callback_copy(var0, got_v0),
                                make_callback_copy(var1, got_v1)});
    func->execute();
    ASSERT_EQ(dtype::Float32{}, got_v0.dtype());
    ASSERT_EQ(cns[0], got_v0.comp_node());
    ASSERT_EQ(dtype::Int32{}, got_v1.dtype());
    ASSERT_EQ(cns[1], got_v1.comp_node());
    MGB_ASSERT_TENSOR_EQ(got_v0, *host_v0);
    MGB_ASSERT_TENSOR_EQ(got_v1, *host_v1);
}

#define GET_OUTPUT_FILE() output_file(ssprintf("TestOprIo.%d", __LINE__))
TEST(TestOprIO, MultipleDeviceTensorWithFormatHolderCpu) {
    // hwcd4 is only supported in naive handle
    NaiveMegDNNHandleScope naive_megdnn_handle;
    auto fname = GET_OUTPUT_FILE();
    auto cn = CompNode::load("cpu0");
    HostTensorGenerator<> gen;
    {
        // dump
        auto graph = ComputingGraph::make();
        graph->options().graph_opt_level = 0;

        auto mkcvar = [&](const char* name, const TensorShape& shp) {
            return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                    .rename(name);
        };
        auto host_x = gen({8, 8, 8, 8}, cn);
        auto x = opr::Host2DeviceCopy::make(*graph, host_x, {"x"});

        opr::Convolution::Param param;
        param.pad_h = param.pad_w = 0;
        auto w1 = mkcvar("w1", {4, 8, 3, 3}),
             conv1 = opr::Convolution::make(x, w1, param);

        auto w2 = mkcvar("w2", {4, 4, 3, 3}),
             conv2 = opr::Convolution::make(conv1, w2, param);

        auto y = opr::Elemwise::make({conv2}, opr::Elemwise::Param::Mode::RELU);
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_nhwcd4();
        SymbolVar y_opt =
                gopt::optimize_for_inference({y}, options)[0].rename("out");

        auto dumper = serialization::GraphDumper::make(
                serialization::OutputFile::make_fs(fname.c_str()));
        serialization::GraphDumper::DumpConfig config;
        config.keep_param_name = true;
        dumper->dump({y_opt}, config);
    }
    auto loader = serialization::GraphLoader::make(
            serialization::InputFile::make_fs(fname.c_str()));

    auto load = [&](CompNode dest_cn) {
        auto dest_cn_loc = dest_cn.locator_logical();
        auto rst = loader->load({
                [&](CompNode::Locator &loc){ loc = dest_cn_loc;}});
        HostTensorND host_z, host_z_expect;
        auto func = rst.graph_compile(
                {make_callback_copy(rst.output_var_map.at("out"), host_z)});
        func->execute();
    };
    load(cn);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
