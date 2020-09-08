/**
 * \file src/gopt/test/inference.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/opr/dnn/local.h"
#include "megbrain/test/helper.h"

#include "megbrain/gopt/basic_arith.h"
#include "megbrain/gopt/gtrans.h"
#include "megbrain/gopt/inference.h"

#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/batch_norm.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/nn_int.h"
#include "megbrain/opr/tensor_gen.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"

#include "./helper.h"
#include "megbrain/comp_node_env.h"

#include "megdnn/tensor_format.h"

#include <random>

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
    mgb_assert(found, "not found opr %s from %s", node_name.c_str(),
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

#if MGB_CUDA
//! this function is only used in TestGoptInference.EnableCHWN4...
void warp_perspective_mat_gen(HostTensorND& mat, size_t N, size_t INP_H,
                              size_t INP_W) {
    static std::mt19937 rng(next_rand_seed());
    auto rand_real = [&](double lo, double hi) {
        return rng() / (std::mt19937::max() + 1.0) * (hi - lo) + lo;
    };
    auto rand_real2 = [&](double range) { return rand_real(-range, range); };
    auto ptr = mat.ptr<float>();
    for (size_t i = 0; i < N; ++i) {
        auto rot = rand_real(0, M_PI * 2), scale = rand_real(0.8, 1.2),
             sheer = rand_real(0.9, 1.1), dy = rand_real2(INP_H * 0.5),
             dx = rand_real2(INP_W * 0.5), ky = rand_real2(0.1 / INP_H),
             kx = rand_real2(0.1 / INP_W), kb = rand_real2(0.1) + 1;
        ptr[0] = ptr[4] = cos(rot) * scale;
        ptr[1] = -(ptr[3] = sin(rot) * scale);
        ptr[3] *= sheer;
        ptr[4] *= sheer;
        ptr[2] = dx;
        ptr[5] = dy;
        ptr[6] = kx;
        ptr[7] = ky;
        ptr[8] = kb;
        ptr += 9;
    }
    mgb_assert(ptr == mat.ptr<float>() + mat.shape().total_nr_elems());
}
#endif
}  // namespace

TEST(TestGoptInference, ParamFuseConstEndPoint) {
    constexpr size_t SIZE = 23;
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE}), host_y = gen({1}), host_p = gen({1});

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto x = opr::SharedDeviceTensor::make(*graph, *host_x),
         y = opr::SharedDeviceTensor::make(*graph, *host_y),
         p = opr::Host2DeviceCopy::make(*graph, host_p), q = p + x, a = y + 3,
         z0 = a + q, z1 = a + 4;

    HostTensorND host_z0, host_z1;

    SymbolVar z0_1, z1_1;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ParamFusePass>()
                          .apply({{z1, z0}})
                          .endpoint_vars(),
                  z1_1, z0_1);

    auto func = graph->compile({make_callback_copy(z0_1, host_z0),
                                make_callback_copy(z1_1, host_z1)});
    func->to_json()->writeto_fpath(
            output_file("TestGoptInference.ParamFuseEndPoint.json"));
    func->execute();

    int nr_opr = 0;
    func->iter_opr_seq([&](cg::OperatorNodeBase*) {
        ++nr_opr;
        return true;
    });
    ASSERT_EQ(8, nr_opr);

    auto px = host_x->ptr<float>(), pz0 = host_z0.ptr<float>();

    auto yv = host_y->ptr<float>()[0], pv = host_p->ptr<float>()[0],
         pz1 = host_z1.ptr<float>()[0];

    for (size_t i = 0; i < SIZE; ++i) {
        MGB_ASSERT_FLOAT_EQ(px[i] + yv + 3 + pv, pz0[i]);
    }
    MGB_ASSERT_FLOAT_EQ(yv + 7, pz1);
}

TEST(TestGoptInference, ParamFuse) {
    constexpr size_t SIZE = 23;
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE}), host_y = gen({1}), host_p = gen({1});

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto x = opr::SharedDeviceTensor::make(*graph, *host_x),
         y = opr::SharedDeviceTensor::make(*graph, *host_y),
         p = opr::Host2DeviceCopy::make(*graph, host_p),
         z = x + y,         // endpoint
            q = x * y + p;  // middle point

    SymbolVar z1, q1;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ParamFusePass>()
                          .apply({{z, q}})
                          .endpoint_vars(),
                  z1, q1);

    ASSERT_TRUE(z1.node()->owner_opr()->same_type<opr::SharedDeviceTensor>());
    ASSERT_NE(q1.node()->owner_opr(), q.node()->owner_opr());
    ASSERT_EQ(q1.node()->owner_opr()->dyn_typeinfo(),
              q.node()->owner_opr()->dyn_typeinfo());

    HostTensorND host_z, host_q;
    auto func = graph->compile(
            {make_callback_copy(z1, host_z), make_callback_copy(q1, host_q)});
    func->execute();

    int nr_opr = 0;
    func->iter_opr_seq([&](cg::OperatorNodeBase*) {
        ++nr_opr;
        return true;
    });
    ASSERT_EQ(6, nr_opr);

    auto px = host_x->ptr<float>(), pz = host_z.ptr<float>(),
         pq = host_q.ptr<float>();
    auto yv = host_y->ptr<float>()[0], pv = host_p->ptr<float>()[0];
    for (size_t i = 0; i < SIZE; ++i) {
        MGB_ASSERT_FLOAT_EQ(px[i] + yv, pz[i]);
        MGB_ASSERT_FLOAT_EQ(px[i] * yv + pv, pq[i]);
    }
}

TEST(TestGoptInference, ParamFuseMultiDeviceTensorHolder) {
    constexpr size_t SIZE = 23;
    HostTensorGenerator<> gen;
    auto host_x = gen({SIZE}), host_y = gen({1}), host_p = gen({1});

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto x = opr::SharedDeviceTensor::make(*graph, *host_x),
         y = opr::SharedDeviceTensor::make(*graph, *host_y),
         p = opr::Host2DeviceCopy::make(*graph, host_p),
         z = x + y,         //! endpoint
            q = x * y + p;  //! middle point

    SymbolVar z1, q1;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ParamMergePass>()
                          .apply({{z}})
                          .endpoint_vars(),
                  z1);

    ASSERT_TRUE(z1.node()
                        ->owner_opr()
                        ->input(0)
                        ->owner_opr()
                        ->same_type<opr::MultipleDeviceTensorHolder>());
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ParamMergePass>()
                          .add_pass<gopt::ParamFusePass>()
                          .apply({{z, q}})
                          .endpoint_vars(),
                  z1, q1);

    ASSERT_TRUE(z1.node()->owner_opr()->same_type<opr::SharedDeviceTensor>());
    ASSERT_NE(q1.node()->owner_opr(), q.node()->owner_opr());
    ASSERT_EQ(q1.node()->owner_opr()->dyn_typeinfo(),
              q.node()->owner_opr()->dyn_typeinfo());

    HostTensorND host_z, host_q;
    auto func = graph->compile(
            {make_callback_copy(z1, host_z), make_callback_copy(q1, host_q)});
    func->execute();

    int nr_opr = 0;
    func->iter_opr_seq([&](cg::OperatorNodeBase* op) {
        ++nr_opr;
        return true;
    });
    ASSERT_EQ(6, nr_opr);

    auto px = host_x->ptr<float>(), pz = host_z.ptr<float>(),
         pq = host_q.ptr<float>();
    auto yv = host_y->ptr<float>()[0], pv = host_p->ptr<float>()[0];
    for (size_t i = 0; i < SIZE; ++i) {
        MGB_ASSERT_FLOAT_EQ(px[i] + yv, pz[i]);
        MGB_ASSERT_FLOAT_EQ(px[i] * yv + pv, pq[i]);
    }
}

TEST(TestGoptInference, ParamFuseMultiRead) {
    HostTensorGenerator<> gen;

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp)).rename(name);
    };

    auto x = mkvar("x", {23}), p0 = mkcvar("p0", {1}), p1 = mkcvar("p1", {1}),
         z0 = x * (p0 + p1) + x / (p0 + p1);

    SymbolVar z1;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ParamFusePass>()
                          .apply({{z0}})
                          .endpoint_vars(),
                  z1);

    ASSERT_NE(z0.node(), z1.node());
    ASSERT_TRUE(z1.node()
                        ->owner_opr()
                        ->input(0)
                        ->owner_opr()
                        ->input(1)
                        ->owner_opr()
                        ->same_type<opr::SharedDeviceTensor>());
    ASSERT_TRUE(z1.node()
                        ->owner_opr()
                        ->input(1)
                        ->owner_opr()
                        ->input(1)
                        ->owner_opr()
                        ->same_type<opr::SharedDeviceTensor>());
    HostTensorND host_z0, host_z1;
    graph->compile({make_callback_copy(z0, host_z0),
                    make_callback_copy(z1, host_z1)})
            ->execute();
    MGB_ASSERT_TENSOR_EQ(host_z0, host_z1);
}

TEST(TestGoptInference, ParamFuseStaticInfer) {
    HostTensorGenerator<> gen;

    auto graph = ComputingGraph::make();

    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp)).rename(name);
    };

    auto a = mkvar("x", {4}),
         b = a.reshape(opr::GetVarShape::make(mkcvar("tshp", {2, 2})));

    SymbolVar b1;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ParamFusePass>()
                          .apply({{b}})
                          .endpoint_vars(),
                  b1);

    ASSERT_EQ(b1, a.reshape({2, 2}));
}

TEST(TestGoptInference, ParamRedistributeConvMul) {
    constexpr size_t N = 4, IC = 3, IH = 5, IW = 4, OC = 4, KH = 3, KW = 2;

    HostTensorGenerator<> gen;
    auto host_x = gen({N, IC, IH, IW}), host_k = gen({IC}),
         host_w = gen({OC, IC, KH, KW});

    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         k = opr::Dimshuffle::make(
                 opr::SharedDeviceTensor::make(*graph, *host_k),
                 {-1, 0, -1, -1}),
         w = opr::SharedDeviceTensor::make(*graph, *host_w),
         y0 = opr::Convolution::make(x * k, w);

    SymbolVar y1;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ParamRedistributePass>()
                          .apply({{y0}})
                          .endpoint_vars(),
                  y1);

    ASSERT_NE(y0.node(), y1.node());

    HostTensorND host_y0, host_y1;
    auto func = graph->compile(
            {make_callback_copy(y0, host_y0), make_callback_copy(y1, host_y1)});
    func->execute();

    MGB_ASSERT_TENSOR_EQ(host_y0, host_y1);
}

TEST(TestGoptInference, ParamRedistributeConvMulUniqReader) {
    constexpr size_t N = 4, C = 3, IH = 5, IW = 4, KH = 1, KW = 1;

    HostTensorGenerator<> gen;
    auto host_x = gen({N, C, IH, IW}), host_k = gen({C}),
         host_w = gen({C, C, KH, KW});

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         k = opr::Dimshuffle::make(
                 opr::SharedDeviceTensor::make(*graph, *host_k) + 2,
                 {-1, 0, -1, -1}),
         w = opr::SharedDeviceTensor::make(*graph, *host_w),
         // y0 should be replaced
            y0 = opr::powf(opr::Convolution::make(x * k, w).rename("y0") + 2,
                           2),
         y0k = (y0 * k).rename("y0k"),
         // y0k is accessed twice, so it should not be replaced
            y1 = opr::Convolution::make(y0k, w).rename("y1"), z0 = y1 / y0k;

    SymbolVar z1;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ParamRedistributePass>()
                          .apply({{z0}})
                          .endpoint_vars(),
                  z1);

    ASSERT_NE(z0.node(), z1.node());
    auto y1_repl = z1.node()->owner_opr()->input(0)->owner_opr();
    ASSERT_TRUE(y1_repl->same_type<opr::Convolution>());
    ASSERT_EQ(y1_repl->input(0), z1.node()->owner_opr()->input(1));

    HostTensorND host_z0, host_z1;
    auto func = graph->compile(
            {make_callback_copy(z0, host_z0), make_callback_copy(z1, host_z1)});
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(host_z0, host_z1, 5e-5);
}

TEST(TestGoptInference, ParamRedistributeMulConvMul) {
    constexpr size_t N = 4, IC = 3, IH = 5, IW = 4, OC = 4, KH = 3, KW = 2;

    HostTensorGenerator<> gen;
    auto host_x = gen({N, IC, IH, IW}), host_k1 = gen({IC}),
         host_k2 = gen({1, OC, 1, 1}), host_w = gen({OC, IC, KH, KW});

    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         k1 = opr::Dimshuffle::make(
                 opr::SharedDeviceTensor::make(*graph, *host_k1),
                 {-1, 0, -1, -1}),
         k2 = opr::SharedDeviceTensor::make(*graph, *host_k2),
         w = opr::SharedDeviceTensor::make(*graph, *host_w),
         y0 = opr::Convolution::make(x * k1, w) * k2;

    SymbolVar y1;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ParamRedistributePass>()
                          .add_pass<gopt::ParamFusePass>()
                          .apply({{y0}})
                          .endpoint_vars(),
                  y1);

    auto y1opr = y1.node()->owner_opr();
    ASSERT_TRUE(y1opr->same_type<opr::Convolution>());
    ASSERT_EQ(y1opr->input(0), x.node());

    HostTensorND host_y0, host_y1;
    auto func = graph->compile(
            {make_callback_copy(y0, host_y0), make_callback_copy(y1, host_y1)});
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(host_y0, host_y1, 5e-6);
}

TEST(TestGoptInference, ParamRedistributeConvAdd) {
    constexpr size_t N = 4, IC = 3, IH = 5, IW = 4, OC = 4, KH = 3, KW = 2;

    HostTensorGenerator<> gen;
    auto host_x = gen({N, IC, IH, IW}), host_b = gen({IC}),
         host_w = gen({OC, IC, KH, KW});

    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         b = opr::Dimshuffle::make(
                 opr::SharedDeviceTensor::make(*graph, *host_b),
                 {-1, 0, -1, -1}),
         w = opr::SharedDeviceTensor::make(*graph, *host_w),
         y0 = opr::Convolution::make(x + b, w);

    SymbolVar y1;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ParamRedistributePass>()
                          .add_pass<gopt::ParamFusePass>()
                          .apply({{y0}})
                          .endpoint_vars(),
                  y1);

    ASSERT_NE(y0.node(), y1.node());

    HostTensorND host_y0, host_y1;
    auto func = graph->compile(
            {make_callback_copy(y0, host_y0), make_callback_copy(y1, host_y1)});
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(host_y0, host_y1, 1e-5);
}

TEST(TestGoptInference, ParamRedistributeDistThenReasso) {
    constexpr size_t N = 4, IC0 = 3, IC1 = 6, IH = 5, IW = 4, OC = 4, KH = 3,
                     KW = 2;

    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp)).rename(name);
    };
    auto x0 = mkvar("x0", {N, IC0, IH, IW}), x1 = mkvar("x1", {N, IC1, IH, IW}),
         k0 = opr::Dimshuffle::make(mkcvar("x1_", {IC0}), {-1, 0, -1, -1})
                      .rename("x1"),
         w0 = mkcvar("w0", {OC, IC0, KH, KW}),
         k1 = mkcvar("k1", {1, IC1, 1, 1}),
         w1 = mkcvar("w1", {OC, IC1, KH, KW}), b0 = mkvar("b0", {1, OC, 1, 1}),
         b1 = mkcvar("b1", {1}), k2 = mkcvar("k2", {1}),
         y0 = (opr::Convolution::make(x0 * k0, w0) +
               opr::Convolution::make(x1 + k1, w1) + b0 + b1) *
              k2;

    SymbolVar y1;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ParamRedistributePass>()
                          .add_pass<gopt::ReorderArithChainPass>(
                                  gopt::ConstVarType::IMMUTABLE_AND_PARAM)
                          .add_pass<gopt::ParamFusePass>()
                          .apply({{y0}})
                          .endpoint_vars(),
                  y1);

    ASSERT_NE(y0.node(), y1.node());
    HostTensorND host_y0, host_y1;
    auto func = graph->compile(
            {make_callback_copy(y0, host_y0), make_callback_copy(y1, host_y1)});
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(host_y0, host_y1, 1e-5);

    auto chain =
            gopt::extract_opr_leaves(y1.node(), [](cg::OperatorNodeBase* opr) {
                return gopt::as_elem_opr(opr, opr::Elemwise::Mode::ADD);
            });
    size_t nr_conv = 0;
    for (auto i : chain) {
        auto opr = i->owner_opr();
        if (opr->same_type<opr::Convolution>()) {
            ++nr_conv;
            ASSERT_TRUE(opr->input(0)
                                ->owner_opr()
                                ->same_type<opr::Host2DeviceCopy>());
            ASSERT_TRUE(opr->input(1)
                                ->owner_opr()
                                ->same_type<opr::SharedDeviceTensor>());
        }
    }
    ASSERT_EQ(2u, nr_conv);
    ASSERT_EQ(4u, chain.size());
}

TEST(TestGoptInference, ParamRedistributeMultiChange) {
    constexpr size_t N = 4, IC = 3, IH = 5, IW = 4, OC = 4, KH = 3, KW = 2;

    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp)).rename(name);
    };
    auto x = mkvar("x", {N, IC, IH, IW}), k0 = mkcvar("k0", {1, IC, 1, 1}),
         b0 = mkcvar("b0", {1, IC, 1, 1}), k1 = mkcvar("k0", {1}),
         b1 = mkcvar("b0", {1}), w = mkcvar("w", {OC, IC, KH, KW}),
         y0 = (opr::Convolution::make(x * k0 + b0, w) + b1) * k1;

    SymbolVar y1;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ParamRedistributePass>()
                          .add_pass<gopt::ParamFusePass>()
                          .apply({{y0}})
                          .endpoint_vars(),
                  y1);

    ASSERT_NE(y0.node(), y1.node());
    HostTensorND host_y0, host_y1;
    auto func = graph->compile(
            {make_callback_copy(y0, host_y0), make_callback_copy(y1, host_y1)});
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(host_y0, host_y1, 1e-5);

    auto y1elem = gopt::as_elem_opr(y1.node(), opr::Elemwise::Mode::ADD);
    ASSERT_TRUE(y1elem);
    auto yconv = y1elem->input(0)->owner_opr();
    if (!yconv->same_type<opr::Convolution>())
        yconv = y1elem->input(1)->owner_opr();
    ASSERT_TRUE(yconv->same_type<opr::Convolution>());
    ASSERT_EQ(x.node(), yconv->input(0));
}

TEST(TestGoptInference, ParamRedistributeMultiReader) {
    constexpr size_t N = 4, IC = 3, IH = 5, IW = 4, OC = 4, KH = 3, KW = 2;

    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };

    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp)).rename(name);
    };

    auto x = mkvar("x", {N, IC, IH, IW}), k = mkcvar("k", {1, OC, 1, 1}),
         w = mkcvar("w", {OC, IC, KH, KW});

    auto conv = opr::Convolution::make(x, w);
    auto t = conv * k;
    auto y0 = t * 4.2f + t * 2.4f;

    SymbolVar y1;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ParamRedistributePass>()
                          .add_pass<gopt::ParamFusePass>()
                          .apply({{y0}})
                          .endpoint_vars(),
                  y1);

    ASSERT_NE(y0.node(), y1.node());
    HostTensorND host_y0, host_y1;
    auto func = graph->compile(
            {make_callback_copy(y0, host_y0), make_callback_copy(y1, host_y1)});
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(host_y0, host_y1, 1e-5);

    auto y1elem = gopt::as_elem_opr(y1.node(), opr::Elemwise::Mode::ADD);
    ASSERT_TRUE(y1elem);
    auto ymul0 = gopt::as_elem_opr(y1elem->input(0), opr::Elemwise::Mode::MUL),
         ymul1 = gopt::as_elem_opr(y1elem->input(1), opr::Elemwise::Mode::MUL);
    ASSERT_TRUE(ymul0);
    ASSERT_TRUE(ymul1);
    auto yconv = ymul0->input(0)->owner_opr();
    if (!yconv->same_type<opr::Convolution>()) {
        yconv = ymul0->input(1)->owner_opr();
    }
    ASSERT_TRUE(yconv->same_type<opr::Convolution>());
    if (ymul1->input(0) != yconv->output(0)) {
        ASSERT_EQ(yconv->output(0), ymul1->input(1));
    }
    ASSERT_EQ(x.node(), yconv->input(0));
}

TEST(TestGoptInference, ParamFuseBiasMerge) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp)).rename(name);
    };
    auto x = mkvar("x", {6, 3, 8, 8}), w1 = mkcvar("w1", {4, 3, 3, 3}),
         w2 = mkcvar("w2", {4, 3, 3, 3}), b1 = mkcvar("b1", {1, 4, 1, 1}),
         b2 = mkcvar("b2", {1, 4, 1, 1}),
         y1 = opr::Convolution::make(x, w1) + b1,
         y2 = opr::Convolution::make(x, w2) + b2, y = y1 + y2;

    SymbolVar y_opt;
    unpack_vector(gopt::optimize_for_inference({y}), y_opt);

    HostTensorND host_y, host_y_opt;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_y, host_y_opt);

    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(
                    output_file("TestGoptInference.ParamFuseConvMerge.json"));

    auto chain = gopt::extract_opr_leaves(
            y_opt.node(), [](cg::OperatorNodeBase* opr) {
                return gopt::as_elem_opr(opr, opr::Elemwise::Mode::ADD);
            });
    ASSERT_EQ(3u, chain.size());
}

TEST(TestGoptInference, Float16IOFloat32Compute) {
    constexpr size_t INP_H = 10, INP_W = 10;
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };
    graph->options().graph_opt_level = 0;
    auto a = mkvar("a", {1, 4, INP_H, INP_W}),
         s0 = mkvar("s0", {20, 3, INP_H, INP_W}),
         s1 = mkvar("s1", {4, 3, 1, 1});
    auto b = opr::Convolution::make(s0, s1, {}, {});
    auto y = a + b;
    y = opr::Concat::make({y, -y}, 0);
    y = opr::Reduce::make(y, {}, y.make_scalar(1));
    SymbolVar y_opt;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_f16_io_f32_comp();
    unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    ASSERT_EQ(y_opt.dtype(), dtype::Float32());

    HostTensorND host_y, host_y_opt;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}

TEST(TestGoptInference, Float16IOFloat32ComputeDeConv) {
    constexpr size_t INP_H = 10, INP_W = 10;
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };
    graph->options().graph_opt_level = 0;

    auto s0 = mkvar("s0", {5, 5, 3, 3}),
         s1 = mkvar("s1", {1, 5, INP_H, INP_W});
    auto y = opr::ConvolutionBackwardData::make(s0, s1, {}, {});
    SymbolVar y_opt;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_f16_io_f32_comp();
    unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    ASSERT_EQ(find_opr<opr::ConvolutionBackwardData>(y_opt).param().compute_mode,
              opr::ConvBias::Param::ConvBias::ComputeMode::FLOAT32);
    ASSERT_EQ(y_opt.dtype(), dtype::Float32());

    HostTensorND host_y, host_y_opt;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-2);
}

TEST(TestGoptInference, Float16IOFloat32ComputeWarpPerspective) {
    constexpr size_t INP_H = 10, INP_W = 10, N = 2;
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };
    graph->options().graph_opt_level = 0;
    auto a = mkvar("a", {N, 4, INP_H, INP_W});
    float value1 = M_PI, value2 = 0.6;
    auto gen_mat = [&](HostTensorND& mat) {
        auto ptr = mat.ptr<float>();
        for (size_t i = 0; i < N; ++i) {
            auto rot = value1, scale = value2, sheer = value1, dy = value2,
                 dx = value2, ky = value2, kx = value2, kb = value2;
            ptr[0] = ptr[4] = cos(rot) * scale;
            ptr[1] = -(ptr[3] = sin(rot) * scale);
            ptr[3] *= sheer;
            ptr[4] *= sheer;
            ptr[2] = dx;
            ptr[5] = dy;
            ptr[6] = kx;
            ptr[7] = ky;
            ptr[8] = kb;
            ptr += 9;
        }
        mgb_assert(ptr == mat.ptr<float>() + mat.shape().total_nr_elems());
    };
    auto mat_host = std::make_shared<HostTensorND>(
            a.node()->comp_node(), TensorShape{N, 3, 3}, dtype::Float32());
    gen_mat(*mat_host);
    auto mat = opr::Host2DeviceCopy::make(*graph, mat_host).rename("mat");
    TensorShape out_shp{20, 20};
    auto y = opr::WarpPerspective::make(a, mat, out_shp);
    SymbolVar y_opt;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_f16_io_f32_comp();
    unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    ASSERT_EQ(y_opt.dtype(), dtype::Float32());
    HostTensorND host_y, host_y_opt;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}

TEST(TestGoptInference, Float16IOFloat32ComputeRemap) {
    auto cn = CompNode::load("cpu1");
    constexpr size_t INP_H = 10, INP_W = 10, N = 2;
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name);
    };
    graph->options().graph_opt_level = 0;
    auto a = mkvar("a", {N, 4, INP_H, INP_W});
    auto gen_map = [&](HostTensorND& mat) {
        auto ptr = mat.ptr<float>();
        for (size_t n = 0; n < N; ++n) {
            for (int h = 0; h < 5; ++h) {
                for (int w = 0; w < 5; ++w) {
                    *ptr++ = (h * 5 * 2) + 5 * 2 + 0;
                    *ptr++ = (h * 5 * 2) + 5 * 2 + 1;
                }
            }
        }
        mgb_assert(ptr == mat.ptr<float>() + mat.shape().total_nr_elems());
    };
    auto map_host = std::make_shared<HostTensorND>(
            a.node()->comp_node(), TensorShape{N, 5, 5, 2}, dtype::Float32());
    gen_map(*map_host);
    auto map = opr::Host2DeviceCopy::make(*graph, map_host).rename("map");
    auto y = opr::Remap::make(a, map);
    SymbolVar y_opt;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_f16_io_f32_comp();
    unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    ASSERT_EQ(y_opt.dtype(), dtype::Float32());
    HostTensorND host_y, host_y_opt;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}

TEST(TestGoptInference, Uint8IOFloat16ComputeWarpPerspective) {
    constexpr size_t INP_H = 10, INP_W = 10, N = 2;
    HostTensorGenerator<dtype::Uint8> gen_uint8;
    auto graph = ComputingGraph::make();
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen_uint8(shp)).rename(name);
    };
    graph->options().graph_opt_level = 0;
    auto a = mkvar("a", {N, 4, INP_H, INP_W});
    float value1 = M_PI, value2 = 0.6;
    auto gen_mat = [&](HostTensorND& mat) {
        auto ptr = mat.ptr<float>();
        for (size_t i = 0; i < N; ++i) {
            auto rot = value1, scale = value2, sheer = value1, dy = value2,
                 dx = value2, ky = value2, kx = value2, kb = value2;
            ptr[0] = ptr[4] = cos(rot) * scale;
            ptr[1] = -(ptr[3] = sin(rot) * scale);
            ptr[3] *= sheer;
            ptr[4] *= sheer;
            ptr[2] = dx;
            ptr[5] = dy;
            ptr[6] = kx;
            ptr[7] = ky;
            ptr[8] = kb;
            ptr += 9;
        }
        mgb_assert(ptr == mat.ptr<float>() + mat.shape().total_nr_elems());
    };
    auto mat_host = std::make_shared<HostTensorND>(
            a.node()->comp_node(), TensorShape{N, 3, 3}, dtype::Float32());
    gen_mat(*mat_host);
    auto mat = opr::Host2DeviceCopy::make(*graph, mat_host).rename("mat");
    TensorShape out_shp{20, 20};
    auto y = opr::WarpPerspective::make(a, mat, out_shp);
    SymbolVar y_opt;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_f16_io_comp();
    unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    ASSERT_EQ(y_opt.dtype(), dtype::Uint8());
    HostTensorND host_y, host_y_opt;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}

TEST(TestGoptInference, Float32TOFloat16) {
    CompNode cn = CompNode::load("cpu0");
    HostTensorGenerator<> gen(0, 1, 0);
    auto host_x0 = gen({1, 4, 16, 8}, cn), host_x1 = gen({2, 3, 16, 8}, cn),
         host_x2 = gen({4, 3, 1, 1}, cn);
    auto graph = ComputingGraph::make();

    auto make_f32_to_f16_graph = [&]() {
        graph->options().graph_opt_level = 0;

        auto d0 = opr::Host2DeviceCopy::make(*graph, host_x0),
             d1 = opr::Host2DeviceCopy::make(*graph, host_x1),
             d2 = opr::SharedDeviceTensor::make(*graph, *host_x2);

        auto b = opr::Convolution::make(d1, d2, {}, {});
        auto y = d0 + b;
        y = opr::Reduce::make(y, {}, y.make_scalar(1));

        SymbolVar y_opt;
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_f16_io_comp();
        unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
        return y_opt;
    };

    auto make_f16_graph = [&]() {
        auto d0 = opr::TypeCvt::make(
                     opr::Host2DeviceCopy::make(*graph, host_x0),
                     dtype::Float16{}),
             d1 = opr::TypeCvt::make(
                     opr::Host2DeviceCopy::make(*graph, host_x1),
                     dtype::Float16{}),
             d2 = opr::TypeCvt::make(
                     opr::SharedDeviceTensor::make(*graph, *host_x2),
                     dtype::Float16{});

        auto b = opr::Convolution::make(d1, d2, {}, {});
        SymbolVar y = d0 + b;
        y = opr::Reduce::make(y, {}, y.make_scalar(1));
        y = opr::TypeCvt::make(y, dtype::Float32{});

        return y;
    };

    auto y_opt = make_f32_to_f16_graph();
    auto y = make_f16_graph();
    ASSERT_EQ(y_opt.dtype(), dtype::Float32{});
    ASSERT_EQ(y.dtype(), dtype::Float32{});

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}

TEST(TestGoptInference, Float32TOFloat16C32) {
    CompNode cn = CompNode::load("cpu0");
    HostTensorGenerator<> gen(0, 1, 0);
    auto host_x0 = gen({1, 4, 1, 1}, cn), host_x1 = gen({2, 3, 16, 8}, cn),
         host_x2 = gen({4, 3, 1, 1}, cn);
    auto graph = ComputingGraph::make();

    auto make_f32_to_f16_graph = [&]() {
        graph->options().graph_opt_level = 0;

        auto d0 = opr::Host2DeviceCopy::make(*graph, host_x0),
             d1 = opr::Host2DeviceCopy::make(*graph, host_x1),
             d2 = opr::SharedDeviceTensor::make(*graph, *host_x2);

        auto y = opr::ConvBias::make(d1, d2, d0);
        y = opr::Reduce::make(y, {}, y.make_scalar(1));

        SymbolVar y_opt;
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_f16_io_f32_comp();
        unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
        return y_opt;
    };

    auto make_f16_graph = [&]() {
        auto d0 = opr::TypeCvt::make(
                     opr::TypeCvt::make(
                             opr::Host2DeviceCopy::make(*graph, host_x0),
                             dtype::Float16{}),
                     dtype::Float32{}),
             d1 = opr::TypeCvt::make(
                     opr::TypeCvt::make(
                             opr::Host2DeviceCopy::make(*graph, host_x1),
                             dtype::Float16{}),
                     dtype::Float32{}),
             d2 = opr::TypeCvt::make(
                     opr::TypeCvt::make(
                             opr::SharedDeviceTensor::make(*graph, *host_x2),
                             dtype::Float16{}),
                     dtype::Float32{});

        auto y = opr::ConvBias::make(d1, d2, d0);
        y = opr::Reduce::make(y, {}, y.make_scalar(1));
        y = opr::TypeCvt::make(opr::TypeCvt::make(y, dtype::Float16{}),
                               dtype::Float32{});

        return y;
    };

    auto y_opt = make_f32_to_f16_graph();
    auto y = make_f16_graph();
    ASSERT_EQ(find_opr<opr::ConvBias>(y_opt).param().compute_mode,
              opr::ConvBias::Param::ConvBias::ComputeMode::FLOAT32);
    ASSERT_EQ(y_opt.dtype(), dtype::Float32{});
    ASSERT_EQ(y.dtype(), dtype::Float32{});

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}

TEST(TestGoptInference, Float32TOFloat16EndpointElemwise) {
    CompNode cn = CompNode::load("cpu0");
    HostTensorGenerator<> gen(0, 1, 0);
    auto host_x0 = gen({1, 4, 16, 8}, cn), host_x1 = gen({2, 3, 16, 8}, cn),
         host_x2 = gen({4, 3, 1, 1}, cn);
    auto graph = ComputingGraph::make();

    auto make_f32_to_f16_graph = [&]() {
        graph->options().graph_opt_level = 0;

        auto d0 = opr::Host2DeviceCopy::make(*graph, host_x0),
             d1 = opr::Host2DeviceCopy::make(*graph, host_x1),
             d2 = opr::SharedDeviceTensor::make(*graph, *host_x2);

        auto b = opr::Convolution::make(d1, d2, {}, {});
        auto y = d0 + b;

        SymbolVar y_opt;
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_f16_io_comp();
        unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
        return y_opt;
    };

    auto make_f16_graph = [&]() {
        auto d0 = opr::TypeCvt::make(
                     opr::Host2DeviceCopy::make(*graph, host_x0),
                     dtype::Float16{}),
             d1 = opr::TypeCvt::make(
                     opr::Host2DeviceCopy::make(*graph, host_x1),
                     dtype::Float16{}),
             d2 = opr::TypeCvt::make(
                     opr::SharedDeviceTensor::make(*graph, *host_x2),
                     dtype::Float16{});

        auto b = opr::Convolution::make(d1, d2, {}, {});
        SymbolVar y = d0 + b;
        y = opr::TypeCvt::make(y, dtype::Float32{});

        return y;
    };

    auto y_opt = make_f32_to_f16_graph();
    auto y = make_f16_graph();
    ASSERT_EQ(y_opt.dtype(), dtype::Float32{});
    ASSERT_EQ(y.dtype(), dtype::Float32{});

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}

TEST(TestGoptInference, Float32TOFloat16Linspace) {
    CompNode cn = CompNode::load("cpu0");
    HostTensorGenerator<> gen(0, 1, 0);
    auto host_x = gen({3, 1}, cn);
    auto graph = ComputingGraph::make();

    auto make_f32_to_f16_graph = [&]() {
        graph->options().graph_opt_level = 0;

        auto x = opr::Host2DeviceCopy::make(*graph, host_x);
        auto xshp = opr::GetVarShape::make(x);

        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto lin = opr::Linspace::make(cv(0), sub(0) - 1, sub(0), {}, {});
        auto shp = opr::Concat::make({sub(1), sub(0)}, 0);
        auto y = opr::Reshape::make(lin, shp);
        auto mm = opr::MatrixMul::make(x, y);

        SymbolVar mm_opt;
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_f16_io_comp();
        unpack_vector(gopt::optimize_for_inference({mm}, options), mm_opt);
        return mm_opt;
    };

    auto make_f16_graph = [&]() {
        auto x = opr::TypeCvt::make(opr::Host2DeviceCopy::make(*graph, host_x),
                                    dtype::Float16());
        auto xshp = opr::GetVarShape::make(x);

        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto lin = opr::Linspace::make(cv(0), sub(0) - 1, sub(0), {}, {});
        lin = opr::TypeCvt::make(lin, dtype::Float16());
        auto shp = opr::Concat::make({sub(1), sub(0)}, 0);
        auto y = opr::Reshape::make(lin, shp);
        auto mm = opr::MatrixMul::make(x, y);

        mm = opr::TypeCvt::make(mm, dtype::Float32{});

        return mm;
    };

    auto y_opt = make_f32_to_f16_graph();
    auto y = make_f16_graph();
    ASSERT_EQ(y_opt.dtype(), dtype::Float32{});
    ASSERT_EQ(y.dtype(), dtype::Float32{});

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}

TEST(TestGoptInference, Float32TOFloat16Endpoints) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();

    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };

    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp)).rename(name);
    };

    graph->options().graph_opt_level = 0;
    opr::Convolution::Param param;
    param.pad_h = param.pad_w = 0;

    auto x = mkvar("x", {8, 8, 8, 8}), y = mkvar("y", {8, 8, 8, 8}),
         w = mkcvar("w", {4, 8, 3, 3}),
         z = opr::Convolution::make(x + y, w, param);

    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_f16_io_f32_comp();
    SymbolVarArray out = gopt::optimize_for_inference({x + y, z}, options);

    ASSERT_EQ(out[0].dtype(), dtype::Float32());
    ASSERT_EQ(out[1].dtype(), dtype::Float32());
    ASSERT_EQ(out[0].node()->owner_opr()->input(0)->dtype(), dtype::Float16());
    ASSERT_EQ(out[1].node()->owner_opr()->input(0)->dtype(), dtype::Float16());
}

TEST(TestGoptInference, ConvertFormatNHWCD4) {
    // hwcd4 is only supported in naive handle
    NaiveMegDNNHandleScope naive_megdnn_handle;

    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                .rename(name);
    };

    auto host_x = gen({8, 8, 8, 8}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    opr::Convolution::Param param;
    param.pad_h = param.pad_w = 0;
    auto w1 = mkcvar("w1", {4, 8, 3, 3}),
         conv = opr::Convolution::make(x, w1, param);
    auto shape_of = opr::GetVarShape::make(conv);
    auto subtensor = opr::Subtensor::make(
            shape_of, {opr::Subtensor::AxisIndexer::make_interval(
                              0, x.make_scalar(2), None, x.make_scalar(1))});

    opr::Resize::Param param_resize;
    param_resize.format = opr::Resize::Param::Format::NCHW;
    auto resize = opr::ResizeForward::make(conv, subtensor * 2, param_resize);
    auto mat = mkcvar("mat", {8, 3, 3}),
         warp = opr::WarpPerspectiveForward::make(
                 resize, mat, nullptr, cg::var_from_tensor_shape(x, {4, 4}));

    auto b = mkvar("b", {1, 4, 1, 1}),
         elem = opr::Elemwise::make({warp + b},
                                    opr::Elemwise::Param::Mode::RELU);
    param.pad_h = param.pad_w = 1;
    auto w2 = mkcvar("w2", {4, 4, 3, 3}),
         y = opr::Convolution::make(elem, w2, param);

    SymbolVar y_opt;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_nhwcd4();
    unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);

    ASSERT_EQ(opr::Convolution::Param::Format::NHWCD4,
              find_opr<opr::Convolution>(y_opt).param().format);

    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(
                    output_file("TestGoptInference.ConvertFormatNHWCD4.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);

    *host_x = *gen({8, 8, 16, 16}, cn);
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}

TEST(TestGoptInference, ConvertFormatNHWCD4LOCAL) {
    // hwcd4 is only supported in naive handle
    NaiveMegDNNHandleScope naive_megdnn_handle;

    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                .rename(name);
    };

    auto host_x = gen({2, 8, 8, 16}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    opr::Convolution::Param param;
    param.pad_h = param.pad_w = 1;
    auto w1 = mkcvar("w1", {4, 8, 3, 3}),
         conv1 = opr::Convolution::make(x, w1, param);

    auto w2 = mkcvar("w2", {8, 16, 4, 3, 3, 4}),
         local = opr::Local::make(conv1, w2, param);

    auto w3 = mkcvar("w3", {4, 4, 3, 3}),
         conv2 = opr::Convolution::make(local, w3, param);

    opr::GroupLocal::Param param_group_local;
    param_group_local.pad_h = param_group_local.pad_w = 1;
    auto w4 = mkcvar("w4", {2, 8, 16, 2, 3, 3, 2}),
         group_local = opr::GroupLocal::make(conv2, w4, param_group_local);

    auto w5 = mkcvar("w5", {4, 4, 3, 3}),
         y = opr::Convolution::make(group_local, w5, param);

    SymbolVar y_opt;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_nhwcd4();
    unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);

    ASSERT_EQ(opr::Convolution::Param::Format::NHWCD4,
              find_opr<opr::Convolution>(y_opt).param().format);

    ASSERT_EQ(opr::Local::Param::Format::NCHW,
              find_opr<opr::Local>(y_opt).param().format);

    ASSERT_EQ(opr::GroupLocal::Param::Format::NCHW,
              find_opr<opr::GroupLocal>(y_opt).param().format);

    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(output_file(
                    "TestGoptInference.ConvertFormatNHWCD4LOCAL.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}

TEST(TestGoptInference, ConvertFormatNHWCD4Deconv) {
    // hwcd4 is only supported in naive handle
    NaiveMegDNNHandleScope naive_megdnn_handle;

    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                .rename(name);
    };

    auto host_x = gen({8, 8, 8, 8}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    opr::Convolution::Param param;
    param.pad_h = param.pad_w = 0;
    auto w0 = mkcvar("w1", {4, 8, 2, 2}),
         conv = opr::Convolution::make(x, w0, param);

    auto w1 = mkcvar("w1", {4, 1, 2, 2}),
         y = opr::ConvolutionBackwardData::make(w1, conv, param, {}, {});

    SymbolVar y_opt;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_nhwcd4();
    unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);

    ASSERT_EQ(opr::Convolution::Param::Format::NCHW,
              find_opr<opr::ConvolutionBackwardData>(y_opt).param().format);
    ASSERT_EQ(opr::Convolution::Param::Format::NHWCD4,
              find_opr<opr::Convolution>(y_opt).param().format);

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}
TEST(TestGoptInference, ConvertFormatNHWCD4Qint8) {
    // hwcd4 is only supported in naive handle
    NaiveMegDNNHandleScope naive_megdnn_handle;

    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    auto mkcvar = [&](const char* name, const TensorShape& shp,
                      const DType& dtype) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                        .rename(name),
                dtype);
    };

    auto host_x = gen({8, 8, 8, 8}, cn);
    auto _x = opr::Host2DeviceCopy::make(*graph, host_x),
         x = opr::TypeCvt::make(_x, dtype::QuantizedS8(0.2f));

    opr::ConvBias::Param param;
    param.pad_h = param.pad_w = 0;
    auto w = mkcvar("w", {4, 8, 3, 3}, dtype::QuantizedS8(0.1f)),
         b = mkcvar("b", {1, 4, 1, 1}, dtype::QuantizedS32(0.02f)),
         y = opr::ConvBias::make(x, w, b, param, {},
                                 OperatorNodeConfig{dtype::QuantizedS8(0.2f)});

    SymbolVar y_opt;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_nhwcd4();
    unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);

    ASSERT_EQ(opr::ConvBias::Param::Format::NHWCD4,
              find_opr<opr::ConvBias>(y_opt).param().format);

    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(output_file(
                    "TestGoptInference.ConvertFormatNHWCD4Qint8.json"));
    auto float_y = opr::TypeCvt::make(y, dtype::Float32()),
         float_y_opt = opr::TypeCvt::make(y_opt, dtype::Float32());

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile({make_callback_copy(float_y, host_y),
                                make_callback_copy(float_y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}
TEST(TestGoptInference, ConvertFormatPadIC) {
    // hwcd4 is only supported in naive handle
    NaiveMegDNNHandleScope naive_megdnn_handle;

    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                .rename(name);
    };

    auto host_inp1 = gen({1, 6, 128, 128}, cn),
         host_inp2 = gen({1, 6, 256, 256}, cn);
    auto inp1 = opr::Host2DeviceCopy::make(*graph, host_inp1),
         inp2 = opr::Host2DeviceCopy::make(*graph, host_inp2);

    auto shape_tmp = mkcvar("tmp", {256, 256});
    auto shape_of = opr::GetVarShape::make(shape_tmp);
    opr::Resize::Param param_resize;
    param_resize.format = opr::Resize::Param::Format::NCHW;
    auto resize = opr::ResizeForward::make(inp1, shape_of, param_resize);

    auto concat = opr::Concat::make({inp2, resize}, 1);

    opr::Convolution::Param param;
    param.pad_h = param.pad_w = 1;
    param.sparse = opr::Convolution::Param::Sparse::DENSE;
    auto w1 = mkcvar("w1", {12, 12, 3, 3});
    auto y = opr::Convolution::make(concat, w1, param);
    SymbolVar y_opt;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_nhwcd4();
    unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}

TEST(TestGoptInference, ConvertBatchNormPass) {
    auto cn = CompNode::load("cpu0");

    HostTensorGenerator<> gen(0, 1, 0);
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                .rename(name);
    };
    using Param = opr::BatchNorm::Param;
    Param param(Param::ParamDim::DIM_1C11, Param::FwdMode::INFERENCE);
    TensorShape shp = {1, 3, 1, 1};
    auto x = mkvar("x", {2, 3, 16, 24}), scale = mkcvar("scale", shp),
         bias = mkcvar("bias", shp), mean = mkcvar("mean", shp);
    auto host_variance = gen(shp, cn);
    for (size_t i = 0; i < shp.total_nr_elems(); ++i) {
        host_variance->ptr<float>()[i] =
                std::abs(host_variance->ptr<float>()[i]);
    }
    auto variance = opr::SharedDeviceTensor::make(*graph, *host_variance)
                            .rename("variance");
    auto y = opr::BatchNorm::make(x, scale, bias, mean, variance, param)[4];
    SymbolVar y_opt;
    unpack_vector(gopt::optimize_for_inference(
                          {y}, gopt::OptimizeForInferenceOptions{}),
                  y_opt);
    ASSERT_EQ(0u, find_opr_num<opr::BatchNorm>(y_opt));
    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(
                    output_file("TestGoptInference.ConvertBatchNormPass.json"));

    HostTensorND host_y, host_y_opt;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-5);
}

TEST(TestGoptInference, ConvBiasNonlinearityFusePass) {
    // hwcd4 is only supported in naive handle
    NaiveMegDNNHandleScope naive_megdnn_handle;

    auto cn = CompNode::load("cpu0");

    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                .rename(name);
    };
    opr::Convolution::Param param;
    auto x = mkvar("x", {5, 8, 16, 24}), w1 = mkcvar("w1", {4, 8, 1, 1}),
         w2 = mkcvar("w2", {4, 4, 3, 3}), b1 = mkcvar("b1", {1, 4, 1, 1}),
         b2 = mkcvar("b2", {1, 4, 1, 1}), w3 = mkcvar("w3", {8, 4, 1, 1}),
         y_cut = opr::Convolution::make(x, w1, param),
         y1 = opr::Elemwise::make({y_cut + b1},
                                  opr::Elemwise::Param::Mode::RELU);
    param.pad_w = param.pad_h = 1;
    auto y2 = opr::Elemwise::make({opr::Convolution::make(y1, w2, param) + b2},
                                  opr::Elemwise::Param::Mode::SIGMOID);
    param.pad_w = param.pad_h = 0;
    auto y3 = opr::Convolution::make(y2, w3, param), y_tmp = y3 + x,
         y_expand =
                 opr::Elemwise::make({y_cut}, opr::Elemwise::Param::Mode::RELU),
         y_y = opr::Convolution::make(y_expand, w3, param), y = y_y + y_tmp;
    SymbolVar y_opt;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_nhwcd4().enable_fuse_conv_bias_nonlinearity();
    unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    ASSERT_EQ(3u, find_opr<opr::ConvBias>(y_opt).input().size());
    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(output_file(
                    "TestGoptInference.FuseConvBiasNonlinPass.json"));

    HostTensorND host_y, host_y_opt;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-4);
}

TEST(TestGoptInference, ParamMerge) {
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto var0 = opr::SharedDeviceTensor::make(*graph, *gen({2, 3}, cns[0])),
         var1 = opr::SharedDeviceTensor::make(*graph, *gen({1, 3}, cns[1])),
         y = var0 + opr::Copy::make(var1, {cns[0]});
    HostTensorND y_expected_val;
    graph->compile({make_callback_copy(y, y_expected_val)})->execute();

    SymbolVar y_opt;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ParamMergePass>()
                          .apply({{y}})
                          .endpoint_vars(),
                  y_opt);
    auto opr = y_opt.node()->owner_opr();
    ASSERT_EQ(2u, opr->input().size());
    ASSERT_EQ(2u,
              find_opr<opr::MultipleDeviceTensorHolder>(y_opt).output().size());
    HostTensorND y_got_val;
    graph->compile({make_callback_copy(y_opt, y_got_val)})->execute();
    MGB_ASSERT_TENSOR_EQ(y_expected_val, y_got_val);
}

TEST(TestGoptInference, ParamMergeFormat) {
    auto cns = load_multiple_xpus(2);

    auto make_dv = [](const HostTensorND& hv) {
        TensorLayout layout{hv.layout(), hv.layout().dtype,
                            megdnn::Image2DPack4TensorFormat::make_raw(1, 64)};
        auto ret = std::make_shared<DeviceTensorND>(hv.comp_node(), layout);
        ret->copy_from_fixlayout(hv).sync();
        return ret;
    };

    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto var0 = opr::SharedDeviceTensorWithFormat::make(
                 *graph, make_dv(*gen({2, 32}, cns[0]))),
         var1 = opr::SharedDeviceTensorWithFormat::make(
                 *graph, make_dv(*gen({1, 32}, cns[1]))),
         y = var0 + opr::Copy::make(var1, {cns[0]});
    HostTensorND y_expected_val;
    graph->compile({make_callback_copy(y, y_expected_val)})->execute();

    SymbolVar y_opt;
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::ParamMergePass>()
                          .apply({{y}})
                          .endpoint_vars(),
                  y_opt);
    auto opr = y_opt.node()->owner_opr();
    ASSERT_EQ(2u, opr->input().size());
    ASSERT_EQ(2u, find_opr<opr::MultipleDeviceTensorWithFormatHolder>(y_opt)
                          .output()
                          .size());
    HostTensorND y_got_val;
    graph->compile({make_callback_copy(y_opt, y_got_val)})->execute();
    MGB_ASSERT_TENSOR_EQ(y_expected_val, y_got_val);
}

#if MGB_ENABLE_FASTRUN
TEST(TestGoptInference, AlgoProfile) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_x = gen({4, 3, 8, 9}), host_y = gen({2, 3, 3, 3});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y),
         z = opr::Convolution::make(x, y);
    auto&& conv = z.node()->owner_opr()->cast_final_safe<opr::Convolution>();
    using S = opr::Convolution::ExecutionPolicy::Strategy;
    ASSERT_EQ(S::HEURISTIC, conv.execution_policy_transient().strategy);
    gopt::enable_opr_algo_profiling_inplace({z + 2.3f});
    ASSERT_EQ(S::PROFILE, conv.execution_policy().strategy);
}
#endif

TEST(TestGoptInference, ProfileCache) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_x = gen({4, 3, 8, 9}), host_y = gen({2, 3, 3, 3});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y),
         z = opr::Convolution::make(x, y);
    auto&& conv = z.node()->owner_opr()->cast_final_safe<opr::Convolution>();
    using S = opr::Convolution::ExecutionPolicy::Strategy;
    ASSERT_EQ(S::HEURISTIC, conv.execution_policy_transient().strategy);
    gopt::enable_opr_use_profiling_cache_inplace({z + 2.3f});
    ASSERT_EQ(S::PROFILE_HEURISTIC, conv.execution_policy().strategy);
}

TEST(TestGoptInference, AlgoWorkspaceLimit) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_x = gen({4, 3, 8, 9}), host_y = gen({2, 3, 3, 3});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = opr::Host2DeviceCopy::make(*graph, host_y),
         z = opr::Convolution::make(x, y);
    auto&& conv = z.node()->owner_opr()->cast_final_safe<opr::Convolution>();
    ASSERT_EQ(std::numeric_limits<uint64_t>::max(),
              conv.execution_policy_transient().workspace_limit);
    gopt::set_opr_algo_workspace_limit_inplace({z + 2.3f}, 10000u);
    ASSERT_EQ(10000u, conv.execution_policy().workspace_limit);
}

TEST_PASS(FuseConvBiasNonlinPass, Basic) {
    auto cn = CompNode::load("xpux");

    HostTensorGenerator<dtype::Int8> gen;
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

    for (auto format : {opr::Convolution::Param::Format::NCHW,
                        opr::Convolution::Param::Format::NHWC,
                        opr::Convolution::Param::Format::NCHW4}) {
        opr::Convolution::Param param;
        param.format = format;
        SymbolVar x, w, b;
        if (format == opr::Convolution::Param::Format::NHWC) {
            x = mkvar("x", {20, 20, 20, 4}, dtype::QuantizedS8(2.5f)),
            w = mkcvar("w1", {24, 1, 1, 4}, dtype::QuantizedS8(2.5f)),
            b = mkcvar("b", {1, 1, 1, 24}, dtype::QuantizedS32(6.25f));
        } else if (format == opr::Convolution::Param::Format::NCHW) {
            x = mkvar("x", {20, 4, 20, 20}, dtype::QuantizedS8(2.5f)),
            w = mkcvar("w1", {24, 4, 1, 1}, dtype::QuantizedS8(2.5f)),
            b = mkcvar("b", {1, 24, 1, 1}, dtype::QuantizedS32(6.25f));
        } else {
            mgb_assert(format == opr::Convolution::Param::Format::NCHW4);
            x = mkvar("x", {20, 1, 20, 20, 4}, dtype::QuantizedS8(2.5f)),
            w = mkcvar("w1", {24, 1, 1, 1, 4}, dtype::QuantizedS8(2.5f)),
            b = mkcvar("b", {1, 6, 1, 1, 4}, dtype::QuantizedS32(6.25f));
        }
        auto y = opr::Convolution::make(x, w, param);
        y = opr::Elemwise::make({y + b}, opr::Elemwise::Param::Mode::RELU);
        y = opr::TypeCvt::make(y, dtype::QuantizedS8(2.5f));

        opr::ConvBias::Param conv_bias_param;
        conv_bias_param.format = format;
        conv_bias_param.nonlineMode = opr::ConvBias::Param::NonlineMode::RELU;
        auto concret_y = opr::ConvBias::make(
                x, w, b, conv_bias_param, {},
                OperatorNodeConfig{dtype::QuantizedS8(2.5f)});

        check(concret_y, y);
    }
}


#if MGB_CUDA

TEST(TestEnableTensorCore, SmallInputShape) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 75) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 75);
        return;
    }

    HostTensorGenerator<dtype::Int8> gen;
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

    auto x = mkvar("x", {32, 16, 4, 8, 4}, dtype::QuantizedS8(2.5f)),
         w = mkcvar("w1", {64, 16, 3, 3, 4}, dtype::QuantizedS8(2.5f)),
         b = mkcvar("b", {1, 16, 1, 1, 4}, dtype::QuantizedS32(6.25f)),
         z = mkcvar("b1", {32, 16, 2, 4, 4}, dtype::QuantizedS8(2.5f));
    opr::ConvBias::Param param;
    param.format = opr::ConvBias::Param::Format::NCHW4;
    param.nonlineMode = opr::ConvBias::Param::NonlineMode::RELU;
    param.stride_h = param.stride_w = 2;
    param.pad_h = param.pad_w = 1;

    auto y = opr::ConvBias::make(x, w, b, z, param, {},
                                 OperatorNodeConfig{dtype::QuantizedS8(2.5f)});
    y = opr::ConvBias::make(y, w, b, param, {},
                            OperatorNodeConfig{dtype::QuantizedS8(2.5f)});
    y = opr::TypeCvt::make(y, dtype::Float32());

    SymbolVar y_opt;
    SymbolVar y_no_tc;
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_nchw32().enable_fuse_conv_bias_nonlinearity();
        unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    }
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_fuse_conv_bias_nonlinearity();
        unpack_vector(gopt::optimize_for_inference({y}, options), y_no_tc);
    }
    auto nr_dimshuffle = find_opr_num<mgb::opr::Dimshuffle>(y_opt);
    ASSERT_EQ(2u, nr_dimshuffle);
    HostTensorND host_y, host_y_opt;
    auto func = graph->compile({make_callback_copy(y_no_tc, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_y, host_y_opt);
}

TEST(TestEnableTensorCore, Nchw4Nchw) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 75) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 75);
        return;
    }

    HostTensorGenerator<dtype::Int8> gen;
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

    auto mkshape = [](opr::ConvBias::Param::Format format, size_t N, size_t C,
                      size_t H, size_t W) -> TensorShape {
        mgb_assert(C % 4 == 0);
        if (format == opr::ConvBias::Param::Format::NCHW4) {
            return {N, C / 4, H, W, 4};
        } else {
            mgb_assert(format == opr::ConvBias::Param::Format::NCHW);
            return {N, C, H, W};
        }
    };

    for (auto format : {opr::ConvBias::Param::Format::NCHW,
                        opr::ConvBias::Param::Format::NCHW4}) {
        auto x = mkvar("x", mkshape(format, 32, 64, 16, 16),
                       dtype::QuantizedS8(2.5f)),
             w = mkcvar("w1", mkshape(format, 64, 64, 3, 3),
                        dtype::QuantizedS8(2.5f)),
             b = mkcvar("b", mkshape(format, 1, 64, 1, 1),
                        dtype::QuantizedS32(6.25f)),
             z = mkcvar("b1", mkshape(format, 32, 64, 8, 8),
                        dtype::QuantizedS8(2.5f));
        opr::ConvBias::Param param;
        param.format = format;
        param.nonlineMode = opr::ConvBias::Param::NonlineMode::RELU;
        param.stride_h = param.stride_w = 2;
        param.pad_h = param.pad_w = 1;

        auto y = opr::ConvBias::make(
                x, w, b, z, param, {},
                OperatorNodeConfig{dtype::QuantizedS8(2.5f)});
        y = opr::ConvBias::make(y, w, b, param, {},
                                OperatorNodeConfig{dtype::QuantizedS8(2.5f)});
        y = opr::TypeCvt::make(y, dtype::Float32());

        SymbolVar y_opt;
        SymbolVar y_no_tc;
        {
            auto options = gopt::OptimizeForInferenceOptions{};
            options.enable_nchw32().enable_fuse_conv_bias_nonlinearity();
            unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
        }
        {
            auto options = gopt::OptimizeForInferenceOptions{};
            options.enable_fuse_conv_bias_nonlinearity();
            unpack_vector(gopt::optimize_for_inference({y}, options), y_no_tc);
        }
        auto nr_dimshuffle = find_opr_num<mgb::opr::Dimshuffle>(y_opt);
        std::string json_name;
        ASSERT_EQ(2u, nr_dimshuffle);
        if (format == opr::ConvBias::Param::Format::NCHW4) {
            json_name = "TestGoptInference.Nchw4Nchw.NCHW4.json";
        } else {
            mgb_assert(format == opr::ConvBias::Param::Format::NCHW);
            json_name = "TestGoptInference.Nchw4Nchw.NCHW.json";
        }

        graph->compile({{y_opt, {}}})
                ->to_json()
                ->writeto_fpath(output_file(json_name.c_str()));
        HostTensorND host_y, host_y_opt;
        auto func = graph->compile({make_callback_copy(y_no_tc, host_y),
                                    make_callback_copy(y_opt, host_y_opt)});
        func->execute();
        MGB_ASSERT_TENSOR_EQ(host_y, host_y_opt);
    }
}

TEST(TestEnableTensorCore, ConvBiasWithZ) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 75) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 75);
        return;
    }

    HostTensorGenerator<dtype::Int8> gen;
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

    auto x = mkvar("x", {32, 16, 16, 16, 4}, dtype::QuantizedS8(2.5f)),
         w = mkcvar("w1", {64, 16, 3, 3, 4}, dtype::QuantizedS8(2.5f)),
         b = mkcvar("b", {1, 16, 1, 1, 4}, dtype::QuantizedS32(6.25f)),
         z = mkvar("b1", {32, 16, 16, 16, 4}, dtype::QuantizedS8(2.5f));
    opr::ConvBias::Param param;
    param.format = opr::ConvBias::Param::Format::NCHW4;
    param.nonlineMode = opr::ConvBias::Param::NonlineMode::RELU;
    param.stride_h = param.stride_w = 1;
    param.pad_h = param.pad_w = 1;

    auto y = opr::ConvBias::make(x, w, b, z, param, {},
                                 OperatorNodeConfig{dtype::QuantizedS8(2.5f)});
    y = opr::TypeCvt::make(y, dtype::Float32());

    SymbolVar y_opt;
    SymbolVar y_no_tc;
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_fuse_conv_bias_nonlinearity().enable_nchw32();
        unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    }
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_fuse_conv_bias_nonlinearity();
        unpack_vector(gopt::optimize_for_inference({y}, options), y_no_tc);
    }
    HostTensorND host_y, host_y_opt;
    auto func = graph->compile({make_callback_copy(y_no_tc, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_y, host_y_opt);
}

TEST(TestGoptInference, EnableTensorCore) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 75) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 75);
        return;
    }

    HostTensorGenerator<dtype::Int8> gen;
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

    auto x = mkvar("x", {32, 16, 16, 16, 4}, dtype::QuantizedS8(2.5f)),
         w = mkcvar("w1", {64, 16, 3, 3, 4}, dtype::QuantizedS8(2.5f)),
         b = mkcvar("b", {1, 16, 1, 1, 4}, dtype::QuantizedS32(6.25f)),
         b1 = mkvar("b1", {32, 16, 16, 16, 4}, dtype::QuantizedS8(2.5f));
    opr::Convolution::Param param;
    param.format = opr::Convolution::Param::Format::NCHW4;
    param.stride_h = param.stride_w = 1;
    param.pad_h = param.pad_w = 1;

    auto y = opr::Convolution::make(x, w, param);
    y = opr::Elemwise::make({y + b}, opr::Elemwise::Param::Mode::RELU);
    y = opr::TypeCvt::make(y, dtype::QuantizedS8(2.5f));

    auto y1 = y + b1, y2 = opr::Convolution::make(y, w, param),
         y3 = opr::Elemwise::make({y - b1}, opr::Elemwise::Param::Mode::RELU);
    y2 = opr::Elemwise::make({y2 + b}, opr::Elemwise::Param::Mode::RELU),
    y2 = opr::TypeCvt::make(y2, dtype::QuantizedS8(2.5f));
    auto y4 = y1 + y2 + y3;
    y4 = opr::TypeCvt::make(y4, dtype::Float32());
    SymbolVar y_opt;
    SymbolVar y_no_tc;
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_fuse_conv_bias_nonlinearity().enable_nchw32();
        unpack_vector(gopt::optimize_for_inference({y4}, options), y_opt);
    }
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_fuse_conv_bias_nonlinearity().enable_nchw32();
        unpack_vector(gopt::optimize_for_inference({y4}, options), y_no_tc);
    }
    auto nr_dimshuffle = find_opr_num<mgb::opr::Dimshuffle>(y_opt);
    ASSERT_EQ(3u, nr_dimshuffle);
    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(
                    output_file("TestGoptInference.EnableTensorCorePass.json"));

    HostTensorND host_y, host_y_opt;
    auto func = graph->compile({make_callback_copy(y_no_tc, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_y, host_y_opt);
}

TEST(FuseConvBiasZPass, BlockFuse) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 61) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 61);
        return;
    }

    HostTensorGenerator<dtype::Int8> gen;
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

    using ElemMultiMode = opr::ElemwiseMultiType::Param::Mode;
    using NonlineMode = opr::ConvBias::Param::NonlineMode;
    for (auto mode :
         {ElemMultiMode::QFUSE_ADD_RELU, ElemMultiMode::QFUSE_ADD_H_SWISH}) {
        auto x = mkvar("x", {32, 16, 16, 16, 4}, dtype::QuantizedS8(2.5f)),
             w1 = mkcvar("w1", {64, 16, 3, 3, 4}, dtype::QuantizedS8(2.5f)),
             b1 = mkcvar("b1", {1, 16, 1, 1, 4}, dtype::QuantizedS32(6.25f)),
             w2 = mkcvar("w2", {64, 16, 3, 3, 4}, dtype::QuantizedS8(2.5f)),
             b2 = mkcvar("b2", {1, 16, 1, 1, 4}, dtype::QuantizedS32(6.25f)),
             w3 = mkcvar("w3", {64, 16, 3, 3, 4}, dtype::QuantizedS8(2.5f)),
             b3 = mkcvar("b3", {1, 16, 1, 1, 4}, dtype::QuantizedS32(3.0f));
        NonlineMode nonline_mode = NonlineMode::RELU;
        if (mode == ElemMultiMode::QFUSE_ADD_H_SWISH) {
            nonline_mode = NonlineMode::H_SWISH;
        }

        opr::ConvBias::Param param;
        param.format = opr::Convolution::Param::Format::NCHW4;
        param.nonlineMode = nonline_mode;
        param.stride_h = param.stride_w = 1;
        param.pad_h = param.pad_w = 1;

        auto y1 = opr::ConvBias::make(
                x, w1, b1, param, {},
                OperatorNodeConfig{dtype::QuantizedS8(2.5f)});
        param.nonlineMode = opr::ConvBias::Param::NonlineMode::IDENTITY;
        auto y2 = opr::ConvBias::make(
                     y1, w2, b2, param, {},
                     OperatorNodeConfig{dtype::QuantizedS8(2.5f)}),
             y3 = opr::ElemwiseMultiType::make(
                     {y1, y2}, {mode},
                     OperatorNodeConfig{dtype::QuantizedS8(1.2f)});
        param.nonlineMode = nonline_mode;
        auto y4 = opr::ConvBias::make(
                     y3, w3, b3, param, {},
                     OperatorNodeConfig{dtype::QuantizedS8(2.5f)}),
             z = opr::ElemwiseMultiType::make(
                     {y3, y4}, {opr::ElemwiseMultiType::Param::Mode::QADD},
                     OperatorNodeConfig{dtype::QuantizedS8(2.5f)});
        z = opr::TypeCvt::make(z, dtype::Float32());

        //! fuse z mannually
        auto z0 = opr::ConvBias::make(
                x, w1, b1, param, {},
                OperatorNodeConfig{dtype::QuantizedS8(2.5f)});
        auto z1 = opr::ConvBias::make(
                     z0, w2, b2, z0, param, {},
                     OperatorNodeConfig{dtype::QuantizedS8(1.2f)}),
             z2 = opr::ConvBias::make(
                     z1, w3, b3, param, {},
                     OperatorNodeConfig{dtype::QuantizedS8(2.5f)}),
             z4 = opr::ElemwiseMultiType::make(
                     {z1, z2}, {opr::ElemwiseMultiType::Mode::QADD},
                     OperatorNodeConfig{dtype::QuantizedS8(2.5f)});
        z4 = opr::TypeCvt::make(z4, dtype::Float32());

        SymbolVar z_fuse;
        SymbolVar z_nonfuse;
        {
            auto options = gopt::OptimizeForInferenceOptions{};
            options.enable_fuse_conv_bias_nonlinearity()
                    .enable_fuse_conv_bias_with_z();
            unpack_vector(gopt::optimize_for_inference({z}, options), z_fuse);
        }
        {
            auto options = gopt::OptimizeForInferenceOptions{};
            options.enable_fuse_conv_bias_nonlinearity();
            unpack_vector(gopt::optimize_for_inference({z4}, options),
                          z_nonfuse);
        }
        auto nr_elem_multi_type =
                find_opr_num<mgb::opr::ElemwiseMultiType>(z_fuse);
        MGB_MARK_USED_VAR(nr_elem_multi_type);
        ASSERT_EQ(1u, nr_elem_multi_type);
        graph->compile({{z_fuse, {}}})
                ->to_json()
                ->writeto_fpath(
                        output_file("FuseConvBiasZPass.BlockFuse_fuse.json"));
        graph->compile({{z_nonfuse, {}}})
                ->to_json()
                ->writeto_fpath(output_file(
                        "FuseConvBiasZPass.BlockFuse_nonfuse.json"));

        HostTensorND host_z_fuse, host_z_nonfuse;
        auto func =
                graph->compile({make_callback_copy(z_nonfuse, host_z_nonfuse),
                                make_callback_copy(z_fuse, host_z_fuse)});
        func->execute();
        MGB_ASSERT_TENSOR_EQ(host_z_fuse, host_z_nonfuse);
    }
}

TEST(TestEnableTensorCore, ShuffleMerge) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 75) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 75);
        return;
    }

    HostTensorGenerator<dtype::Int8> gen;
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

    auto x = mkvar("x", {32, 64, 16, 16}, dtype::QuantizedS8(2.5f)),
         w = mkcvar("w1", {64, 64, 3, 3}, dtype::QuantizedS8(2.5f)),
         b = mkcvar("b", {1, 64, 1, 1}, dtype::QuantizedS32(6.25f)),
         z = mkvar("b1", {32, 64, 16, 16}, dtype::QuantizedS8(2.5f));
    x = nchw2nchw4(x), w = nchw2nchw4(w), b = nchw2nchw4(b), z = nchw2nchw4(z);
    opr::ConvBias::Param param;
    param.format = opr::ConvBias::Param::Format::NCHW4;
    param.nonlineMode = opr::ConvBias::Param::NonlineMode::RELU;
    param.stride_h = param.stride_w = 1;
    param.pad_h = param.pad_w = 1;

    auto y = opr::ConvBias::make(x, w, b, z, param, {},
                                 OperatorNodeConfig{dtype::QuantizedS8(2.5f)});
    y = nchw42nchw(y);
    y = opr::TypeCvt::make(y, dtype::Float32());

    SymbolVar y_opt;
    SymbolVar y_no_tc;
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_fuse_conv_bias_nonlinearity().enable_nchw32();
        unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    }
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_fuse_conv_bias_nonlinearity();
        unpack_vector(gopt::optimize_for_inference({y}, options), y_no_tc);
    }
    auto nr_dimshuffle = find_opr_num<mgb::opr::Dimshuffle>(y_opt);
    ASSERT_EQ(3u, nr_dimshuffle);
    HostTensorND host_y, host_y_opt;
    auto func = graph->compile({make_callback_copy(y_no_tc, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_y, host_y_opt);
}

#endif

TEST(FuseConvBiasZPass, Basic) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");

    HostTensorGenerator<dtype::Int8> gen;
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

    auto format = opr::Convolution::Param::Format::NCHW4;

    auto x = mkvar("x", {32, 16, 16, 16, 4}, dtype::QuantizedS8(2.5f)),
         w = mkcvar("w1", {64, 16, 3, 3, 4}, dtype::QuantizedS8(2.5f)),
         b = mkcvar("b", {1, 16, 1, 1, 4}, dtype::QuantizedS32(6.25f)),
         b1 = mkvar("b1", {32, 16, 16, 16, 4}, dtype::QuantizedS8(2.5f)),
         b2 = mkvar("b2", {32, 16, 16, 16, 4}, dtype::QuantizedS8(2.5f));

    opr::ConvBias::Param conv_bias_param;
    conv_bias_param.format = format;
    conv_bias_param.stride_h = conv_bias_param.stride_w = 1;
    conv_bias_param.pad_h = conv_bias_param.pad_w = 1;

    auto y = opr::ConvBias::make(x, w, b, conv_bias_param, {},
                                 OperatorNodeConfig{dtype::QuantizedS8(2.5f)});

    SymbolVar y_opt;

    // check fuse mode
    for (auto mode : {opr::ElemwiseMultiType::Param::Mode::QADD,
                      opr::ElemwiseMultiType::Param::Mode::QMUL,
                      opr::ElemwiseMultiType::Param::Mode::QFUSE_ADD_RELU}) {
        auto y1 = opr::ElemwiseMultiType::make(
                {y, b1}, {mode}, OperatorNodeConfig{dtype::QuantizedS8(2.5f)});
        {
            auto options = gopt::OptimizeForInferenceOptions{};
            options.enable_fuse_conv_bias_nonlinearity()
                    .enable_fuse_conv_bias_with_z()
                    .enable_nchw32();
            unpack_vector(gopt::optimize_for_inference({y1}, options), y_opt);
        }
        auto nr_elemwisemultitype = find_opr_num<opr::ElemwiseMultiType>(y_opt);
        if (mode == opr::ElemwiseMultiType::Param::Mode::QMUL) {
            ASSERT_NE(0u, nr_elemwisemultitype);
        } else
            ASSERT_EQ(0u, nr_elemwisemultitype);
        // fuse convbiasz and z
        if (mode == opr::ElemwiseMultiType::Param::Mode::QADD) {
            auto y2 = opr::ElemwiseMultiType::make(
                    {y1, b2}, {mode},
                    OperatorNodeConfig{dtype::QuantizedS8(2.5f)});
            {
                auto options = gopt::OptimizeForInferenceOptions{};
                options.enable_fuse_conv_bias_nonlinearity()
                        .enable_fuse_conv_bias_with_z()
                        .enable_nchw32();
                unpack_vector(gopt::optimize_for_inference({y2}, options),
                              y_opt);
            }
            auto nr_elemwisemultitype =
                    find_opr_num<opr::ElemwiseMultiType>(y_opt);
            ASSERT_NE(0u, nr_elemwisemultitype);
        }
    }
}

#if MGB_CUDA
TEST(TestGoptInference, EnableCHWN4) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 61) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 61);
        return;
    }

    HostTensorGenerator<dtype::Int8> gen;
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
    auto mkshape = [](opr::ConvBias::Param::Format format, size_t N, size_t C,
                      size_t H, size_t W) -> TensorShape {
        mgb_assert(C % 4 == 0);
        if (format == opr::ConvBias::Param::Format::NCHW4) {
            return {N, C / 4, H, W, 4};
        } else {
            mgb_assert(format == opr::ConvBias::Param::Format::NCHW);
            return {N, C, H, W};
        }
    };

    for (auto format : {opr::ConvBias::Param::Format::NCHW,
                        opr::ConvBias::Param::Format::NCHW4}) {
        auto x = mkvar("x", mkshape(format, 32, 64, 16, 16),
                       dtype::QuantizedS8(2.5f)),
             w = mkcvar("w1", mkshape(format, 64, 64, 3, 3),
                        dtype::QuantizedS8(2.5f)),
             b = mkcvar("b", mkshape(format, 1, 64, 1, 1),
                        dtype::QuantizedS32(6.25f)),
             b1 = mkvar("b1", mkshape(format, 32, 64, 16, 16),
                        dtype::QuantizedS8(2.5f));
        opr::ConvBias::Param param;
        param.format = format;
        param.stride_h = param.stride_w = 1;
        param.pad_h = param.pad_w = 1;
        param.nonlineMode = opr::ConvBias::Param::NonlineMode::RELU;

        auto y = opr::ConvBiasForward::make(
                x, w, b, param, {},
                OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
        auto y1 = opr::ElemwiseMultiType::make(
                {y, b1}, opr::ElemwiseMultiType::Mode::QFUSE_ADD_RELU,
                OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
        auto y2 = opr::ConvBiasForward::make(
                y, w, b, param, {},
                OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
        auto y3 = opr::ElemwiseMultiType::make(
                {y, b1}, opr::ElemwiseMultiType::Param::Mode::QSUB,
                OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
        auto y4 = opr::ElemwiseMultiType::make(
                {y1, y2}, opr::ElemwiseMultiType::Param::Mode::QADD,
                OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
        y4 = opr::ElemwiseMultiType::make(
                {y3, y4}, opr::ElemwiseMultiType::Param::Mode::QADD,
                OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
        y4 = opr::TypeCvt::make(y4, dtype::Float32());
        SymbolVar y_opt;
        SymbolVar y_cudnn;
        {
            auto options = gopt::OptimizeForInferenceOptions{};
            options.enable_chwn4();
            unpack_vector(gopt::optimize_for_inference({y4}, options), y_opt);
        }
        unpack_vector(gopt::GraphOptimizer{}
                              .add_pass<gopt::FuseConvBiasNonlinPass>()
                              .add_pass<gopt::FuseConvBiasZPass>()
                              .apply({{y4}})
                              .endpoint_vars(),
                      y_cudnn);

        ASSERT_EQ(opr::ConvBias::Param::Format::CHWN4,
                  find_opr<opr::ConvBias>(y_opt).param().format);
        HostTensorND host_y, host_y_opt;
        auto func = graph->compile({make_callback_copy(y_cudnn, host_y),
                                    make_callback_copy(y_opt, host_y_opt)});
        func->execute();
        MGB_ASSERT_TENSOR_EQ(host_y, host_y_opt);
    }
}

TEST(TestGoptInference, EnableCHWN4WarpPespective) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 61) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 61);
        return;
    }

    HostTensorGenerator<dtype::Int8> gen;
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
    std::shared_ptr<HostTensorND> mat = std::make_shared<HostTensorND>(
            cn, TensorShape{32, 3, 3}, dtype::Float32());
    warp_perspective_mat_gen(*mat, 32, 16, 16);
    auto mat_var = opr::Host2DeviceCopy::make(*graph, mat).rename("mat");

    auto x = mkvar("x", {32, 16, 16, 16, 4}, dtype::QuantizedS8(2.5f)),
         w = mkcvar("w1", {64, 16, 3, 3, 4}, dtype::QuantizedS8(2.5f)),
         b = mkcvar("b", {1, 16, 1, 1, 4}, dtype::QuantizedS32(6.25f));
    opr::ConvBias::Param param;
    param.format = opr::ConvBias::Param::Format::NCHW4;
    param.stride_h = param.stride_w = 1;
    param.pad_h = param.pad_w = 1;
    param.nonlineMode = opr::ConvBias::Param::NonlineMode::RELU;

    auto y = opr::ConvBiasForward::make(
            x, w, b, param, {}, OperatorNodeConfig{dtype::QuantizedS8{2.5f}});

    opr::WarpPerspective::Param warp_param;
    warp_param.format = opr::WarpPerspective::Param::Format::NCHW4;
    auto y1 = opr::WarpPerspective::make(y, mat_var, TensorShape{16, 16},
                                         warp_param);
    y1 = opr::TypeCvt::make(y1, dtype::Float32());
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
    y1 = nchw42nchw(y1);
    warp_param.format = opr::WarpPerspective::Param::Format::NCHW;
    auto y2 = opr::WarpPerspective::make(y1, mat_var, TensorShape{16, 16},
                                         warp_param);
    SymbolVar y_opt;
    SymbolVar y_cudnn;
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_chwn4();
        unpack_vector(gopt::optimize_for_inference({y2}, options), y_opt);
    }
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::FuseConvBiasNonlinPass>()
                          .add_pass<gopt::FuseConvBiasZPass>()
                          .apply({{y2}})
                          .endpoint_vars(),
                  y_cudnn);

    HostTensorND host_y, host_y_opt;
    auto func = graph->compile({make_callback_copy(y_cudnn, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_y, host_y_opt);
}

TEST(TestGoptInference, EnableCHWN4Pooling) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 61) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 61);
        return;
    }

    HostTensorGenerator<dtype::Int8> gen;
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

    auto x = mkvar("x", {32, 16, 16, 16, 4}, dtype::QuantizedS8(2.5f)),
         w = mkcvar("w1", {64, 16, 3, 3, 4}, dtype::QuantizedS8(2.5f)),
         b = mkcvar("b", {1, 16, 1, 1, 4}, dtype::QuantizedS32(6.25f));
    opr::ConvBias::Param param;
    param.format = opr::ConvBias::Param::Format::NCHW4;
    param.stride_h = param.stride_w = 1;
    param.pad_h = param.pad_w = 1;
    param.nonlineMode = opr::ConvBias::Param::NonlineMode::RELU;

    auto y = opr::ConvBiasForward::make(
            x, w, b, param, {}, OperatorNodeConfig{dtype::QuantizedS8{2.5f}});

    opr::Pooling::Param pool_param;
    pool_param.format = opr::Pooling::Param::Format::NCHW4;
    y = opr::Pooling::make(y, pool_param);
    y = opr::TypeCvt::make(y, dtype::Float32());

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
    y = nchw42nchw(y);
    pool_param.format = opr::Pooling::Param::Format::NCHW;
    auto y1 = opr::Pooling::make(y, pool_param);

    SymbolVar y_opt;
    SymbolVar y_cudnn;
    unpack_vector(
            gopt::GraphOptimizer{}
                    .add_pass<gopt::FuseConvBiasNonlinPass>()
                    .add_pass(gopt::EnableCHWN4Pass::make_chwn4_converter())
                    .add_pass<gopt::FuseConvBiasZPass>()
                    .apply({{y1}})
                    .endpoint_vars(),
            y_opt);
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::FuseConvBiasNonlinPass>()
                          .add_pass<gopt::FuseConvBiasZPass>()
                          .apply({{y1}})
                          .endpoint_vars(),
                  y_cudnn);

    HostTensorND host_y, host_y_opt;
    auto func = graph->compile({make_callback_copy(y_cudnn, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_y, host_y_opt);
}

TEST(TestGoptInference, EnableCHWN4ShuffleRemove) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 61) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 61);
        return;
    }

    HostTensorGenerator<dtype::Int8> gen;
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

    auto x = mkvar("x", {32, 64, 16, 16}, dtype::QuantizedS8(2.5f)),
         w = mkcvar("w1", {64, 16, 3, 3, 4}, dtype::QuantizedS8(2.5f)),
         b = mkcvar("b", {1, 16, 1, 1, 4}, dtype::QuantizedS32(6.25f)),
         b1 = mkcvar("b1", {32, 16, 16, 16, 4}, dtype::QuantizedS8{2.5f});
    x = nchw2nchw4(x);
    opr::ConvBias::Param param;
    param.format = opr::ConvBias::Param::Format::NCHW4;
    param.stride_h = param.stride_w = 1;
    param.pad_h = param.pad_w = 1;
    param.nonlineMode = opr::ConvBias::Param::NonlineMode::RELU;

    auto y = opr::ConvBiasForward::make(
            x, w, b, param, {}, OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    auto y1 = opr::ElemwiseMultiType::make(
            {y, b1}, opr::ElemwiseMultiType::Mode::QFUSE_ADD_RELU,
            OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    auto y2 = opr::ConvBiasForward::make(
            y, w, b, param, {}, OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    auto y3 = opr::ElemwiseMultiType::make(
            {y, b1}, opr::ElemwiseMultiType::Param::Mode::QSUB,
            OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    auto y4 = opr::ElemwiseMultiType::make(
            {y1, y2}, opr::ElemwiseMultiType::Param::Mode::QADD,
            OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    y4 = opr::ElemwiseMultiType::make(
            {y3, y4}, opr::ElemwiseMultiType::Param::Mode::QADD,
            OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    y4 = opr::TypeCvt::make(y4, dtype::Float32());
    y4 = nchw42nchw(y4);

    SymbolVar y_opt;
    SymbolVar y_cudnn;
    unpack_vector(
            gopt::GraphOptimizer{}
                    .add_pass<gopt::ParamRedistributePass>()
                    .add_pass<gopt::ParamFusePass>()
                    .add_pass<gopt::FuseConvBiasNonlinPass>()
                    .add_pass<gopt::FuseConvBiasZPass>()
                    .add_pass(gopt::EnableCHWN4Pass::make_chwn4_converter())
                    .add_pass<gopt::ShuffleShuffleRemovePass>()
                    .add_pass<gopt::ParamFusePass>()
                    .apply({{y4}})
                    .endpoint_vars(),
            y_opt);
    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(output_file(
                    "TestGoptInference.EnableCHWN4ShuffleRemove.json"));
    auto nr_dimshuffle = find_opr_num<mgb::opr::Dimshuffle>(y_opt);
    ASSERT_EQ(2u, nr_dimshuffle);
    auto nr_reformat = find_opr_num<mgb::opr::RelayoutFormat>(y_opt);
    ASSERT_EQ(0u, nr_reformat);
    unpack_vector(gopt::GraphOptimizer{}
                          .add_pass<gopt::FuseConvBiasNonlinPass>()
                          .add_pass<gopt::FuseConvBiasZPass>()
                          .apply({{y4}})
                          .endpoint_vars(),
                  y_cudnn);

    HostTensorND host_y, host_y_opt;
    auto func = graph->compile({make_callback_copy(y_cudnn, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_y, host_y_opt);
}

TEST(TestGoptInference, ConvertFormatNCHW4GPU) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    cn.activate();
    auto&& prop = CompNodeEnv::from_comp_node(cn).cuda_env().device_prop;
    auto sm_ver = prop.major * 10 + prop.minor;
    if (sm_ver < 61) {
        printf("This testcast ignored due to insufficient cuda cap(got: %d, "
               "expected: %d)\n",
               sm_ver, 61);
        return;
    }

    HostTensorGenerator<dtype::Int8> gen;
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

    auto x = mkvar("x", {2, 4, 16, 16}, dtype::QuantizedS8(2.5f));
    opr::ConvBias::Param param_conv_bias;
    param_conv_bias.format = opr::ConvBias::Param::Format::NCHW;
    param_conv_bias.stride_h = param_conv_bias.stride_w = 1;
    param_conv_bias.pad_h = param_conv_bias.pad_w = 1;
    param_conv_bias.nonlineMode = opr::ConvBias::Param::NonlineMode::RELU;
    // dense
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::DENSE;
    auto w1 = mkcvar("w1", {8, 4, 3, 3}, dtype::QuantizedS8(2.5f)),
         b1 = mkcvar("b1", {1, 8, 1, 1}, dtype::QuantizedS32(6.25f));
    auto conv1 = opr::ConvBiasForward::make(
            x, w1, b1, param_conv_bias, {},
            OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    // group
    // icpg != 1 && ocpg != 1
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::GROUP;
    auto w2 = mkcvar("w2", {2, 4, 4, 3, 3}, dtype::QuantizedS8(2.5f)),
         b2 = mkcvar("b2", {1, 8, 1, 1}, dtype::QuantizedS32(6.25f));
    auto conv2 = opr::ConvBiasForward::make(
            conv1, w2, b2, param_conv_bias, {},
            OperatorNodeConfig{dtype::QuantizedS8{2.5f}});

    auto y = opr::TypeCvt::make(conv2, dtype::Float32());

    SymbolVar y_opt;
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_nchw4();
        unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    }

    ASSERT_EQ(opr::ConvBias::Param::Format::NCHW4,
              find_opr<opr::ConvBias>(y_opt).param().format);
    auto nr_reshape = find_opr_num<mgb::opr::Reshape>(y_opt);
    ASSERT_EQ(2u, nr_reshape);

    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(output_file(
                    "TestGoptInference.ConvertFormatNCHW4GPU.json"));

    HostTensorND host_y, host_y_opt;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_EQ(host_y, host_y_opt);
}

#endif

TEST(TestGoptInference, ConvertFormatNCHW4NonConvOpr) {
    auto cn = CompNode::load("xpu0");
    HostTensorGenerator<dtype::Int8> gen;
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
    auto mkcvarf32 = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                .rename(name);
    };

    auto x = mkvar("x", {2, 4, 16, 16}, dtype::QuantizedS8(2.5f));
    opr::ConvBias::Param param_conv_bias;
    param_conv_bias.format = opr::ConvBias::Param::Format::NCHW;
    param_conv_bias.stride_h = param_conv_bias.stride_w = 1;
    param_conv_bias.pad_h = param_conv_bias.pad_w = 1;
    param_conv_bias.nonlineMode = opr::ConvBias::Param::NonlineMode::RELU;
    // dense
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::DENSE;
    auto w1 = mkcvar("w1", {8, 4, 3, 3}, dtype::QuantizedS8(2.5f)),
         b1 = mkcvar("b1", {1, 8, 1, 1}, dtype::QuantizedS32(6.25f));
    auto conv1 = opr::ConvBiasForward::make(
            x, w1, b1, param_conv_bias, {},
            OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    // test Resize
    auto shape_of = opr::GetVarShape::make(x);
    auto subtensor = opr::Subtensor::make(
            shape_of, {opr::Subtensor::AxisIndexer::make_interval(
                              0, x.make_scalar(2), None, x.make_scalar(1))});
    opr::Resize::Param param_resize;
    param_resize.format = opr::Resize::Param::Format::NCHW;
    auto resize = opr::ResizeForward::make(conv1, subtensor * 2, param_resize);
    // test WarpPerspective
    auto mat = mkcvarf32("mat", {2, 3, 3}),
         warp = opr::WarpPerspectiveForward::make(
                 resize, mat, nullptr, cg::var_from_tensor_shape(x, {32, 32}));
    opr::Pooling::Param pool_param;
    pool_param.format = opr::Pooling::Param::Format::NCHW;
    // test Pooling
    auto pool = opr::Pooling::make(warp, pool_param);
    // group
    // icpg != 1 && ocpg != 1
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::GROUP;
    auto w2 = mkcvar("w2", {2, 4, 4, 3, 3}, dtype::QuantizedS8(2.5f)),
         b2 = mkcvar("b2", {1, 8, 1, 1}, dtype::QuantizedS32(6.25f));
    auto conv2 = opr::ConvBiasForward::make(
            pool, w2, b2, param_conv_bias, {},
            OperatorNodeConfig{dtype::QuantizedS8{2.5f}});

    auto add = opr::ElemwiseMultiType::make(
            {conv1, conv2}, {opr::ElemwiseMultiType::Param::Mode::QADD},
            OperatorNodeConfig{dtype::QuantizedS8{1.2f}});
    auto y = opr::TypeCvt::make(add, dtype::Float32());

    SymbolVar y_opt;
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_nchw4();
        unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    }
    auto nr_dimshuffle = find_opr_num<mgb::opr::Dimshuffle>(y_opt);
    ASSERT_EQ(2u, nr_dimshuffle);
    ASSERT_EQ(opr::ConvBias::Param::Format::NCHW4,
              find_opr<opr::ConvBias>(y_opt).param().format);
    ASSERT_EQ(opr::ResizeForward::Param::Format::NCHW4,
              find_opr<opr::ResizeForward>(y_opt).param().format);
    ASSERT_EQ(opr::WarpPerspectiveForward::Param::Format::NCHW4,
              find_opr<opr::WarpPerspectiveForward>(y_opt).param().format);
    ASSERT_EQ(opr::PoolingForward::Param::Format::NCHW4,
              find_opr<opr::PoolingForward>(y_opt).param().format);
}

TEST(TestGoptInference, ConvertFormatNCHW4) {
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                .rename(name);
    };

    auto x = mkvar("x", {2, 4, 16, 16});
    // ConvBias test dense
    opr::ConvBias::Param param_conv_bias;
    param_conv_bias.pad_h = param_conv_bias.pad_w = 1;
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::DENSE;
    auto w1 = mkcvar("w1", {8, 4, 3, 3}), b1 = mkcvar("b1", {1, 8, 1, 1});
    auto conv1 = opr::ConvBias::make(x, w1, b1, param_conv_bias);
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::GROUP;
    auto w2 = mkcvar("w2", {2, 4, 4, 3, 3}), b2 = mkcvar("b2", {1, 8, 1, 1});
    auto conv2 = opr::ConvBias::make(conv1, w2, b2, param_conv_bias);
    // Convolution
    opr::Convolution::Param param_conv;
    param_conv.pad_h = param_conv.pad_w = 1;
    param_conv.sparse = opr::Convolution::Param::Sparse::DENSE;
    auto w3 = mkcvar("w3", {8, 8, 3, 3});
    auto y = opr::Convolution::make(conv2, w3, param_conv);

    SymbolVar y_opt;
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_nchw4();
        unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    }

    ASSERT_EQ(opr::ConvBias::Param::Format::NCHW,
              find_opr<opr::ConvBias>(y_opt).param().format);

    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(
                    output_file("TestGoptInference.ConvertFormatNCHW4.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}

TEST(TestGoptInference, ConvertFormatNCHW4Ic3) {
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
                opr::SharedDeviceTensor::make(*graph, *gen(shp)).rename(name),
                dtype);
    };

    auto x = mkvar("x", {2, 3, 16, 16}, dtype::QuantizedS8(2.5f));
    // ConvBias test dense
    opr::ConvBias::Param param_conv_bias;
    param_conv_bias.pad_h = param_conv_bias.pad_w = 1;
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::DENSE;
    auto w1 = mkcvar("w1", {8, 3, 3, 3}, dtype::QuantizedS8(2.5f)),
         b1 = mkcvar("b1", {1, 8, 1, 1}, dtype::QuantizedS32(6.25f));
    auto conv1 =
            opr::ConvBias::make(x, w1, b1, param_conv_bias, {},
                                OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::GROUP;
    auto w2 = mkcvar("w2", {2, 4, 4, 3, 3}, dtype::QuantizedS8(2.5f)),
         b2 = mkcvar("b2", {1, 8, 1, 1}, dtype::QuantizedS32(6.25f));
    auto conv2 =
            opr::ConvBias::make(conv1, w2, b2, param_conv_bias, {},
                                OperatorNodeConfig{dtype::QuantizedS8{2.5f}});
    auto y = opr::TypeCvt::make(conv2, dtype::Float32());

    SymbolVar y_opt;
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_nchw4();
        unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    }

    ASSERT_EQ(opr::ConvBias::Param::Format::NCHW4,
              find_opr<opr::ConvBias>(y_opt).param().format);

    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(output_file(
                    "TestGoptInference.ConvertFormatNCHW4Ic3.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-3);
}

TEST(TestGoptInference, ConvertFormatNCHW88) {
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                .rename(name);
    };

    auto host_x = gen({2, 3, 16, 16}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    //! Hybrid nchw88 mode
    opr::Convolution::Param param_conv;
    param_conv.pad_h = param_conv.pad_w = 1;
    auto w1 = mkcvar("w1", {8, 3, 3, 3}),
         conv1 = opr::Convolution::make(x, w1, param_conv, {},
                                        OperatorNodeConfig("conv1"));
    //! channel wise
    opr::ConvBias::Param param_conv_bias;
    param_conv_bias.pad_h = param_conv_bias.pad_w = 1;
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::GROUP;
    auto w2 = mkcvar("w2", {8, 1, 1, 3, 3}), b2 = mkcvar("b2", {1, 8, 1, 1}),
         conv2 = opr::ConvBias::make(conv1, w2, b2, param_conv_bias);
    //! group
    auto w3 = mkcvar("w3", {1, 8, 8, 3, 3}), b3 = mkcvar("b3", {1, 8, 1, 1}),
         conv3 = opr::ConvBias::make(conv2, w3, b3, param_conv_bias);

    auto shape_of = opr::GetVarShape::make(conv3);
    auto subtensor = opr::Subtensor::make(
            shape_of, {opr::Subtensor::AxisIndexer::make_interval(
                              0, x.make_scalar(2), None, x.make_scalar(1))});
    opr::Resize::Param param_resize;
    param_resize.format = opr::Resize::Param::Format::NCHW;
    auto resize = opr::ResizeForward::make(conv3, subtensor * 2, param_resize);
    auto mat = mkcvar("mat", {2, 3, 3}),
         warp = opr::WarpPerspectiveForward::make(
                 resize, mat, nullptr, cg::var_from_tensor_shape(x, {4, 4}));

    auto b = mkvar("b", {1, 8, 1, 1}),
         elem = opr::Elemwise::make({warp + b},
                                    opr::Elemwise::Param::Mode::RELU);
    //! Dense
    param_conv_bias.pad_h = param_conv_bias.pad_w = 1;
    auto w4 = mkcvar("w4", {2, 6, 4, 3, 3}), b4 = mkcvar("b4", {1, 12, 1, 1}),
         conv4 = opr::ConvBias::make(elem, w4, b4, param_conv_bias);
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::DENSE;
    auto w5 = mkcvar("w5", {8, 12, 3, 3}), b5 = mkcvar("b5", {1, 8, 1, 1}),
         conv5 = opr::ConvBias::make(conv4, w5, b5, param_conv_bias);
    auto w6 = mkcvar("w6", {8, 8, 3, 3}), b6 = mkcvar("b6", {1, 8, 1, 1}),
         y = opr::ConvBias::make(conv5, w6, b6, param_conv_bias);

    SymbolVar y_opt;
    {
        auto options = gopt::OptimizeForInferenceOptions{};
        options.enable_nchw88();
        unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);
    }
    ASSERT_EQ(opr::ConvBias::Param::Format::NCHW88,
              find_opr<opr::Convolution>(y_opt, "conv1").param().format);
    ASSERT_EQ(opr::ConvBias::Param::Format::NCHW88,
              find_opr<opr::ConvBias>(y_opt).param().format);

    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(
                    output_file("TestGoptInference.ConvertFormatNCHW88.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    //! meybe go to winograd in x86-32, so set error 1e-1
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-1);

    *host_x = *gen({2, 3, 32, 32}, cn);
    func->execute();
    //! meybe go to winograd in x86-32, so set error 1e-1
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-1);
}

TEST(TestGoptInference, ConvertFormatNCHW44) {
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                .rename(name);
    };
    auto mkcvar_dtype = [&](const char* name, const TensorShape& shp,
                            const DType& dtype) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                        .rename(name),
                dtype);
    };

    auto host_x = gen({2, 3, 16, 16}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    //! Hybrid nchw44 mode
    opr::Convolution::Param param_conv;
    param_conv.pad_h = param_conv.pad_w = 1;
    auto w1 = mkcvar("w1", {8, 3, 3, 3}),
         conv1 = opr::Convolution::make(x, w1, param_conv, {},
                                        OperatorNodeConfig("conv1"));

    //! no supported hybrid nchw44
    opr::ConvBias::Param param_conv_bias_pad0;
    param_conv_bias_pad0.pad_h = param_conv_bias_pad0.pad_w = 0;
    auto w1_f1 = mkcvar("w1_1", {8, 3, 1, 1});
    auto conv1_f1 = opr::ConvBias::make(x, w1_f1, param_conv_bias_pad0, {},
                                        OperatorNodeConfig("conv1_f1"));

    auto conv1_add = conv1_f1 * conv1;
    auto conv_1_q8 = opr::TypeCvt::make(conv1_add, dtype::QuantizedS8(2.5f));

    //! s8 dense conv
    opr::ConvBias::Param param_conv_bias;
    param_conv_bias.pad_h = param_conv_bias.pad_w = 1;
    auto w1_2 = mkcvar_dtype("w1_2", {8, 8, 3, 3}, dtype::QuantizedS8(2.5f));
    auto b1_2 = mkcvar_dtype("b1_2", {1, 8, 1, 1}, dtype::QuantizedS32(6.25f));
    auto conv_1_2 = opr::ConvBias::make(
            conv_1_q8, w1_2, b1_2, param_conv_bias, {},
            OperatorNodeConfig{"conv_1_2", cn, dtype::QuantizedS8{6.25f}});
    auto conv_1_2_fp32 = opr::TypeCvt::make(conv_1_2, dtype::Float32());

    //! channel wise
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::GROUP;
    auto w2 = mkcvar("w2", {8, 1, 1, 3, 3}), b2 = mkcvar("b2", {1, 8, 1, 1}),
         conv2 = opr::ConvBias::make(conv_1_2_fp32, w2, b2, param_conv_bias);
    //! group
    auto w3 = mkcvar("w3", {2, 4, 4, 3, 3}), b3 = mkcvar("b3", {1, 8, 1, 1}),
         conv3 = opr::ConvBias::make(conv2, w3, b3, param_conv_bias);

    auto shape_of = opr::GetVarShape::make(conv3);
    auto subtensor = opr::Subtensor::make(
            shape_of, {opr::Subtensor::AxisIndexer::make_interval(
                              0, x.make_scalar(2), None, x.make_scalar(1))});
    opr::Resize::Param param_resize;
    param_resize.format = opr::Resize::Param::Format::NCHW;
    auto resize = opr::ResizeForward::make(conv3, subtensor * 2, param_resize);
    auto mat = mkcvar("mat", {2, 3, 3}),
         warp = opr::WarpPerspectiveForward::make(
                 resize, mat, nullptr, cg::var_from_tensor_shape(x, {4, 4}));

    auto b = mkvar("b", {1, 8, 1, 1}),
         elem = opr::Elemwise::make({warp + b},
                                    opr::Elemwise::Param::Mode::RELU);
    //! Dense
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::DENSE;
    param_conv_bias.pad_h = param_conv_bias.pad_w = 1;
    auto w3_2 = mkcvar("w3_2", {16, 8, 3, 3}),
         b3_2 = mkcvar("b3_2", {1, 16, 1, 1}),
         conv3_2 = opr::ConvBias::make(elem, w3_2, b3_2, param_conv_bias, {},
                                       OperatorNodeConfig("conv3_2"));
    //! s8 group conv
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::GROUP;
    auto conv3_2_q8 = opr::TypeCvt::make(conv3_2, dtype::QuantizedS8(2.5f));
    auto w3_3 = mkcvar_dtype("w3_3", {4, 8, 4, 3, 3}, dtype::QuantizedS8(2.5f)),
         b3_3 = mkcvar_dtype("b3_3", {1, 32, 1, 1}, dtype::QuantizedS32(6.25f)),
         conv3_3_q = opr::ConvBias::make(
                 conv3_2_q8, w3_3, b3_3, param_conv_bias, {},
                 OperatorNodeConfig{"conv_3_3_q", cn,
                                    dtype::QuantizedS8{6.25f}});
    auto conv3_3 = opr::TypeCvt::make(conv3_3_q, dtype::Float32());

    //! Dense
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::DENSE;
    auto w4 = mkcvar("w4", {16, 32, 3, 3}), b4 = mkcvar("b4", {1, 16, 1, 1}),
         conv4 = opr::ConvBias::make(conv3_3, w4, b4, param_conv_bias, {},
                                     OperatorNodeConfig("conv4"));
    auto w4_1 = mkcvar("w4_1", {16, 32, 1, 1}),
         b4_1 = mkcvar("b4_1", {2, 16, 4, 4}),
         conv4_1 =
                 opr::ConvBias::make(conv3_3, w4_1, b4_1, param_conv_bias_pad0,
                                     {}, OperatorNodeConfig("conv4_1"));
    auto conv4_add = conv4 + conv4_1;

    auto w5 = mkcvar("w5", {6, 16, 3, 3}), b5 = mkcvar("b5", {1, 6, 1, 1}),
         conv5 = opr::ConvBias::make(conv4_add, w5, b5, param_conv_bias, {},
                                     OperatorNodeConfig("conv5"));
    auto w6 = mkcvar("w6", {4, 6, 3, 3}), b6 = mkcvar("b6", {1, 4, 1, 1}),
         y = opr::ConvBias::make(conv5, w6, b6, param_conv_bias, {},
                                 OperatorNodeConfig("conv6"));

    SymbolVar y_opt;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_fuse_conv_bias_nonlinearity();
    options.enable_nchw44();
    unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);

    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44,
              find_opr<opr::Convolution>(y_opt, "conv1").param().format);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW,
              find_opr<opr::ConvBias>(y_opt, "conv1_f1").param().format);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44,
              find_opr<opr::ConvBias>(y_opt, "conv_1_2").param().format);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44,
              find_opr<opr::ConvBias>(y_opt, "conv3_2").param().format);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44,
              find_opr<opr::ConvBias>(y_opt, "conv_3_3_q").param().format);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44,
              find_opr<opr::ConvBias>(y_opt, "conv4").param().format);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW,
              find_opr<opr::ConvBias>(y_opt, "conv5").param().format);

    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(
                    output_file("TestGoptInference.ConvertFormatNCHW44.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    //! meybe go to winograd in x86-32, so set error 1e-1
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-1);

    *host_x = *gen({2, 3, 32, 32}, cn);
    func->execute();
    //! meybe go to winograd in x86-32, so set error 1e-1
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-1);
}

TEST(TestGoptInference, ConvertFormatNCHW44MultiInput) {
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                .rename(name);
    };

    auto host_x1 = gen({1, 8, 16, 16}, cn);
    auto host_x2 = gen({1, 1, 16, 16}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x1);
    opr::Convolution::Param param_conv;
    param_conv.pad_h = param_conv.pad_w = 1;
    auto w1 = mkcvar("w1", {8, 8, 3, 3}),
         conv1 = opr::Convolution::make(x, w1, param_conv);

    auto b = mkvar("b", {1, 1, 16, 16}),
         y = opr::Elemwise::make({conv1 + b}, opr::Elemwise::Param::Mode::RELU);

    SymbolVar y_opt;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_nchw44();
    unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);

    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44,
              find_opr<opr::Convolution>(y_opt).param().format);

    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(output_file(
                    "TestGoptInference.ConvertFormatNCHW44MultiInput.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    //! meybe go to winograd in x86-32, so set error 1e-1
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-1);
}

TEST(TestGoptInference, ConvertFormatNCHW44Reshape) {
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                .rename(name);
    };

    auto host_x1 = gen({1, 8, 16, 16}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x1);
    opr::Convolution::Param param_conv;
    param_conv.pad_h = param_conv.pad_w = 1;
    auto w1 = mkcvar("w1", {8, 8, 3, 3}),
         conv1 = opr::Convolution::make(x, w1, param_conv);
    auto y = opr::Reshape::make(conv1, {8, 16 * 16});

    SymbolVar y_opt;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_nchw44();
    unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);

    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44,
              find_opr<opr::Convolution>(y_opt).param().format);

    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(output_file(
                    "TestGoptInference.ConvertFormatNCHW44Reshape.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    //! meybe go to winograd in x86-32, so set error 1e-1
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-1);
}

TEST(TestGoptInference, ConvertFormatNCHW44_DOT) {
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpu0");
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name);
    };
    auto mkcvar = [&](const char* name, const TensorShape& shp) {
        return opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                .rename(name);
    };
    auto mkcvar_dtype = [&](const char* name, const TensorShape& shp,
                            const DType& dtype) {
        return opr::TypeCvt::make(
                opr::SharedDeviceTensor::make(*graph, *gen(shp, cn))
                        .rename(name),
                dtype);
    };

    auto host_x = gen({2, 3, 16, 16}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    //! Hybrid nchw44 mode
    opr::Convolution::Param param_conv;
    param_conv.pad_h = param_conv.pad_w = 1;
    auto w1 = mkcvar("w1", {8, 3, 3, 3}),
         conv1 = opr::Convolution::make(x, w1, param_conv, {},
                                        OperatorNodeConfig("conv1"));
    printf("create conv1 %s\n",
           conv1.node()->owner_opr()->dyn_typeinfo()->name);
    param_conv.pad_h = param_conv.pad_w = 1;
    //! no supported hybrid nchw44
    opr::ConvBias::Param param_conv_bias_pad0;
    param_conv_bias_pad0.pad_h = param_conv_bias_pad0.pad_w = 0;
    auto b1 = mkcvar("b1", {1, 8, 1, 1});
    auto w1_f1 = mkcvar("w1_1", {8, 3, 1, 1});
    auto conv1_f1 = opr::ConvBias::make(x, w1_f1, b1, param_conv_bias_pad0, {},
                                        OperatorNodeConfig("conv1_f1"));

    //! hybrid dot
    auto x_s = opr::TypeCvt::make(x, dtype::QuantizedS8(2.5f));
    auto w1_3 = mkcvar_dtype("w1_3", {8, 3, 3, 3}, dtype::QuantizedS8(2.5f));
    auto conv1_3_q = opr::Convolution::make(
            x_s, w1_3, param_conv, {},
            OperatorNodeConfig{"conv1_3_q", cn, dtype::QuantizedS8{6.25f}});
    auto conv1_3 = opr::TypeCvt::make(conv1_3_q, dtype::Float32());

    auto conv1_add = conv1_f1 * conv1 * conv1_3;
    auto conv_1_q8 = opr::TypeCvt::make(conv1_add, dtype::QuantizedS8(2.5f));

    //! s8 dense conv
    opr::ConvBias::Param param_conv_bias;
    param_conv_bias.pad_h = param_conv_bias.pad_w = 1;
    auto w1_2 = mkcvar_dtype("w1_2", {8, 8, 3, 3}, dtype::QuantizedS8(2.5f));
    auto conv_1_2 = opr::ConvBias::make(
            conv_1_q8, w1_2, param_conv_bias, {},
            OperatorNodeConfig{"conv_1_2", cn, dtype::QuantizedS8{6.25f}});
    auto conv_1_2_fp32 = opr::TypeCvt::make(conv_1_2, dtype::Float32());

    //! channel wise
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::GROUP;
    auto w2 = mkcvar("w2", {8, 1, 1, 3, 3}), b2 = mkcvar("b2", {1, 8, 1, 1}),
         conv2 = opr::ConvBias::make(conv_1_2_fp32, w2, b2, param_conv_bias);
    //! group
    auto w3 = mkcvar("w3", {2, 4, 4, 3, 3}), b3 = mkcvar("b3", {1, 8, 1, 1}),
         conv3 = opr::ConvBias::make(conv2, w3, b3, param_conv_bias);

    auto shape_of = opr::GetVarShape::make(conv3);
    auto subtensor = opr::Subtensor::make(
            shape_of, {opr::Subtensor::AxisIndexer::make_interval(
                              0, x.make_scalar(2), None, x.make_scalar(1))});
    opr::Resize::Param param_resize;
    param_resize.format = opr::Resize::Param::Format::NCHW;
    auto resize = opr::ResizeForward::make(conv3, subtensor * 2, param_resize);
    auto mat = mkcvar("mat", {2, 3, 3}),
         warp = opr::WarpPerspectiveForward::make(
                 resize, mat, nullptr, cg::var_from_tensor_shape(x, {4, 4}));

    auto b = mkvar("b", {1, 8, 1, 1}),
         elem = opr::Elemwise::make({warp + b},
                                    opr::Elemwise::Param::Mode::RELU);
    //! Dense
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::DENSE;
    param_conv_bias.pad_h = param_conv_bias.pad_w = 1;
    auto w3_2 = mkcvar("w3_2", {16, 8, 3, 3}),
         b3_2 = mkcvar("b3_2", {1, 16, 1, 1}),
         conv3_2 = opr::ConvBias::make(elem, w3_2, b3_2, param_conv_bias, {},
                                       OperatorNodeConfig("conv3_2"));
    //! s8 group conv
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::GROUP;
    auto conv3_2_q8 = opr::TypeCvt::make(conv3_2, dtype::QuantizedS8(2.5f));
    auto w3_3 = mkcvar_dtype("w3_3", {4, 8, 4, 3, 3}, dtype::QuantizedS8(2.5f)),
         b3_3 = mkcvar_dtype("b3_3", {1, 32, 1, 1}, dtype::QuantizedS32(6.25f)),
         conv3_3_q = opr::ConvBias::make(
                 conv3_2_q8, w3_3, b3_3, param_conv_bias, {},
                 OperatorNodeConfig{"conv_3_3_q", cn,
                                    dtype::QuantizedS8{6.25f}});
    auto conv3_3 = opr::TypeCvt::make(conv3_3_q, dtype::Float32());

    //! Dense
    param_conv_bias.sparse = opr::ConvBias::Param::Sparse::DENSE;
    auto w4 = mkcvar("w4", {4, 32, 3, 3}), b4 = mkcvar("b4", {1, 4, 1, 1}),
         conv4 = opr::ConvBias::make(conv3_3, w4, b4, param_conv_bias, {},
                                     OperatorNodeConfig("conv4"));

    auto w5 = mkcvar("w5", {6, 4, 3, 3}), b5 = mkcvar("b5", {1, 6, 1, 1}),
         conv5 = opr::ConvBias::make(conv4, w5, b5, param_conv_bias, {},
                                     OperatorNodeConfig("conv5"));
    auto w6 = mkcvar("w6", {4, 6, 3, 3}), b6 = mkcvar("b6", {1, 4, 1, 1}),
         y = opr::ConvBias::make(conv5, w6, b6, param_conv_bias, {},
                                 OperatorNodeConfig("conv6"));

    SymbolVar y_opt;
    auto options = gopt::OptimizeForInferenceOptions{};
    options.enable_fuse_conv_bias_nonlinearity();
    options.enable_nchw44_dot();
    unpack_vector(gopt::optimize_for_inference({y}, options), y_opt);

    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44,
              find_opr<opr::Convolution>(y_opt, "conv1").param().format);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44_DOT,
              find_opr<opr::Convolution>(y_opt, "conv1_3_q").param().format);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW,
              find_opr<opr::ConvBias>(y_opt, "conv1_f1").param().format);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44_DOT,
              find_opr<opr::ConvBias>(y_opt, "conv_1_2").param().format);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44,
              find_opr<opr::ConvBias>(y_opt, "conv3_2").param().format);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44_DOT,
              find_opr<opr::ConvBias>(y_opt, "conv_3_3_q").param().format);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW44,
              find_opr<opr::ConvBias>(y_opt, "conv4").param().format);
    ASSERT_EQ(opr::Convolution::Param::Format::NCHW,
              find_opr<opr::ConvBias>(y_opt, "conv5").param().format);

    graph->compile({{y_opt, {}}})
            ->to_json()
            ->writeto_fpath(output_file(
                    "TestGoptInference.ConvertFormatNCHW44_DOT.json"));

    HostTensorND host_y_opt, host_y;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_opt, host_y_opt)});
    func->execute();
    //! meybe go to winograd in x86-32, so set error 1e-1
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-1);

    *host_x = *gen({2, 3, 32, 32}, cn);
    func->execute();
    //! meybe go to winograd in x86-32, so set error 1e-1
    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_opt, 1e-1);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
