/**
 * \file src/gopt/test/gtrans.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"
#include "megbrain/gopt/gtrans.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/dnn/convolution.h"

using namespace mgb;

using gopt::BinaryTrans20;
using BinaryOp = std::function<SymbolVar(SymbolVar, SymbolVar)>;

#define BOP(_expr) [&](SymbolVar a, SymbolVar b) -> SymbolVar { return _expr; }

namespace {

    //! check that fop(gop(a, b), c) has been changed
    void run_binary_trans20_test(
            BinaryTrans20 &trans,
            const TensorShape &sa, const TensorShape &sb, const TensorShape &sc,
            const BinaryOp &fop, const BinaryOp &gop,
            bool expect_succ,
            float err=5e-6) {

        HostTensorGenerator<> gen;
        auto graph = ComputingGraph::make();
        graph->options().graph_opt_level = 0;
        auto mkvar = [&](const char *name, const TensorShape &shp) {
            return opr::SharedDeviceTensor::make(
                    *graph, *gen(shp)).rename(name);
        };

        auto a = mkvar("a", sa), b = mkvar("b", sb),
             ab = gop(a, b),
             c = mkvar("c", sc),
             f = fop(ab, c);
        auto ret = trans.apply(f.node()->owner_opr());
        if (!expect_succ) {
            ASSERT_FALSE(ret.valid());
            return;
        }
        ASSERT_TRUE(ret.valid());

        auto ft = ret->result;
        ASSERT_NE(f.node(), ft);

        HostTensorND host_f, host_ft;
        graph->compile({make_callback_copy(f, host_f),
                make_callback_copy(ft, host_ft)})->execute();
        MGB_ASSERT_TENSOR_NEAR(host_f, host_ft, err);
    }

}

TEST(TestGoptGtrans, ExtractOprLeaves) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto v = [&](int idx) {
        auto hv = gen({1});
        return opr::Host2DeviceCopy::make(*graph, hv).rename(
                ssprintf("v%d", idx));
    };
    auto v0 = v(0), v1 = v(1), v2 = v(2), v3 = v(3),
         v4 = v(4), v5 = v(5), v6 = v(6);

    using Mode = opr::Elemwise::Mode;
    auto vt = opr::Elemwise::make(
            {(v0 + v1) - (v2 - v3),
            opr::Elemwise::make({v0, v5, v6}, Mode::FUSE_MUL_ADD3),
            v4 / v3 * v5},
            Mode::COND_LEQ_MOV);

    std::unordered_set<Mode, enumhash> allowed_modes;
    for (size_t i = 0; i < megdnn::param::Elemwise::MODE_NR_MEMBER; ++ i) {
        allowed_modes.insert(static_cast<Mode>(i));
    }
    auto pred = [&](cg::OperatorNodeBase *opr) -> bool{
        auto elem = gopt::try_cast_as_op<opr::Elemwise>(opr);
        if (elem)
            return allowed_modes.count(elem->param().mode);
        return false;
    };
    auto chain = gopt::extract_opr_leaves(vt.node(), pred);

    SymbolVarArray chain_expect = {
        v0, v1, v2, v3,
        v0, v5, v6,
        v5, v4, v3
    };
    ASSERT_EQ(chain_expect.size(), chain.size());
    for (size_t i = 0; i < chain.size(); ++ i) {
        ASSERT_EQ(chain_expect[i].node(), chain[i]);
    }
}

TEST(TestGoptGtrans, BinaryTrans20Elem) {
    run_binary_trans20_test(
            BinaryTrans20::associtive(),
            {5}, {5}, {5},
            BOP(a + b),
            BOP(a + b),
            true);
}

TEST(TestGoptGtrans, BinaryTrans20ConvMul) {
    run_binary_trans20_test(
            BinaryTrans20::associtive(),
            {2, 3, 6, 7}, {1, 3, 1, 1}, {5, 3, 3, 2},
            BOP(opr::Convolution::make(a, b)),
            BOP(a * b),
            true);
}

TEST(TestGoptGtrans, BinaryTrans20GroupConvMul) {
    opr::Convolution::Param p;
    p.sparse = opr::Convolution::Param::Sparse::GROUP;
    run_binary_trans20_test(
            BinaryTrans20::associtive(),
            {2, 6, 6, 7}, {1, 6, 1, 1}, {2, 2, 3, 3, 2},
            BOP(opr::Convolution::make(a, b, p)),
            BOP(a * b),
            true);
}

TEST(TestGoptGtrans, BinaryTrans20GroupConvMulScalar) {
    opr::Convolution::Param p;
    p.sparse = opr::Convolution::Param::Sparse::GROUP;
    run_binary_trans20_test(
            BinaryTrans20::associtive(),
            {2, 6, 6, 7}, {1}, {2, 2, 3, 3, 2},
            BOP(opr::Convolution::make(a, b, p)),
            BOP(a * b),
            true);
}

TEST(TestGoptGtrans, BinaryTrans20MatmulMul) {
    run_binary_trans20_test(
            BinaryTrans20::associtive(),
            {5, 3}, {1}, {3, 5},
            BOP(opr::MatrixMul::make(a, b, {false, false})),
            BOP(a * b),
            true);

    run_binary_trans20_test(
            BinaryTrans20::associtive(),
            {5, 3}, {1, 3}, {3, 5},
            BOP(opr::MatrixMul::make(a, b, {false, false})),
            BOP(a * b),
            true);

    run_binary_trans20_test(
            BinaryTrans20::associtive(),
            {5, 3}, {5, 1}, {3, 5},
            BOP(opr::MatrixMul::make(a, b, {false, false})),
            BOP(a * b),
            false);

    run_binary_trans20_test(
            BinaryTrans20::associtive(),
            {5, 3}, {5, 1}, {3, 5},
            BOP(opr::MatrixMul::make(a, b, {true, true})),
            BOP(a * b),
            true);
}

TEST(TestGoptGtrans, BinaryTrans20MulConv) {
    run_binary_trans20_test(
            BinaryTrans20::associtive(),
            {2, 3, 6, 7}, {2, 3, 3, 2}, {1, 2, 1, 1},
            BOP(a * b),
            BOP(opr::Convolution::make(a, b)),
            true);
}

TEST(TestGoptGtrans, BinaryTrans20MulGroupConv) {
    opr::Convolution::Param p;
    p.sparse = opr::Convolution::Param::Sparse::GROUP;
    run_binary_trans20_test(
            BinaryTrans20::associtive(),
            {2, 6, 6, 7}, {2, 2, 3, 3, 2}, {1, 4, 1, 1},
            BOP(a * b),
            BOP(opr::Convolution::make(a, b, p)),
            true);
}

TEST(TestGoptGtrans, BinaryTrans20MulGroupConvScalar) {
    opr::Convolution::Param p;
    p.sparse = opr::Convolution::Param::Sparse::GROUP;
    run_binary_trans20_test(
            BinaryTrans20::associtive(),
            {2, 6, 6, 7}, {2, 2, 3, 3, 2}, {1},
            BOP(a * b),
            BOP(opr::Convolution::make(a, b, p)),
            true);
}

TEST(TestGoptGtrans, BinaryTrans20MulMatmul) {
    run_binary_trans20_test(
            BinaryTrans20::associtive(),
            {5, 3}, {3, 5}, {1},
            BOP(a * b),
            BOP(opr::MatrixMul::make(a, b, {false, false})),
            true);

    run_binary_trans20_test(
            BinaryTrans20::associtive(),
            {5, 3}, {3, 5}, {1, 5},
            BOP(a * b),
            BOP(opr::MatrixMul::make(a, b, {false, false})),
            true);

    run_binary_trans20_test(
            BinaryTrans20::associtive(),
            {5, 3}, {3, 5}, {1, 3},
            BOP(a * b),
            BOP(opr::MatrixMul::make(a, b, {true, true})),
            true);
}

TEST(TestGoptGtrans, BinaryTrans20ConvAdd) {
    run_binary_trans20_test(
            BinaryTrans20::distributive_add(),
            {2, 3, 6, 7}, {1, 3, 1, 1}, {5, 3, 3, 2},
            BOP(opr::Convolution::make(a, b)),
            BOP(a + b),
            true);
}

TEST(TestGoptGtrans, BinaryTrans20GroupConvAdd) {
    opr::Convolution::Param p;
    p.sparse = opr::Convolution::Param::Sparse::GROUP;
    run_binary_trans20_test(
            BinaryTrans20::distributive_add(),
            {2, 6, 6, 7}, {1, 6, 1, 1}, {2, 2, 3, 3, 2},
            BOP(opr::Convolution::make(a, b, p)),
            BOP(a + b),
            true);
}


TEST(TestGoptGtrans, BinaryTrans20MatmulAdd) {
    run_binary_trans20_test(
            BinaryTrans20::distributive_add(),
            {5, 3}, {1}, {3, 5},
            BOP(opr::MatrixMul::make(a, b, {false, false})),
            BOP(a + b),
            true);

    run_binary_trans20_test(
            BinaryTrans20::distributive_add(),
            {5, 3}, {1, 3}, {3, 5},
            BOP(opr::MatrixMul::make(a, b, {false, false})),
            BOP(a + b),
            true);

    run_binary_trans20_test(
            BinaryTrans20::distributive_add(),
            {5, 3}, {1, 3}, {3, 5},
            BOP(opr::MatrixMul::make(a, b, {true, true})),
            BOP(a + b),
            false);

    run_binary_trans20_test(
            BinaryTrans20::distributive_add(),
            {5, 3}, {5, 1}, {3, 5},
            BOP(opr::MatrixMul::make(a, b, {true, true})),
            BOP(a + b),
            true);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

