/**
 * \file src/gopt/test/reformat_manager.cpp
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

#include "megbrain/gopt/reformat_manager.h"
#include "megbrain/opr/tensor_manip.h"

using namespace mgb;
using namespace gopt;

TEST(TestReformatManager, Feature) {
    constexpr size_t N = 16, C = 128, H = 7, W = 7;
    HostTensorGenerator<> gen;
    using ReformatKey = ReformatManager::ReformatKey;
    auto src_format = TensorFormats::NHWC, dst_format = TensorFormats::NCHWc64;
    ReformatKey key{src_format, dst_format};
    auto reformat = ReformatManager::instance().get(key);

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    auto r = [](VarNode* inp) {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make(
                {sub(0), sub(1), sub(2), sub(3) / 64, cv(64)}, 0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 3, 1, 2, 4});
        return y1;
    };

    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };
    auto x = mkvar("x", {N, H, W, C});
    auto y1 = SymbolVar(reformat({x.node()}));
    auto y2 = r(x.node());
    size_t nr_shapeof = 0;
    size_t nr_reshape = 0;
    cg::DepOprIter{[&nr_shapeof, &nr_reshape](cg::OperatorNodeBase* o) {
        if (o->same_type<opr::GetVarShape>())
            nr_shapeof++;
        if (o->same_type<opr::Reshape>())
            nr_reshape++;
    }}
            .add(y1.node()->owner_opr());
    ASSERT_EQ(nr_shapeof, 1);
    ASSERT_EQ(nr_reshape, 1);
    HostTensorND t1, t2;
    auto func1 = graph->compile({make_callback_copy(y1, t1)});
    func1->execute();
    auto func2 = graph->compile({make_callback_copy(y2, t2)});
    func2->execute();
    MGB_ASSERT_TENSOR_EQ(t1, t2);
}

TEST(TestReformatManager, Weight) {
    constexpr size_t G = 8, K = 128, C = 128, R = 3, S = 3;
    HostTensorGenerator<> gen;
    using ReformatKey = ReformatManager::ReformatKey;
    auto src_format = TensorFormats::GKCRS,
         dst_format = TensorFormats::GKCRSk4c4;
    ReformatKey key{src_format, dst_format};
    auto reformat = ReformatManager::instance().get(key);

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    auto r = [](VarNode* inp) {
        auto x = SymbolVar(inp);
        auto xshp = opr::GetVarShape::make(x);
        auto cv = [&x](int v) { return x.make_scalar(v); };
        auto sub = [&xshp, &cv](int idx) {
            return opr::IndexAt::make(xshp, {{0, cv(idx)}});
        };
        auto tshp0 = opr::Concat::make({sub(0), sub(1) / 4, cv(4), sub(2) / 4,
                                        cv(4), sub(3), sub(4)},
                                       0),
             tshp1 = opr::Concat::make({sub(0), sub(1) / 4, sub(2) / 4, sub(3),
                                        sub(4), cv(4), cv(4)},
                                       0);
        auto y0 = opr::Reshape::make(x, tshp0);
        auto y1 = opr::Dimshuffle::make(y0, {0, 1, 3, 5, 6, 2, 4});
        auto y2 = opr::Reshape::make(y1, tshp1);
        return y2;
    };

    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };
    auto w = mkvar("w", {G, K / G, C / G, R, S});
    auto y1 = SymbolVar(reformat({w.node()}));
    auto y2 = r(w.node());
    size_t nr_shapeof = 0;
    size_t nr_reshape = 0;
    cg::DepOprIter{[&nr_shapeof, &nr_reshape](cg::OperatorNodeBase* o) {
        if (o->same_type<opr::GetVarShape>())
            nr_shapeof++;
        if (o->same_type<opr::Reshape>())
            nr_reshape++;
    }}
            .add(y1.node()->owner_opr());
    ASSERT_EQ(nr_shapeof, 1);
    ASSERT_EQ(nr_reshape, 1);
    HostTensorND t1, t2;
    auto func1 = graph->compile({make_callback_copy(y1, t1)});
    func1->execute();
    auto func2 = graph->compile({make_callback_copy(y2, t2)});
    func2->execute();
    MGB_ASSERT_TENSOR_EQ(t1, t2);
}

TEST(TestReformatManager, InvalidKey) {
    using ReformatKey = ReformatManager::ReformatKey;
    using Attribute = ReformatKey::Attribute;
    auto src_format = TensorFormats::GKCRS,
         dst_format = TensorFormats::GKCRSk4c4;
    Attribute attribute = Attribute::IMAGE2D;
    ReformatKey key{src_format, dst_format, attribute};
    ASSERT_THROW(ReformatManager::instance().get(key), AssertionError);
}

TEST(TestReformatManager, InputChannelSmall) {
    constexpr size_t N = 16, C = 3, H = 224, W = 224;
    auto cn = CompNode::load("cpux");
    HostTensorGenerator<> gen;
    using ReformatKey = ReformatManager::ReformatKey;
    using Attribute = ReformatKey::Attribute;
    auto src_format = TensorFormats::NCHW, dst_format = TensorFormats::NCHWc4;
    ReformatKey key{src_format, dst_format, Attribute::IC_SMALL};
    auto reformat = ReformatManager::instance().get(key);

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    auto r = [](VarNode* inp) {
        auto x = SymbolVar(inp);
        auto y = opr::RelayoutFormat::make(
                x, megdnn::param::RelayoutFormat::Mode::NCHW_NCHW4_IC_SMALL);
        return y;
    };

    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp, cn)).rename(name);
    };
    auto x = mkvar("x", {N, C, H, W});
    auto y1 = SymbolVar(reformat({x.node()}));
    auto y2 = r(x.node());
    HostTensorND t1, t2;
    auto func1 = graph->compile({make_callback_copy(y1, t1)});
    func1->execute();
    auto func2 = graph->compile({make_callback_copy(y2, t2)});
    func2->execute();
    MGB_ASSERT_TENSOR_EQ(t1, t2);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
