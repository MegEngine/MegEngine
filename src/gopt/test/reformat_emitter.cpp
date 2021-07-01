/**
 * \file src/gopt/test/reformat_emitter.cpp
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

#include "megbrain/gopt/reformat_emitter.h"
#include "megbrain/opr/tensor_manip.h"

using namespace mgb;

TEST(TestReformatEmitter, Basic) {
    constexpr size_t N = 12, C = 64, H = 7, W = 7;
    HostTensorGenerator<> gen;
    using NamedTensorShape = megdnn::NamedTensorShape;
    auto dest = NamedTensorShape::make_named_tensor_shape(
            NamedTensorShape::Format::NCHW4);
    auto src = NamedTensorShape::make_named_tensor_shape(
            NamedTensorShape::Format::NCHW32);
    auto reformat = gopt::ReformatEmitter(src, dest).emit();

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;

    auto nchw32_to_nchw4 = [](VarNode* in) {
        auto x = SymbolVar(in);
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
        return y2.node();
    };

    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };
    auto x = mkvar("x", {N, C / 32, H, W, 32});
    auto y1 = SymbolVar(reformat(x.node()));
    auto y2 = SymbolVar(nchw32_to_nchw4(x.node()));
    HostTensorND t1, t2;
    auto func1 = graph->compile({make_callback_copy(y1, t1)});
    func1->execute();
    auto func2 = graph->compile({make_callback_copy(y2, t2)});
    func2->execute();
    MGB_ASSERT_TENSOR_EQ(t1, t2);
}

TEST(TestReformatEmitter, MoreComplicated) {
    constexpr size_t N = 16, C = 64, H = 7, W = 7;
    HostTensorGenerator<> gen;
    using NamedTensorShape = megdnn::NamedTensorShape;
    auto src = NamedTensorShape::make_named_tensor_shape(
            NamedTensorShape::Format::NCHW64);
    auto dest = NamedTensorShape::make_named_tensor_shape(
            NamedTensorShape::Format::NCHW88);
    auto reformat = gopt::ReformatEmitter(src, dest).emit();

    auto graph = ComputingGraph::make();
    graph->options().graph_opt_level = 0;
    auto mkvar = [&](const char* name, const TensorShape& shp) {
        return opr::Host2DeviceCopy::make(*graph, gen(shp)).rename(name);
    };
    auto x = mkvar("x", {N, C / 64, H, W, 64});
    auto y = SymbolVar(reformat(x.node()));
    HostTensorND t;
    auto func = graph->compile({make_callback_copy(y, t)});
    func->execute();
}
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
