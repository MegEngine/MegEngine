/**
 * \file src/opr/test/tensor_gen.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/opr/tensor_gen.h"
#include "megbrain/opr/io.h"

using namespace mgb;
using namespace opr;

TEST(TestTensorGen, Alloc) {
    auto host_x = std::make_shared<HostTensorND>(
            CompNode::load("xpu0"), dtype::Int32());
    auto graph = ComputingGraph::make();
    auto x = opr::Host2DeviceCopy::make_no_value_infer(*graph, host_x),
         y = opr::Alloc::make(x, dtype::Float32());
    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    auto px = host_x->resize({3}).ptr<int>();
    px[0] = 2;
    px[1] = 3;
    px[2] = 5;
    func->execute();

    ASSERT_EQ(TensorShape({2, 3, 5}), host_y.shape());
}

TEST(TestTensorGen, Linspace) {
    auto host_num = std::make_shared<HostTensorND>(
        CompNode::load("xpu0"), dtype::Int32());
    host_num->resize({1}).ptr<int>()[0] = 30;
    using Checker = AutoOprChecker<2, 1>;
    for (auto endpoint: {false, true}) {
        auto make_graph = [endpoint, &host_num](
                const Checker::SymInpArray &inputs) -> Checker::SymOutArray {
            auto num = opr::Host2DeviceCopy::make(
                    *inputs[0].node()->owner_graph(), host_num).rename("num");
            return {opr::Linspace::make(
                    inputs[0].rename("start"),
                    inputs[1].rename("stop"),
                    num, {endpoint}).rename("linspace")};
        };

        auto fwd = [&](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
            size_t num = host_num->ptr<int>()[0];
            auto ptr = dest[0].resize({num}).ptr<float>();
            auto start = *inp[0]->ptr<float>(), stop = *inp[1]->ptr<float>(),
                 step = (stop - start) / std::max<int>((num - endpoint), 1);
            for (size_t i = 0; i < num; ++ i)
                ptr[i] = start + step * i;
        };
        Checker::RunOptions opt;
        opt.numdiff_eps = 1; // large eps because all linear
        std::array<TensorShape, 2> ishp{TensorShape{1}, {1}};
        Checker checker(make_graph, fwd);
        host_num->ptr<int>()[0] = 30;
        checker.
            run(ishp, opt).
            run(ishp, opt);
        host_num->ptr<int>()[0] = 1;
        checker.run(ishp, opt);
    }
}

TEST(TestTensorGen, Eye) {
    auto graph = ComputingGraph::make();
    auto x = opr::Eye::make(
            SymbolVar::make_scalar(5, *graph, CompNode::load("xpu0")),
            {-1, DTypeEnum::Int32});
    HostTensorND host_x;
    auto func = graph->compile({make_callback_copy(x, host_x)});
    func->execute();

    ASSERT_EQ(TensorShape({5, 5}), host_x.shape());
    auto ptr = host_x.ptr<int>();
    for (int i = 0; i < 5; ++ i) {
        for (int j = 0; j < 5; ++ j)
            ASSERT_EQ(*(ptr ++), i - j - 1 == 0);
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

