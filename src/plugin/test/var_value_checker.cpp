/**
 * \file src/plugin/test/var_value_checker.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/io.h"
#include "megbrain/opr/utility.h"
#include "megbrain/plugin/var_value_checker.h"
#include "megbrain/test/helper.h"

using namespace mgb;

TEST(TestVarValueChecker, Simple) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    VarValueChecker checker(graph.get(), 2);
    bool should_fail = false;
    auto cb = [&should_fail](DeviceTensorND &dv) {
        if (!should_fail)
            return;
        HostTensorND hv;
        hv.copy_from(dv).sync();
        hv.ptr<float>()[0] += 1;
        dv.copy_from(hv).sync();
    };

    auto host_x = gen({3});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x),
         y = x + 1,
         z = opr::CallbackInjector::make(y, cb);
    auto func = graph->compile({{z, {}}});
    func->execute();
    should_fail = true;
    for (int i = 0; i < 6; ++ i) {
        // run 6 times becore x, ADD, IMM(1) are not modified
        func->execute();
    }
    ASSERT_THROW(func->execute(), MegBrainError);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
