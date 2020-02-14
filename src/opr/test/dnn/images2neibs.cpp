/**
 * \file src/opr/test/dnn/images2neibs.cpp
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
#include "megbrain/test/megdnn_helper.h"
#include "megbrain/opr/dnn/images2neibs.h"

#include "megdnn/oprs.h"

using namespace mgb;

TEST(TestOprDNN, Images2Neibs) {
    using Checker = AutoOprChecker<1, 1>;

    opr::Images2Neibs::Param param;
    param.pad_h = 1;
    param.pad_w = 2;
    param.stride_w = 2;
    param.window_h = 4;

    auto make_graph = [&](const Checker::SymInpArray &inputs) ->
        Checker::SymOutArray {
        return {opr::Images2Neibs::make(inputs[0], param)};
    };

    auto fwd = [&](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto opr = megdnn_naive_handle()->
            create_operator<megdnn::Images2Neibs>();
        opr->param() = param;
        TensorLayout dest_layout;
        opr->deduce_layout(inp[0]->layout(), dest_layout);
        std::vector<dt_byte> workspace(
                opr->get_workspace_in_bytes(inp[0]->layout(), dest_layout));
        dest[0].dtype(dtype::Float32()).
            comp_node(inp[0]->comp_node()).resize(dest_layout);
        opr->exec(inp[0]->as_megdnn(), dest[0].as_megdnn(),
                {workspace.data(), workspace.size()});
    };
    Checker::RunOptions opt;
    opt.numdiff_eps = 1;

    Checker(make_graph, fwd).
        run({TensorShape{2, 3, 8, 7}}, opt).
        run({TensorShape{4, 5, 6, 3}}, opt).
        run({TensorShape{3, 2, 7, 5}}, opt);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
