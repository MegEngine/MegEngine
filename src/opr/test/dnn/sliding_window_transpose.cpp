/**
 * \file src/opr/test/dnn/sliding_window_transpose.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/megdnn_helper.h"
#include "megbrain/opr/dnn/sliding_window_transpose.h"

#include "megdnn/oprs.h"

using namespace mgb;

TEST(TestOprDNN, SlidingWindowTranspose) {
    using Checker = AutoOprChecker<1, 1>;

    opr::SlidingWindowTranspose::Param param;
    param.pad_h = 1;
    param.pad_w = 2;
    param.stride_w = 2;
    param.window_h = 4;
    param.dilate_h = 2;
    unsigned long ih = 16, iw = 15;
    unsigned long oh = (ih + 2 * param.pad_h - param.dilate_h * (param.window_h-1)-1) / param.stride_h + 1;
    unsigned long ow = (iw + 2 * param.pad_w - param.dilate_w * (param.window_w-1)-1) / param.stride_w + 1;
    param.out_h = ih;
    param.out_w = iw;

    auto make_graph = [&](const Checker::SymInpArray &inputs) ->
        Checker::SymOutArray {
        return {opr::SlidingWindowTranspose::make(inputs[0], param)};
    };

    auto fwd = [&](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto opr = megdnn_naive_handle()->
            create_operator<megdnn::SlidingWindowTranspose>();
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
    Checker checker{make_graph, fwd};
    
    checker.
        run({TensorShape{2, 3, oh, ow, param.window_h, param.window_w}}, opt).
        run({TensorShape{4, 5, oh, ow, param.window_h, param.window_w}}, opt).
        run({TensorShape{3, 2, oh, ow, param.window_h, param.window_w}}, opt);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
