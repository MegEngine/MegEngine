/**
 * \file src/opr/test/dnn/tqt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/opr/dnn/tqt.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/test/autocheck.h"

using namespace std;
using namespace mgb;

namespace {

void run() {
    using Checker = AutoOprChecker<2, 1>;

    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        auto o0 = opr::TQTForward::make(inputs[0], inputs[1]);
        return {o0};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto opr = MegDNNHandle::get(
                           CompNodeEnv::from_comp_node(CompNode::default_cpu()))
                           ->create_operator<megdnn::TQTForward>();
        dest[0].dtype(dtype::Float32())
                .comp_node(inp[0]->comp_node())
                .resize(inp[0]->shape());
        opr->exec(inp[0]->as_megdnn(), inp[1]->as_megdnn(), dest[0].as_megdnn(),
                  {});
    };

    auto gen = [&](HostTensorND& src) {
        HostTensorGenerator<dtype::Float32, RandomDistribution::GAUSSIAN>
                src_gen(10.f);
        src = *src_gen(src.shape(), src.comp_node());
    };

    Checker::RunOptions opt;
    opt.numdiff_max_err = 1e-5;

    Checker checker{make_graph, fwd};
    checker.set_input_generator(0, gen)
            .set_input_generator(1, gen)
            .set_input_allow_grad(0, false)
            .set_input_allow_grad(1, false)
            .set_output_allow_grad(0, false);
    checker.run({TensorShape{1, 2, 3, 4}, TensorShape{1}}, opt)
            .run({TensorShape{2, 3, 8, 8}, TensorShape{1}}, opt)
            .run({TensorShape{1, 3, 4, 4}, TensorShape{1}}, opt);
}

}  // anonymous namespace

TEST(TestOprDNN, TQTForward) {
    REQUIRE_GPU(1);
    run();
}
