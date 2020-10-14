/**
 * \file src/opr/test/dnn/adaptive_pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/opr/dnn/adaptive_pooling.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/megdnn_helper.h"
#include "megdnn/dtype.h"
#include "megdnn/opr_param_defs.h"

using namespace std;
using namespace mgb;

namespace {

using Param = opr::AdaptivePoolingForward::Param;
void run(Param::Mode mode) {
    using Checker = AutoOprChecker<2, 1>;
    Param param{mode};

    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        auto o0 = opr::GetVarShape::make(inputs[1]);
        auto o1 = opr::AdaptivePoolingForward::make(inputs[0], o0, param);
        return {o1};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto opr = MegDNNHandle::get(
                           CompNodeEnv::from_comp_node(CompNode::default_cpu()))
                           ->create_operator<megdnn::AdaptivePoolingForward>();
        opr->param() = param;
        size_t N = inp[0].get()->shape(0), C = inp[0].get()->shape(1);
        size_t OH = inp[1].get()->shape(0), OW = inp[1].get()->shape(1);
        dest[0].resize(TensorShape{N, C, OH, OW});
        opr->exec(inp[0]->as_megdnn(), dest[0].as_megdnn(), {});
    };

    auto gen = [&](HostTensorND& src) {
        if (mode == Param::Mode::MAX) {
            HostTensorGenerator<dtype::Float32, RandomDistribution::CONSECUTIVE>
                    src_gen(1.0f, 0.1f);
            src = *src_gen(src.shape(), src.comp_node());
        } else {
            HostTensorGenerator<dtype::Float32, RandomDistribution::GAUSSIAN>
                    src_gen(10.f);
            src = *src_gen(src.shape(), src.comp_node());
        }
    };

    Checker::RunOptions opt;
    opt.numdiff_max_err = 1e-2;

    Checker checker{make_graph, fwd};
    checker.set_input_allow_grad(1, false)
           .set_input_generator(0, gen);
    checker.run({TensorShape{1, 1, 10, 7}, TensorShape{5, 4}}, opt);
    checker.run({TensorShape{1, 1, 9, 7}, TensorShape{5, 4}}, opt);
    checker.run({TensorShape{1, 2, 8, 9}, TensorShape{3, 4}}, opt);
}

}  // anonymous namespace

TEST(TestOprDNN, AdaptivePoolingMax) {
    run(Param::Mode::MAX);
}

TEST(TestOprDNN, AdaptivePoolingAverage) {
    run(Param::Mode::AVERAGE);
}

TEST(TestOprDNN, AdaptivePoolingAverageCountExcludePadding) {
    run(Param::Mode::AVERAGE_COUNT_EXCLUDE_PADDING);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
