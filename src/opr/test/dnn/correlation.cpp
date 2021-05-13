/**
 * \file src/opr/test/dnn/correlation.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/opr/dnn/correlation.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/helper.h"
#include "megbrain/test/megdnn_helper.h"

#include "megdnn/oprs.h"

#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>

using namespace mgb;

namespace {
using Param = opr::CorrelationForward::Param;

void run_forward(bool is_multiply) {
    RNGxorshf rng{next_rand_seed()};
    using Checker = AutoOprChecker<2, 1>;

    Param param;
    param.format = Param::Format::NCHW;
    param.is_multiply = is_multiply;
    param.kernel_size = 3;
    param.max_displacement = 2;
    param.pad_size = 1;
    param.stride1 = 2;
    param.stride2 = 2;

    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        auto o0 = opr::CorrelationForward::make(inputs[0], inputs[1], param);
        return {o0};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto opr = megdnn_naive_handle()
                           ->create_operator<megdnn::CorrelationForward>();
        opr->param() = param;
        auto inp_shape = inp[0]->shape();
        auto num = inp_shape[0];
        auto height = inp_shape[2];
        auto width = inp_shape[3];

        uint32_t pad_size = param.pad_size;
        uint32_t kernel_size = param.kernel_size;
        uint32_t stride1 = param.stride1;
        uint32_t stride2 = param.stride2;
        uint32_t max_displacement = param.max_displacement;

        int paddedbottomheight = height + 2 * pad_size;
        int paddedbottomwidth = width + 2 * pad_size;
        uint32_t kernel_radius = (kernel_size - 1) / 2;
        uint32_t border_size = max_displacement + kernel_radius;
        uint32_t top_width =
                ceil(static_cast<float>(paddedbottomwidth - border_size * 2) /
                     static_cast<float>(stride1));
        uint32_t top_height =
                ceil(static_cast<float>(paddedbottomheight - border_size * 2) /
                     static_cast<float>(stride1));
        uint32_t neighborhood_grid_radius = max_displacement / stride2;
        uint32_t neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;
        uint32_t top_channels =
                neighborhood_grid_width * neighborhood_grid_width;
        megdnn::TensorShape target_shape{num, top_channels, top_height,
                                         top_width};

        dest[0].dtype(dtype::Float32())
                .comp_node(inp[0]->comp_node())
                .resize(target_shape);
        opr->exec(inp[0]->as_megdnn(), inp[1]->as_megdnn(), dest[0].as_megdnn(),
                  {});
    };

    auto rand_real = [&](float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        return dist(rng);
    };
    auto gen_inp1 = [&](HostTensorND &inp) {
        auto ptr = inp.ptr<float>();
        for (size_t i = 0; i < inp.shape().total_nr_elems(); ++i) {
            ptr[i] = rand_real(0.06f, 0.1f);
        };
    };

    auto gen_inp2 = [&](HostTensorND &inp) {
        auto ptr = inp.ptr<float>();
        for (size_t i = 0; i < inp.shape().total_nr_elems(); ++i) {
            ptr[i] = rand_real(0.01f, 0.04f);
        };
    };

    Checker::RunOptions option;
    option.numdiff_eps = 1e-3;
    option.numdiff_max_err = 1e-2;
    Checker checker{make_graph, fwd};

    checker.set_input_generator(0, gen_inp1);
    checker.set_input_generator(1, gen_inp2);

    checker.run({TensorShape{1, 1, 10, 10}, TensorShape{1, 1, 10, 10}}, option)
            .run({TensorShape{1, 3, 50, 50}, TensorShape{1, 3, 50, 50}}, option)
            .run({TensorShape{1, 1, 100, 100}, TensorShape{1, 1, 100, 100}},
                 option);
}

TEST(TestOprDNN, CorrelationForwardMultiply) {
    // TODO: fix me, add correct backward of cpu
    REQUIRE_GPU(1);
    run_forward(true);
}

TEST(TestOprDNN, CorrelationForwardSubstract) {
    // TODO: fix me, add correct backward of cpu
    REQUIRE_GPU(1);
    run_forward(false);
}

}  // anonymous namespace

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
