/**
 * \file src/opr/test/dnn/roi_pooling.cpp
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
#include "megbrain/opr/dnn/roi_pooling.h"

#include "megdnn/oprs.h"

#include <cmath>
#include <random>
#include <sstream>
#include <iomanip>

using namespace mgb;

namespace {
using Param = opr::ROIPoolingForward::Param;

void run(Param::Mode mode) {
    RNGxorshf rng{next_rand_seed()};

    using Checker = AutoOprChecker<2, 2>;

    size_t N = 2, C = 3, M = 4;

    TensorShape rois_shp{M, 5};
    TensorShape dst_shp{M, C, 2, 3};

    Param param{mode, 6.f};

    auto make_graph = [&](const Checker::SymInpArray &inputs) ->
        Checker::SymOutArray {

        auto o0 = opr::ROIPoolingForward::make(inputs[0], inputs[1],
                TensorShape{dst_shp.shape[2], dst_shp.shape[3]},
                param);
        return {o0, o0.node()->owner_opr()->output(1)};
    };

    auto fwd = [&](Checker::NumOutArray &dest, Checker::NumInpArray inp) {
        auto opr = megdnn_naive_handle()->create_operator<megdnn::ROIPooling>();
        opr->param() = param;
        dest[0].dtype(dtype::Float32()).
            comp_node(inp[0]->comp_node()).resize(dst_shp);
        dest[1].dtype(dtype::Int32()).
            comp_node(inp[0]->comp_node()).resize(dst_shp);
        opr->exec(inp[0]->as_megdnn(), inp[1]->as_megdnn(),
                dest[0].as_megdnn(), dest[1].as_megdnn(), {});
    };
    auto rand_int = [&](int lo, int hi) {
        std::uniform_int_distribution<int> dist(lo, hi);
        return dist(rng);
    };
    auto rand_real = [&](float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        return dist(rng);
    };
    auto gen_rois = [&](HostTensorND &rois) {
        auto ptr = rois.ptr<float>();
        for (size_t i = 0; i < M; ++i) {
            ptr[0] = rand_int(0, N-1);
            ptr[1] = rand_real(0.0f, 1.0f);
            ptr[2] = rand_real(0.0f, 1.0f);
            ptr[3] = rand_real(0.0f, 1.0f);
            ptr[4] = rand_real(0.0f, 1.0f);
            if (ptr[1] > ptr[3])
                std::swap(ptr[1], ptr[3]);
            if (ptr[2] > ptr[4])
                std::swap(ptr[2], ptr[4]);
            ptr += 5;
        };
        mgb_assert(ptr == rois.ptr<float>() + rois.shape().total_nr_elems());
    };
    Checker::RunOptions option;
    option.numdiff_eps = 1e-3;
    option.numdiff_max_err = 1e-2;
    Checker checker{make_graph, fwd};

    checker.
        set_input_generator(1, gen_rois).
        // we cannot take gradient wrt. rois
        set_input_allow_grad(1, false).
        // we cannot take gradient wrt. temporary output
        set_output_allow_grad(1, false);

    if (mode == Param::Mode::AVERAGE)
        checker.set_output_allow_check(1, false);

    checker.
        run({TensorShape{M, C, 5, 6}, rois_shp}, option).
        run({TensorShape{M, C, 7, 8}, rois_shp}, option).
        run({TensorShape{M, C, 4, 2}, rois_shp}, option);
}
} // anonymous namespace

TEST(TestOprDNN, ROIPoolingMax) {
    run(Param::Mode::MAX);
}

TEST(TestOprDNN, ROIPoolingAverage) {
    run(Param::Mode::AVERAGE);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
