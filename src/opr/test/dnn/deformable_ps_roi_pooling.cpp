/**
 * \file src/opr/test/dnn/deformable_ps_roi_pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/dnn/roi_pooling.h"
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

auto gen_rois_helper = [](HostTensorND& rois, size_t M, size_t N) {
    RNGxorshf rng{next_rand_seed()};
    auto ptr = rois.ptr<float>();

    auto rand_int = [](RNGxorshf& rng, int lo, int hi) {
        std::uniform_int_distribution<int> dist(lo, hi);
        return dist(rng);
    };
    auto rand_real = [](RNGxorshf& rng, float lo, float hi) {
        std::uniform_real_distribution<float> dist(lo, hi);
        return dist(rng);
    };

    for (size_t i = 0; i < M; ++i) {
        ptr[0] = rand_int(rng, 0, N - 1);
        ptr[1] = rand_real(rng, 0.0f, 1.0f);
        ptr[2] = rand_real(rng, 0.0f, 1.0f);
        ptr[3] = rand_real(rng, 0.0f, 1.0f);
        ptr[4] = rand_real(rng, 0.0f, 1.0f);
        if (ptr[1] > ptr[3])
            std::swap(ptr[1], ptr[3]);
        if (ptr[2] > ptr[4])
            std::swap(ptr[2], ptr[4]);
        ptr += 5;
    };
    mgb_assert(ptr == rois.ptr<float>() + rois.shape().total_nr_elems());
};

namespace deformable_ps_roi_pooling {

using Param = opr::DeformablePSROIPooling::Param;
void run(size_t N, size_t C, size_t M, size_t PH, size_t PW, bool no_trans,
         size_t nr_cls, size_t part_sz, size_t sample_per_part, float trans_std,
         float spatial_scale) {
    using Checker = AutoOprChecker<3, 2>;

    TensorShape rois_shp{M, 5};
    TensorShape trans_shp{nr_cls, 2, PH, PW};
    TensorShape dst_shp{M, C, PH, PW};

    Param param;
    param.no_trans = no_trans;
    param.pooled_h = PH;
    param.pooled_w = PW;
    param.trans_std = trans_std;
    param.spatial_scale = spatial_scale;
    param.part_size = part_sz;
    param.sample_per_part = sample_per_part;

    auto gen_rois = [&](HostTensorND& rois) { gen_rois_helper(rois, M, N); };
    auto make_graph =
            [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {

        auto o0 = opr::DeformablePSROIPoolingForward::make(inputs[0], inputs[1],
                                                           inputs[2], param);
        return {o0, o0.node()->owner_opr()->output(1)};
    };
    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto opr = megdnn_naive_handle()
                           ->create_operator<megdnn::DeformablePSROIPooling>();
        opr->param() = param;
        dest[0].dtype(dtype::Float32())
                .comp_node(inp[0]->comp_node())
                .resize(dst_shp);
        dest[1].dtype(dtype::Float32())
                .comp_node(inp[0]->comp_node())
                .resize(dst_shp);
        opr->exec(inp[0]->as_megdnn(), inp[1]->as_megdnn(), inp[2]->as_megdnn(),
                  dest[0].as_megdnn(), dest[1].as_megdnn(), {});
    };

    Checker checker{make_graph, fwd};

    Checker::RunOptions option;
    option.numdiff_eps = 1e-3;
    option.numdiff_max_err = 1e-1;

    //! we canot take gradient wrt. rois and output_count
    checker.disable_grad_check();
    checker.set_input_generator(1, gen_rois)
            .set_input_allow_grad(1, false)
            .set_output_allow_grad(1, false);

    checker.run({TensorShape{M, C, 5, 6}, rois_shp, trans_shp}, option)
            .run({TensorShape{M, C, 7, 8}, rois_shp, trans_shp}, option)
            .run({TensorShape{M, C, 8, 9}, rois_shp, trans_shp}, option);
}
}  // namespace deformable_ps_roi_pooling
}  // anonymous namespace

TEST(TestOprDNN, DeformablePSROIPoolingForward) {
    using namespace ::deformable_ps_roi_pooling;
    run(2, 3, 4, 2, 3, false, 2, 1, 1, 1.5f, 1.2f);
    run(2, 3, 4, 2, 3, true, 2, 1, 1, 1.5f, 1.2f);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
