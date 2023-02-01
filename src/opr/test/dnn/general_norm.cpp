#include "megbrain/opr/dnn/general_norm.h"
#include "megbrain/comp_node_env.h"
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
using Param = opr::GeneralNormForward::Param;

void run_forward(bool is_affine, size_t normalized_size, size_t normalized_axis) {
    using Checker = AutoOprChecker<3, 3>;

    printf("for test\n");
    Param param;
    param.eps = 1e-5;
    param.affine = is_affine;
    param.normalized_axis = normalized_axis;

    auto make_graph = [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        auto out = opr::GeneralNormForward::make(inputs[0], inputs[1], inputs[2], param);
        return {out[0], out[1], out[2]};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto opr =
                MegDNNHandle::get(CompNodeEnv::from_comp_node(CompNode::default_cpu()))
                        ->create_operator<megdnn::GeneralNormForward>();
        auto inp_shape = inp[0]->shape();
        auto n_slices = inp_shape[0];
        auto slice_len = inp_shape[1];

        opr->param() = param;

        dest[0].dtype(dtype::Float32())
                .comp_node(inp[0]->comp_node())
                .resize({n_slices, slice_len});
        dest[1].dtype(dtype::Float32())
                .comp_node(inp[0]->comp_node())
                .resize({n_slices});
        dest[2].dtype(dtype::Float32())
                .comp_node(inp[0]->comp_node())
                .resize({n_slices});
        opr->exec(
                inp[0]->as_megdnn(), inp[1]->as_megdnn(), inp[2]->as_megdnn(),
                dest[0].as_megdnn(), dest[1].as_megdnn(), dest[2].as_megdnn(), {});
    };

    auto gen = [&](HostTensorND& src) {
        HostTensorGenerator<dtype::Float32, RandomDistribution::GAUSSIAN> src_gen(0.f);
        src = *src_gen(src.shape(), src.comp_node());
    };

    Checker::RunOptions option;
    option.numdiff_max_err = 1e-4;
    Checker checker{make_graph, fwd};

    checker.set_input_generator(0, gen);
    checker.set_input_generator(1, gen);
    checker.set_input_generator(2, gen);
    checker.set_input_allow_grad(0, false);
    checker.set_input_allow_grad(1, false);
    checker.set_input_allow_grad(2, false);
    checker.set_output_allow_grad(0, false);
    checker.set_output_allow_grad(1, false);
    checker.set_output_allow_grad(2, false);

    checker.run({TensorShape{normalized_size, normalized_size},
                 TensorShape{normalized_size}, TensorShape{normalized_size}},
                option)
            .run({TensorShape{normalized_size, normalized_size},
                  TensorShape{normalized_size}, TensorShape{normalized_size}},
                 option)
            .run({TensorShape{normalized_size, normalized_size},
                  TensorShape{normalized_size}, TensorShape{normalized_size}},
                 option);
}

TEST(TestOprDNN, GeneralNormForwardAffine) {
    REQUIRE_GPU(1);
    run_forward(true, 1, 0);
    run_forward(true, 16, 0);
    run_forward(true, 17, 0);
}

}  // anonymous namespace

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
