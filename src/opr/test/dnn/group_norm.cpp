#include "megbrain/opr/dnn/group_norm.h"
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
using Param = opr::GroupNormForward::Param;

void run_forward(bool is_affine) {
    using Checker = AutoOprChecker<3, 3>;

    Param param;
    param.eps = 1e-5;
    param.affine = is_affine;
    param.group = 3;

    auto make_graph = [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        auto out = opr::GroupNormForward::make(inputs[0], inputs[1], inputs[2], param);
        return {out[0], out[1], out[2]};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto opr =
                MegDNNHandle::get(CompNodeEnv::from_comp_node(CompNode::default_cpu()))
                        ->create_operator<megdnn::GroupNormForward>();
        auto inp_shape = inp[0]->shape();
        auto n_slices = inp_shape[0];

        opr->param() = param;

        dest[0].dtype(dtype::Float32())
                .comp_node(inp[0]->comp_node())
                .resize(inp_shape);
        dest[1].dtype(dtype::Float32())
                .comp_node(inp[0]->comp_node())
                .resize({n_slices, param.group});
        dest[2].dtype(dtype::Float32())
                .comp_node(inp[0]->comp_node())
                .resize({n_slices, param.group});
        std::vector<dt_byte> workspace(opr->get_workspace_in_bytes(
                inp[0]->layout(), inp[1]->layout(), inp[2]->layout(), dest[0].layout(),
                dest[1].layout(), dest[2].layout()));
        opr->exec(
                inp[0]->as_megdnn(), inp[1]->as_megdnn(), inp[2]->as_megdnn(),
                dest[0].as_megdnn(), dest[1].as_megdnn(), dest[2].as_megdnn(),
                {workspace.data(), workspace.size()});
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

    checker.run({TensorShape{2, 6, 2, 1}, TensorShape{6}, TensorShape{6}}, option)
            .run({TensorShape{2, 6, 2, 1}, TensorShape{6}, TensorShape{6}}, option)
            .run({TensorShape{2, 6, 2, 1}, TensorShape{6}, TensorShape{6}}, option);
}

TEST(TestOprDNN, GroupNormForward) {
    REQUIRE_GPU(1);
    run_forward(true);
}

}  // anonymous namespace

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
