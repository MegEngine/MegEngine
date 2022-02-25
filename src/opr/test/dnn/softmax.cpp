#include "megbrain/opr/dnn/softmax.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/test/autocheck.h"

using namespace std;
using namespace mgb;

namespace {
using Param = opr::SoftmaxForward::Param;
void run(int32_t axis) {
    using Checker = AutoOprChecker<1, 1>;
    Param param{axis};

    auto make_graph = [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        auto o0 = opr::SoftmaxForward::make(inputs[0], param);
        return {o0};
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto opr =
                MegDNNHandle::get(CompNodeEnv::from_comp_node(CompNode::default_cpu()))
                        ->create_operator<megdnn::SoftmaxForward>();
        opr->param() = param;
        dest[0].dtype(dtype::Float32())
                .comp_node(inp[0]->comp_node())
                .resize(inp[0]->shape());
        size_t wk_size =
                opr->get_workspace_in_bytes(inp[0]->layout(), dest[0].layout());
        std::unique_ptr<dt_byte[]> wk_store{new dt_byte[wk_size]};
        opr->exec(inp[0]->as_megdnn(), dest[0].as_megdnn(), {wk_store.get(), wk_size});
    };

    auto gen = [&](HostTensorND& src) {
        HostTensorGenerator<dtype::Float32, RandomDistribution::GAUSSIAN> src_gen(10.f);
        src = *src_gen(src.shape(), src.comp_node());
    };

    Checker::RunOptions opt;
    opt.numdiff_max_err = 1e-4;

    Checker checker{make_graph, fwd};
    checker.set_input_generator(0, gen);
    checker.run({TensorShape{1, 2, 3, 4}}, opt)
            .run({TensorShape{2, 3, 8, 8}}, opt)
            .run({TensorShape{1, 3, 4, 4}}, opt);
}

}  // anonymous namespace

TEST(TestOprDNN, SoftmaxForward) {
    REQUIRE_GPU(1);
    run(1);
}