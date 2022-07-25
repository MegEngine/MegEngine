#include "./legacy_checker.h"
#include "megbrain/comp_node_env.h"

#include "megbrain/gopt/inference.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/helper.h"
#include "megbrain/test/megdnn_helper.h"
#include "megdnn/algorithm_cache.h"
#include "megdnn/dtype.h"
#include "megdnn/oprs/base.h"

#include <gmock/gmock.h>

#include <cmath>
#include <memory>
#include <random>

using namespace mgb;

TEST(TestOprDNN, REGIONCONV_FWD_CPU_WRAPPER) {
    using Checker = AutoOprChecker<4, 1>;
    megdnn::RegionRestrictedConvolution::Param param;
    param.sparse = opr::RegionRestrictedConvolution::Param::Sparse::DENSE;

    auto make_graph = [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        return {opr::RegionRestrictedConvolutionForward::make(
                inputs[0], inputs[1], inputs[2], inputs[3], param)};
    };

    Checker::RunOptions option;
    option.numdiff_eps = 0.1;
    option.numdiff_max_err = 1e-2;

    auto mask_gen = [&](HostTensorND& src) {
        HostTensorGenerator<dtype::Int32, RandomDistribution::CONSTANT> gen(1);
        src = *gen(src.shape(), src.comp_node());
    };
    auto float_gen = [&](HostTensorND& src) {
        HostTensorGenerator<dtype::Float32, RandomDistribution::GAUSSIAN> gen;
        src = *gen(src.shape(), src.comp_node());
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto opr =
                megdnn_naive_handle()
                        ->create_operator<megdnn::RegionRestrictedConvolutionForward>();
        opr->param() = param;
        TensorLayout dest_layout;
        opr->deduce_layout(
                inp[0]->layout(), inp[1]->layout(), inp[2]->layout(), inp[3]->layout(),
                dest_layout);
        std::vector<dt_byte> workspace(opr->get_workspace_in_bytes(
                inp[0]->layout(), inp[1]->layout(), inp[2]->layout(), inp[3]->layout(),
                dest_layout));
        dest[0].dtype(inp[0]->dtype())
                .comp_node(inp[0]->comp_node())
                .resize(dest_layout);
        opr->exec(
                inp[0]->as_megdnn(), inp[1]->as_megdnn(), inp[2]->as_megdnn(),
                inp[3]->as_megdnn(), dest[0].as_megdnn(),
                {workspace.data(), workspace.size()});
    };

    Checker(make_graph, fwd, CompNode::load("cpu0"))
            .set_input_dtype(0, dtype::Float32())
            .set_input_dtype(1, dtype::Float32())
            .set_input_dtype(2, dtype::Int32())
            .set_input_dtype(3, dtype::Int32())
            .set_input_generator(0, float_gen)
            .set_input_generator(1, float_gen)
            .set_input_generator(2, mask_gen)
            .set_input_generator(3, mask_gen)
            .set_input_allow_grad(2, false)
            .set_input_allow_grad(3, false)
            // {n,ic,ih,iw}, {oc,ic,fh,fw}, {n,ih,iw}, {n,oh,ow}
            .run({TensorShape{1, 2, 2, 2}, TensorShape{1, 2, 2, 2},
                  TensorShape{1, 2, 2}, TensorShape{1, 1, 1}},
                 option)
            .run({TensorShape{1, 2, 3, 3}, TensorShape{1, 2, 3, 3},
                  TensorShape{1, 3, 3}, TensorShape{1, 1, 1}},
                 option)
            .run({TensorShape{1, 1, 4, 4}, TensorShape{1, 1, 2, 2},
                  TensorShape{1, 4, 4}, TensorShape{1, 3, 3}},
                 option)
            .run({TensorShape{2, 2, 8, 8}, TensorShape{4, 2, 2, 2},
                  TensorShape{2, 8, 8}, TensorShape{2, 7, 7}},
                 option)
            .run({TensorShape{4, 4, 8, 8}, TensorShape{4, 4, 2, 2},
                  TensorShape{4, 8, 8}, TensorShape{4, 7, 7}},
                 option);
}

#if MGB_CUDA
TEST(TestOprDNN, REGIONCONV_FWD_GPU_WRAPPER) {
    using Checker = AutoOprChecker<4, 1>;
    megdnn::RegionRestrictedConvolution::Param param;
    param.sparse = opr::RegionRestrictedConvolution::Param::Sparse::GROUP;

    auto make_graph = [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
        return {opr::RegionRestrictedConvolutionForward::make(
                inputs[0], inputs[1], inputs[2], inputs[3], param)};
    };

    Checker::RunOptions option;
    option.numdiff_eps = 0.1;
    option.numdiff_max_err = 1e-2;

    auto mask_gen = [&](HostTensorND& src) {
        HostTensorGenerator<dtype::Int32, RandomDistribution::CONSTANT> gen(1);
        src = *gen(src.shape(), src.comp_node());
    };
    auto uint8_mask_gen = [&](HostTensorND& src) {
        HostTensorGenerator<dtype::Uint8, RandomDistribution::CONSTANT> gen(1);
        src = *gen(src.shape(), src.comp_node());
    };
    auto float_gen = [&](HostTensorND& src) {
        HostTensorGenerator<dtype::Float32, RandomDistribution::GAUSSIAN> gen;
        src = *gen(src.shape(), src.comp_node());
    };

    auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
        auto opr =
                megdnn_naive_handle()
                        ->create_operator<megdnn::RegionRestrictedConvolutionForward>();
        opr->param() = param;
        TensorLayout dest_layout;
        opr->deduce_layout(
                inp[0]->layout(), inp[1]->layout(), inp[2]->layout(), inp[3]->layout(),
                dest_layout);
        std::vector<dt_byte> workspace(opr->get_workspace_in_bytes(
                inp[0]->layout(), inp[1]->layout(), inp[2]->layout(), inp[3]->layout(),
                dest_layout));
        dest[0].dtype(inp[0]->dtype())
                .comp_node(inp[0]->comp_node())
                .resize(dest_layout);
        opr->exec(
                inp[0]->as_megdnn(), inp[1]->as_megdnn(), inp[2]->as_megdnn(),
                inp[3]->as_megdnn(), dest[0].as_megdnn(),
                {workspace.data(), workspace.size()});
    };

    Checker(make_graph, fwd, CompNode::load("gpu0"))
            .set_input_dtype(0, dtype::Float32())
            .set_input_dtype(1, dtype::Float32())
            .set_input_dtype(2, dtype::Int32())
            .set_input_dtype(3, dtype::Int32())
            .set_input_generator(0, float_gen)
            .set_input_generator(1, float_gen)
            .set_input_generator(2, mask_gen)
            .set_input_generator(3, mask_gen)
            .set_input_allow_grad(2, false)
            .set_input_allow_grad(3, false)
            // {n,ic,ih,iw}, {oc,ic,fh,fw}, {n,ih,iw}, {n,oh,ow}
            .run({TensorShape{1, 2, 2, 2}, TensorShape{2, 1, 1, 2, 2},
                  TensorShape{1, 2, 2}, TensorShape{1, 1, 1}},
                 option)
            .run({TensorShape{1, 2, 3, 3}, TensorShape{2, 1, 1, 3, 3},
                  TensorShape{1, 3, 3}, TensorShape{1, 1, 1}},
                 option)
            .run({TensorShape{1, 4, 4, 4}, TensorShape{4, 1, 1, 2, 2},
                  TensorShape{1, 4, 4}, TensorShape{1, 3, 3}},
                 option)
            .run({TensorShape{2, 4, 8, 8}, TensorShape{4, 1, 1, 2, 2},
                  TensorShape{2, 8, 8}, TensorShape{2, 7, 7}},
                 option);

    Checker(make_graph, fwd, CompNode::load("gpu0"))
            .set_input_dtype(0, dtype::Float32())
            .set_input_dtype(1, dtype::Float32())
            .set_input_dtype(2, dtype::Uint8())
            .set_input_dtype(3, dtype::Uint8())
            .set_input_generator(0, float_gen)
            .set_input_generator(1, float_gen)
            .set_input_generator(2, uint8_mask_gen)
            .set_input_generator(3, uint8_mask_gen)
            .set_input_allow_grad(2, false)
            .set_input_allow_grad(3, false)
            // {n,ic,ih,iw}, {oc,ic,fh,fw}, {n,ih,iw}, {n,oh,ow}
            .run({TensorShape{1, 2, 4, 4}, TensorShape{2, 1, 1, 1, 1},
                  TensorShape{1, 4, 4}, TensorShape{1, 4, 4}},
                 option)
            .run({TensorShape{1, 2, 8, 8}, TensorShape{2, 1, 1, 1, 1},
                  TensorShape{1, 8, 8}, TensorShape{1, 8, 8}},
                 option)
            .run({TensorShape{1, 4, 8, 8}, TensorShape{4, 1, 1, 5, 5},
                  TensorShape{1, 8, 8}, TensorShape{1, 4, 4}},
                 option)
            .run({TensorShape{2, 4, 8, 8}, TensorShape{4, 1, 1, 1, 1},
                  TensorShape{2, 8, 8}, TensorShape{2, 8, 8}},
                 option);
}
#endif
