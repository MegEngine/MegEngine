#include "megbrain/comp_node_env.h"
#include "megbrain/graph/symbol_var.h"
#include "megbrain/opr/rand.h"
#include "megbrain/test/autocheck.h"
#include "megbrain/test/helper.h"
#include "megbrain/test/megdnn_helper.h"

#include "megdnn/basic_types.h"
#include "megdnn/oprs.h"

#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>

using namespace mgb;

namespace {
using Param = opr::MultiHeadAttn::Param;

void run_forward() {
    using Checker = AutoOprChecker<7, 4>;

    Param param;
    auto run_kernel = [&](size_t batch_size, size_t seq_qlen, size_t seq_klen,
                          size_t num_heads, size_t qsize, size_t ksize, size_t vsize,
                          size_t qproj_size, size_t kproj_size, size_t vproj_size,
                          size_t oproj_size, bool qbias, bool kbias, bool vbias,
                          bool obias, bool maybe_cudnn, bool have_mask, bool train) {
        auto make_graph =
                [&](const Checker::SymInpArray& inputs) -> Checker::SymOutArray {
            if (have_mask) {
                auto out = opr::MultiHeadAttn::make(
                        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], param);
                return {out[0], out[1], out[2], out[3]};
            } else {
                auto out = opr::MultiHeadAttn::make(
                        inputs[0], inputs[1], inputs[2], inputs[3], param);
                return {out[0], out[1], out[2], out[3]};
            }
        };

        auto fwd = [&](Checker::NumOutArray& dest, Checker::NumInpArray inp) {
            auto opr = MegDNNHandle::get(
                               CompNodeEnv::from_comp_node(CompNode::default_cpu()))
                               ->create_operator<megdnn::MultiHeadAttn>();
            opr->param() = param;
            TensorLayout o0, o1, o2, o3;
            TensorLayout empty_layout(inp[0]->dtype());
            megdnn::TensorND empty_tensor;
            if (!have_mask) {
                opr->deduce_layout(
                        inp[0]->layout(), inp[1]->layout(), inp[2]->layout(),
                        inp[3]->layout(), empty_layout, empty_layout, empty_layout, o0,
                        o1, o2, o3);

                dest[0].dtype(dtype::Float32())
                        .comp_node(inp[0]->comp_node())
                        .resize(o0);
                dest[1].dtype(dtype::Float32())
                        .comp_node(inp[0]->comp_node())
                        .resize(o1);
                dest[2].dtype(dtype::Byte()).comp_node(inp[0]->comp_node()).resize(o2);
                dest[3].dtype(dtype::Float32())
                        .comp_node(inp[0]->comp_node())
                        .resize(o3);

                size_t wk_size = opr->get_workspace_in_bytes(
                        inp[0]->layout(), inp[1]->layout(), inp[2]->layout(),
                        inp[3]->layout(), empty_layout, empty_layout, empty_layout,
                        dest[0].layout(), dest[1].layout(), dest[2].layout(),
                        dest[3].layout());
                std::unique_ptr<dt_byte[]> wk_store{new dt_byte[wk_size]};
                opr->exec(
                        inp[0]->as_megdnn(), inp[1]->as_megdnn(), inp[2]->as_megdnn(),
                        inp[3]->as_megdnn(), empty_tensor, empty_tensor, empty_tensor,
                        dest[0].as_megdnn(), dest[1].as_megdnn(), dest[2].as_megdnn(),
                        dest[3].as_megdnn(), {wk_store.get(), wk_size});
            } else {
                opr->deduce_layout(
                        inp[0]->layout(), inp[1]->layout(), inp[2]->layout(),
                        inp[3]->layout(), inp[4]->layout(), empty_layout, empty_layout,
                        o0, o1, o2, o3);

                dest[0].dtype(dtype::Float32())
                        .comp_node(inp[0]->comp_node())
                        .resize(o0);
                dest[1].dtype(dtype::Float32())
                        .comp_node(inp[0]->comp_node())
                        .resize(o1);
                dest[2].dtype(dtype::Byte()).comp_node(inp[0]->comp_node()).resize(o2);
                dest[3].dtype(dtype::Float32())
                        .comp_node(inp[0]->comp_node())
                        .resize(o3);

                size_t wk_size = opr->get_workspace_in_bytes(
                        inp[0]->layout(), inp[1]->layout(), inp[2]->layout(),
                        inp[3]->layout(), inp[4]->layout(), empty_layout, empty_layout,
                        dest[0].layout(), dest[1].layout(), dest[2].layout(),
                        dest[3].layout());
                std::unique_ptr<dt_byte[]> wk_store{new dt_byte[wk_size]};
                opr->exec(
                        inp[0]->as_megdnn(), inp[1]->as_megdnn(), inp[2]->as_megdnn(),
                        inp[3]->as_megdnn(), inp[4]->as_megdnn(), empty_tensor,
                        empty_tensor, dest[0].as_megdnn(), dest[1].as_megdnn(),
                        dest[2].as_megdnn(), dest[3].as_megdnn(),
                        {wk_store.get(), wk_size});
            }
        };

        auto gen = [&](HostTensorND& src) {
            HostTensorGenerator<dtype::Float32, RandomDistribution::GAUSSIAN> src_gen(
                    0.f);
            src = *src_gen(src.shape(), src.comp_node());
        };
        auto get_weight_len = [&](size_t num_heads, size_t qsize, size_t ksize,
                                  size_t vsize, size_t qproj_size, size_t kproj_size,
                                  size_t vproj_size, size_t oproj_size, bool qbias,
                                  bool kbias, bool vbias, bool obias) {
            size_t weight_len = 0;
            if (qproj_size > 0)
                weight_len += qsize * qproj_size + (qbias ? qproj_size : 0);
            if (kproj_size > 0)
                weight_len += ksize * kproj_size + (kbias ? kproj_size : 0);
            if (vproj_size > 0)
                weight_len += vsize * vproj_size + (vbias ? vproj_size : 0);
            if (oproj_size > 0 and vproj_size > 0)
                weight_len += vproj_size * oproj_size + (obias ? oproj_size : 0);
            else if (oproj_size > 0 and vproj_size == 0)
                weight_len += num_heads * vsize * oproj_size + (obias ? oproj_size : 0);
            return weight_len;
        };

        auto get_smscalar = [&](size_t num_heads, size_t qsize, size_t qproj_size) {
            size_t head_dim = qproj_size ? qsize / num_heads : qsize;
            return 1 / std::sqrt(head_dim);
        };
        param.need_weights = !maybe_cudnn;
        param.num_heads = num_heads;
        param.embeding_size = qsize;
        param.k_size = ksize;
        param.v_size = vsize;
        param.qbias = qbias && qproj_size;
        param.kbias = kbias && kproj_size;
        param.vbias = vbias && vproj_size;
        param.obias = obias && oproj_size;
        param.qproj_size = qproj_size;
        param.kproj_size = kproj_size;
        param.vproj_size = vproj_size;
        param.oproj_size = oproj_size;
        if (param.qproj_size == 0 && param.kproj_size > 0)
            param.embeding_size = param.embeding_size / num_heads;
        if (param.qproj_size > 0 && param.kproj_size == 0)
            param.k_size = param.k_size / num_heads;
        param.sm_scaler =
                get_smscalar(param.num_heads, param.embeding_size, param.qproj_size);
        size_t weight_len = get_weight_len(
                param.num_heads, param.embeding_size, param.k_size, param.v_size,
                param.qproj_size, param.kproj_size, param.vproj_size, param.oproj_size,
                param.qbias, param.kbias, param.vbias, param.obias);
        TensorShape attn_mask{};
        if (have_mask) {
            param.attn_mask_type =
                    Param::MultiHeadAttn::AttnMaskType::USER_DEFINED_MASK;
            param.tensor_combination_type =
                    Param::MultiHeadAttn::TensorCombinationType::ONLY_MASK;
            attn_mask = {batch_size * num_heads, seq_qlen, seq_klen};
        } else {
            param.attn_mask_type = Param::MultiHeadAttn::AttnMaskType::NO_MASK;
            param.tensor_combination_type =
                    Param::MultiHeadAttn::TensorCombinationType::NONE;
        }
        TensorShape query{batch_size, seq_qlen, param.embeding_size};
        TensorShape key{batch_size, seq_klen, param.k_size};
        TensorShape value{batch_size, seq_klen, param.v_size};
        TensorShape weight{weight_len};

        Checker::RunOptions option;
        option.outputs_max_err = 10e-3;
        option.numdiff_max_err = 10e-2;
        Checker checker{make_graph, fwd};

        checker.set_input_generator(0, gen);
        checker.set_input_generator(1, gen);
        checker.set_input_generator(2, gen);
        checker.set_input_generator(3, gen);
        checker.set_input_allow_grad(4, false);
        checker.set_input_allow_grad(5, false);
        checker.set_input_allow_grad(6, false);
        checker.set_output_allow_grad(1, false);
        checker.set_output_allow_grad(2, false);
        checker.set_output_allow_grad(3, false);

        checker.set_output_allow_check(1, false);
        checker.set_output_allow_check(2, false);
        checker.set_output_allow_check(3, false);

        if (train) {
            checker.set_input_allow_grad(0, true);
            checker.set_input_allow_grad(1, true);
            checker.set_input_allow_grad(2, true);
            checker.set_input_allow_grad(3, true);
            checker.set_output_allow_grad(0, true);
        } else {
            checker.set_input_allow_grad(0, false);
            checker.set_input_allow_grad(1, false);
            checker.set_input_allow_grad(2, false);
            checker.set_input_allow_grad(3, false);
            checker.set_output_allow_grad(0, false);
        }
        checker.run({query, key, value, weight, attn_mask, TensorShape{},
                     TensorShape{}},
                    option)
                .run({query, key, value, weight, attn_mask, TensorShape{},
                      TensorShape{}},
                     option)
                .run({query, key, value, weight, attn_mask, TensorShape{},
                      TensorShape{}},
                     option);
    };
    {
        size_t embeding_size = 4;
        size_t k_size = 4;
        size_t v_size = 4;
        for (size_t batch_size : {5})
            for (size_t seq_qlen : {1, 11})
                for (size_t seq_klen : {1, 12})
                    for (size_t num_heads : {1, 2, 4})
                        for (size_t qproj_size : {0, 4})
                            for (size_t kproj_size : {0, 4})
                                for (size_t vproj_size : {0, 8})
                                    for (size_t oproj_size : {0, 12})
                                        for (bool maybe_cudnn : {false, true})
                                            for (bool have_mask : {false, true}) {
                                                run_kernel(
                                                        batch_size, seq_qlen, seq_klen,
                                                        num_heads, embeding_size,
                                                        k_size, v_size, qproj_size,
                                                        kproj_size, vproj_size,
                                                        oproj_size, false, false, false,
                                                        false, maybe_cudnn, have_mask,
                                                        false);
                                            }
    }
    {
        size_t embeding_size = 4;
        size_t k_size = 5;
        size_t v_size = 6;
        size_t qproj_size = 4;
        size_t kproj_size = 4;
        size_t vproj_size = 4;
        size_t oproj_size = 4;
        for (size_t batch_size : {1, 10})
            for (size_t seq_qlen : {1, 11})
                for (size_t seq_klen : {1, 6, 12})
                    for (size_t num_heads : {1, 2, 4})
                        for (bool qbias : {false, true})
                            for (bool kbias : {false, true})
                                for (bool vbias : {false, true})
                                    for (bool obias : {false, true}) {
                                        run_kernel(
                                                batch_size, seq_qlen, seq_klen,
                                                num_heads, embeding_size, k_size,
                                                v_size, qproj_size, kproj_size,
                                                vproj_size, oproj_size, qbias, kbias,
                                                vbias, obias, false, false, false);
                                    }
    }
}

TEST(TestOprDNN, MultiHeadAttn) {
    REQUIRE_GPU(1);
    run_forward();
}

}  // anonymous namespace

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
