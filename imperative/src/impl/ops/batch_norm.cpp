/**
 * \file imperative/src/impl/ops/batch_norm.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/dnn/batch_norm.h"
#include "../blob_manager_impl.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"
#include "megbrain/imperative/graph_builder.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/utility.h"
#include "megbrain/imperative/proxy_graph_detail.h"
#include "megbrain/imperative/subgraph_detail.h"
#include "megbrain/tensor.h"

namespace mgb {
namespace imperative {
namespace {

EncodedSubgraph generate_batchnorm_backward_graph(DType dtype, CompNode device) {
    Subgraph::Builder<LogicalTensorDesc> builder{
            [](std::shared_ptr<OpDef> op, SmallVector<LogicalTensorDesc> inputs,
               size_t nr_outputs) {
                auto [outputs, validated] =
                        OpDef::infer_output_attrs_fallible(*op, inputs);
                mgb_assert(outputs.size() == nr_outputs, "nr_outputs mismatch");
                return outputs;
            }};
    auto f = [&](auto&& op, auto... args) {
        return builder.write_expr(
                op, Subgraph::vars_t({(Subgraph::var_t)args...}), 1)[0];
    };

    auto prod = Reduce::make(megdnn::param::Reduce(Reduce::Mode::PRODUCT, 0), true);
    auto sum = Reduce::make(megdnn::param::Reduce(Reduce::Mode::SUM), true);
    auto sub = Elemwise::make(Elemwise::Mode::SUB);
    auto mul = Elemwise::make(Elemwise::Mode::MUL);
    auto div = Elemwise::make(Elemwise::Mode::TRUE_DIV);
    auto floor_div = Elemwise::make(Elemwise::Mode::FLOOR_DIV);
    auto broadcast = Broadcast::make();

    auto c = [&](TensorPtr tensor, DType dtype) {
        auto result = builder.write_constant(
                tensor, {TensorLayout{tensor->dtype()}, tensor->comp_node()});
        if (tensor->dtype() != dtype) {
            result = f(TypeCvt::make(dtype), result);
        }
        return result;
    };
    auto ci = [&](megdnn::dt_int32 value) {
        return c(Tensor::make_scalar(DTypeScalar(value), device), dtype::Int32());
    };
    auto cf = [&](megdnn::dt_float32 value) {
        return c(Tensor::make_scalar(DTypeScalar(value), device), dtype);
    };

    auto desc = LogicalTensorDesc{TensorLayout{dtype}, device};
    auto x = builder.write_input(desc);
    auto y_grad = builder.write_input(desc);
    auto save_mean = builder.write_input(desc);
    auto save_invstd = builder.write_input(desc);
    auto weight = builder.write_input(desc);
    auto reserved = builder.write_input(desc);
    MGB_MARK_USED_VAR(reserved);

    // assert x.ndim == 4
    auto input_shape = f(GetVarShape::make(), x);
    auto channels = f(GetVarShape::make(1), x);
    auto reduce_shape = f(Concat::make(0, device), ci(1), channels, ci(1), ci(1));
    auto input_elems = f(prod, input_shape);
    auto reduce_size = f(floor_div, input_elems, channels);
    auto reduce_size_f = f(TypeCvt::make(dtype), reduce_size);
    auto mean = f(broadcast, save_mean, input_shape);
    auto invstd = save_invstd;
    auto norm = f(div, cf(1), reduce_size_f);
    auto output_grad_sum = f(sum, y_grad, reduce_shape);
    auto dot_p = f(sum, f(mul, y_grad, f(sub, x, mean)), reduce_shape);
    auto mean_grad = f(broadcast, f(mul, output_grad_sum, norm), input_shape);
    auto proj_scale =
            f(broadcast, f(mul, f(mul, dot_p, norm), f(mul, invstd, invstd)),
              input_shape);
    auto grad_scale = f(
            mul, f(broadcast, invstd, input_shape), f(broadcast, weight, input_shape));
    auto proj = f(mul, f(sub, x, mean), proj_scale);
    auto x_grad = f(mul, f(sub, f(sub, y_grad, proj), mean_grad), grad_scale);
    auto weight_grad = f(mul, dot_p, invstd);
    auto bias_grad = output_grad_sum;

    builder.add_outputs({weight_grad, bias_grad, x_grad});

    auto bn_backward = builder.encode();
    return bn_backward;
}

namespace bn {

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::BatchNorm>();
    return BatchNorm::make(node->param());
}

cg::OperatorNodeBase* apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& bn_opr = def.cast_final_safe<BatchNorm>();
    size_t nr_inp = inputs.size();
    mgb_assert(
            nr_inp == 3 || nr_inp == 5,
            "BatchNorm expects 3 or 5 inputs; got %lu actually", nr_inp);
    OperatorNodeConfig config{bn_opr.make_name()};
    if (nr_inp == 3) {
        return opr::BatchNorm::make(
                       inputs[0], inputs[1], inputs[2], bn_opr.param(), config)[0]
                .node()
                ->owner_opr();
    } else {
        return opr::BatchNorm::make(
                       inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                       bn_opr.param(), config)[0]
                .node()
                ->owner_opr();
    }
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& op_def = def.cast_final_safe<BatchNorm>();
    size_t nr_inp = inputs.size();
    mgb_assert(
            nr_inp == 3 || nr_inp == 5,
            "BatchNorm expects 3 or 5 inputs; got %lu actually", nr_inp);
    // need running mean/variance
    bool need_stat = (nr_inp == 5) && op_def.fwd_mode == BatchNorm::FwdMode::TRAINING;
    size_t nr_out = need_stat ? 6 : 4;
    SmallVector<LogicalTensorDesc> out_shapes(nr_out);
    auto&& i0 = inputs[0];
    auto&& i1 = inputs[1];
    // [running_mean, running_var,] save_mean, save_variance
    for (size_t i = 0; i < nr_out - 2; ++i) {
        out_shapes[i] = {i1.layout, i1.comp_node};
    }
    out_shapes[nr_out - 2] = {
            TensorLayout({0}, dtype::Byte()), i0.comp_node};  // reserve
    out_shapes[nr_out - 1] = {i0.layout, i0.comp_node};       // output
    return {out_shapes, out_shapes[nr_out - 1].layout.ndim != 0};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    auto&& op_def = def.cast_final_safe<BatchNorm>();
    auto&& comp_node = inputs[0]->comp_node();

    using TensorND = megdnn::TensorND;

    SmallVector<TensorND> inp_tensornds(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        inp_tensornds[i] = inputs[i]->dnn_tensor();
    }

    DnnOprCaller<megdnn::BN> dnn_opr(comp_node);
    dnn_opr.op->param() = op_def.param();

    TensorLayout src_layout = inputs[0]->layout();
    TensorLayout scale_layout = inputs[1]->layout();
    bool empty_input = src_layout.is_empty();
    size_t nr_inp = inputs.size();

    DeviceTensorND reserve;
    size_t sz = 0, rsz = 0;

    TensorLayout w_layout({sz}, dtype::Byte());
    TensorLayout r_layout({rsz}, dtype::Byte());

    if (!empty_input) {
        sz = dnn_opr.op->get_workspace_in_bytes(
                src_layout, src_layout, src_layout, src_layout, src_layout, src_layout,
                src_layout, src_layout, src_layout);
        rsz = dnn_opr.op->get_reserve_in_bytes(src_layout);

        w_layout = TensorLayout({sz}, dtype::Byte());
        r_layout = TensorLayout({rsz}, dtype::Byte());
    }
    auto dnn_wk = dnn_opr.create_workspace(w_layout);
    reserve = BlobManager::inst()->alloc_workspace_with_defrag(comp_node, r_layout);

    // alloc memory
    DeviceTensorND y =
            BlobManager::inst()->alloc_workspace_with_defrag(comp_node, src_layout);

    DeviceTensorND save_mean =
            BlobManager::inst()->alloc_workspace_with_defrag(comp_node, scale_layout);
    DeviceTensorND save_variance =
            BlobManager::inst()->alloc_workspace_with_defrag(comp_node, scale_layout);

    if (op_def.fwd_mode == ::megdnn::param::BN::FwdMode::INFERENCE) {
        if (!empty_input)
            dnn_opr.op->exec(
                    inp_tensornds[0], inp_tensornds[1], inp_tensornds[2],
                    inp_tensornds[3], inp_tensornds[4], save_mean.as_megdnn(),
                    save_variance.as_megdnn(), reserve.as_megdnn(), y.as_megdnn(),
                    dnn_wk);
        return {inputs[3], inputs[4], Tensor::make(reserve), Tensor::make(y)};
    } else {
        DeviceTensorND mean, variance;
        if (nr_inp == 5) {
            mean = BlobManager::inst()->alloc_workspace_with_defrag(
                    comp_node, scale_layout);
            variance = BlobManager::inst()->alloc_workspace_with_defrag(
                    comp_node, scale_layout);

            megdnn::RefPtr src_ptr1(
                    inp_tensornds[3].get_ref_ptr().get_ptr(), inputs[3]->offset());
            megdnn::RefPtr dst_ptr1(
                    mean.storage().get_ref_ptr(), mean.storage().offset(), false);
            comp_node.peer_copy_to_ref(
                    comp_node, dst_ptr1, src_ptr1, scale_layout.span().high_byte);

            megdnn::RefPtr src_ptr2(
                    inp_tensornds[4].get_ref_ptr().get_ptr(), inputs[4]->offset());
            megdnn::RefPtr dst_ptr2(
                    variance.storage().get_ref_ptr(), variance.storage().offset(),
                    false);
            comp_node.peer_copy_to_ref(
                    comp_node, dst_ptr2, src_ptr2, scale_layout.span().high_byte);

            if (!empty_input)
                dnn_opr.op->exec(
                        inp_tensornds[0], inp_tensornds[1], inp_tensornds[2],
                        mean.as_megdnn(), variance.as_megdnn(), save_mean.as_megdnn(),
                        save_variance.as_megdnn(), reserve.as_megdnn(), y.as_megdnn(),
                        dnn_wk);

            return {Tensor::make(mean),      Tensor::make(variance),
                    Tensor::make(save_mean), Tensor::make(save_variance),
                    Tensor::make(reserve),   Tensor::make(y)};
        }

        TensorLayout m_layout({0}, scale_layout.dtype);
        mean = BlobManager::inst()->alloc_workspace_with_defrag(comp_node, m_layout);
        variance =
                BlobManager::inst()->alloc_workspace_with_defrag(comp_node, m_layout);

        if (!empty_input) {
            dnn_opr.op->exec(
                    inp_tensornds[0], inp_tensornds[1], inp_tensornds[2],
                    mean.as_megdnn(), variance.as_megdnn(), save_mean.as_megdnn(),
                    save_variance.as_megdnn(), reserve.as_megdnn(), y.as_megdnn(),
                    dnn_wk);
        }

        return {Tensor::make(save_mean), Tensor::make(save_variance),
                Tensor::make(reserve), Tensor::make(y)};
    }
}

OP_TRAIT_REG(BatchNorm, BatchNorm, opr::BatchNorm)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();

}  // namespace bn

namespace bn_backward {

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::BatchNormBackward>();
    return BatchNormBackward::make(node->param());
}

VarNodeArray apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto& op = def.cast_final_safe<BatchNormBackward>();
    cg::SymbolVar x, y_grad, save_mean, save_variance, weight, reserve;
    x = inputs[0];
    y_grad = inputs[1];
    save_mean = inputs[2];
    save_variance = inputs[3];
    weight = inputs[4];
    if (inputs.size() == 6) {
        reserve = inputs[5];
    }
    return opr::BatchNormBackward::make(
                   x, y_grad, save_mean, save_variance, weight, reserve, op.param())[0]
            .node()
            ->owner_opr()
            ->usable_output();
}

EncodedSubgraph make_backward_graph(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad) {
    def.cast_final_safe<BatchNormBackward>();
    size_t nr_inputs = 6;
    size_t nr_outputs = 3;
    mgb_assert(inputs.size() == nr_inputs);
    mgb_assert(input_requires_grad.size() == nr_inputs);
    mgb_assert(output_has_grad.size() == nr_outputs);
    auto dtype = inputs[0].layout.dtype;
    auto device = inputs[0].comp_node;
    auto bn_backward = generate_batchnorm_backward_graph(dtype, device);
    auto bn_double_backward = subgraph_detail::make_backward_graph_from_forward(
            bn_backward, inputs, input_requires_grad, output_has_grad);
    return bn_double_backward;
}

OP_TRAIT_REG(BatchNormBackward, BatchNormBackward, opr::BatchNormBackward)
        .make_from_op_node(make_from_op_node)
        .apply_on_var_node(apply_on_var_node)
        .make_backward_graph(make_backward_graph)
        .fallback();
}  // namespace bn_backward

}  // anonymous namespace
}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
