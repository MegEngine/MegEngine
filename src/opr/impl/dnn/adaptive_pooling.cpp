/**
 * \file src/opr/impl/dnn/adaptive_pooling.cpp
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
#include "../internal/megdnn_opr_wrapper.inl"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/utility.h"

#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs/nn.h"

using namespace mgb;
using namespace opr;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(AdaptivePoolingForward);
AdaptivePoolingForward::AdaptivePoolingForward(VarNode* src, VarNode* out_shape,
                                               const Param& param,
                                               const OperatorNodeConfig& config)
        : Super(OperatorNodeBaseCtorParam{src->owner_graph(),
                                          config,
                                          "adaptive_pooling",
                                          {src, out_shape}}) {
    init_megdnn_opr(*this, param);
    add_input({src, out_shape});
    outshape_by_symvar_enable(1, 1);
}

SymbolVar AdaptivePoolingForward::make(SymbolVar src, SymbolVar out_shape,
                                       const Param& param,
                                       const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<AdaptivePoolingForward>(
            src.node(), out_shape.node(), param, config);
}

void AdaptivePoolingForward::scn_do_execute() {
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),
                       output(0)->dev_tensor().as_megdnn(),
                       intl::get_megdnn_workspace_from_var(output().back()));
}

void AdaptivePoolingForward::outshape_by_symvar_do_get_output_shape(
        TensorShape& dest, const ShapeInferInfo& shpinfo) {
    TensorShape oshp2d;
    cg::copy_tensor_value_to_shape(oshp2d, *shpinfo.shpval_inp_val.at(0));
    auto src = shpinfo.shape_inp_shp.at(0);
    mgb_assert(src.ndim == 4 && oshp2d.ndim == 2,
               "shape mismatch for AdaptivePooling: src=%s, out2d=%s",
               src.to_string().c_str(), oshp2d.to_string().c_str());

    mgb_assert(param().format == Param::Format::NCHW,
               "AdaptivePooling only support NCHW");
    dest.ndim = 4;
    dest.shape[0] = src.shape[0];
    dest.shape[1] = src.shape[1];
    dest.shape[2] = oshp2d.shape[0];
    dest.shape[3] = oshp2d.shape[1];
}

size_t AdaptivePoolingForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    return megdnn_opr()->get_workspace_in_bytes(
            {input_shapes[0], this->input(0)->dtype(),
             this->input(0)->format()},
            {output_shapes[0], this->output(0)->dtype(),
             this->output(0)->format()});
}

void AdaptivePoolingForward::init_output_dtype() {
    output(0)->dtype(input(0)->dtype());
}

void AdaptivePoolingForward::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

void AdaptivePoolingForward::init_output_static_infer_desc() {
    Super::init_output_static_infer_desc();
    init_output_static_infer_desc_workspace(false);
}

void AdaptivePoolingForward::record_execute_deps(ExecDependencyArray& deps) {
    record_megdnn_opr(deps);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(AdaptivePoolingForward) {
    if (wrt_idx == 0) {
        // wrt src
        SymbolVar grad = AdaptivePoolingBackward::make(
                opr.input(0), opr.input(1), opr.output(0), out_grad[0],
                opr.param());
        return grad.node();
    } else {
        mgb_assert(wrt_idx == 1);
        return InvalidGrad::make(opr, wrt_idx);
    }
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(AdaptivePoolingBackward);
AdaptivePoolingBackward::AdaptivePoolingBackward(
        VarNode* src, VarNode* out_shape, VarNode* dst, VarNode* diff,
        const Param& param, const OperatorNodeConfig& config)
        : Super(OperatorNodeBaseCtorParam{src->owner_graph(),
                                          config,
                                          "adaptive_pooling_bwd",
                                          {src}},
                0, true) {
    init_megdnn_opr(*this, param);
    add_input({src, out_shape, dst, diff});
}

SymbolVar AdaptivePoolingBackward::make(SymbolVar src, SymbolVar out_shape,
                                        SymbolVar dst, SymbolVar diff,
                                        const Param& param,
                                        const OperatorNodeConfig& config) {
    return src.insert_single_output_opr<AdaptivePoolingBackward>(
            src.node(), out_shape.node(), dst.node(), diff.node(), param,
            config);
}

void AdaptivePoolingBackward::scn_do_execute() {
    megdnn_opr()->exec(input(0)->dev_tensor().as_megdnn(),
                       input(2)->dev_tensor().as_megdnn(),
                       input(3)->dev_tensor().as_megdnn(),
                       output(0)->dev_tensor().as_megdnn(),
                       intl::get_megdnn_workspace_from_var(output().back()));
}
size_t AdaptivePoolingBackward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    return megdnn_opr()->get_workspace_in_bytes(
            {input_shapes[0], input(0)->dtype(), input(0)->format()},
            {input_shapes[2], input(2)->dtype(), input(2)->format()},
            {input_shapes[3], input(3)->dtype(), input(3)->format()},
            {output_shapes[0], output(0)->dtype(), output(0)->format()});
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
