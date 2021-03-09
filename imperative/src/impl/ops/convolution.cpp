/**
 * \file imperative/src/impl/ops/dnn/convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/dnn/convolution.h"

#include "../op_trait.h"

namespace mgb {
namespace imperative {

namespace { namespace convolution {
std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::Convolution>();
    return Convolution::make(node->param(), node->execution_policy());
}

auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& conv = static_cast<const Convolution&>(def);
    OperatorNodeConfig config{conv.make_name()};
    return opr::Convolution::make(inputs[0], inputs[1], conv.param(), conv.policy(), config);
}

OP_TRAIT_REG(Convolution, Convolution, opr::Convolution)
    .make_from_op_node(make_from_op_node)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // convolution

namespace { namespace convolution_backward_data {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& conv = static_cast<const ConvolutionBackwardData&>(def);
    OperatorNodeConfig config{conv.make_name()};
    if (inputs.size() == 2) {
        return opr::ConvolutionBackwardData::make(inputs[0], inputs[1], conv.param(), conv.policy(), config);
    } else {
        mgb_assert(inputs.size() == 3);
        return opr::ConvolutionBackwardData::make(inputs[0], inputs[1], inputs[2], conv.param(), conv.policy(), config);
    }
}

OP_TRAIT_REG(ConvolutionBackwardData, ConvolutionBackwardData)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // convolution_backward_data

namespace { namespace convolution3d {
std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node_) {
    auto* node = &node_->cast_final_safe<opr::Convolution3D>();
    return Convolution3D::make(node->param(), node->execution_policy());
}

auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& conv = static_cast<const Convolution3D&>(def);
    return opr::Convolution3D::make(inputs[0], inputs[1], conv.param(), conv.policy());
}

OP_TRAIT_REG(Convolution3D, Convolution3D, opr::Convolution3D)
    .make_from_op_node(make_from_op_node)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // convolution3d

}
}
